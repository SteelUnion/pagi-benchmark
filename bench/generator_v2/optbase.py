"""
optbase —— 选项底层构造（每项独立随机函数）

════════════════════════════════════════════════════════════════
【构造：y → x（逆运算，两层，两个不同的反函数）】
════════════════════════════════════════════════════════════════

    y 指派（一个数）

        第 1 步： f1 从 59 个函数里随机抽
                  c1 算出来，使 c1·y 落进 f1 的 y_range
                  v = f1⁻¹( c1 · y )           ← 反函数

        第 2 步： f2 从 59 个函数里【再随机抽一次】，和 f1 独立
                  c2 算出来，使 c2·v 落进 f2 的 y_range
                  x = f2⁻¹( c2 · v )           ← 反函数

    ⚠️ 弗兰克：「两步定义成不同的反函数。」
    ⇒ f1 和 f2 是两个独立抽取的反函数，不是同一个。

════════════════════════════════════════════════════════════════
【正向：x → y（正运算）】
════════════════════════════════════════════════════════════════

    y = f1( f2( x ) / c2 ) / c1

    ⚠️ 同一个 c，逆的时候乘、正的时候除。
    ⚠️ 函数用 f1、f2 的【正运算】，和老代码 _mk 里成对的那个。

════════════════════════════════════════════════════════════════
【老代码的配对规则（照搬）】
════════════════════════════════════════════════════════════════

    registry._mk(fwd_s, inv_s, search):
        有解析逆 → return lambda y: inv_s(y)
        没有     → return lambda y: solve_inverse(fwd_s, y, search)

    ⇒ 每个函数自带 (正运算, 逆运算) 一对，packs 里已经成对。
      p._fwd  / p._inv   就是这一对。

    ⚠️ solve_inverse 只吃【一维张量】，而且自己处理多解
       （找所有变号区间 → 随机挑一个根 → 二分）。
       所以每次调用只传一维，不传二维。

════════════════════════════════════════════════════════════════
【随机性（弗兰克反复强调）】
════════════════════════════════════════════════════════════════

    60 个选项，每项都不同、没规律：
        · f1、f2 各自从 59 个里等概率抽，每项独立
        · c1、c2 各自独立算
        · x 各自不同
    ⚠️ 不允许「60 项共用同一对函数」—— 那是偷工。

════════════════════════════════════════════════════════════════
【禁止搭配】
════════════════════════════════════════════════════════════════

    弗兰克：「这种的没办法，我建议是把这个搭配 ban 掉，
              因为我们的搭配很多，要有个 ban 的功能。」

    实测出来的坏搭配见 BAN_PAIRS。
"""

from __future__ import annotations

import hashlib
import math
import sys

import torch

sys.path.insert(0, r'<BENCH>\generator_v2')

import config
import packs
import skeleton
from mathpack import DT, DEV

__all__ = [
    'P_OPTION', 'BAN_PAIRS', 'func_pool', 'func_code', 'pick_c',
    'Option', 'build_one', 'build_options', 'build_from_batch', 'self_test',
]

P_OPTION = 'O'


def opt_pid(i: int) -> str:
    return '%s%03d' % (P_OPTION, i)


# ══════════════════════════════════════════════════════════════════
# 一、禁止搭配
# ══════════════════════════════════════════════════════════════════

BAN_PAIRS: set[tuple[str, str]] = {
    ('log', 'pow3'),
}

# ⚠️ 整函数 ban（2026-11-08 实测：两边全败）
#
#   判据来自 trace/enum_1794155718.jsonl（3481 对 × 2 个 y 的完整枚举）：
#       作前层 59/59 全败、作后层 59/59 全败 → 和谁都配不上
BAN_FUNCS: set[str] = {
    # ⚠️ 2026-11-09 更正：昨天 ban 的 ellipe 实测正常（8/8）。
    #    它当时是按「同时判 search 和 y_range」的错判据筛掉的。
    'erfcx',       # 实测：在自己的 y_range 上 8 个点全部 f⁻¹ 给 nan
}

# ⚠️ 只禁【作后层】（作前层还能用）
#
#   ellipk  后 59/59 全败、前 9/59 → 只禁后
#   expi    后 47/59 全败          → 只禁后
#   acos    后 28/59 全败          → 只禁后
BAN_AS_F2: set[str] = set()   # ⚠️ 2026-11-09 全部解禁（实测都正常）


# ⚠️ 整函数禁用（2026-11-08 实测出来的）
#
#   第一轮：反解在【多个 y 上一个都对不上】的
#   判据：v = f⁻¹(c·y) 之后，回代 f(v)/c 必须 == y；做不到的进来。
#
#   ⚠️ 这些不是我们代码的问题，是老代码 solve_inverse 的局限：
#      它靠「扫描网格找变号 + 二分」，根落在扫描缝隙、擦边、
#      或者函数在区间上近似常数时，就找不到根，返回 nan。



def is_banned(f1: str, f2: str) -> bool:
    return (f1, f2) in BAN_PAIRS or (f2, f1) in BAN_PAIRS


# ══════════════════════════════════════════════════════════════════
# 二、函数池与编码
# ══════════════════════════════════════════════════════════════════

def func_pool() -> list[str]:
    """
    ⭐ 可用函数池：59 个（inverse_trusted=True）。

    ⚠️ 走 config.func_pool()，不自己列。
    """
    names = [n for n in config.func_pool()]
    out = []
    for n in names:
        if n in BAN_FUNCS:
            continue
        p = packs.get(n)
        if p is not None and p.inverse_trusted:
            out.append(n)
    return out


def func_code(name: str) -> str:
    """⭐ 函数名 → 编码（E001..），带反向校验。"""
    p = packs.get(name)
    if p is None:
        raise KeyError('函数 %r 不在数理包里' % (name,))
    code = getattr(p, 'code', '')
    back = packs.BY_CODE.get(code)
    if back is None or back.name != p.name:
        raise ValueError('函数编码错位：%s → %s' % (name, code))
    return code


# ══════════════════════════════════════════════════════════════════
# 三、值域 / 定义域
# ══════════════════════════════════════════════════════════════════

def y_range(p, pad: float = 0.02) -> tuple[float, float]:
    """
    ⭐ f⁻¹ 能算的输入范围 = f 的【值域】。

    ⚠️⚠️ 2026-11-08 弗兰克纠正的根本性错误：

        p._inv(x) 解的是  f(v) = x
        ⇒ x 是 f 的【输出】，必须落在 f 的 y_range 里。
        ⇒ 判据是 y_range，不是 search。

        我为了「统一写法」把它改成 search，方向反了：
            i0 search=[0.01,20]，但值域是 [1.02, 4.27e7]
            y=1 时 c·y=1 明明在 search 里，却够不着值域 → 反解 nan
    """
    lo, hi = p.y_range
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (-float('inf'), float('inf'))
    # ⚠️ 2026-11-08 弗兰克：余量要【按比例】留，不能按区间总宽的百分比。
    #    [0.000102, 9800] 按绝对差留 2% 会砍成 [196, 9604]，
    #    把能吃的小值段整段切掉 —— 这正是 1752 对「进不了」的根因。
    if lo > 0 and hi > 0:
        return (lo * (1 + 0.02), hi * (1 - 0.02))
    if lo < 0 and hi < 0:
        return (lo * (1 - 0.02), hi * (1 + 0.02))
    w = (hi - lo) * 0.02
    return (lo + w, hi - w)


def _search_of(p, pad: float = 0.02) -> tuple[float, float]:
    """⭐ f 能接受的输入范围（search，按比例留余量）。"""
    lo, hi = p.search
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return (-float('inf'), float('inf'))
    # ⚠️ 2026-11-08 弗兰克：余量要【按比例】留，不能按区间总宽的百分比。
    #    [0.000102, 9800] 按绝对差留 2% 会砍成 [196, 9604]，
    #    把能吃的小值段整段切掉 —— 这正是 1752 对「进不了」的根因。
    if lo > 0 and hi > 0:
        return (lo * (1 + 0.02), hi * (1 - 0.02))
    if lo < 0 and hi < 0:
        return (lo * (1 - 0.02), hi * (1 + 0.02))
    w = (hi - lo) * 0.02
    return (lo + w, hi - w)


# ⚠️ 细调档：2 的 m 次方
#    弗兰克 2026-11-09：
#    「直接用第二个 c，不需要微调的时候，m=0，这样 2 的 m 次方就是 1，
#      需要的时候，微调到一个精度损失不严重的值。」
#
#    ⇒ c = ±1 × 10^k × 2^m   —— 一个数，不是两个系数
#        m = 0 → 退化成粗调 ±10^k
#        m ≠ 0 → 半个数量级内微调
#    ⚠️ 2^m 在 float64 里【精确】（整数次幂），不引入十进制那种舍入。
FINE_EXP = (0, -1, 1, -2, 2, -3, 3)


def c_candidates(fine: bool = True) -> list[float]:
    """
    ⭐ c 的全部候选：±1 × 10^k × 2^m，按 |k| 递增、|m| 递增排。

    ⚠️ 顺序 = 优先级：先 m=0（粗调），再 |m|=1,2,3（细调）。
       这样「取第一个合格的」自然是「最温和的那个」。
    ⚠️ fine=False 时只用 m=0，等价于老的 ±10^k 表。
    """
    ks = sorted(range(-30, 31), key=abs)
    ms = FINE_EXP if fine else (0,)
    out = []
    for k in ks:
        b = 10.0 ** k
        for m in ms:
            v = b * (2.0 ** m)
            out.append(v)
            out.append(-v)
    return out


# ══════════════════════════════════════════════════════════════════
# 四、挑 c（逐个算，一维进一维出）
# ══════════════════════════════════════════════════════════════════

def pick_c(v: float, p, lo: float, hi: float,
           rng: torch.Generator | None = None) -> tuple[float, bool]:
    """
    ⭐ 给一个数 v 挑 c = ±10^k，使 c·v 落进 [lo, hi]。

    四级判定（弗兰克 2026-11-08）：
        ① v 本来就在区间里 → c = 1
        ② 上下界能用       → 留下
        ③ 中间有档         → 留下
        ④ 多个合格 → 随机挑一个；零个 → 失败

    ⚠️ 逐个调用 p._inv（一维），因为 solve_inverse 只吃一维，
       而且它自己会处理多解（随机取一个根）。
    ⚠️ 判据是「c·v 在区间内」，这是【输入】，不是函数输出。

    返回 (c, ok)。
    """
    if not math.isfinite(v):
        return (1.0, False)

    # ① 本来就在区间里
    if lo <= v <= hi:
        return (1.0, True)

    good: list[float] = []
    for c in c_candidates():
        arg = v * c
        if arg == 0 or abs(arg) < 1e-12:      # ⚠️ 不接受 0
            continue
        if not (lo <= arg <= hi):
            continue
        good.append(c)

    if not good:
        return (1.0, False)

    # ⚠️ 2026-11-09 弗兰克：「先验证，缩小范围后满足再随机。」
    #   从合格 c 里逐个真算 f⁻¹ 并回代，只留真正走通通的。
    #   ⚠️ 最多验 8 个（随机抽样），避免碰上数值反解的函数太慢。
    import random as _r
    cand = good if len(good) <= 8 else _r.sample(good, 8)
    verified = [c for c in cand if _usable_c(v, p, c)]
    if verified:
        pick_from = verified
    else:
        # 一个都没验过 → 退回原始的合格集（不静默失败，后续会报往返误差）
        pick_from = good

    if rng is None:
        return (pick_from[0], True)
    # ⚠️ rng 可能是 CPU 生成器，别硬塞 cuda device
    _dev = rng.device if hasattr(rng, 'device') else 'cpu'
    i = int(torch.randint(0, len(pick_from), (1,), generator=rng,
                          device=_dev)[0])
    return (pick_from[i], True)


def _inv_one(p, arg: float) -> float:
    """
    ⭐ 一维调一次 f⁻¹。

    ⚠️ solve_inverse 只吃一维张量；多解由它内部随机取一个。
    """
    with torch.no_grad():
        t = torch.tensor([arg], device=DEV, dtype=DT)
        out = p._inv(t)
    return float(out[0])


def _fwd_one(p, arg: float) -> float:
    with torch.no_grad():
        t = torch.tensor([arg], device=DEV, dtype=DT)
        out = p._fwd(t)
    return float(out[0])


# ══════════════════════════════════════════════════════════════════
# 五、一个选项
# ══════════════════════════════════════════════════════════════════

class Option:
    """⭐ 一个选项的底。⚠️ 全部固定值。"""

    __slots__ = ('idx', 'x', 'y', 'v', 'u', 'arg1', 'arg2', 'trace',
                 'f1', 'f2', 'f1_code', 'f2_code', 'c1', 'c2',
                 'err', 'ok', 'why', 'pid', 'digest')

    def __init__(self, idx: int):
        self.idx = idx
        self.pid = opt_pid(idx)
        self.x = float('nan')
        self.y = float('nan')
        self.v = float('nan')      # f1⁻¹ 的结果（真正的中间量）
        self.u = float('nan')
        self.arg1 = float('nan')   # c1·y，f1⁻¹ 的输入
        self.arg2 = float('nan')   # c2·v，f2⁻¹ 的输入
        self.f1 = ''
        self.f2 = ''
        self.f1_code = ''
        self.f2_code = ''
        self.c1 = 1.0
        self.c2 = 1.0
        self.err = float('inf')
        self.ok = False
        self.why = ''
        self.digest = ''
        self.trace = {}

    @property
    def codes(self) -> dict:
        p = self.pid
        return {'O': p, 'X': 'X-%s' % p, 'Y': 'Y-%s' % p,
                'F1': 'F1-%s' % p, 'F2': 'F2-%s' % p,
                'C1': 'C1-%s' % p, 'C2': 'C2-%s' % p}

    def payload(self) -> str:
        return '|'.join([self.pid, self.f1, self.f1_code,
                         self.f2, self.f2_code,
                         '%.17g' % self.c1, '%.17g' % self.c2])

    def make_digest(self, prev: str = '') -> str:
        self.digest = hashlib.sha256(
            (prev + '||' + self.payload()).encode('utf-8')).hexdigest()
        return self.digest

    # ── 式子 ──────────────────────────────────────

    def _term(self, fname: str, arg: str) -> str:
        sk = skeleton.get_skeleton(fname)
        return (sk.plain % arg) if (sk is not None and sk.plain) \
            else '%s(%s)' % (fname, arg)

    def formula(self) -> str:
        """
        ⭐ 模型要解的式子：

            Y = f1( f2( X ) ÷ c2 ) ÷ c1
        """
        inner = self._term(self.f2, 'X')
        if self.c2 != 1.0:
            inner = '(%s) ÷ %g' % (inner, self.c2)
        body = self._term(self.f1, inner)
        if self.c1 != 1.0:
            body = '(%s) ÷ %g' % (body, self.c1)
        return 'Y = ' + body

    def public_view(self) -> dict:
        """⚠️ 不给 y 的值，只给编码。"""
        return {'pid': self.pid, 'y_code': self.codes['Y'],
                'x': self.x, 'formula': self.formula()}

    def show(self) -> str:
        L = ['%s  x=%.12g  y=%.12g' % (self.pid, self.x, self.y)]
        L.append('   f1=%s(%s) c1=%g   f2=%s(%s) c2=%g'
                 % (self.f1, self.f1_code, self.c1,
                    self.f2, self.f2_code, self.c2))
        L.append('   ' + self.formula())
        if not self.ok:
            L.append('   ❌ ' + self.why)
        return '\n'.join(L)


# ══════════════════════════════════════════════════════════════════
# 六、造一项
# ══════════════════════════════════════════════════════════════════

def O_fwd(p, val: float) -> float:
    """⭐ 一维调一次 f（trace 用）。"""
    return _fwd_one(p, val)


def _usable_c(v: float, p, c: float, tol: float = 1e-9) -> bool:
    """
    ⭐ 这个 c 配这个值，能不能正逆走通？

    照老代码 anchor.usable_anchor 的做法：
        v' = f⁻¹(c · v)
        v'' = f(v')              应当 ≈ c·v
        
    ⚠️ 弗兰克 2026-11-09：「先验证，缩小范围后满足再随机。」
    ⚠️ 这一步是专门针对「c 太小 → 值贴 0 → 反解在远尾精度不足」的。
       不验的话，挑到 1e-10 这种 c，回代误差能到 1e-5。
    """
    if not math.isfinite(v) or v == 0:
        return False
    arg = c * v
    if arg == 0 or abs(arg) < 1e-12:
        return False
    try:
        x = _inv_one(p, arg)
        if not math.isfinite(x):
            return False
        back = _fwd_one(p, x)
        if not math.isfinite(back):
            return False
        return abs(back - arg) / abs(arg) < tol
    except Exception:                                  # noqa: BLE001
        return False


def _can_scale_into(v: float, lo: float, hi: float) -> bool:
    """
    ⭐ 有没有某个 c = ±10^k 使 c·v 落进 [lo, hi]？

    ⚠️ 这是「抽函数时就校验」的快速判据：不看 f⁻¹ 能不能算，
       只看区间缩放可行性。用来在早期挡掉注定失败的 (f1, f2)。
    """
    if not math.isfinite(v) or v == 0:
        return False
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return True
    for c in c_candidates():
        a = c * v
        if a == 0 or abs(a) < 1e-12:
            continue
        if lo <= a <= hi:
            return True
    return False


def build_one(idx: int, y: float, f1: str, f2: str,
              rng: torch.Generator | None = None) -> Option:
    """
    ⭐ 造一个选项：y 给定，f1/f2 给定，算出 c1、c2、x。

        v = f1⁻¹( c1 · y )
        x = f2⁻¹( c2 · v )
        验 y' = f1( f2(x) / c2 ) / c1  ?=  y
    """
    op = Option(idx)
    op.y = float(y)
    op.f1, op.f2 = f1, f2
    try:
        op.f1_code = func_code(f1)
        op.f2_code = func_code(f2)
    except Exception as e:                            # noqa: BLE001
        op.why = '函数编码出错：%s' % e
        return op

    p1, p2 = packs.get(f1), packs.get(f2)

    # ── 第 1 步：v = f1⁻¹(c1·y) ────────────────────
    lo1, hi1 = y_range(p1)
    if op.y == 0 or abs(op.y) < 1e-12:
        op.why = 'RANGE_UNFIXABLE y=%.6g 太接近 0' % op.y
        return op
    c1, ok1 = pick_c(op.y, p1, lo1, hi1, rng)
    if not ok1:
        op.why = ('y=%.6g 乘任何 ±10^k 都进不了 %s 的 y_range '
                  '[%.4g, %.4g]' % (op.y, f1, lo1, hi1))
        return op
    op.c1 = c1
    # ⚠️⚠️ 2026-11-09 修：op.v 原来存的是「反解函数的输入」c1*y，
    #    不是反解结果 —— 日志验算全对不上就是因为记错了字段。
    #    现在分开：arg1 = 输入，v = 反解结果。
    op.arg1 = c1 * op.y
    v = _inv_one(p1, op.arg1)
    op.v = v                                   # 真正的第 1 层产物
    if not math.isfinite(v):
        op.why = '%s⁻¹(%.6g) 得不出有限值' % (f1, op.arg1)
        return op

    # ⚠️⚠️ 2026-11-09 弗兰克：
    #   「生成 f1/f2 随机的时候，就校验一下，f1 反函数计算出的 v，
    #    能不能通过 f2 的值域，这样可以在生成那层就避免出错。」
    #
    #   ⇒ 挑完 c1、算出 v 之后，立刻校验「有没有 c2 让 c2·v 进 f2.y_range」。
    #     进不去 → 这一对 (f1, f2) 不可用，让上层换一对重新抽，
    #     而不是拖到挑 c2 才发现（那时 c1、v 都白算了）。
    _lo2, _hi2 = y_range(p2)
    if not _can_scale_into(v, _lo2, _hi2):
        op.why = ('RANGE_UNFIXABLE v=%.6g 乘任何 ±10^k 都进不了 %s '
                  'y_range [%.4g, %.4g]' % (v, f2, _lo2, _hi2))
        return op

    # ── ⭐ 升级校验（弗兰克 2026-11-08）─────────────────
    #
    #   「应该是就只去看第一层的输出能不能进第二层定义。
    #     因为这两个函数自身有系数帮我们看着，不需要看。」
    #
    #   ⇒ 不再做「两函数 y_range 交集」校验；
    #     改成「第 1 层的输出 v 能不能进第 2 层的 search」。
    #     ⚠️ 判 search（f2 能吃什么输入），不是 y_range。
    # ⚠️⚠️ 2026-11-08 弗兰克纠正：先缩放，再看能不能进。
    #
    #   我原来直接判「v 进不进 f2.search」，进不去就失败 —— 错了。
    #   系数 c2 就是干这个的：先把 v 缩放，缩完能进就能算。
    #   只有【所有 c2 都缩不进去】才是真正不可用。
    s_lo, s_hi = _search_of(p2)
    lo2, hi2 = y_range(p2)

    # 枚举 c2：要求 c2·v 既在 f2 的 search 里，又在它的 y_range 里
    #
    # ⚠️ 2026-11-08 弗兰克：「给他一个设置不等于零的情况呗，
    #    不然搞出 0 不好计算。」
    #    ⇒ 缩放后的值必须【非零】且不能贴 0，否则 f⁻¹ 会退化成常数
    #      （log⁻¹(0)=1、pow2⁻¹(0)=0），后面全废。
    ZERO_GUARD = 1e-12
    good_c2: list[float] = []
    if math.isfinite(v) and v != 0:
        for c in c_candidates():
            a = c * v
            if a == 0 or abs(a) < ZERO_GUARD:
                continue
            # ⚠️ 2026-11-09：只判 y_range，不判 search
            if lo2 <= a <= hi2 and a != 0:
                good_c2.append(c)
    if not good_c2:
        op.why = ('RANGE_UNFIXABLE v=%.6g 乘任何 ±10^k 都进不了 %s '
                  'search[%.4g,%.4g] ∩ y_range[%.4g,%.4g]'
                  % (v, f2, s_lo, s_hi, lo2, hi2))
        return op
    c2 = good_c2[int(torch.randint(0, len(good_c2), (1,), generator=rng)[0])] \
        if rng is not None else good_c2[0]
    op.c2 = c2
    op.arg2 = c2 * v
    x = _inv_one(p2, op.arg2)
    if not math.isfinite(x):
        op.why = '%s⁻¹(%.6g) 得不出有限值' % (f2, op.arg2)
        return op
    op.x = x

    # ── 验算：正运算走回去 ─────────────────────────
    a = _fwd_one(p2, x) / c2
    y2 = _fwd_one(p1, a) / c1
    op.trace = {'f1': f1, 'f2': f2, 'f1_code': op.f1_code,
                'f2_code': op.f2_code, 'y': op.y,
                'c1': c1, 'arg1': op.arg1, 'v': v,
                'c2': c2, 'arg2': op.arg2, 'x': x,
                'f2x': O_fwd(p2, x), 'y2': y2}
    if not math.isfinite(y2) or op.y == 0:
        op.why = '正运算得不出有限值'
        return op
    op.err = abs(y2 - op.y) / abs(op.y)
    if op.err >= 1e-9:
        op.why = '往返误差 %.3e 超容差' % op.err
        return op

    op.ok = True
    return op


# ══════════════════════════════════════════════════════════════════
# 七、批量造
# ══════════════════════════════════════════════════════════════════

def build_from_batch(batch: dict, seed: int = 20261108,
                     max_try: int = 40) -> list[Option]:
    """
    ⭐ 直接从 assign2.build_batch() 的结果造 60 个选项。

    前 10 项是真值（链值），后 50 项是干扰。
    ⚠️ 每项独立随机抽 f1、f2，抽到不合规（v 进不了 f2）就换一组重抽。
    """
    all_ys = batch['ys_all'].tolist()
    return build_options(all_ys, seed=seed, max_try=max_try)


def build_options(ys: list[float], seed: int = 20261108,
                  max_try: int = 40) -> list[Option]:
    """
    ⭐ 一批 y → 一批选项，每项独立随机抽 f1、f2。

    ⚠️ 60 项每项都不同 —— 不允许共用同一对函数。
    ⚠️ 抽到坏搭配或算不出来的，换一组重抽（最多 max_try 次），
       仍不行就留下失败记录，不静默跳过。
    """
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)

    pool = func_pool()
    pool_f1 = [n for n in pool]
    pool_f2 = [n for n in pool if n not in BAN_AS_F2]

    # ⚠️ 2026-11-09 弗兰克：「每生成一个用掉一种组合，
    #    在其他的里面继续随机，这样这种组合应该不会频繁出现。」
    #
    #   ⇒ 建一个组合池，抽走就不放回。
    #     60 项最多用 60 种组合，不重复。
    #     ⚠️ 只管这一道题，跨题重新开始。
    combo_pool = [(a, b) for a in pool_f1 for b in pool_f2
                  if not is_banned(a, b)]
    n_combo = len(combo_pool)
    used_idx: set[int] = set()

    out: list[Option] = []
    prev = ''

    for i, y in enumerate(ys):
        op = None
        for _try in range(max_try):
            # 还没用过的组合
            avail = [k for k in range(n_combo) if k not in used_idx]
            if not avail:
                used_idx.clear()          # 池子用完了才允许重用
                avail = list(range(n_combo))
            k = avail[int(torch.randint(0, len(avail), (1,),
                                        generator=rng)[0])]
            f1, f2 = combo_pool[k]
            if is_banned(f1, f2):
                continue
            cand = build_one(i + 1, float(y), f1, f2, rng)
            if cand.ok:
                op = cand
                used_idx.add(k)          # ⚠️ 用掉了
                break
        if op is None:
            op = Option(i + 1)
            op.y = float(y)
            op.why = '试了 %d 次都没凑出可用组合' % max_try
        prev = op.make_digest(prev)
        out.append(op)
    return out


# ══════════════════════════════════════════════════════════════════
# 八、自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0

    def ck(name, cond, extra: str = ''):
        nonlocal fails
        if verbose:
            print(('  ✅ ' if cond else '  ❌ ') + name
                  + (('   ' + extra) if extra else ''))
        if not cond:
            fails += 1

    if verbose:
        print('=' * 96)
        print('optbase 自检（每项独立随机函数）')
        print('=' * 96)
        print('-' * 96)

    # ① 函数池
    pool = func_pool()
    n_all = len([n for n in config.func_pool()
                 if packs.get(n) and packs.get(n).inverse_trusted])
    ck('函数池 = %d - ban 数' % n_all,
       len(pool) == n_all - len(BAN_FUNCS),
       '%d 个（总 %d，ban %d）' % (len(pool), n_all, len(BAN_FUNCS)))
    ck('exp → E008', func_code('exp') == 'E008')
    badc = []
    for nm in pool:
        try:
            func_code(nm)
        except Exception:                              # noqa: BLE001
            badc.append(nm)
    ck('59 个函数编码全部往返一致', not badc, str(badc[:3]))

    # ② 单次构造
    op = build_one(1, 2.0, 'exp', 'pow2')
    ck('单次构造成功', op.ok, op.why[:40])
    ck('单次构造往返误差 < 1e-9', op.err < 1e-9, '%.2e' % op.err)

    # ③ 手算核对
    if op.ok:
        p1, p2 = packs.get('exp'), packs.get('pow2')
        v = _inv_one(p1, op.c1 * op.y)
        x = _inv_one(p2, op.c2 * v)
        ck('手算 x 与代码一致', abs(x - op.x) / abs(x) < 1e-12,
           'x=%.10g' % x)
        y2 = _fwd_one(p1, _fwd_one(p2, op.x) / op.c2) / op.c1
        ck('手算 y 与代码一致', abs(y2 - op.y) / abs(op.y) < 1e-12,
           'y=%.10g' % y2)

    # ④ 两个反函数不同
    rng = torch.Generator(device='cpu')
    rng.manual_seed(3)
    diff = 0
    for _ in range(20):
        a = int(torch.randint(0, len(pool), (1,), generator=rng)[0])
        b = int(torch.randint(0, len(pool), (1,), generator=rng)[0])
        if pool[a] != pool[b]:
            diff += 1
    ck('随机抽两次，多数时候两函数不同', diff >= 15, '%d/20' % diff)

    # ⑤ 60 项，每项独立
    ys = [1.0 + 0.13 * i for i in range(60)]
    ops = build_options(ys)
    n_ok = sum(1 for o in ops if o.ok)
    ck('60 项造出', len(ops) == 60)
    ck('通过 >= 50 项', n_ok >= 50, '%d/60' % n_ok)
    n_pair = len({(o.f1, o.f2) for o in ops if o.ok})
    ck('60 项的函数对不是同一对', n_pair > 20, '共 %d 种组合' % n_pair)

    # ⑥ 式子能算出 y
    badf = []
    for o in ops:
        if not o.ok:
            continue
        try:
            p1, p2 = packs.get(o.f1), packs.get(o.f2)
            y2 = _fwd_one(p1, _fwd_one(p2, o.x) / o.c2) / o.c1
            if abs(y2 - o.y) / abs(o.y) > 1e-6:
                badf.append(o.pid)
        except Exception:                              # noqa: BLE001
            badf.append(o.pid)
    ck('按 (f1, f2, c1, c2, x) 能算出 y', not badf, str(badf[:3]))

    # ⑦ 不给 y 值
    pv = ops[0].public_view()
    ck('对外视图不含 y 值', 'y' not in pv)
    ck('对外视图有 y 编码', pv['y_code'] == 'Y-O001')

    if verbose:
        print('-' * 96)
        for o in ops[:4]:
            print(o.show())
            print()
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
