"""
锚点层（Anchor）—— 把「区间对区间」换成「单值对单值」

════════════════════════════════════════════════════════════════
【为什么要有这一层】（研究者 2026-11-06 22:20 定）
════════════════════════════════════════════════════════════════

    原话：

        「你说的这个问题真实存在，不能靠简单系数解决，我的看法是，
          要么把这个系数生成的值每次取一个单个值，比如我们改一下叫
          ellipe max，然后这种区间的都取上界，要么取下界，
          这样的话，我们可以针对单个数进行调整。
          如果我们老是调区间会很麻烦，因为区间有的是包含关系，不好调。
          如果我们把区间变成单值，并且正逆向能运算的话，那么就好调了。」

    ⭐ 问题所在（实测）：

        区间对区间：
            ellipe 输出 [1.03, 1.57]  ×10⁻¹  →  [0.103, 0.157]
            ellipk 能接 [0.01, 0.98]
            → 整个区间落进去了，看起来通。但换个 cn 就出现「部分重叠」，
              部分重叠意味着**有的样本能过、有的不能**，
              而区间还有包含关系、交叉关系，调一个就动一串。

        单值对单值：
            只问「这个数进不进得去」，进或不进，二元。
            进不去就把 cnk 调一格，调完再判一次，不会牵扯别的层。

════════════════════════════════════════════════════════════════
【⭐ 关键前提：正逆向都要能运算】
════════════════════════════════════════════════════════════════

    研究者特别强调「并且正逆向能运算」。

    ⚠️ 所以锚点不是「随便挑个数」——
       必须满足：  f⁻¹(anchor) 存在且有限，且 f(f⁻¹(anchor)) ≈ anchor

    ⚠️ 而且取上界/下界时要注意开区间：
       ellipk 的 search 上界是 0.98（开区间），
       f(0.98) 会很大；取 0.98 当作**输入**是合法的（在 search 内），
       但如果取 y_range 的上界当输入，就可能出域。

════════════════════════════════════════════════════════════════
【三种锚点 + 怎么选】
════════════════════════════════════════════════════════════════

    LO   取下界：search[0] 或 y_range[0]
    HI   取上界：search[1] 或 y_range[1]
    MID  取中点（几何平均，适配跨数量级的函数）

    ⭐ 选法：**不预先规定，逐个试「能不能正逆走通」，能走通就用。**
       研究者说的「针对单个数进行调整」就是这个意思。

════════════════════════════════════════════════════════════════
【怎么用】
════════════════════════════════════════════════════════════════

    from anchor import ANCHORS, plan_anchors, verify_chain

    plan = plan_anchors(['tanh', 'ellipe', 'ellipk'])
    # → 每层一个锚点 + 一个 cnk
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from mathpack import MathPack, DT, DEV

__all__ = ['Anchor', 'ANCHORS', 'anchor_candidates', 'pick_anchor',
           'usable_anchor', 'plan_anchors', 'verify_chain']

LO, MID, HI = 'LO', 'MID', 'HI'
EXTRA = 'EX'   # 扩充的随机锚点（2026-11-07）

# ⭐ 边界判定（2026-11-07 01:35）：离区间端点不到 15% 就算「贴边界」，
#    这时才值得用连续 n 微调。够远就不动（cn=1），保持题面干净。
_EDGE_FRAC = 0.15


# ================================================================
# 一、锚点候选
# ================================================================


def anchor_candidates(pack: MathPack, n_extra: int = 0,
                      seed: int | None = None) -> list[tuple[str, float]]:
    """
    ⭐ 给一个包列出「可以拿去做锚点」的单值。

    ⚠️ 只用 search 里的值（那是它真正能接受的输入），
       并且往内收，避开开区间的边界
       —— 取到边界上 f⁻¹ 可能不收敛（实测 ellipk 在 0.9999 处是 8.1e3）。

    ⚠️ 2026-11-07 01:10 加大内收幅度：0.5% → 5%。
       实测 k=10 有一条链把 pow8 的输入推到 0.011（下界 0.01），
       那里 pow8 的输出 2.2e-14 和其他点挤在一起，反推不回来。
       ⭐ 根因不是算法，是**深链把值推到了定义域边缘，那里分辨率低**。
       多留余量，边缘就不进了。
    """
    lo, hi = pack.search
    if not (lo < hi):
        return []

    span = hi - lo
    a = lo + span * 0.05
    b = hi - span * 0.05

    out = [(LO, a), (HI, b)]
    # 中点：跨数量级用几何平均，否则算术平均
    if a > 0 and b / a > 100:
        out.append((MID, math.sqrt(a * b)))
    else:
        out.append((MID, (a + b) / 2))

    # ── 扩充：区间内随机取点（默认 0 个，不影响旧行为）────
    # ⚠️ 2026-11-07 12:30 加（Frank 定）：
    #    原版永远只返回 LO/HI/MID 三个固定点，pick_anchor 永远取第一个。
    #    后果：pow2 与 sqrt 的 search 区间相同 → 候选 LO 都是 5.0095
    #          → 链里一旦同时出现，v0 = v2 **必然成立**，重试多少次都一样。
    #    实测：200 条 k=10 的链，99% 因节点重复被淘汰，且 1.57 秒/条。
    if n_extra > 0 and b > a:
        rng = random.Random(seed) if seed is not None else random.Random()
        same_sign = (a > 0 and b > 0) or (a < 0 and b < 0)
        for i in range(int(n_extra)):
            if same_sign and a > 0 and b / a > 100:
                v = math.exp(rng.uniform(math.log(a), math.log(b)))
            else:
                v = rng.uniform(a, b)
                if not same_sign and abs(v) < span * 1e-6:
                    v = a * 0.5 if a != 0 else b * 0.5
            out.append(('%s#%d' % (EXTRA, i + 1), v))

    return out


# ================================================================
# 二、单个锚点能不能用
# ================================================================


def usable_anchor(pack: MathPack, v: float, tol: float = 1e-9):
    """
    ⭐ 这个单值能不能正逆走通。

    三件事都要成立：
        ① f(v)·c 有限
        ② f⁻¹(f(v)·c / c) 有限          ← ⚠️ 先除包系数 c
        ③ f(f⁻¹(...))·c ≈ f(v)·c       ← 回来的值对得上

    ⚠️ 2026-11-06 22:40 修：第一版反向是 `pack._inv(f)`，
       **漏了除包系数 c**。
       实测：pow4 的 c=100，x=0.05995 → f·c=0.00129，
             _inv(0.00129) 给 0.1896（应为 0.0599），相对误差 99。
       于是所有候选都判 ok=False，pow4+exprel 整条链「锚点挑不出来」。

    ⭐ 这是同一个错误在今天的**第三次**：
       前两次在 layerchain（正向两个乘法、反向只除一个），
       这次在 anchor。**凡是正向乘了 c 的地方，反向就必须先除 c。**
    """
    t = torch.tensor([v], device=DEV, dtype=DT)
    with torch.no_grad():
        # ① 正向：f(v) · c
        f = pack._fwd(t) * pack.c
        if not torch.isfinite(f).all():
            return (False, float('nan'), float('inf'))

        # ② 反向：先除 c，再 f⁻¹
        inv = pack._inv(f / pack.c)
        if not torch.isfinite(inv).all():
            return (False, float(f[0]), float('inf'))

        # ③ 再正向，比对
        back = pack._fwd(inv) * pack.c
        if not torch.isfinite(back).all():
            return (False, float(f[0]), float('inf'))

        err = abs(float(back[0]) - float(f[0])) / max(abs(float(f[0])), 1e-300)
    return (err < tol, float(f[0]), err)


def pick_anchor(pack: MathPack, prefer: str | None = None,
                want: tuple[float, float] | None = None,
                positive_only: bool = False,
                n_extra: int = 0,
                seed: int | None = None):
    """
    ⭐ 挑一个能正逆走通的锚点。

    ⭐ 研究者的意思：不预设上界还是下界，**能走通的那个就用**。

    ⚠️ 2026-11-06 22:30 修：不能只看「这一层自己能不能走通」，
       还要看「它×cn 之后能不能被**下一层**接住」。
       实测：tanh 取 LO 锚点 -4.95 → 输出 -0.9999 → ellipe 不吃负数，
       整条链断在第 2 层。而 tanh 自己是好的，问题在**选错了侧**。

    ⚠️ 2026-11-06 22:50 补：positive_only=True 时，只考虑输出为正的锚点。
       ⭐ 因为整条链里只要有包下界 ≥ 0，上游输出就必须为正，
          这个约束要一路传到最内层。见 plan_anchors 的注释。

    want: 下一层能接受的正值区间（None = 最后一层，不约束）
    """
    cands = anchor_candidates(pack, n_extra=n_extra, seed=seed)
    if not cands:
        return None

    # ⭐ 先筛出「下游能接」的候选，再按 prefer 排序
    fitting = []
    for side, v in cands:
        ok, fv, err = usable_anchor(pack, v)
        if not ok:
            continue
        # ⭐ 正性约束：这一层的输出（f·c）必须为正
        if positive_only and not (fv > 0):
            continue
        if want is not None:
            # ⚠️ 直接判带符号的值，不取绝对值（负值不能被伪装成正值）
            cn, kk = fit_cn_to_next(fv, want[0], want[1])
            if not (want[0] <= fv * cn <= want[1]):
                continue          # 下游接不住
        fitting.append({'pack': pack.name, 'side': side, 'in': v,
                        'out': fv, 'err': err})

    if not fitting:
        return None

    if prefer:
        # ⚠️ 排序稳定：只把 prefer 侧提到前面，同侧内部保持原顺序
        fitting.sort(key=lambda x: 0 if x['side'] == prefer else 1)

    # ⭐ 2026-11-07 12:45（Frank 定）：扩充候选存在时，在合格候选里随机取。
    #
    # ⚠️ 原版恒取 fitting[0]，而默认候选恒为 [LO, HI, MID]，
    #    所以同一个包每次都返回同一个点 —— 链里出现同一区间形状的包必然重复。
    #    ⭐ 只有 n_extra > 0 时才随机，**默认 n_extra=0 行为与昨天完全一致**。
    if n_extra > 0:
        # 只从 prefer 侧（若指定）里随机；没指定就全体随机
        head = fitting
        if prefer:
            same = [x for x in fitting if x['side'] == prefer]
            head = same if same else fitting
        rng = random.Random(seed) if seed is not None else random.Random()
        return rng.choice(head)

    return fitting[0]


# ================================================================
# 三、⚠️ 关键：单值能不能接住下一个单值
# ================================================================


@dataclass
class Anchor:
    """一条链上某一层的锚点。"""

    pack_name: str
    side: str            # LO / MID / HI
    x: float             # 这一层的输入（锚点本身）
    f: float             # f(x)·c，即「未缩放输出」
    cn: float = 1.0      # ⭐ 本层系数（可带负号，可为 ±10^n 实数指数）
    cn_k: float = 0.0    # ⭐ 指数 n（实数，2026-11-07 起不再限定整数）
    ok: bool = True
    why: str = ''


def fit_cn_to_next(f: float, nxt_lo: float, nxt_hi: float,
                   continuous: bool = True) -> tuple[float, float]:
    """
    ⭐ 单值拟合：找一个 cn = ±10^n，使 f·cn 落进 [nxt_lo, nxt_hi]。

    ⚠️ 2026-11-06 22:35 修：**不能对 f 取绝对值**。
       第一版用 abs(f) 去找区间，于是 tanh 的输出 -0.9999 被当成 +0.9999，
       判成「能落进 [0.01, 0.98]」，实际上 ellipe 根本不吃负数。

    ⭐ 2026-11-07 00:45 改：**cn 允许带负号**（研究者提的）。
       原来 cn 只能是 10^k（恒正），于是「负输出撞只收正的下游」必然断。
       实测断了 5 条：betaln / logit / erf / log2 / ndtri，
       ✅ cn = −1 一乘全部接上，成对误差 0.0e+00。

    ⭐ 2026-11-07 01:35 改：**n 可以是实数**（研究者提的）。
       原话：「这个 n 不再取整数，而是在远离边界的范围内随机……
              值域内看是不是在边界，不是在边界的用 1，
              边界的用一个随机的非整数值微调，然后正负号保留。」

       规则三层：
           ① 在区间内且远离边界 → cn = 1
           ② 在区间内但靠近边界 → cn = ±10^n，n 实数，推到区间中部
           ③ 在区间外           → cn = ±10^k，k 整数，先拉回来

    ⭐ 2026-11-07 01:35 试过「n 取实数」（连续微调），**已回退**。
       实测对比（每档 40 条随机链）：

           k      整数 cn    连续 cn
           2      100%       100%
           3      100%       88%    ↓
           4      100%       100%
           5       98%       92%    ↓
           6       98%       98%
           8      100%       90%    ↓
          10       90%       92%    ↑

       ⚠️ 平均反而下降，而且题面变丑：
              整数：  0.1x = f(y)
              连续：  0.0217893x = f(y)
          后者失去了「十进制精确表示」，成了一个带舍入误差的任意实数。

       ⭐ 结论：负号是决定性的（k=8 从 38%→92%，正向断彻底消失），
          连续微调收益不稳定还牺牲题面 —— 所以只用整数。

       ⚠️ 连续版代码留在 contcoef.py，不启用。以后真需要可以接回来。

    返回 (cn, n)，cn 带符号，n 是整数（记作 float 方便统一类型）。
    """
    if not math.isfinite(f) or f == 0:
        return (1.0, 0.0)

    # ⭐ 整数 k 搜索（带符号）。按量级由近到远，同量级先试正号（题面更干净）。
    best = None
    for k in range(0, 31):
        mags = [1.0] if k == 0 else [10.0 ** k, 10.0 ** -k]
        for mag in mags:
            for s in (1.0, -1.0):
                v = f * mag * s
                if nxt_lo <= v <= nxt_hi:
                    score = (abs(math.log10(mag)), 0 if s > 0 else 1)
                    if best is None or score < best[0]:
                        best = (score, mag * s, float(k))
        if best is not None:
            break          # 找到最近量级就停，不再往远处找
    if best is None:
        return (1.0, 0.0)
    return (best[1], best[2])


# ================================================================
# 四、整链规划：每层一个锚点 + 一个 cnk
# ================================================================


def plan_anchors(names: list[str], prefer: str | None = None,
                 n_extra: int = 0, seed: int | None = None) -> dict | None:
    """
    ⭐ 为一条链逐层规划「锚点 + cnk」。

    方向：从**最内层往外**走（y → x），每层：
        ① 挑一个能正逆走通的锚点
        ② 算它的输出 f
        ③ 找 cnk 让 f·cnk 落进下一层的 search
        ④ 把 f·cnk 当作下一层的输入，重复

    ⚠️ 下一层能不能接，判的是它自己的 search（能接受的输入），
       不是 y_range（它的输出）。

    ⚠️ 2026-11-06 22:50 补：**符号要一路传到最内层**。
       原来只在挑最内层锚点时看「紧邻的下一层」，中段就断了。
       实测失败的四条链全是同一个原因：
           tan 输出 -5.35   → 下一层只收正 → ✗
           erf 输出 -1      → 下一层只收正 → ✗
           expi 输出 -17.4  → 下一层只收正 → ✗
           betaln 输出 -0.698 → 下一层只收正 → ✗
       ⭐ 74 个包里大量包的下界是 0.01（只收正）。
          所以一旦链里有这样的包，**上游所有输出都必须是正的**。
       ✅ 做法：从最外层往里推「需要正数」这个约束，传到最内层选锚点。
    """
    ps = [packs_get(n) for n in names]
    out = []
    cur = None

    # ⭐ 符号约束：链里只要有包「下界 ≥ 0」，上游就必须产正数
    #
    # ⚠️ 2026-11-07 00:45 改：**不再需要了**。
    #    原来这条是因为 cn 只能是正的，负值翻不回来。
    #    现在 cn 可以取 −10^k，负输出乘 −1 就成正的。
    #    ⭐ 所以 need_positive 恒为 False —— 保留字段是为了兼容旧调用，
    #       也在日志里留痕：这个约束曾经存在过，现在被负号 cn 取代了。
    need_positive = False

    for i, p in enumerate(ps):
        if cur is None:
            # 最内层：自己挑锚点
            want = ps[1].search if len(ps) > 1 else None
            picked = pick_anchor(p, prefer, want,
                                 positive_only=need_positive,
                                 n_extra=n_extra, seed=seed)
            if picked is None:
                return None
            a = Anchor(pack_name=p.name, side=picked['side'],
                       x=picked['in'], f=picked['out'])
        else:
            # 往上层：输入已被上一层定死，只验证能不能用
            side = 'IN'
            ok, fv, err = usable_anchor(p, cur)
            a = Anchor(pack_name=p.name, side=side, x=cur, f=fv,
                       ok=ok, why='' if ok else '输入不可用（f 或 f⁻¹ 走不通）')

        # 定这一层的 cnk：让它的输出落进下一层的 search
        if i + 1 < len(ps):
            nxt = ps[i + 1]
            cn, kk = fit_cn_to_next(a.f, nxt.search[0], nxt.search[1])
            a.cn, a.cn_k = cn, kk
            cur = a.f * cn
            # 校验：真的落进去了吗（带符号判，不取绝对值）
            if not (nxt.search[0] <= cur <= nxt.search[1]):
                a.ok = False
                a.why = '输出 ×cn 后仍进不了下一层 search'
        else:
            # 最后一层：输出就是 x，落进它自己的 y_range
            cn, kk = fit_cn_to_next(a.f, p.y_range[0], p.y_range[1])
            a.cn, a.cn_k = cn, kk
            cur = a.f * cn

        out.append(a)

    return {'names': names, 'anchors': out, 'x_final': cur,
            'need_positive': need_positive}


# ================================================================
# 五、验：整条链从锚点出发，正逆能不能走通
# ================================================================


def verify_chain(names: list[str], prefer: str | None = None,
                 n_extra: int = 0, seed: int | None = None) -> dict:
    """
    ⭐ 用锚点值真的跑一遍正逆，判定可行性。

    ⚠️ 这里**只测一个值**（锚点），不比区间 —— 这正是单值方案的意义。
    """
    plan = plan_anchors(names, prefer, n_extra=n_extra, seed=seed)
    if plan is None:
        return {'names': names, 'ok': False, 'why': '锚点挑不出来'}

    ps = [packs_get(n) for n in names]
    anchors: list[Anchor] = plan['anchors']

    # 起步：最内层锚点
    v = torch.tensor([anchors[0].x], device=DEV, dtype=DT)

    # ---- 正向 ----
    fwd_ok = True
    for i, p in enumerate(ps):
        f = p._fwd(v) * p.c
        v = f * anchors[i].cn
        if not torch.isfinite(v).all():
            fwd_ok = False
            break

    # ---- 反向（查同一张 cn 表）----
    inv_ok = True
    if fwd_ok:
        for i in range(len(ps) - 1, -1, -1):
            v = p_inv(ps[i], v / anchors[i].cn)
            if not torch.isfinite(v).all():
                inv_ok = False
                break

    err = float('inf')
    if fwd_ok and inv_ok:
        y0 = anchors[0].x
        err = abs(float(v[0]) - y0) / max(abs(y0), 1e-300)

    ok = fwd_ok and inv_ok and err < 1e-9
    return {'names': names, 'ok': ok, 'err': err,
            'anchors': anchors, 'x_final': float(plan['x_final'] or 0),
            'why': '' if ok else ('正向断' if not fwd_ok else
                                  '反向断' if not inv_ok else '回不来')}


# ---- 小工具 ----
def packs_get(name):
    import packs
    return packs.get(name)


def p_inv(p, v):
    return p._inv(v / p.c)


# ================================================================
# 六、显示
# ================================================================


def show(names: list[str], prefer: str | None = None) -> bool:
    r = verify_chain(names, prefer)
    tag = '+'.join(names)
    print('=' * 88)
    print('%s   prefer=%s' % (tag, prefer or '（不预设）'))
    print('=' * 88)

    if 'anchors' not in r:
        print('  ❌ %s' % r.get('why'))
        print()
        return False

    for i, a in enumerate(r['anchors']):
        # ⚠️ cn 现在可能带负号，显示要体现出来
        cn_txt = ('%+.3g' % a.cn) if a.cn != 0 else '0'
        print('  第%d层 %-9s 锚点[%-3s]=%-11.6g f·c=%-14.6g cn=%-9s %s'
              % (i + 1, a.pack_name, a.side, a.x, a.f, cn_txt,
                 '✅' if a.ok else '❌ ' + a.why))

    print()
    print('  最终 x = %.6g' % r['x_final'])
    print('  正逆往返误差 = %.3e' % r['err'])
    print('  %s' % ('✅ 这条链立得住' if r['ok'] else '❌ ' + r.get('why', '')))
    print()
    return r['ok']
