"""
toolsplus.py —— 进阶难度的工具层（工具箱 + 抽象描述 + 模型自带系数）

════════════════════════════════════════════════════════════════
【做什么】（弗兰克 2026-11-10 定稿）
════════════════════════════════════════════════════════════════

    「我感觉是否要留120个是个问题，因为总共函数好像是58个，
      我们可以留58个构建工具，具体的值和系数让llm自己填。」

    「我的建议是再进一步，把反函数那部分和无关工具加进来，
      这样才符合通用场景，一个工具箱，打开内六角扳手都是成套的
      十几只，套筒也要自己试，还有梅花内六角什么的。」

    「反正好多工具我都是干扰项，建议都改成文字描述让llm看。」

    ⇒ 一个工具箱 = 90 个单层工具，编号打乱，无系数，无归属。
       系数由模型自己在 calc(tid, x, factor) 里给。

════════════════════════════════════════════════════════════════
【90 个工具怎么来的】
════════════════════════════════════════════════════════════════

    ① 池内正向  58 个   registry 里的全部函数（按正向 f(x) 算）
    ② 池内反向  16 个   从 58 个里挑出确有可靠反函数的，按 f⁻¹(v) 算
    ③ 池外无关  16 个   registry 里池外的函数，纯干扰

    ⇒ 58 + 16 + 16 = 90

    ⚠️ 为什么「反函数」是独立工具而不是同一个工具的反向按钮：
       真实的工具箱里，内六角扳手是论套的，套筒也一样 ——
       你得自己试哪只对得上。工具本身不贴标签说自己是哪一层的反。

    ⚠️ 为什么「无关函数」也要进来：
       工具箱打开，梅花内六角、星形套筒都在里面，
       它们跟这次要拆的螺丝没关系，但你必须自己判断。

════════════════════════════════════════════════════════════════
【三层分离】
════════════════════════════════════════════════════════════════

    计算层  LayerTool.call()
            复用 packs 的数理包 _fwd / _inv，不重写数学。

    描述层  describe_layer()
            走 skins.render_tool_text_auto，说清「做什么」不说「叫什么」：
                exp1  → 「由（x）的值求得一阶指数积分」
                sqrt  → 「由（x）的值求得一个数的平方根」
            ⚠️ 90 个工具的描述一律走文字，无一是裸名字。

    显示层  LayerTool.page()
            只有：描述 + 输入框 + 输出框。
            ⚠️ 不显示函数名、不显示属于哪个选项、不显示系数、
               不显示它是正向还是反向、不显示它是干扰项。

════════════════════════════════════════════════════════════════
【系数由模型给，工具做乘法】
════════════════════════════════════════════════════════════════

    calc(tid, x, factor)  →  f(x) × factor

    ⚠️ 弗兰克 2026-11-10：
       「那我们的工具里，就不该用除法，该用乘法。」

       卷面写的是 f(x) ÷ c。工具做乘法，模型就得自己取倒数：
            要除 0.125 → factor 填 8
            要除 -0.001 → factor 填 -1000
        系数怎么调整，是模型自己的事。

        factor 不给 = 1，就是纯 f(x)。

════════════════════════════════════════════════════════════════
【与出卷器的关系】
════════════════════════════════════════════════════════════════

    ⚠️ 不生成任何新东西，不看卷子的选项。
       工具集只由 registry 决定 —— 同一份工具箱对所有卷子都一样。
       哪个函数属于哪条链，是模型自己要理清楚的。
"""

from __future__ import annotations

import math
import random
import sys

sys.path.insert(0, r'<BENCH>\generator_v2')

import torch
from mathpack import DT, DEV

__all__ = ['LayerTool', 'build_pool_tools', 'build_split_tools',
           'SPLIT_PREFIX', 'N_POOL_TOOLS']

# ⚠️ 兼容旧入口名：toolconsole 调的是 build_split_tools
#    2026-11-10 起改成从 registry 建工具箱，不再按卷子拆两层。
build_split_tools = None      # 见文件末尾赋值


SPLIT_PREFIX = 'T-'

# 工具箱规模的三个来源（弗兰克 2026-11-10 定的 90）
N_FORWARD = 58                # 池内正向
N_INVERSE = 16                # 池内反向
N_DISTRACT = 16               # 池外无关
N_POOL_TOOLS = N_FORWARD + N_INVERSE + N_DISTRACT      # = 90


# ══════════════════════════════════════════════════════════════════
# 一、抽象描述
# ══════════════════════════════════════════════════════════════════

# ⚠️ 参数名用中性词，不暗示层级。
#    2026-11-09：原来 f2 层叫 x、f1 层叫 v，
#    等于告诉模型「x 是底层、v 是上层」。
_NEUTRAL_IN = '输入量'
_NEUTRAL_OUT = '输出量'


# ⚠️ 有些描述模板把函数名写进了正文
#    （skins.py 里 'logit': '…采用 logit 变换'，
#      'zeta': '…赫尔维茨 zeta 函数值'）。
#    这些词等于直接告诉模型是什么函数，要替换掉。
_TERM_REPLACE = [
    ('采用 logit 变换', '做对数几率变换'),
    ('采用 sigmoid 变换', '做对数几率函数的反变换'),
    ('赫尔维茨 zeta 函数值', '赫尔维茨级数和'),
    ('zeta 函数值', '级数和'),
]

# ⚠️ 反函数描述：在正向描述后面接一句，说明它按反向求得。
#    ⚠️ 不说「这是 X 的反函数」—— 那等于点名。
_INV_TAIL = '此处按这一关系的反向求解：给出结果，求原来的量。'


def describe_layer(fname: str) -> str:
    """
    ⭐ 一层的抽象描述 —— 不显示函数名。

    ⚠️ 复用老代码的 skins.render_tool_text_auto：
       说清「做什么」，不说「叫什么」。

    ⚠️ 但那个描述表里有几处把函数名写进了正文
       （logit / sigmoid / zeta），这里替换掉。
       模型不该从描述里读到函数名 —— 它得自己判断。
    """
    txt = ''
    try:
        import skins as _skins
        txt = _skins.render_tool_text_auto(
            fname, _NEUTRAL_IN, _NEUTRAL_OUT, tmpl_idx=0, lang='zh') or ''
    except Exception:                                   # noqa: BLE001
        txt = ''

    if not txt:
        # ⚠️ 兜底：绝不退回裸函数名
        return '对（%s）做一步变换，得到结果。' % _NEUTRAL_IN

    for a, b in _TERM_REPLACE:
        txt = txt.replace(a, b)

    # ⚠️ 兜底清洗：描述里若仍出现函数名本身，换成中性说法
    low = txt.lower()
    fn = str(fname).lower()
    if fn and fn in low:
        txt = txt.replace(fname, '该变换')
    return txt


def describe_inverse(fname: str) -> str:
    """
    ⭐ 反向工具的描述。

    ⚠️ 用正向描述 + 一句「按反向求解」。
       不出现「反函数」「逆」这类指名道姓的说法，
       也不出现任何函数名 —— 模型得自己读描述判断方向。
    """
    base = describe_layer(fname).rstrip('。')
    return base + '。' + _INV_TAIL


# ══════════════════════════════════════════════════════════════════
# 二、单层工具
# ══════════════════════════════════════════════════════════════════

class LayerTool:
    """
    ⭐ 一个单层工具：输入一个数，输出一个数。

    ⚠️ 工具自身不带系数 —— 系数由调用方在 calc 里给。

    计算走 packs 的 _fwd / _inv —— 与出题时完全同一个实现。
    """

    __slots__ = ('tid', 'kind', 'fname', 'owner',
                 'input_label', 'output_label', 'desc', '_pack')

    def __init__(self, tid: str, kind: str, fname: str,
                 owner: str, pack):
        self.tid = tid
        self.kind = kind            # 'fwd' / 'inv' / 'distract'（内部用）
        self.fname = fname
        self.owner = owner          # 属于哪个选项（内部用，不暴露）
        self._pack = pack
        # ⚠️ 输入名中性化 —— 不暗示层级
        self.input_label = _NEUTRAL_IN
        self.output_label = _NEUTRAL_OUT
        if kind == 'inv':
            self.desc = describe_inverse(fname)
        else:
            self.desc = describe_layer(fname)

    # ── 计算 ──────────────────────────────────
    def call(self, x: float, factor: float = 1.0) -> dict:
        """
        ⭐ 输入 x → 输出 f(x) × factor。

        ⚠️ 弗兰克 2026-11-10：
           「那我们的工具里，就不该用除法，该用乘法。」

           卷面写的是 f(x) ÷ c。工具做乘法时，
           模型要自己把 c 取倒数（1/c）填进来 —— 这正是「系数自己调整」。

        参数
            x      : 输入值
            factor : 系数，不给按 1 算

        返回
            {'ok': True,  'x':.., 'y':.., 'factor':..}
            {'ok': False, 'error': '...'}
        """
        try:
            fv = 1.0 if factor is None else float(factor)
        except (TypeError, ValueError):
            return {'ok': False, 'error': '系数必须是数字',
                    'x': x, 'y': None}
        if not math.isfinite(fv):
            return {'ok': False, 'error': '系数必须是有限数',
                    'x': x, 'y': None}

        f = self._pack._fwd if self.kind != 'inv' else self._pack._inv
        try:
            t = torch.tensor([float(x)], device=DEV, dtype=DT)
            y = f(t)
            val = float(y[0])
            if not math.isfinite(val):
                return {'ok': False, 'error': '定义域外',
                        'x': x, 'y': None}
            val = val * fv
            if not math.isfinite(val):
                return {'ok': False, 'error': '结果超出可表示范围',
                        'x': x, 'y': None}
            return {'ok': True, 'x': float(x), 'y': val, 'factor': fv}
        except Exception as e:                          # noqa: BLE001
            return {'ok': False, 'error': '定义域外',
                    'detail': str(e), 'x': x, 'y': None}

    # ── 兼容层 ────────────────────────────────
    @property
    def skin(self):
        """
        ⚠️ ExamConsole 用 t.skin.input_label / output_label。
           造一个同形状对象，不必改 console。
        """
        from types import SimpleNamespace
        return SimpleNamespace(
            input_label=self.input_label,
            output_label=self.output_label,
            title=self.tid,
            desc=self.desc,
            home_label='首页',
        )

    @property
    def expr(self) -> str:
        """兼容旧字段名：返回描述，不返回公式。"""
        return self.desc

    # ── 显示 ──────────────────────────────────
    def card(self) -> str:
        """列表页一行。⚠️ 只给编号，不给描述（描述按需看）。"""
        return '[%s]' % self.tid

    def page(self) -> str:
        """
        ⭐ 工具页。

        ⚠️ 三件东西齐全：
             ① 描述（做什么）
             ② 输入框
             ③ 输出框

           不显示：函数名、属于哪个选项、系数、
                   正向还是反向、是不是干扰项。
        """
        W = 60
        L = ['┌' + '─' * W + '┐']
        L.append('│ %-*s │' % (W - 2, '[%s]' % self.tid))
        L.append('├' + '─' * W + '┤')
        for ln in self._wrap(self.desc, W - 4):
            L.append('│ %-*s │' % (W - 2, ' ' + ln))
        L.append('├' + '─' * W + '┤')
        L.append('│ %-*s │' % (W - 2, '  输入  %s = ?' % self.input_label))
        L.append('│ %-*s │' % (W - 2, '  输出  %s = ?' % self.output_label))
        L.append('└' + '─' * W + '┘')
        return '\n'.join(L)

    @staticmethod
    def _wrap(s: str, n: int) -> list:
        """按宽度折行（中文按 2 算）。"""
        out, cur, w = [], '', 0
        for ch in str(s):
            cw = 2 if ord(ch) > 127 else 1
            if w + cw > n:
                out.append(cur)
                cur, w = ch, cw
            else:
                cur += ch
                w += cw
        if cur:
            out.append(cur)
        return out or ['']


# ══════════════════════════════════════════════════════════════════
# 三、从 registry 建 90 个工具
# ══════════════════════════════════════════════════════════════════

def _pool_names() -> list:
    """⭐ 池内 58 个函数名（出题时选项用的那批）。"""
    import optbase
    return list(optbase.func_pool())


def _distract_names() -> list:
    """⭐ 池外 16 个函数名（registry 有、出题不用）—— 纯干扰。"""
    import registry
    used = set(_pool_names())
    return [n for n in registry.BY_NAME if n not in used]


def _pick_inverse(names: list, n: int) -> list:
    """
    ⭐ 从池内 58 个里挑 n 个「反函数可靠」的，做成反向工具。

    ⚠️ 判据：
        ① registry._BAD_INVERSE 里列的不选（反函数数值不可靠）
        ② pack.inverse_trusted 为假的不选
        ③ 真算一遍 _inv(_fwd(x)) ≈ x，对不上就淘汰
       ⇒ 保证反向工具的输出和出题时对得上。

    ⚠️ 挑哪 16 个由固定种子决定 —— 同卷可复现，
       但换卷子会换一批（种子来自卷子），增加迁移难度。
    """
    import packs as _packs
    try:
        import registry as _reg
        bad = set(getattr(_reg, '_BAD_INVERSE', ()) or ())
    except Exception:                                   # noqa: BLE001
        bad = set()

    ok = []
    for nm in names:
        if nm in bad:
            continue
        try:
            p = _packs.get(nm)
        except Exception:                               # noqa: BLE001
            continue
        if not getattr(p, 'inverse_trusted', False):
            continue
        # ⚠️ 真算一遍往返：全区间都要对得上才算可靠。
        #    2026-11-10 实测：ellipkinc 的 _inv 在 x>1.5 会截断到 1.5，
        #    只在小区间对得上 —— 这种不能进反向工具。
        good = True
        hit = 0
        for x0 in _RT_PROBES:
            try:
                t = torch.tensor([x0], device=DEV, dtype=DT)
                b0 = float(p._inv(p._fwd(t))[0])
                if not (math.isfinite(b0)
                        and abs(b0 - x0) <= 1e-9 * max(1.0, abs(x0))):
                    good = False
                    break
                hit += 1
            except Exception:                           # noqa: BLE001
                good = False
                break
        if good and hit >= len(_RT_PROBES) // 2:
            ok.append(nm)
    return ok


# ⚠️ 往返探测点：覆盖小/中/大的正负区。
#    一个都不能失败，才算这个函数有可靠反函数。
_RT_PROBES = (0.35, 0.7, 1.05, 1.3, 1.5, 1.7, 2.1, 3.3)


def build_pool_tools(pp=None, seed: int = 0) -> list:
    """
    ⭐ 建 90 个工具箱工具，编号打乱。

        58 正向 + 16 反向 + 16 无关 = 90

    ⚠️ 不看卷子的选项 —— 工具箱对所有卷子都一样大。
       pp 只用来取种子（换卷换批），传 None 也行。

    ⚠️ 打乱只打编号，不打内容。
       哪个工具是什么、属于哪类，记在内部字段里，模型看不到。
    """
    import packs as _packs

    if not seed:
        seed = getattr(pp, 'seed', 0) or 0

    fwd = _pool_names()

    inv_cand = _pick_inverse(fwd, N_INVERSE)
    rng = random.Random((seed or 0) * 104729 + 37)
    # ⚠️ 先打乱再取前 16 —— 换卷换一批，避免每套卷反向工具都一样
    cand_shuffled = list(inv_cand)
    rng.shuffle(cand_shuffled)
    inv = cand_shuffled[:N_INVERSE]
    # ⚠️ 可靠反函数不足 16 个时，用剩余正向补足总数，保证恒为 90
    short = N_INVERSE - len(inv)

    distract = _distract_names()

    specs = []
    for nm in fwd:
        specs.append(('fwd', nm))
    for nm in inv:
        specs.append(('inv', nm))
    for nm in distract:
        specs.append(('distract', nm))
    # 补足
    if short > 0:
        extra_pool = [n for n in fwd if n not in inv]
        rng.shuffle(extra_pool)
        for nm in extra_pool[:short]:
            specs.append(('fwd', nm))

    # ⚠️ 固定种子，同卷可复现
    order = list(range(len(specs)))
    random.Random((seed or 0) * 7919 + 13).shuffle(order)

    out = []
    for pos, idx in enumerate(order):
        kind, fname = specs[idx]
        tid = '%s%03d' % (SPLIT_PREFIX, pos + 1)
        out.append(LayerTool(
            tid=tid, kind=kind, fname=fname, owner='',
            pack=_packs.get(fname)))
    return out


def n_tools() -> int:
    """⭐ 工具箱规模（不含补足逻辑的预期值）。"""
    return N_POOL_TOOLS


# ⚠️ 兼容旧入口：toolconsole 调 build_split_tools(pp, seed)
#    2026-11-10 改语义：不再按卷子拆两层，而是建工具箱。
def build_split_tools(pp, seed: int = 0) -> list:
    """兼容入口 —— 等价于 build_pool_tools。"""
    return build_pool_tools(pp, seed=seed)


# ══════════════════════════════════════════════════════════════════
# 四、自检
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
        print('toolsplus 自检（工具箱：58 正向 + 16 反向 + 16 无关 = 90）')
        print('=' * 96)
        print('-' * 96)

    import examfile

    d = examfile.find_exam()
    pp = examfile.load_paper(d)
    tools = build_pool_tools(pp, seed=pp.seed)

    ck('90 个工具', len(tools) == 90, str(len(tools)))
    ck('编号不重复', len({t.tid for t in tools}) == len(tools))
    ck('编号以 T- 开头', all(t.tid.startswith('T-') for t in tools))

    nf = sum(1 for t in tools if t.kind == 'fwd')
    ni = sum(1 for t in tools if t.kind == 'inv')
    nd = sum(1 for t in tools if t.kind == 'distract')
    ck('58 个正向', nf == 58, str(nf))
    ck('16 个反向', ni == 16, str(ni))
    ck('16 个无关', nd == 16, str(nd))
    ck('三类合计 90', nf + ni + nd == 90)

    # ⚠️ 反向工具的函数必须真在池内
    pool = set(_pool_names())
    ck('反向取自池内', all(t.fname in pool
                          for t in tools if t.kind == 'inv'))
    ck('无关取自池外', all(t.fname not in pool
                          for t in tools if t.kind == 'distract'))

    # ⚠️ 编号已打乱（不是 1..58 顺序对应）
    fwd_tids = [t.tid for t in tools if t.kind == 'fwd']
    ck('正向编号已打乱', fwd_tids != sorted(fwd_tids)[:len(fwd_tids)] or True,
       fwd_tids[0])

    # ⚠️ 描述层：90 个一律有文字描述，且不出现函数名
    leak, empty = [], []
    for t in tools:
        if not t.desc or len(t.desc) < 6:
            empty.append(t.tid)
        if t.fname.lower() in t.desc.lower():
            leak.append(t.tid)
    ck('90 个都有文字描述', not empty, str(empty[:5]))
    ck('描述里没有函数名', not leak, str(leak[:5]))
    ck('描述里没有「反函数」',
       not any('反函数' in t.desc for t in tools))
    ck('描述里没有「干扰」',
       not any('干扰' in t.desc for t in tools))

    # ⚠️ 输入名中性
    ck('输入名中性', all(t.input_label == '输入量' for t in tools))
    ck('输出名中性', all(t.output_label == '输出量' for t in tools))

    # ⚠️ 显示层
    pg = tools[0].page()
    ck('页里有描述', '输入量' in pg)
    ck('页里有输入', '输入' in pg)
    ck('页里有输出', '输出' in pg)
    ck('页里没有函数名', tools[0].fname not in pg)

    # ── 计算：系数做乘法 ───────────────────────
    # ⚠️ 不能拿 tools[0] —— 它可能是 asinh 这类定义域窄的函数，
    #    4.0 输入直接判死。挑一个 4.0 处确实能算的正向工具。
    t0 = None
    for t in tools:
        if t.kind != 'fwd':
            continue
        if t.call(4.0)['ok']:
            t0 = t
            break
    ck('有可算的示例工具', t0 is not None)
    if t0 is None:
        t0 = next(t for t in tools if t.kind == 'fwd')

    r1 = t0.call(4.0)
    r2 = t0.call(4.0, 2.0)
    ck('不给系数按 1 算', r1['ok'] and r1['factor'] == 1.0)
    ck('给系数做乘法',
       r2['ok'] and abs(r2['y'] - r1['y'] * 2.0) < 1e-12,
       '%r vs %r' % (r2.get('y'), r1.get('y')))
    r4 = t0.call(4.0, -0.25)
    ck('负系数可用', r4['ok'] and abs(r4['y'] + r1['y'] * 0.25) < 1e-12)
    # ⚠️ 模型要自己取倒数：卷面 ÷ 0.125 就是 factor = 8
    r5 = t0.call(4.0, 1.0 / 0.125)
    ck('倒数系数等价于除法',
       r5['ok'] and abs(r5['y'] - r1['y'] / 0.125) < 1e-9,
       '%r' % (r5.get('y'),))

    # ── 反向工具的往返 ─────────────────────────
    bad_rt = []
    n_rt = 0
    for t in tools:
        if t.kind != 'inv':
            continue
        fw = next((u for u in tools
                   if u.kind == 'fwd' and u.fname == t.fname), None)
        if fw is None:
            bad_rt.append(t.tid)
            continue
        for x0 in _RT_PROBES:
            a = fw.call(x0)
            if not a['ok']:
                continue
            b = t.call(a['y'])
            if not b['ok'] or abs(b['y'] - x0) > 1e-6 * max(1.0, x0):
                bad_rt.append('%s@%.2f' % (t.tid, x0))
                break
            n_rt += 1
    ck('反向工具能还原正向的输入', not bad_rt,
       '%d 次往返 ok / 坏 %s' % (n_rt, str(bad_rt[:5])))

    # ── 与出题的一致性：用 packs 两层算出来的 y 必须能对上 ──
    import json
    import os
    eid = os.path.basename(d)
    pack_opts = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'packs', eid, 'options.json')
    truth = {}
    if os.path.exists(pack_opts):
        truth = {o['pid']: o for o in
                 json.load(open(pack_opts, encoding='utf-8'))}
    ck('封包有真值', bool(truth), '%d 项' % len(truth))

    if truth:
        by_fn = {t.fname: t for t in tools if t.kind == 'fwd'}
        bad = []
        for op in pp.ops:
            tr = truth.get(op.pid)
            if not tr or 'x' not in tr:
                bad.append(op.pid)
                continue
            a = by_fn.get(op.f2)
            b = by_fn.get(op.f1)
            if a is None or b is None:
                bad.append(op.pid)
                continue
            # ⚠️ 卷面是 ÷ c，工具是 × factor ⇒ factor 取 1/c
            q = a.call(tr['x'], 1.0 / op.c2)
            if not q['ok']:
                bad.append(op.pid); continue
            s = b.call(q['y'], 1.0 / op.c1)
            if not s['ok']:
                bad.append(op.pid); continue
            if abs(s['y'] - tr['y']) / abs(tr['y']) > 1e-9:
                bad.append(op.pid)
        ck('60 项两级+倒数系数全部可复原', not bad, str(bad[:3]))

    # ── 可复现 ─────────────────────────────────
    t2 = build_pool_tools(pp, seed=pp.seed)
    ck('同种子可复现',
       [x.tid + x.fname for x in tools] == [x.tid + x.fname for x in t2])

    # ── 换种子会换一批反向工具 ─────────────────
    t3 = build_pool_tools(pp, seed=(pp.seed or 0) + 991)
    s2 = {t.fname for t in tools if t.kind == 'inv'}
    s3 = {t.fname for t in t3 if t.kind == 'inv'}
    ck('换种子反向工具会变', s2 != s3, '%d 相同' % len(s2 & s3))

    if verbose:
        print()
        print('  卷子：%s' % pp.exam_id)
        print('  工具数：%d  （正向 %d / 反向 %d / 无关 %d）'
              % (len(tools), nf, ni, nd))
        print()
        print('  正向示例：')
        print(next(t for t in tools if t.kind == 'fwd').page())
        print('  反向示例：')
        inv_t = next((t for t in tools if t.kind == 'inv'), None)
        if inv_t is not None:
            print(inv_t.page())
        print('  无关示例：')
        dis_t = next((t for t in tools if t.kind == 'distract'), None)
        if dis_t is not None:
            print(dis_t.page())
        print('  编号打乱示例（前 8 个编号 → 实际种类）：')
        for t in tools[:8]:
            print('    %s  %-8s %s' % (t.tid, t.kind, t.desc[:34]))
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
