"""
工具登记表 —— 71 个工具，全部在 CUDA GPU 上，正逆成对

════════════════════════════════════════════════════════════
【这是唯一有效的工具库】
════════════════════════════════════════════════════════════
⚠️ 旧版本（`pairs.py` 等 99 个工具的 CPU 版）已移入 `history/01_CPU时代_作废/`。
   原因见 `history/README.md`。

════════════════════════════════════════════════════════════
【六条筛选标准】
════════════════════════════════════════════════════════════
    ① 真实存在（公开库）
    ② 输出为标量
    ③ 取值可由构造确定
    ④ 有实际计算量
    ⑤ 领域典型
    ⑥ ⭐ **必须能在 CUDA GPU 上批量求值**

⭐ 第⑥条是硬约束 —— 因此「为什么选它」无需另证，只需证明它能跑。

════════════════════════════════════════════════════════════
【组成】
════════════════════════════════════════════════════════════
    torch 原生      64 个   （torch.special + torch 基本运算）
    自己实现          7 个   （gpu_extra.py：椭圆积分 / 指数积分 / 朗伯 W）

════════════════════════════════════════════════════════════
【量级纪律】
════════════════════════════════════════════════════════════
    中间值与终点：1e-50 ~ 1e50   （Frank：小数点前后 50 位还算能键入）
    零值纪律：     MIN_ABS = 1e-3，采样不取 0
    精度：         全程 float64

════════════════════════════════════════════════════════════
【用法】
════════════════════════════════════════════════════════════
    from registry import TOOLS, get
    t = get('E001')
    y = t.forward(1.5)              # 标量进，标量出
    xs = t.inverse_scalar(3.14)     # 标量反解
    # 批量（推荐）：
    x0, y = t.build_batch(1024, rng_seed=7)   # 摇初始值 + 正算
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from gpu_extra import (
    ellipk, ellipk_inverse, ellipe, ellipe_inverse,
    ellipkinc, ellipkinc_inverse,
    lambertw, lambertw_inverse,
    exp1, exp1_inverse, expn, expn_inverse, expi, expi_inverse,
)

DT = torch.float64
DEV = 'cuda'

LO = 1e-50
HI = 1e50
MIN_ABS = 1e-3


# ================================================================
# 一、工具定义
# ================================================================


@dataclass
class Tool:
    """
    一个工具 = 正运算 + 逆运算 + 编码。

    ⚠️ 正逆必须成对 —— 不成对就做不了逆运算，模型也就无法从 y 回到 x。
    """

    code: str                    # 唯一编码 E001 / H001
    name: str                    # 函数名
    kind: str                    # ELEMENTARY / HIGHER
    domain: str                  # 学科领域
    note: str                    # 说明

    # 正运算：张量进张量出
    _fwd: Callable = None
    # 逆运算：张量进张量出（解集里随机取一个）
    _inv: Callable = None
    # 反解的搜索区间
    search: tuple[float, float] = (0.01, 20.0)
    # 含除法/对数（要避开 0）
    has_div: bool = False

    # ⭐ 值域（Frank 2026-11-06 定）：y 只允许摇在这个范围里。
    #    默认 (1e-50, 1e50) = 全域；有界函数必须收窄。
    #    ⚠️ 不登记值域 = 摇出工具取不到的 y = solve_inverse 一定失败。
    #    实测值来自 diagnose_tools.py（logs/diagnose_*.jsonl）。
    y_range: tuple[float, float] = (1e-50, 1e50)

    # ⭐ 逆运算可不可信（2026-11-06）：
    #    True  = 单调单根，二分求根可靠
    #    False = 非单调/多根/跳变，逆运算只能给「一个」解，不保证唯一
    #    ⚠️ 这个标记不影响正运算，也不影响值域 —— 只影响能不能拿它求逆。
    inverse_trusted: bool = True

    # ════════════════════════════════════════════════════════════
    # ⚠️ 2026-11-06：本类是**原料**，不再承担系数与正逆配对。
    #    系数和正逆已经打包进 mathpack.MathPack —— 见 packs.py。
    #
    #    这里删掉的旧接口（已由 MathPack 取代）：
    #        scale 字段          → MathPack.coef_value（打包时定死）
    #        _coef(y)            → MathPack.coef（按输入算）
    #        forward(x)          → MathPack.forward
    #        apply_inverse(y)    → MathPack.inverse
    #        inverse(y)          → MathPack.inverse
    #
    #    ⭐ 为什么必须删：旧的 _coef 按**输入**算 c，
    #       正向按 y 算出 c=10、反向按 x 算出 c=100，
    #       两边不抵消反而叠加 —— 实测 tan/cos/digamma/polygamma
    #       回验误差 90%~100%。打包成固定 c 后全部降到 1e-16。
    # ════════════════════════════════════════════════════════════

    def forward_scalar(self, x: float) -> float:
        return float(self._fwd(torch.tensor(x, device=DEV, dtype=DT)))

    def inverse_scalar(self, y: float) -> float:
        return float(self._inv(torch.tensor(y, device=DEV, dtype=DT)))

    # ------------------------------------------------------------
    # ⭐ 链上每步的「能不能接住」检查（2026-11-06 新增）
    # ------------------------------------------------------------

    def accepts(self, v: torch.Tensor) -> torch.Tensor:
        """
        ⭐ 这个值能不能当我的输入（落在我的值域里吗）。

        ⚠️ 链上每步的输出 = 下一步的输入，所以每步都要检查一次。
           不检查的话，中间值一旦落在下一个工具的值域外，
           那一步就必然解不出 —— 这正是那 20.6% 链失败的根源。
        """
        lo, hi = safe_range(self)
        return (v >= lo) & (v <= hi)

    def domain_image(self, y_lo: float, y_hi: float,
                     grid: int = 4000) -> tuple[float, float] | None:
        """
        ⭐ 值域区间 [y_lo, y_hi] 对应的 x 区间（在 search 里反查）。

        用途：链的可行性判断。
            这一步的输入在 [y_lo, y_hi] 里 → 它对应的 x 在哪个区间？
            那个 x 区间就是**前一个工具**必须能接住的值域。

        ⚠️ 非单调工具会有多个不连续段，这里取**外包络**（min..max），
           偏保守 —— 宁可拒绝一条可行的链，也不放行一条必崩的链。
        """
        lo, hi = self.search
        xs = torch.linspace(lo, hi, grid, device=DEV, dtype=DT)
        with torch.no_grad():
            ys = self._fwd(xs)
        m = torch.isfinite(ys) & (ys >= y_lo) & (ys <= y_hi)
        if not m.any():
            return None
        sel = xs[m]
        return (float(sel.min().item()), float(sel.max().item()))

    def build_batch(self, n: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        摇 n 个初始值 + 正算，返回 (x0, y)。

        ⚠️ 零值纪律：x0 不取 0；y 超出量级区间的不返回。
        """
        g = torch.Generator(device=DEV)
        if seed is not None:
            g.manual_seed(seed)
        x0 = torch.empty(0, device=DEV, dtype=DT)
        while x0.numel() < n:
            k = max(n * 2, 64)
            v = torch.rand(k, device=DEV, dtype=DT, generator=g) \
                * (self.search[1] - self.search[0]) + self.search[0]
            v = v[v.abs() >= MIN_ABS]
            x0 = torch.cat([x0, v])
        x0 = x0[:n]
        y = self.forward(x0)
        keep = torch.isfinite(y) & (y.abs() >= LO) & (y.abs() <= HI)
        return x0[keep], y[keep]

    def supports_value(self, v: torch.Tensor) -> torch.Tensor:
        """这个 y 落不落在这个工具能取到的范围里（逐元素布尔）。"""
        lo, hi = safe_range(self)
        return (v >= lo) & (v <= hi)


# ================================================================
# ⭐ 按值域摇 y（2026-11-06 新增）
# ================================================================

# ⭐ 值域跨零的工具在回填时被跳过了，仍是默认 (1e-50, 1e50)。
#    但摇 1e42 这种 y 对它们同样荒唐 —— 兜底用一个「典型量级窗」。
CROSS_ZERO_WINDOW = (-1e8, 1e8)


def safe_range(t: Tool) -> tuple[float, float]:
    """
    ⭐ 摇 y 实际使用的值域（2026-11-06 修）。

    ⚠️ 问题：值域跨零的工具（sin / cos / erf / pow3 ...）回填时被跳过，
       仍是默认 (1e-50, 1e50)，于是又摇出 1e42 这种它根本取不到的值。

    ⭐ 修法：未登记的一律用 CROSS_ZERO_WINDOW (-1e8, 1e8) 兜底。
       这仍然远小于 1e50，足以避开荒唐的 y。
    ⚠️ 这是兜底；最终应补全全部 74 个工具的实测值域。
    """
    lo, hi = t.y_range
    if (lo, hi) != (1e-50, 1e50):
        return (lo, hi)
    return CROSS_ZERO_WINDOW


# ================================================================
# ⭐ 按值域摇 y（2026-11-06 新增）
# ================================================================


def sample_y_for(tools: list[Tool], n: int,
                 g: torch.Generator | None = None) -> torch.Tensor:
    """
    ⭐ 按链上工具的真实值域摇 y。

    为什么要这么做：
        有界函数（sin ∈ [-1,1]、expit ∈ (0,1)、ellipe ∈ [1.03,1.57]）
        摇出 1e42 这种 y，逆运算**必然**找不到根 —— 不是算法坏了，
        是这个 y 本来就不可能由一个 f(x) 产生。

    ⭐ 规则：取链上所有工具 y_range 的**交集**。
       交集为空 → 这条链不可用（抛错，让调用方换链）。
       否则 → 正区间跨 3 个数量级用对数均匀，否则线性均匀。

    ⚠️ 交集为空是**链的设计问题**，不是工具的问题 ——
       两个值域不相交的工具不能接在一起。
    """
    lo = -float('inf')
    hi = float('inf')
    for t in tools:
        a, b = safe_range(t)
        lo = max(lo, a)
        hi = min(hi, b)
    if not (lo < hi):
        raise ValueError(
            '工具值域无交集，不能成链：'
            + ', '.join(f'{t.name}[{safe_range(t)[0]:.3g},{safe_range(t)[1]:.3g}]'
                        for t in tools))

    if g is None:
        g = torch.Generator(device=DEV)
    e = torch.rand(n, device=DEV, dtype=DT, generator=g)

    if lo > 0 and hi / lo > 1e3:
        a_, b_ = math.log(lo), math.log(hi)          # 正区间宽 → 对数均匀
        return torch.exp(e * (b_ - a_) + a_)
    if hi < 0 and lo / hi > 1e3:
        a_, b_ = math.log(-hi), math.log(-lo)        # 负区间宽 → 对数均匀
        return -torch.exp(e * (b_ - a_) + a_)
    return e * (hi - lo) + lo                        # 窄区间或跨零 → 线性


def chain_ok(tools: list[Tool]) -> tuple[bool, str]:
    """这条链能不能用（值域有交集吗）。"""
    lo, hi = -float('inf'), float('inf')
    for t in tools:
        a, b = safe_range(t)
        lo = max(lo, a)
        hi = min(hi, b)
    if lo < hi:
        return True, f'值域交集 [{lo:.3g}, {hi:.3g}]'
    return False, f'值域无交集（下界 {lo:.3g} > 上界 {hi:.3g}）'


def chain_trusted(tools: list[Tool]) -> tuple[bool, str]:
    """
    ⭐ 这条链的逆运算可不可信（2026-11-06 新增）。

    非单调/多根的工具，逆运算只能给「一个」解，不能保证它是唯一解。
    做题生成阶段这必须拦下 —— 否则答案的唯一性论证就废了。
    """
    bad = [t.name for t in tools if not t.inverse_trusted]
    if not bad:
        return True, '全部单调单根'
    return False, '逆运算不唯一：' + ', '.join(bad)


def chain_feasible(tools: list[Tool],
                   n_probe: int = 200) -> tuple[bool, str, tuple[float, float] | None]:
    """
    ⭐ 链的可行性预检（2026-11-06 新增，Frank：「先收集清楚信息再改」）。

    ⚠️ 问题：只看「y 落在所有工具值域的交集里」是不够的。
       中间的传递才是真关卡：

           y  在末端工具的值域里       ✅
           x₁ = 末端工具的逆(y)         ← 这一步的输出
           x₁ 必须在**上一个工具**的值域里   ← ⚠️ 这里常常断

       例：ive → log1p
           交集 [0.092, 0.970] 里的 y=0.9 是合法的，
           但 log1p⁻¹(0.9) = 1.46，而 ive 的值域是 [0.092, 0.970]
           → 1.46 落在 ive 值域外 → 第 2 步必崩。

    ⭐ 做法：倒着推一遍值域区间，每步都用 domain_image 把
       「这一步能接的值」换算成「对应的 x 区间」，交给前一个工具检查。

    ⚠️ 2026-11-06 第一版写错过一次：把「上一步输出的 x 区间」又用
       本工具自己的值域重新收窄了一遍，等于把真正的约束丢掉，
       结果所有链都判「可行」。正确做法是 x 区间**原样传给前一步**。

    ⭐ 返回 (能不能用, 说明, 可用的 y 起始区间)
    """
    # 从末端工具开始：y 允许的范围 = 末端工具的值域
    lo, hi = safe_range(tools[-1])
    if not (lo < hi):
        return False, f'{tools[-1].name} 值域非法', None

    for k in range(len(tools) - 1, -1, -1):
        t = tools[k]
        # ① 这一步的输入范围必须落在本工具的值域里
        a, b = safe_range(t)
        lo2, hi2 = max(lo, a), min(hi, b)
        if not (lo2 < hi2):
            return (False,
                    f'第{k+1}步 {t.name}：值域 [{a:.3g}, {b:.3g}] '
                    f'接不住输入 [{lo:.3g}, {hi:.3g}]', None)
        lo, hi = lo2, hi2

        # ② 换算成 x 区间（本步能解的 x 范围）
        img = t.domain_image(lo, hi)
        if img is None:
            return (False,
                    f'第{k+1}步 {t.name}：值域 [{lo:.3g}, {hi:.3g}] '
                    f'在 search 里找不到对应的 x', None)

        # ③ ⭐ 这个 x 区间原样变成前一个工具的输入范围
        lo, hi = img

    return True, '可行', (lo, hi)


# ================================================================
# 二、GPU 批量反解：变号扫描 + 二分
# ================================================================


def solve_inverse(fwd: Callable, y: torch.Tensor,
                  search: tuple[float, float],
                  grid: int = 2000, n_iter: int = 60,
                  gen: torch.Generator | None = None) -> torch.Tensor:
    """
    批量反解：对每个 y，在 search 区间里随机找一个根。

    ⭐ 两步：
        ① 一次算完 B×G 网格，找所有变号区间
        ② 每个样本随机挑一个变号区间，做二分

    ⚠️ 全在 GPU 上；不调用任何 scipy。
    """
    B = y.shape[0]
    lo0, hi0 = search
    xs = torch.linspace(lo0, hi0, grid, device=DEV, dtype=DT)

    X = xs.view(1, -1).expand(B, grid)
    D = fwd(X) - y.view(-1, 1)
    s1 = torch.sign(D[:, :-1])
    s2 = torch.sign(D[:, 1:])
    change = (s1 * s2) < 0                     # (B, G-1)
    has = change.any(dim=1)

    # 每个样本随机挑一个变号位置
    w = torch.rand(B, grid - 1, device=DEV, dtype=DT, generator=gen)
    w = torch.where(change, w, torch.full_like(w, -1.0))
    idx = torch.argmax(w, dim=1)

    xsv = xs.expand(B, grid)
    lo = torch.gather(xsv, 1, idx.unsqueeze(1)).squeeze(1)
    hi = torch.gather(xsv, 1, (idx + 1).unsqueeze(1)).squeeze(1)

    f_lo = fwd(lo)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        f_mid = fwd(mid)
        same = (f_mid - y) * (f_lo - y) > 0
        lo = torch.where(same, mid, lo)
        hi = torch.where(same, hi, mid)
        f_lo = torch.where(same, f_mid, f_lo)

    nan = torch.full_like(lo, float('nan'))
    return torch.where(has, (lo + hi) / 2, nan)


def _mk(fwd_s, inv_s, search):
    """
    造一对 GPU 逆运算：

    ⭐ 有解析逆（inv_s 不为 None）→ 直接用
       没有 → 走 solve_inverse
    """
    if inv_s is not None:
        return lambda y: inv_s(y)
    return lambda y: solve_inverse(fwd_s, y, search)


# ================================================================
# 三、71 个工具登记
# ================================================================
# 格式：(编码, 名字, 类型, 领域, 说明, 正运算, 解析逆或None, 搜索区间, 含除法)


def _s(v):
    return torch.tensor(v, device=DEV, dtype=DT)


_ELEM: list[tuple] = [
    # ---- 幂族 ----
    ('pow2', '通用', 'x²',           lambda x: x ** 2,            lambda y: torch.sqrt(y),        (0.01, 100.0), False),
    ('pow3', '通用', 'x³',           lambda x: x ** 3,            lambda y: torch.sign(y) * y.abs() ** (1 / 3), (-21.0, 21.0), False),
    ('pow4', '通用', 'x⁴',           lambda x: x ** 4,            lambda y: y ** 0.25,            (0.01, 10.0), False),
    ('pow5', '通用', 'x⁵',           lambda x: x ** 5,            lambda y: torch.sign(y) * y.abs() ** 0.2, (-6.3, 6.3), False),
    ('pow8', '通用', 'x⁸',           lambda x: x ** 8,            lambda y: y ** 0.125,           (0.01, 3.1), False),
    ('sqrt', '通用', '平方根',        torch.sqrt,                  lambda y: y ** 2,               (0.01, 100.0), False),
    ('cbrt', '通用', '立方根',        lambda x: torch.sign(x) * x.abs() ** (1 / 3),
                                                                  lambda y: y ** 3,               (-21.0, 21.0), False),

    # ---- 指数对数 ----
    ('exp',   '通用', '指数',         torch.exp,                   torch.log,                      (0.01, 9.0), True),
    ('log',   '通用', '对数',         torch.log,                   torch.exp,                      (0.01, 80.0), True),
    ('expm1', '通用', 'e^x−1',       torch.expm1,                 torch.log1p,                    (0.01, 9.0), False),
    ('log1p', '通用', 'log(1+x)',     torch.log1p,                 torch.expm1,                    (0.01, 80.0), True),
    ('log2',  '通用', '以2为底',      torch.log2,                  lambda y: 2.0 ** y,             (0.01, 60.0), True),
    ('log10', '通用', '以10为底',     torch.log10,                 lambda y: 10.0 ** y,            (0.01, 18.0), True),
    ('exp2',  '通用', '2^x',         torch.exp2,                  torch.log2,                     (0.01, 60.0), False),
    ('exp10', '通用', '10^x',        lambda x: 10.0 ** x,         torch.log10,                    (0.01, 18.0), False),
    ('exprel', '通用', '(e^x−1)/x',  lambda x: torch.expm1(x) / x, None,                          (0.01, 9.0), True),
    ('softplus', '通用', 'softplus', torch.nn.functional.softplus, None,                         (0.01, 12.0), False),
    ('loggamma', '核工程', 'log Γ',   torch.special.gammaln,       None,                           (0.01, 20.0), True),
    ('gammaln',  '核工程', 'log|Γ|',  torch.special.gammaln,       None,                           (0.01, 20.0), True),

    # ---- 三角双曲 ----
    ('sin',   '通用', '正弦',         torch.sin,                   lambda y: torch.arcsin(y.clamp(-0.999, 0.999)), (-6.0, 6.0), False),
    ('cos',   '通用', '余弦',         torch.cos,                   lambda y: torch.arccos(y.clamp(-0.999, 0.999)), (-6.0, 6.0), False),
    ('tan',   '通用', '正切',         torch.tan,                   torch.atan,                     (-1.4, 1.4), True),
    ('atan',  '通用', '反正切',       torch.atan,                  torch.tan,                      (-20.0, 20.0), False),
    ('asin',  '通用', '反正弦',       torch.asin,                  torch.sin,                      (-0.95, 0.95), False),
    ('acos',  '通用', '反余弦',       torch.acos,                  torch.cos,                      (-0.95, 0.95), False),
    ('sinh',  '通用', '双曲正弦',     torch.sinh,                  torch.asinh,                    (-6.0, 6.0), False),
    ('cosh',  '通用', '双曲余弦',     torch.cosh,                  torch.acosh,                    (0.01, 6.0), False),
    ('tanh',  '通用', '双曲正切',     torch.tanh,                  torch.atanh,                    (-5.0, 5.0), False),
    ('asinh', '通用', '反双曲正弦',   torch.asinh,                 torch.sinh,                     (-6.0, 6.0), False),
    ('acosh', '通用', '反双曲余弦',   torch.acosh,                 torch.cosh,                     (1.01, 20.0), False),
    ('atanh', '通用', '反双曲正切',   torch.atanh,                 torch.tanh,                     (-0.95, 0.95), True),
    ('hypot', '通用', '√(x²+1.5²)',  lambda x: torch.hypot(x, _s(1.5)), None,                     (0.01, 20.0), False),

    # ---- 误差函数族 ----
    # ⚠️ 2026-11-07 01:00 收窄定义域：这三个在边界处饱和，反推回不到原值。
    #
    #    erf   实测：x=±4.0 误差 2.85e-11 ✅
    #               x=±4.9 误差 1.06e-07 ❌（尾部饱和，浮点只剩几位）
    #         → (-5, 5) 收到 (-4, 4)
    #
    #    erfc  实测：x=±3.0 误差 4.03e-13 ✅
    #               x=±4.0 反推直接 NaN ❌
    #         → (-4, 4) 收到 (-3, 3)
    #
    #    erfcx 实测：x=-4.0 误差 0 ✅   x=-5.0 反推 NaN ❌
    #         → (-5, 5) 收到 (-4, 4)
    ('erf',    '热传导', '误差函数',   torch.erf,                 torch.erfinv,                    (-4.0, 4.0), False),
    ('erfc',   '热传导', '互补误差',   torch.erfc,                None,                            (-3.0, 3.0), False),
    ('erfinv', '热传导', 'erf逆',      torch.erfinv,              torch.erf,                       (-0.95, 0.95), False),
    ('erfcx',  '热传导', '缩放erfc',   torch.special.erfcx,       None,                            (-4.0, 4.0), False),
    ('expit',  '信息论', 'sigmoid',    torch.special.expit,       torch.special.logit,             (-10.0, 10.0), False),
    ('logit',  '信息论', 'logit',      torch.special.logit,       torch.special.expit,             (0.01, 0.99), False),
    # ⚠️ 2026-11-07 00:05：ndtr 保持原实现，**收窄 search**。
    #
    #    曾尝试「尾部稳定版」(_ndtr_tail)，用 1 - 0.5·erfc(x/√2)。
    #    实测失败，而且比原来的还差：
    #        x=+7.9   原 5.69e-03  →  改后 1.14e-03
    #        x=-7.9   原 6.04e-04  →  改后 7.64e-05
    #
    #    ⭐ 原因：丢信息的**不是算法，是存储格式**。
    #       Φ(x) 在 x 大时必须返回「贴着 1 的数」——
    #       只要答案是 0.99999999999999856，float64 就只剩 1 位有效数字。
    #       换什么公式都救不回来。
    #    ⚠️ 连 double-double 也救不全：dd 能把 1-tail 存得更准，
    #       但 ndtr 的值本身设计上就是「1 减去一个极小量」，
    #       要保信息只能**不返回 Φ，返回 tail** —— 那就不是 ndtr 了。
    #
    #    ✅ 因此选方案 A：收窄到 (-6, 6)。
    #       x=6 时 1-Φ = 9.87e-10，float64 还剩约 6.7 位有效数字，够用。
    #       x=7 时只剩 3.8 位，x=7.9 只剩 0.8 位 —— 不可用。
    ('ndtr',   '统计',   '正态CDF',    torch.special.ndtr,        torch.special.ndtri,             (-6.0, 6.0), False),
    ('ndtri',  '统计',   '正态PPF',    torch.special.ndtri,       torch.special.ndtr,              (0.01, 0.99), False),
    ('log_ndtr', '统计', 'log CDF',    torch.special.log_ndtr,    None,                            (-10.0, 10.0), False),
    ('log_expit', '信息论', 'log σ(x)', torch.nn.functional.logsigmoid, None,                      (-10.0, 10.0), False),

    # ---- Γ 与数论 ----
    ('digamma',   '数论', 'ψ',        torch.special.digamma,     None,                            (0.01, 20.0), False),
    ('polygamma', '数论', 'ψ⁽²⁾',     lambda x: torch.special.polygamma(2, x), None,               (0.01, 20.0), True),
    ('zeta',      '数论', 'ζ',        lambda x: torch.special.zeta(_s(2.5), x), None,             (0.01, 20.0), False),
    ('xlogy',     '信息论', 'x·log y', lambda x: torch.special.xlogy(_s(2.5), x), None,           (0.01, 20.0), True),
    ('xlog1py',   '信息论', 'x·log(1+y)', lambda x: torch.special.xlog1py(_s(2.5), x), None,      (0.01, 20.0), True),
    ('beta',      '热传导', 'B 函数',  lambda x: torch.exp(torch.special.gammaln(_s(1.5)) + torch.special.gammaln(x) - torch.special.gammaln(_s(1.5) + x)), None, (0.01, 20.0), True),
    ('betaln',    '热传导', 'log B',   lambda x: torch.special.gammaln(_s(1.5)) + torch.special.gammaln(x) - torch.special.gammaln(_s(1.5) + x), None, (0.01, 20.0), True),
    ('gammainc',  '核工程', 'P(a,x)',  lambda x: torch.special.gammainc(_s(1.5), x), None,        (0.01, 20.0), False),
    ('gammaincc', '核工程', 'Q(a,x)',  lambda x: torch.special.gammaincc(_s(1.5), x), None,       (0.01, 20.0), False),

    # ---- 信息论损失 ----
    ('entr',      '信息论', '熵项',     torch.special.entr,       None,                            (0.01, 20.0), False),
    # ⚠️ 2026-11-06 剔除 logsumexp / huber
    #    原因：它们不是「标量→标量」的逐元素函数
    #      logsumexp(x, dim) 是张量归约
    #      huber(x, zeros)   是损失函数
    #    用作「工具」不符标准②（输出为标量、可进链）
    # ⚠️ 2026-11-07 核对说明（Frank 定：最小化改动，只加注释不改实现）：
    #
    #    本函数的**名字**沿用自 scipy.special.pseudo_huber，但**实际形式不同**：
    #
    #        本实现：  f(x) = 1.5 · √( 1 − e^(−(x/1.5)²) )
    #        scipy：   δ² · ( √(1 + (x/δ)²) − 1 )
    #
    #    本实现**有界**（上限 1.5），scipy 那个**无界**（随 x 增长）。
    #    实测：x=5 时本实现 1.5000，scipy 定义 5.5802 —— 完全不同的两个函数。
    #
    #    ⭐ 不改实现的理由：
    #       · 它是链上的**处理器**（虽然 inverse_trusted=False 不能入链，
    #         但作为工具池中的计算器，值域已与其他组件配合）
    #       · 改实现会改变值域与逆运算，牵动已通过的链测试
    #
    #    ⭐ 不改描述的理由（Frank 2026-11-07）：
    #       「我看那串中文我能写出算式来，这就是抽象能力。」
    #       → 描述精确到能写出算式，**正是要测的东西**；
    #         若简化成「求伪 Huber 损失」，模型靠记住名字就能过，反而是泄露。
    #
    #    ⚠️ **阅读本文件时勿按 scipy 文档理解此函数。**
    #    出处核对记录见 `论文\基准测试\_参考文献\词汇标准\函数描述核对.md`
    ('pseudo_huber', '信息论', '伪Huber', lambda x: torch.sqrt(_s(2.25) * (-torch.expm1(-(x / _s(1.5)) ** 2))), None, (-8.0, 8.0), False),
]


_HIGH: list[tuple] = [
    # ---- 贝塞尔族 ----
    ('jv',  '流体力学', 'J₀',   lambda x: torch.special.bessel_j0(x), None,                (0.01, 20.0), False),
    ('yv',  '流体力学', 'Y₀',   lambda x: torch.special.bessel_y0(x), None,                (0.01, 20.0), False),
    ('iv',  '流体力学', 'I₀',   lambda x: torch.special.modified_bessel_i0(x), None,       (0.01, 20.0), False),
    ('kv',  '流体力学', 'K₀',   lambda x: torch.special.modified_bessel_k0(x), None,       (0.01, 20.0), False),
    ('ive', '流体力学', 'I₀缩放', lambda x: torch.special.i0e(x), None,                    (0.01, 20.0), False),
    ('kve', '流体力学', 'K₀缩放', lambda x: torch.special.scaled_modified_bessel_k0(x), None, (0.01, 20.0), False),
    ('i0',  '流体力学', 'i0',   lambda x: torch.special.i0(x), None,                       (0.01, 20.0), False),
    ('i1',  '流体力学', 'i1',   lambda x: torch.special.i1(x), None,                       (0.01, 20.0), False),

    # ---- 艾里 ----
    ('airy', '光学', '艾里函数', lambda x: torch.special.airy_ai(x), None,                 (-8.0, 6.0), False),

    # ---- 正交多项式 ----
    ('eval_hermite',    '量子力学', '厄米 H₃', lambda x: torch.special.hermite_polynomial_h(x, 3), None,   (-5.0, 5.0), False),
    ('eval_laguerre',   '量子力学', '拉盖尔 L₃', lambda x: torch.special.laguerre_polynomial_l(x, 3), None, (-5.0, 5.0), False),
    ('eval_chebyt',     '数值分析', '切比雪夫 T₃', lambda x: torch.special.chebyshev_polynomial_t(x, 3), None, (-1.0, 1.0), False),
    ('eval_chebyu',     '数值分析', '切比雪夫 U₃', lambda x: torch.special.chebyshev_polynomial_u(x, 3), None, (-1.0, 1.0), False),
    ('eval_legendre',   '量子力学', '勒让德 P₃', lambda x: torch.special.legendre_polynomial_p(x, 3), None, (-1.0, 1.0), False),

    # ---- 椭圆积分（自己实现，正逆成对） ----
    ('ellipk',    '材料力学', '第一类完全',    ellipk,           ellipk_inverse,     (0.01, 0.98), False),
    ('ellipe',    '材料力学', '第二类完全',    ellipe,           ellipe_inverse,     (0.01, 0.98), False),
    ('ellipkinc', '材料力学', '第一类不完全',  lambda p: ellipkinc(p, _s(0.5)),
                                                            lambda v: ellipkinc_inverse(v, _s(0.5)), (0.05, 1.4), False),

    # ---- 指数积分（自己实现） ----
    ('exp1', '核工程', 'E1',       exp1,                        exp1_inverse,                (0.05, 8.0), True),
    ('expn', '核工程', 'E₂',       lambda x: expn(2, x),        lambda v: expn_inverse(2, v), (0.05, 8.0), False),
    ('expi', '核工程', 'Ei',       expi,                        expi_inverse,                (0.05, 8.0), False),

    # ---- 化工 ----
    ('lambertw', '化工', '朗伯 W', lambertw, lambertw_inverse, (0.01, 20.0), True),
]


def _build() -> list[Tool]:
    out: list[Tool] = []
    for i, (name, dom, note, fwd, inv, search, div) in enumerate(_ELEM, 1):
        out.append(Tool(
            code=f'E{i:03d}', name=name, kind='ELEMENTARY', domain=dom, note=note,
            _fwd=fwd, _inv=_mk(fwd, inv, search),
            search=search, has_div=div,
        ))
    for i, (name, dom, note, fwd, inv, search, div) in enumerate(_HIGH, 1):
        out.append(Tool(
            code=f'H{i:03d}', name=name, kind='HIGHER', domain=dom, note=note,
            _fwd=fwd, _inv=_mk(fwd, inv, search),
            search=search, has_div=div,
        ))
    return out


TOOLS: list[Tool] = _build()
BY_CODE: dict[str, Tool] = {t.code: t for t in TOOLS}
BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def get(key: str) -> Tool:
    """
    按名字取工具。

    ⚠️ 优先按名字 —— 编码会随登记顺序变化，名字稳定。
    """
    if key in BY_NAME:
        return BY_NAME[key]
    if key in BY_CODE:
        return BY_CODE[key]
    raise KeyError(key)


# ================================================================
# 四、自检
# ================================================================


def main() -> int:
    from collections import Counter
    import time

    print('=' * 88)
    print('工具登记表')
    print('=' * 88)
    print(f'torch {torch.__version__}  设备 {torch.cuda.get_device_name(0)}  float64')
    k = Counter(t.kind for t in TOOLS)
    print(f'总计 {len(TOOLS)}   初等 {k["ELEMENTARY"]}   高等 {k["HIGHER"]}')
    print(f'量级区间 [{LO:g}, {HI:g}]   零值纪律 MIN_ABS={MIN_ABS}')
    print()

    # ---- 逐个跑通：批量 256，正算 + 反算 + 回验 ----
    print(f'{"编码":<6} {"函数":<16} {"领域":<8} {"样本":>5} {"复现率":>8} {"最大误差":>11}')
    print('-' * 88)

    bad = []
    t_all = time.time()
    for t in TOOLS:
        t0 = time.time()
        x0, y = t.build_batch(256, seed=hash(t.code) & 0xFFFF)
        if y.numel() < 16:
            print(f'{t.code:<6} {t.name:<16} {t.domain:<8}  样本不足')
            bad.append((t.code, t.name, '样本不足'))
            continue
        x1 = t.inverse(y)
        ok = torch.isfinite(x1)
        if ok.sum() < 4:
            print(f'{t.code:<6} {t.name:<16} {t.domain:<8}  反解全失败')
            bad.append((t.code, t.name, '反解失败'))
            continue
        y2 = t.forward(x1[ok])
        err = ((y2 - y[ok]).abs() / y[ok].abs().clamp(min=1e-300))
        rate = (err <= 1e-8).float().mean().item()
        mx = err.max().item()
        flag = '✅' if rate >= 0.95 else '⚠️'
        if rate < 0.95:
            bad.append((t.code, t.name, f'复现率 {rate*100:.1f}%'))
        print(f'{flag} {t.code:<5} {t.name:<16} {t.domain:<8} {int(ok.sum()):>5} '
              f'{rate*100:>7.1f}% {mx:>11.2e}')

    print('-' * 88)
    print(f'总耗时 {(time.time()-t_all):.1f} s（含逐个编译/预热）')
    print()

    print('=' * 88)
    print('按领域')
    print('=' * 88)
    for dom, n in Counter(t.domain for t in TOOLS).most_common():
        print(f'  {dom:<12} {n:>3}')

    if bad:
        print()
        print('=' * 88)
        print(f'⚠️ 有问题的 {len(bad)} 个')
        print('=' * 88)
        for c, n, r in bad:
            print(f'  {c:<6} {n:<16} {r}')
    else:
        print()
        print('⭐ 全部 71 个工具通过（正算 + 反算 + 回验）')

    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(main())

# ================================================================
# # 五、值域登记表（2026-11-06 实测回填）
# ================================================================
# 来源：diagnose_tools.py 在 search 区间上 20000 点密采样，取 [min,max] 再按比例收 2%。
# ⚠️ 登记的是「y 可以摇到哪」，不是「函数定义域」。
#    没登记的工具 = 值域跨零 = 沿用默认 (1e-50, 1e50)。
#
# ⭐ 值域和「能不能二分求根」是两件事：
#    值域 = y 能取到哪（所有工具都有）
#    二分 = 逆运算能不能靠变号找根（只有单调单根的能）
#    下面 _BAD_INVERSE 里的 15 个工具，值域照登记，但逆运算不可信。

_MEASURED_RANGE: dict[str, tuple[float, float]] = {
# --- 单调单根（逆运算可信）---
    'pow2': (0.00010200000000000001, 9800.0),
    'pow4': (1.0200000000000002e-08, 9800.0),
    'pow8': (1.0200000000000003e-16, 8358.332166921802),
    'sqrt': (0.10200000000000001, 9.8),
    'exp': (1.0302511704258512, 7941.022249023877),
    'expm1': (0.01025117042585142, 7940.042249023876),
    'log1p': (0.010149337470231445, 4.30656017157899),
    'exp2': (1.0270946610578533, 1.12986307451471e+18),
    'exp10': (1.0437588521263692, 9.8e+17),
    'exprel': (1.025117042585142, 882.2269165582085),
    'softplus': (0.7121228741180196, 11.760006021309607),
    'acos': (0.32391163787735194, 2.7675515798123063),
    'cosh': (1.0200510004250012, 197.68132340000676),
    'acosh': (0.14412984487536162, 3.6144887900140708),
    'hypot': (1.5300339996222307, 19.65504769772895),
    'erfc': (1.5725603058285618e-08, 1.9599999848910872),
    'erfcx': (0.11291873048772999, 141129602701.16782),
    'expit': (4.630582607648309e-05, 0.9799555100886717),
    'ndtr': (6.228351168147129e-16, 0.9799999999999993),
    'log_ndtr': (-52.16665944750223, -7.772250084643707e-24),
    'log_expit': (-9.800044490921232, -4.630687720120194e-05),
    'zeta': (0.007893665456624056, 98001.28749823317),
    'xlog1py': (0.025373343675578613, 7.459079972422386),
    'beta': (0.00992179641543356, 97.40384326308177),
    'gammainc': (0.0007627104461790219, 0.9799999895580114),
    'gammaincc': (1.0868192140940964e-08, 0.9792671997673966),
    'iv': (1.0200255001593754, 42687116.90836246),
    'kv': (5.856062571643254e-10, 4.626819835557873),
    'ive': (0.09157591812252253, 0.9702730934467135),
    'kve': (0.28411577419032585, 4.6733201479735715),
    'i0': (1.0200255001593754, 42687116.90836246),
    'i1': (0.005100063750265626, 41605873.91742522),
    'ellipk': (1.606240472747703, 3.2870586167851754),
    'ellipe': (1.0491664094113895, 1.535524703181235),
    'ellipkinc': (0.05101062566299065, 1.5814110781085517),
    'exp1': (3.841893530565521e-05, 2.418540518739775),
    'expn': (3.4820398015320404e-05, 0.811277810073711),
    'lambertw': (0.010099503320466913, 2.1609032124635785),
        # --- 值域跨零（回填时被跳过，这里补上实测值）---
    'pow3': (-9307.304999999998, 9214.695),
    'pow5': (-9973.987257149998, 9874.74360285),
    'cbrt': (-2.772718797263026, 2.745129555499215),
    'log': (-4.628196036918031, 4.360116501500512),
    'log2': (-6.677075470723597, 5.877356142630475),
    'log10': (-2.0099999999999993, 1.2489961425777896),
    'loggamma': (-0.12209368232872272, 39.14318476626349),
    'gammaln': (-0.12209368232872272, 39.14318476626349),
    'sin': (-1.0049999985690437, 0.994999998583282),
    'cos': (-1.0049999907629765, 0.9949999552205224),
    'tan': (-5.826873134060301, 5.768894296905472),
    'atan': (-1.5284421207283185, 1.5132337414175892),
    'asin': (-1.259502076990892, 1.2469697180158583),
    'sinh': (-202.72172315713058, 200.70459158342783),
    'tanh': (-1.004908750283908, 0.9949096582412821),
    'asinh': (-2.5042387519081366, 2.479320953381688),
    'atanh': (-1.8409397271801466, 1.8226219189494985),
    'erf': (-1.0049999999984547, 0.9949999999984702),
    'erfinv': (-1.392833343471426, 1.3789743052279293),
    'logit': (-4.618095449385263, 4.572144250883916),
    'ndtri': (-2.337979613411045, 2.3147161346706366),
    'digamma': (-101.06368988515798, 2.9556713722809382),
    'xlogy': (-11.570490092295076, 7.451884030465552),
    'betaln': (-4.655988030666473, 4.576073033862038),
    'entr': (-60.21421869843521, 0.36604004174192356),
    'jv': (-0.40477315626870286, 0.9949751251554683),
    'yv': (-3.0204829152690635, 0.5181824762285747),
    'airy': (-0.421110554304865, 0.532978356831117),
    'eval_hermite': (-944.6999999999999, 935.3),
    'eval_laguerre': (-0.7357110455551022, 73.96166666666666),
    'eval_chebyt': (-1.005, 0.995),
    'eval_chebyu': (-4.02, 3.98),
    'eval_legendre': (-1.005, 0.995),
    'expi': (-2.379724021572271, 438.178000037164),
# --- 非单调/多根（值域可信，逆运算不可信）---
    'polygamma': (-1960002.2935907037, -0.002680684850360947),
    'pseudo_huber': (0.00040802039376595527, 1.4699999999996742),
}

# ⭐ 逆运算不可信的工具（非单调 / 多根 / 跳变）—— 不能只用二分
_BAD_INVERSE: set[str] = {
    'loggamma',
    'gammaln',
    'sin',
    'cos',
    'polygamma',
    'entr',
    'pseudo_huber',
    'jv',
    'yv',
    'airy',
    'eval_hermite',
    'eval_laguerre',
    'eval_chebyt',
    'eval_chebyu',
    'eval_legendre',
}

for _t in TOOLS:
    if _t.name in _MEASURED_RANGE:
        _t.y_range = _MEASURED_RANGE[_t.name]
    _t.inverse_trusted = _t.name not in _BAD_INVERSE
