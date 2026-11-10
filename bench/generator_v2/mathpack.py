"""
数理包（MathPack）—— 正运算 + 逆运算 + 系数，三者成一个整体

════════════════════════════════════════════════════════════════
【为什么要有这一层】（Frank 2026-11-06 定）
════════════════════════════════════════════════════════════════

    之前的问题：函数、逆函数、系数 c 是三个**散着放**的东西，
    c 挂在 Tool.scale 上、生成时动态设、还要用 c_list 跨步记录顺序。
    结果——串用、顺序错乱、缩放打架，一个接一个。

    ✅ 打包后：c 进包里，外面**看不见也改不了**。
       正运算和逆运算各自从包里取同一个 c，天然成对。

════════════════════════════════════════════════════════════════
【核心恒等式】（Frank 反复强调的那一条）
════════════════════════════════════════════════════════════════

        这边放大 100 倍  → 那边计算前就得缩 100 倍

    方程：       c · f(y) = x
        ↓ 两边同除 c（⭐ 不除就变成 x = c·f(y) 或 x = f(y)，全错）
                f(y) = x / c
        ↓ 反解
                y = f⁻¹(x / c)

    ⭐ 数理包的职责就是**保证这一步永远做对**：
       forward:  y --×c·f--> x
       inverse:  x --÷c, f⁻¹--> y
       两个方向用的是**同一个 c**，因为 c 只在包里。

════════════════════════════════════════════════════════════════
【可颠倒】（Frank：这样组合还能翻一倍）
════════════════════════════════════════════════════════════════

    同一个包既能正着用也能倒着用：

        正向：  v ──forward──▶ w         （出题：算 x）
        反向：  w ──inverse──▶ v         （解题/验算：算 y）

    两种方向都合法 → 组合数翻倍。

════════════════════════════════════════════════════════════════
【写法】
════════════════════════════════════════════════════════════════

    内部默认写法（简单、好验算）：      x/c = f(y)
    对外可复杂化的等价写法（考模型）：   0.01x = f(y)、0.1x = 10·f(y) ...
    ⚠️ 无论怎么变，解出来的 y 必须是同一个数 —— 数量关系不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

DT = torch.float64
DEV = 'cuda'


# ================================================================
# 一、数理包
# ================================================================


@dataclass
class MathPack:
    """
    数理包 = 正运算 + 逆运算 + 系数，绑成一个整体。

    ⚠️ 三者不可分离：
        · 没有 f⁻¹ 就解不出（选项无法验算）
        · 没有 c  就分不清「y」和「x」（数量关系错）
    """

    code: str                          # 编码 E001 / H001
    name: str                          # 函数名
    kind: str                          # ELEMENTARY / HIGHER
    domain: str                        # 学科领域
    note: str                          # 说明

    _fwd: Callable = None              # f
    _inv: Callable = None              # f⁻¹
    search: tuple[float, float] = (0.01, 20.0)   # f⁻¹ 的求解区间
    y_range: tuple[float, float] = (1e-50, 1e50) # y 能取到哪
    inverse_trusted: bool = True       # 逆运算是否唯一（非单调工具 = False）
    has_div: bool = False              # 含除法/对数（要避开 0）

    # ------------------------------------------------------------
    # ⭐ 系数：打包时定死，正逆共用（2026-11-06 Frank 定）
    # ------------------------------------------------------------
    #
    #   打包时看一次：这个包的值要不要缩放？
    #       值在定义域内  →  c = 1
    #       超出定义域    →  c = 10^k（缩回域内）
    #
    #   ⭐ c 一旦定下就是包的属性，**正运算和逆运算都用这一个 c**，
    #     两边同时处理才能抵消。运行时绝不重算 c。
    #
    #   ⚠️ 之前把 c 写成「按输入算」是错的：
    #      正向 coef(y) 得 10、反向 coef(x) 得 100 → 两边不抵消，反而越缩越乱。
    coef_value: float = 1.0            # ⭐ 打包时定死的系数
    coef_code: str = ''                # ⭐ 系数的独立编码（如 C000）

    @property
    def c(self) -> float:
        """系数（打包时定死，正逆共用）。"""
        return self.coef_value

    # ------------------------------------------------------------
    # ⭐ 两个方向（同一个 c）
    # ------------------------------------------------------------

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        正向：y ──×c·f──▶ x

        ⭐ c 是包的属性，不按输入重算。
        """
        c = self.coef_value
        if c == 1.0:
            x = self._fwd(y)
        else:
            x = self._fwd(y) * c
        return x, c

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        反向：x ──÷c, f⁻¹──▶ y

        ⭐ 关键：**计算前先除以同一个 c**。不除就变成 x = c·f(y) 或 x = f(y)，全错。
        """
        c = self.coef_value
        v = x / c if c != 1.0 else x
        y = self._inv(v)
        return y, c

    # ------------------------------------------------------------
    # ⭐ 自洽检查
    # ------------------------------------------------------------

    def self_test(self, n: int = 256, seed: int = 0) -> dict:
        """
        ⭐ 正逆自洽：摇 x → 正算得 y → 逆算回 x′ → 判 y′ == y。

        ⚠️ 2026-11-06 修：第一版在 y_range 里摇，把「函数值」当成了
           「函数输入」用，第一步就错了。正确做法是**在 search 里摇 x**。

        ⭐ 流程（和 diagnose_inverse.py 一致，那一次 74/74 全过）：
            ①  x ∈ search            ← 这才是函数输入
            ②  y = c·f(x)            ← 正向
            ③  x′ = f⁻¹(y/c)         ← 反向
            ④  判 c·f(x′) == y       ← 回来的 y 对不对

        ⚠️ 判 y 不判 x —— 多解工具可能给出另一个合法解，但 y 必须回来。
        """
        g = torch.Generator(device=DEV)
        g.manual_seed(seed)
        slo, shi = self.search
        pad = (shi - slo) * 0.02
        x0 = torch.rand(n, device=DEV, dtype=DT, generator=g) \
            * (shi - slo - 2 * pad) + (slo + pad)

        y, _c1 = self.forward(x0)                 # ① ②
        ok1 = torch.isfinite(y)
        if not ok1.any():
            return {'ok': False, 'n': n, 'rate': 0.0, 'med': float('inf'),
                    'max': float('inf')}

        x1, _c2 = self.inverse(torch.where(ok1, y, torch.ones_like(y)))   # ③
        ok = ok1 & torch.isfinite(x1) & torch.isfinite(x1)
        if not ok.any():
            return {'ok': False, 'n': n, 'rate': 0.0, 'med': float('inf'),
                    'max': float('inf')}

        y2, _c3 = self.forward(torch.where(ok, x1, torch.ones_like(x1)))  # ④
        ok = ok & torch.isfinite(y2)
        if not ok.any():
            return {'ok': False, 'n': n, 'rate': 0.0, 'med': float('inf'),
                    'max': float('inf')}

        err = (y2[ok] - y[ok]).abs() / y[ok].abs().clamp(min=1e-300)
        rate = float((err < 1e-9).float().mean().item())
        return {'ok': rate >= 0.99, 'n': int(ok.sum().item()), 'rate': rate,
                'med': float(err.median().item()),
                'max': float(err.max().item())}


# ================================================================
# 二、⭐ 打包时定系数
# ================================================================


def decide_coef(y_range: tuple[float, float],
                search: tuple[float, float]) -> tuple[float, int]:
    """
    ⭐ 打包时决定系数 c（Frank 2026-11-06 定）。

    规则只有一条：
        值在定义域内  →  c = 1
        超出定义域    →  c = 10^k，把值缩回域内

    ⭐ 判据用「值域的典型量级」和「搜索区间的典型量级」比：
       f⁻¹ 的求解要在 search 里扫，所以 y 的典型量级不该离 search 太远。

    ⚠️ 这是**打包时算一次**，之后写死在包里，正逆共用。

    返回 (c, k)：c = 10^k
    """
    lo, hi = y_range
    if not (lo < hi) or not math.isfinite(lo) or not math.isfinite(hi):
        return (1.0, 0)

    # 值域的典型量级（对数量级的中点）
    def _mag(v: float) -> float:
        a = abs(v)
        return math.log10(a) if a > 1e-300 else -300.0

    # 域内判据：值域的上下界都在「能直接算」的范围内
    if lo > -1e30 and hi < 1e30 and lo >= 1e-3 and hi <= 1e30:
        return (1.0, 0)                     # 已经在域内 → 不缩放

    # 越界了：按上界的量级缩
    top = max(abs(lo), abs(hi))
    k = int(math.floor(math.log10(top))) - 1
    k = max(k, 0)
    return (float(10.0 ** k), k)
