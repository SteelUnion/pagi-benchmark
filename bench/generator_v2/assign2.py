"""
assign2 —— GPU 内部的自动生成：首尾值 + 10 个链值 + 50 个干扰项

════════════════════════════════════════════════════════════════
【要造什么】（弗兰克 2026-11-08 定）
════════════════════════════════════════════════════════════════

    y0 × y1 × y2 × … × y10 = y11

        y0   首值（题面给）
        y1..y10  10 个链值（要模型挑出来的答案）
        y11  尾值（题面给，= 前面 11 个的乘积）

    另加 50 个干扰项：和 12 个 y、以及 y0~y10 的所有组合乘积都无关。

    一共 60 个选项。

════════════════════════════════════════════════════════════════
【和老代码的关系：针对我们的链节做了升级】
════════════════════════════════════════════════════════════════

    老代码 registry.sample_y_for：
        取链上所有工具 y_range 的【交集】
        交集空 → 链不可用
        正区间宽 → 对数均匀；负区间宽 → 对数均匀；否则线性均匀

    ⭐ 我们的升级（弗兰克 2026-11-08）：
        「应该是就只去看第一层的输出能不能进第二层定义。
          因为这两个函数自身有系数帮我们看着，不需要看。」

        ⇒ 不再做「两函数 y_range 交集」校验。
        ⇒ 改成「第 1 层的输出 v 能不能进第 2 层的 search」。

    ⚠️ 为什么能这么改：我们有系数 c（= ±10^k），
       负责把值搬进函数能算的范围。
       老代码没有这套逐项的 c，所以才需要预先挑交集。

    ⚠️ 对数均匀保留：
       正区间跨 3 个数量级以上 → 对数均匀，避免摇出贴 0 或极大的值。

════════════════════════════════════════════════════════════════
【为什么全在 GPU】
════════════════════════════════════════════════════════════════

    弗兰克：「这次我们把东西都放 gpu 内部计算。」

    ⇒ 摇 y、算乘积、造干扰、查 2047 个组合，全在张量里跑。
       Python 只做「一层的循环」，不做逐样本循环。
"""

from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, r'<BENCH>\generator_v2')

import config
import packs
from mathpack import DT, DEV

__all__ = ['pick_range', 'sample_uniform', 'assign_chain', 'avoid_products',
           'make_decoys', 'build_batch', 'self_test']


# ══════════════════════════════════════════════════════════════════
# 一、摇值：对数均匀 / 线性均匀
# ══════════════════════════════════════════════════════════════════

# ⚠️ 单个选项的往返精度上限（弗兰克 2026-11-09 定）
#    超过这个的选项不采用
EPS_LIMIT = 1e-6

# ⚠️ 精度隔离带的安全倍数
SAFE_K = 10.0

# ⚠️ 跨这么多个数量级就用对数均匀
LOG_SPAN = 1e3


def sample_uniform(n: int, lo: float, hi: float,
                   gen: torch.Generator) -> torch.Tensor:
    """
    ⭐ 在 [lo, hi] 里摇 n 个数。

    规则（老代码 sample_y_for 照搬）：
        正区间，hi/lo > 1e3  → 对数均匀
        负区间，lo/hi > 1e3  → 对数均匀
        窄区间或跨零        → 线性均匀

    ⚠️ 对数均匀的意义：不会摇出贴 0 的 1e-29 这种值，
       那种值喂进 f⁻¹ 会得到 f(v)=0，后面全是 0。
    """
    e = torch.rand(n, device=DEV, dtype=DT, generator=gen)

    if lo > 0 and hi > 0 and hi / lo > LOG_SPAN:
        a, b = math.log(lo), math.log(hi)
        return torch.exp(e * (b - a) + a)
    if hi < 0 and lo < 0 and lo / hi > LOG_SPAN:
        a, b = math.log(-hi), math.log(-lo)
        return -torch.exp(e * (b - a) + a)
    return e * (hi - lo) + lo


def pick_range(p, pad: float = 0.02) -> tuple[float, float]:
    """
    ⭐ 一个函数能处理的输入范围（search，留余量）。

    ⚠️ 这里判 search，因为问的是「什么东西能喂给 f」。
       （f⁻¹ 的输入判 y_range，那是另一回事，见 optbase.y_range）
    """
    lo, hi = p.search
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return (0.0, 1.0)
    w = (hi - lo) * pad
    return (lo + w, hi - w)


# ══════════════════════════════════════════════════════════════════
# 二、指派链值
# ══════════════════════════════════════════════════════════════════

def assign_chain(n_true: int | None = None,
                 seed: int = 20261108,
                 gen: torch.Generator | None = None) -> dict:
    """
    ⭐ 摇 y0 ~ y10（11 个数），算 y11 = 乘积。

    ⚠️ 这是「还没绑定具体函数」的版本：
       先摇出数值，绑函数时用系数把它们搬进函数能算的范围。

    返回 dict：y0, ys(10,), y11, 全部在 GPU 张量里。
    """
    k = config.CHAIN_N_TRUE if n_true is None else int(n_true)
    if gen is None:
        gen = torch.Generator(device=DEV)
        gen.manual_seed(seed)

    # ⚠️ 首值和链值都用 [1, 9]：
    #    这个区间对绝大多数函数的 search 都在内部，
    #    系数只需要微调，不会挑到 1e-29 这种极端值。
    ys = sample_uniform(k, 1.0, 9.0, gen)
    y0 = sample_uniform(1, 1.0, 9.0, gen)

    prod = y0.clone()
    for v in ys:
        prod = prod * v
    y11 = prod

    return {'y0': y0, 'ys': ys, 'y11': y11, 'n': k}


# ══════════════════════════════════════════════════════════════════
# 三、干扰项：避开 2^11 − 1 个组合乘积
# ══════════════════════════════════════════════════════════════════

def avoid_products(nodes: torch.Tensor) -> torch.Tensor:
    """
    ⭐ m 个数的全部非空子集乘积 → 2^m − 1 个。

    ⚠️ 乘法没有顺序，是组合不是排列。
    ⚠️ 弗兰克：「乘法是没有顺序的，他们之间的所有组合都要排除，
                  这也是为什么我把计算放到 gpu，因为 cpu 做不完。」
    """
    m = int(nodes.numel())
    idx = torch.arange(1, 1 << m, device=nodes.device, dtype=torch.long)
    bits = ((idx.unsqueeze(1) >> torch.arange(m, device=nodes.device)
             .unsqueeze(0)) & 1).to(nodes.dtype)
    vals = torch.where(bits > 0, nodes.unsqueeze(0), torch.ones_like(bits))
    return vals.prod(dim=1)


def make_decoys(nodes: torch.Tensor, n_fake: int | None = None,
                lo: float = 1.0, hi: float = 9.0,
                tol: float = 1e-9, seed: int = 20261108,
                max_round: int = 64,
                eps: torch.Tensor | None = None,
                safe_k: float = 10.0) -> dict:
    """
    ⭐ 造 n_fake 个干扰项，每个都和 2^m−1 个组合乘积【全不一样】。

    ⚠️ 全在 GPU：候选 (M, 2047) 一次比完。

    ⚠️⚠️ 2026-11-09 弗兰克：精度隔离带
        「干扰项生成的时候，要避开这些精度区间，
          同时，干扰项自身要互相避开精度区间，以免生成近似的 y 选项。」

        ⇒ 每个元素有它自己的精度带 eps[i]。
           组合乘积的精度带 = 参与元素的 eps 之和（误差累积）。
           干扰项不能落进任何一个「值 ± safe_k × 精度带」里。
    """
    # 每个元素的精度带
    # ⚠️ 弗兰克 2026-11-09：「可以多一个数量级」
    #    ⇒ 用精度【上限】1e-6 再乘安全系数 10 → 隔离带 1e-5
    if eps is None:
        eps = torch.full_like(nodes, EPS_LIMIT)
    n_fake = config.CHAIN_N_FAKE if n_fake is None else int(n_fake)
    dev = nodes.device
    avoid = avoid_products(nodes)

    # ⭐ 每个组合乘积的精度带 = 参与元素的 eps 之和
    m = int(nodes.numel())
    idx = torch.arange(1, 1 << m, device=nodes.device, dtype=torch.long)
    bits = ((idx.unsqueeze(1) >> torch.arange(m, device=nodes.device)
             .unsqueeze(0)) & 1).to(nodes.dtype)
    avoid_eps = (bits * eps.unsqueeze(0)).sum(dim=1)      # (2047,)

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + 7)

    picked: list[torch.Tensor] = []
    have = 0
    rnd = 0
    while have < n_fake and rnd < max_round:
        rnd += 1
        m = n_fake * 4
        cand = sample_uniform(m, lo, hi, gen)
        # ① 与禁止集合比：不能相同，也不能落在精度带里
        d = (cand.unsqueeze(1) - avoid.unsqueeze(0)).abs()
        same = d <= (avoid.unsqueeze(0).abs() * tol
                     + safe_k * avoid_eps.unsqueeze(0))
        keep = ~same.any(dim=1)

        # ② 与本批已选中的干扰项比：互相也不能落在精度带里
        # ② 与本批已选中的干扰项比：互相也不能落在精度带里
        #    ⚠️ 已枚举出的干扰项，用「两者精度带之和」的距离来隔离
        if picked:
            prev = torch.cat(picked)
            dd = (cand.unsqueeze(1) - prev.unsqueeze(0)).abs()
            thr = (prev.unsqueeze(0).abs() * tol
                   + safe_k * (EPS_LIMIT + EPS_LIMIT))
            keep = keep & ~(dd <= thr).any(dim=1)

        good = cand[keep]
        if good.numel():
            picked.append(good)
            have += int(good.numel())

    fakes = (torch.cat(picked)[:n_fake] if picked
             else torch.zeros(0, device=dev, dtype=DT))
    return {'fakes': fakes, 'avoid': avoid,
            'ok': int(fakes.numel()) >= n_fake,
            'n_round': rnd, 'n_avoid': int(avoid.numel())}


# ══════════════════════════════════════════════════════════════════
# 四、一次造一整批：11 个链值 + 50 个干扰
# ══════════════════════════════════════════════════════════════════

def build_batch(seed: int = 20261108) -> dict:
    """
    ⭐ 一次造出：y0、y1..y10、y11、50 个干扰项。

    全部在 GPU。返回的 ys_all 是 60 个数（10 真 + 50 干扰）。
    """
    a = assign_chain(seed=seed)
    k = a['n']

    nodes = torch.cat([a['y0'].reshape(1), a['ys']])     # 11 个
    d = make_decoys(nodes, seed=seed)

    all_ys = torch.cat([a['ys'], d['fakes']])            # 10 + 50 = 60
    return {
        'y0': float(a['y0'][0]),
        'ys_true': a['ys'],           # (10,)
        'y11': float(a['y11'][0]),
        'ys_all': all_ys,             # (60,)
        'n_true': k,
        'n_fake': int(d['fakes'].numel()),
        'avoid': d['avoid'],
        'decoy_ok': d['ok'],
    }


# ══════════════════════════════════════════════════════════════════
# 五、自检
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
        print('assign2 自检（GPU 内部：链值 + 干扰项）')
        print('=' * 96)
        print('-' * 96)

    b = build_batch()
    ck('10 个链值', b['n_true'] == 10)
    ck('50 个干扰', b['n_fake'] == 50, '%d' % b['n_fake'])
    ck('60 个候选', b['ys_all'].numel() == 60)

    # y11 = 乘积
    p = b['y0']
    for v in b['ys_true'].tolist():
        p *= v
    ck('y11 = y0 × y1 × … × y10',
       abs(p - b['y11']) / abs(b['y11']) < 1e-12, 'y11=%.10g' % b['y11'])

    # 2047 个组合
    ck('禁止集合 = 2^11−1 = 2047', b['avoid'].numel() == 2047)
    ck('y11 在禁止集合里',
       bool((torch.abs(b['avoid'] - b['y11'])
             <= b['avoid'].abs().clamp(min=1e-300) * 1e-12).any()))

    # 干扰项
    f = b['ys_all'][10:]
    d = (f.unsqueeze(1) - b['avoid'].unsqueeze(0)).abs()
    same = d <= (b['avoid'].unsqueeze(0).abs() + 1e-300) * 1e-9
    ck('干扰项和 2047 个乘积全不同', not bool(same.any()),
       '撞上 %d 格' % int(same.sum()))

    ff = (f.unsqueeze(1) - f.unsqueeze(0)).abs()
    dup = ff <= (f.unsqueeze(0).abs() + 1e-300) * 1e-9
    dup.fill_diagonal_(False)
    ck('干扰项彼此不重复', not bool(dup.any()))

    # 都用 GPU
    ck('候选值在 GPU', b['ys_all'].device.type == 'cuda',
       str(b['ys_all'].device))

    # 可复现
    b2 = build_batch()
    ck('同种子可复现', bool(torch.equal(b['ys_all'], b2['ys_all'])))

    # 对数均匀
    x1 = sample_uniform(2000, 1e-6, 1e6,
                        torch.Generator(device=DEV).manual_seed(1))
    ck('对数均匀不贴 0（最小 > 1e-6）', float(x1.min()) > 1e-6,
       'min=%.3e max=%.3e' % (float(x1.min()), float(x1.max())))

    if verbose:
        print('-' * 96)
        print('y0  = %.10g' % b['y0'])
        print('y11 = %.10g' % b['y11'])
        print('链值 : %s' % [round(v, 4) for v in b['ys_true'].tolist()])
        print('干扰前 5: %s' % [round(v, 4) for v in f[:5].tolist()])
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
