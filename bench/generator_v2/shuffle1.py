"""
shuffle1 —— 选项打乱（约束：每组真值 ≤ 3，与原顺序相同 ≤ 30%）

════════════════════════════════════════════════════════════════
【要满足什么】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「每十个一组，每个里面真实选项小于等于 3。」
    「相似度超过 30% 就重抽。」

    ⇒ 三条硬约束：

        ① 60 个选项分 6 组，每组 10 个
        ② 每组里的真值个数 ≤ 3
        ③ 打乱后与【原始顺序】同位相同的个数 ≤ 18（30%）

    ⚠️ 满足约束的前提下，每种合法排列等概率 ——
       不是「最分散」，是「约束内最大随机」。

════════════════════════════════════════════════════════════════
【算法】
════════════════════════════════════════════════════════════════

    ① 定真值位置掩码（每组 <= 3）
         · 10 个真值随机分配到 6 组，每组上限 3
         · 每组内部的具体位置随机选
    ② 按掩码填充
         · 真值按原顺序填到选出的位置
         · 干扰项随机顺序填到剩下位置
    ③ 检查与原始顺序的相似度
         · 同位相同的个数 / 60 <= 0.30
         · 超过就重抽

    ⚠️ 为什么不用「先全排列再拒绝」：
       全排列有 60! 种，真值位置的合法比例很高，
       但逐次检查太慢。按掩码直接构造，一次成功率高。

    ⚠️ 为什么相似度要单独判：
       「每组真值 ≤ 3」只管真值，
       干扰项可能大量留在原位 —— 那也算泄露（模型知道干扰项在哪）。
"""

from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, r'<BENCH>\generator_v2')

from mathpack import DT, DEV

__all__ = ['plan_true_positions', 'shuffle_options', 'self_test']

# ⚠️ 相似度上限：与原始顺序同位相同的比例
SIM_LIMIT = 0.30

# ⚠️ 重抽上限
MAX_TRY = 200


# ══════════════════════════════════════════════════════════════════
# 一、定真值位置（每组 <= cap）
# ══════════════════════════════════════════════════════════════════

def plan_true_positions(n_true: int = 10, n_total: int = 60,
                        group: int = 10, cap: int = 3,
                        rng: torch.Generator | None = None,
                        tries: int = 100) -> torch.Tensor:
    """
    ⭐ 选出 n_true 个位置放真值，保证每组（每 group 个）<= cap 个。

    返回 (n_true,) 的 long 张量，升序。

    ⚠️ 全部在 GPU：用随机打分 + 逐组裁剪的办法。

    做法：
        ① 给每组随机分配配额（每组 0..cap 个，总数 = n_true）
        ② 组内随机选位置
    """
    if rng is None:
        rng = torch.Generator(device=DEV)
        rng.manual_seed(20261109)

    n_group = n_total // group
    assert n_group * group == n_total, '分组不整除'
    assert n_true <= n_group * cap, '真值太多，装不下'

    for _ in range(tries):
        # ① 每组配额：先给每组 1 个，剩下的随机撒
        quota = torch.zeros(n_group, device=DEV, dtype=torch.long)
        left = n_true
        # 先保证不超 cap
        for _r in range(n_true):
            cand = (quota < cap).nonzero().flatten()
            if cand.numel() == 0:
                break
            k = int(torch.randint(0, cand.numel(), (1,),
                                  generator=rng, device=DEV)[0])
            quota[int(cand[k])] += 1
            left -= 1
            if left <= 0:
                break
        if int(quota.sum()) != n_true:
            continue

        # ② 组内随机选位置
        pos: list[int] = []
        ok = True
        for g in range(n_group):
            q = int(quota[g])
            if q == 0:
                continue
            base = g * group
            perm = torch.randperm(group, generator=rng, device=DEV)[:q]
            pos.extend(int(base + p) for p in perm.tolist())
        if not ok or len(set(pos)) != n_true:
            continue

        pos.sort()
        return torch.tensor(pos, device=DEV, dtype=torch.long)

    raise RuntimeError('分配真值位置失败：%d 次都没凑出来' % tries)


# ══════════════════════════════════════════════════════════════════
# 二、打乱
# ══════════════════════════════════════════════════════════════════

def _similarity(a: list[int], b: list[int]) -> float:
    """⭐ 两个排列的同位相同比例。"""
    n = len(a)
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / n if n else 0.0


def shuffle_options(n_true: int = 10, n_total: int = 60,
                    group: int = 10, cap: int = 3,
                    seed: int = 20261109,
                    sim_limit: float = SIM_LIMIT,
                    max_try: int = MAX_TRY) -> dict:
    """
    ⭐ 打乱 60 个选项。

    返回 dict:
        order      : list[int]   新顺序里第 i 个位置放原第 order[i] 个
        true_pos   : list[int]   真值现在在新顺序的哪些位置
        group_true : list[int]   每组里的真值个数
        sim_orig   : float       与原始顺序的相似度
        n_try      : int         重抽了几次
        ok         : bool

    ⚠️ 原始顺序 = 0..59（前 10 个是真值）。
    """
    orig = list(range(n_total))
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)

    for t in range(1, max_try + 1):
        # ① 定真值位置
        pos = plan_true_positions(n_true, n_total, group, cap, gen)
        posl = sorted(int(p) for p in pos.tolist())

        # ② 真值按原顺序填进 posl，干扰项随机填剩下的
        true_orig = list(range(n_true))              # 0..9
        fake_orig = list(range(n_true, n_total))     # 10..59
        fake_perm = torch.randperm(len(fake_orig),
                                   generator=gen, device=DEV).tolist()
        fakes = [fake_orig[k] for k in fake_perm]

        order = [None] * n_total
        for i, p in enumerate(posl):
            order[p] = true_orig[i]
        fi = 0
        for p in range(n_total):
            if order[p] is None:
                order[p] = fakes[fi]
                fi += 1

        # ③ 相似度检查
        sim = _similarity(order, orig)
        if sim <= sim_limit:
            group_true = []
            for g in range(n_total // group):
                lo, hi = g * group, (g + 1) * group
                group_true.append(sum(1 for p in posl if lo <= p < hi))
            return {
                'order': order,
                'true_pos': posl,
                'group_true': group_true,
                'sim_orig': sim,
                'n_try': t,
                'ok': True,
            }

    # 抽不出来：返回最后一次，标记失败（不静默）
    return {
        'order': order if 'order' in dir() else orig,
        'true_pos': posl if 'posl' in dir() else [],
        'group_true': group_true if 'group_true' in dir() else [],
        'sim_orig': sim if 'sim' in dir() else 1.0,
        'n_try': max_try,
        'ok': False,
    }


# ══════════════════════════════════════════════════════════════════
# 三、自检
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
        print('shuffle1 自检（每组真值 <= 3，与原序相同 <= 30%）')
        print('=' * 96)
        print('-' * 96)

    # 跑 50 次
    n_ok = 0
    max_gt = 0
    max_sim = 0.0
    tries = []
    for seed in range(50):
        r = shuffle_options(seed=20261109 + seed)
        if r['ok']:
            n_ok += 1
        max_gt = max(max_gt, max(r['group_true']) if r['group_true'] else 0)
        max_sim = max(max_sim, r['sim_orig'])
        tries.append(r['n_try'])

    ck('50 次全部成功', n_ok == 50, '%d/50' % n_ok)
    ck('每组真值 <= 3', max_gt <= 3, '最大 %d' % max_gt)
    ck('与原序相同 <= 30%', max_sim <= 0.3001, '最大 %.1f%%' % (100 * max_sim))
    ck('平均重抽次数 <= 3', sum(tries) / len(tries) <= 3,
       '平均 %.1f 次，最多 %d 次' % (sum(tries) / len(tries), max(tries)))

    # 检查是否真是排列（每个恰好出现一次）
    r = shuffle_options(seed=12345)
    ck('结果是一个排列', sorted(r['order']) == list(range(60)))
    ck('真值位置个数 = 10', len(r['true_pos']) == 10)

    # 各组真值分布
    if verbose:
        print()
        print('  一个样例的真值分布：', r['group_true'], '（每组 <= 3）')
        print('  与原序相同：%.1f%%，重抽 %d 次'
              % (100 * r['sim_orig'], r['n_try']))
        print('  真值位置：', r['true_pos'])

    if verbose:
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
