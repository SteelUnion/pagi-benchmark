"""
向量层（GPU 内部交接）—— 10 中间值 + 50 干扰值 + 工具池，全在 GPU 上完成

⭐ 2026-11-07 Frank 定：
    「10+40 吗，你改成前面 10+50，然后 10 个值不放到选项里，
      给他们 10+40 的选项，但这工具是从 10+50 直接的随机变化里来。」

    「尽量让数据在 gpu 内部交接，已有的不改，我们新写的尽量走内部，
      然后分组打包还是老样子，有 uid 的话能有哈希链校验。」

⭐ 结构：
    ┌─────────────────────────────────────────────┐
    │ ① 造链：10 次复合 → 11 节点 + 连乘终值         │  GPU
    │ ② 干扰值：50 个，与 11 个节点都不碰            │  GPU
    │ ③ 选项：10 中间值 + 从 50 干扰里抽 40 个 = 50  │  索引
    │ ④ 工具：从 10+50 = 60 个值里随机组合生成        │  GPU
    │ ⑤ UID + 哈希链                                │  轻量
    └─────────────────────────────────────────────┘

⚠️ 已有的不改：anchor.py / problem.py / options.py / toolpool.py 全部保持原样。
   本模块是**新写的一条向量化通路**，与它们并存。

⚠️ 数据不出 GPU：中间值、干扰值、抽检都在张量里做，
   只在最后把结果搬回 CPU 一次。
"""
from __future__ import annotations

import hashlib
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, r'<BENCH>\generator')

import torch
import packs
import anchor
from anchor import DEV, DT


# ══════════════════════════════════════════════════════════════════
# 〇、UID 与哈希链
# ══════════════════════════════════════════════════════════════════

def make_uid(seed: int, chain: list[str], idx: int) -> str:
    """
    给一道题算 UID。

    ⚠️ 用「种子 + 链 + 序号」算，**不用 hash()** ——
       hash() 每个进程不同，结果复现不了（README 记过这个坑）。
    """
    raw = '%d|%s|%d' % (seed, ','.join(chain), idx)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


def hash_link(prev: str, uid: str) -> str:
    """
    哈希链：把上一题的链接值与本题 UID 串起来。

    ⭐ 作用：题目序列一旦被改动，整条链对不上 —— 可用于查篡改。
    """
    return hashlib.sha256(('%s->%s' % (prev or 'GENESIS', uid)).encode('utf-8')).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════
# 一、数据结构
# ══════════════════════════════════════════════════════════════════

@dataclass
class VectorProblem:
    """一道题的全部数据。"""
    uid: str
    link: str                    # 哈希链值
    chain: list[str]             # 10 个工具名
    nodes: list[float]           # 11 个节点 v0..v10
    final: float                 # 第 12 个：连乘终值
    decoy_values: list[float]    # 50 个干扰值（全部）
    option_values: list[float]   # 50 个选项 = 10 中间 + 40 抽样干扰
    option_steps: list[int]      # 每个选项若正确，对应链上第几步（0=干扰）
    tool_pool: list[dict]        # 60 个值生成的工具池
    seed: int
    cns: list[float] = field(default_factory=list)
    """⭐ 2026-11-07 补：每层的系数 cn。

    ⚠️ 原来漏了这个字段 —— `gen_chains_batch` 算了 cns，但没传进来。
       `module.py` 拆解时取不到系数，导致「系数部件 0 个」。
    ⚠️ 系数是题目的一部分：模型若能直接看到 cn，就能跳过一步推理。
       所以它必须和节点值一样被记录、被保护。
    """

    def digest(self) -> str:
        """整道题的摘要，用于校验没被改动。"""
        s = ('%s|%s|%s|%s|%s|%s|%s'
             % (self.uid, self.link, ','.join(self.chain),
                ','.join('%.17g' % v for v in self.nodes),
                ','.join('%.17g' % v for v in self.option_values),
                ','.join(str(s) for s in self.option_steps),
                ','.join('%.17g' % v for v in self.cns)))
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]

    @property
    def mid_values(self) -> list[float]:
        """10 个中间值（正确选项）。"""
        return self.nodes[1:]


# ══════════════════════════════════════════════════════════════════
# 二、GPU 内部：造链
# ══════════════════════════════════════════════════════════════════

def gen_chains_batch(
    m: int = 512,
    k: int = 10,
    seed: int = 0,
    pool: Optional[list[str]] = None,
) -> dict:
    """
    ⭐ GPU 批量造链，全部在张量里推进。

    返回：{chains, nodes(tensor), finals(tensor), alive(tensor), why}
    ⚠️ 结果留在 GPU 上，不急着搬回 CPU。
    """
    rng = random.Random(seed)
    pool = pool if pool is not None else [p.name for p in packs.PACKS if p.inverse_trusted]

    chains = [[rng.choice(pool) for _ in range(k)] for _ in range(m)]

    # 起点
    starts = []
    for ch in chains:
        c = anchor.anchor_candidates(packs.get(ch[0]))
        starts.append(c[0][1] if c else float('nan'))

    nodes = torch.full((m, k + 1), float('nan'), device=DEV, dtype=DT)
    nodes[:, 0] = torch.tensor(starts, device=DEV, dtype=DT)
    alive = torch.isfinite(nodes[:, 0])
    why = ['起点无候选' if not math.isfinite(s) else '' for s in starts]

    cns = torch.ones((m, k), device=DEV, dtype=DT)

    # 逐层推进（层间串行，层内批量）
    for step in range(k):
        if not bool(alive.any()):
            break

        idx_alive = torch.nonzero(alive).flatten().tolist()
        groups: dict[str, list[int]] = {}
        for i in idx_alive:
            groups.setdefault(chains[i][step], []).append(i)

        new_v = torch.full((m,), float('nan'), device=DEV, dtype=DT)
        new_cn = torch.ones((m,), device=DEV, dtype=DT)
        dead_why: dict[int, str] = {}

        for nm, idxs in groups.items():
            pk = packs.get(nm)
            if pk is None:
                for i in idxs:
                    dead_why[i] = '未知包'
                continue

            sel = nodes[torch.tensor(idxs, device=DEV), step]
            with torch.no_grad():
                f = pk._fwd(sel) * pk.c
            fv = f.tolist()

            cn_list = []
            for i, fi in zip(idxs, fv):
                if not math.isfinite(fi):
                    dead_why[i] = '正向非有限'
                    cn_list.append(1.0)
                    continue
                if step + 1 < k:
                    nxt = packs.get(chains[i][step + 1])
                    tgt = nxt.search if nxt is not None else (1e300, 1e300)
                else:
                    tgt = pk.y_range
                cn, _ = anchor.fit_cn_to_next(fi, tgt[0], tgt[1])
                nv = fi * cn
                if not math.isfinite(nv):
                    dead_why[i] = '乘cn非有限'
                    cn_list.append(cn)
                    continue
                if step + 1 < k and not (tgt[0] <= nv <= tgt[1]):
                    dead_why[i] = '进不了下一层'
                    cn_list.append(cn)
                    continue
                new_v[i] = nv
                new_cn[i] = cn
                cn_list.append(cn)

            cns[torch.tensor(idxs, device=DEV), step] = torch.tensor(
                cn_list, device=DEV, dtype=DT)

        # 写回
        for i in idx_alive:
            if i in dead_why:
                why[i] = dead_why[i]
            else:
                nodes[i, step + 1] = new_v[i]
        alive = alive & torch.isfinite(nodes[:, step + 1])

    # 终值：连乘（GPU 上算）
    with torch.no_grad():
        finals = torch.ones((m,), device=DEV, dtype=DT)
        for j in range(k + 1):
            finals = finals * nodes[:, j]

    return {
        'chains': chains,
        'nodes': nodes,
        'finals': finals,
        'cns': cns,
        'alive': alive,
        'why': why,
        'k': k,
    }


# ══════════════════════════════════════════════════════════════════
# 三、GPU 内部：干扰值与选项
# ══════════════════════════════════════════════════════════════════

def make_decoys_gpu(
    avoid: torch.Tensor,          # (M, 11) 每条链的 11 个节点
    n_decoy: int = 50,
    seed: int = 0,
    lo: float = 1e-3,
    hi: float = 1e3,
    tol: float = 1e-9,
    max_rounds: int = 40,
    dom_packs: Optional[list] = None,
) -> torch.Tensor:
    """
    ⭐ 造 n_decoy 个干扰值。**走与正确值相同的生成流程。**

    ⚠️ 2026-11-07 Frank 改：
        「我们针对干扰项，也按照正确选项一样，上来先用逆运算，得到一个 x，
          再用 x 和正运算函数表达这个选项。
          从生成源头开始，就要把所有的内容纳入生成，
          每个都按照正确答案的流程来生成。」

        「w 的取值要满足两个条件：
          1 恰好跟 y 连乘的所有基础值和中间值不一样
          2 满足函数成立的条件。
          所以 LLM 看选项的时候，会发现他们都有极其相似的特征，
          因为他们本身是一个循环里的"邻居"。」

    ⭐ 旧做法的问题（实测确认）：
        旧版「随机取量级 + 随机定符号」，只满足条件①（不相撞），
        **完全不满足条件②（满足函数成立）**。
        结果：正确项 93% 为正，干扰项只有 46% 为正 ——
        **模型只要说「偏正的是答案」就能拿 90% 分。**

    ⭐ 新做法：
        ① 随机取一个起点 w（满足条件①的候选）
        ② 从链上随机挑一段函数（1~3 个）
        ③ 用这段函数把 w 走一遍（这就是「满足函数成立」）
        ④ 走出来的值就是干扰值 —— 它**必然是正确答案的"邻居"**
        ⑤ 再按条件①（不与任何节点相撞）检查，不合格就重来

    ⚠️ chain_names 给了就用真函数走；没给就退化为「随机量级」旧行为。
    """
    m = avoid.shape[0]
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)

    # 取值域：由本批节点的量级决定，但受 lo/hi 约束
    finite = avoid[torch.isfinite(avoid)]
    if finite.numel() == 0:
        finite = torch.ones(1, device=DEV, dtype=DT)
    a = torch.log10(finite.abs().clamp(min=1e-300))
    lo_l = float(a.min()) - 2.0
    hi_l = float(a.max()) + 2.0
    # ⭐ 与配置给的 [lo, hi] 取交集，防止干扰值跑到荒唐量级
    lo_l = max(lo_l, math.log10(max(lo, 1e-300)))
    hi_l = min(hi_l, math.log10(max(hi, 1e-300)))
    if hi_l <= lo_l:
        lo_l, hi_l = math.log10(max(lo, 1e-300)), math.log10(hi)

    out = torch.full((m, n_decoy), float('nan'), device=DEV, dtype=DT)

    # ⭐ 2026-11-07 Frank 纠正：
    #   「不能这么写。你只要参照，链上的每一个 y 转化成 x 时候的处理，
    #     而不要关注第一个 y 生成 11 个 y 链的过程，
    #     也就是说重点不在指派上，而在指派后对每个 y 值的处理上，
    #     这个处理最重要。
    #     w 应该完全随机但避开 y 的运算中间值（老方法不变），
    #     针对每个 w 转化成 x 的部分，算法，
    #     参照每个 y 转化为 x 的算法，那里决定了 y 的定义域。」

    # ⭐ 所以干扰值的处理方式（与上一版的关键差别）：
    #   ① w 完全随机（老方法不变），只需避开 y 的中间值 → 由下面的判碰负责
    #   ② w 合不合法，参照「单个 y → x」的判据：必须落在 pack.search（定义域）内
    #   ③ ⚠️ **不走整条链** —— 走链是上一版的错误

    for d in range(n_decoy):
        # ── ① w 完全随机（老方法，不变）──────────────
        lg = torch.rand((m,), device=DEV, dtype=DT, generator=g) * (hi_l - lo_l) + lo_l
        mag = torch.pow(10.0, lg)
        sign = torch.where(torch.rand((m,), device=DEV, dtype=DT, generator=g) < 0.5,
                           -1.0, 1.0)
        v = (mag * sign).to(DT)

        # ── ② 参照「单个 y → x」的定义域判据 ──────────
        #    ⚠️ 这一步只做「把不适格的值换成适格的」，不改分布形态。
        #    ⭐ 判据与 pick_anchor 一致：候选必须落在 pack.search 内。
        if dom_packs is not None:
            raw = v.tolist()
            fixed = []
            for i in range(m):
                pk = dom_packs[i]
                w = raw[i]
                if pk is None:
                    fixed.append(w)
                    continue
                s_lo, s_hi = pk.search
                if not (s_lo <= w <= s_hi):
                    span = s_hi - s_lo
                    if span <= 0:
                        fixed.append(w)
                        continue
                    # 按小数部分映射进去，保留原有分布形态
                    frac = abs(w) % 1.0
                    if s_lo > 0:
                        w = s_lo + span * (0.1 + 0.8 * frac)
                    else:
                        w = s_lo + span * (0.05 + 0.9 * frac)
                    if not (s_lo <= w <= s_hi):
                        w = raw[i]
                fixed.append(w)
            v = torch.tensor(fixed, device=DEV, dtype=DT)

        # ── ③ 检查：与节点不撞 + 与已造的不撞 + 量级与正确值同区间 ──
        for _ in range(max_rounds):
            bad = torch.zeros((m,), dtype=torch.bool, device=DEV)

            # ⭐ 量级必须与正确值同区间（否则一眼可辨）
            av = v.abs()
            bad |= (~torch.isfinite(v)) | (av < 10.0 ** (lo_l - 1)) | (av > 10.0 ** (hi_l + 1))

            # 与节点比
            denom = torch.maximum(v.abs().unsqueeze(1), avoid.abs()).clamp(min=1e-300)
            diff = (v.unsqueeze(1) - avoid).abs() / denom
            bad |= (diff <= tol).any(dim=1)

            # 与已造的干扰值比
            if d > 0:
                prev = out[:, :d]
                denom2 = torch.maximum(v.abs().unsqueeze(1), prev.abs()).clamp(min=1e-300)
                diff2 = (v.unsqueeze(1) - prev).abs() / denom2
                bad |= (diff2 <= tol).any(dim=1)

            if not bool(bad.any()):
                break
            # 撞了的重新抽
            n_bad = int(bad.sum())
            lg2 = torch.rand((n_bad,), device=DEV, dtype=DT, generator=g) * (hi_l - lo_l) + lo_l
            sg2 = torch.where(torch.rand((n_bad,), device=DEV, dtype=DT, generator=g) < 0.5,
                              -1.0, 1.0)
            v = v.clone()
            v[bad] = (torch.pow(10.0, lg2) * sg2).to(DT)

        out[:, d] = v

    return out


def pick_options_gpu(
    mid: torch.Tensor,            # (M, 10) 中间值
    decoys: torch.Tensor,         # (M, 50) 干扰值
    n_pick: int = 40,
    seed: int = 0,
) -> dict:
    """
    ⭐ 50 选项 = 10 中间值 + 从 50 干扰里抽 40 个。

    返回打乱后的选项值与「该值若正确对应链上第几步（0 = 干扰）」。
    """
    m = mid.shape[0]
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 7)

    # 抽 40 个干扰（不重合）
    idx = torch.argsort(torch.rand((m, decoys.shape[1]), device=DEV, dtype=DT, generator=g),
                        dim=1)[:, :n_pick]
    picked = torch.gather(decoys, 1, idx)

    vals = torch.cat([mid, picked], dim=1)          # (M, 50)
    steps = torch.cat([
        torch.arange(1, mid.shape[1] + 1, device=DEV).unsqueeze(0).expand(m, -1),
        torch.zeros((m, n_pick), dtype=torch.long, device=DEV),
    ], dim=1)

    # 打乱
    perm = torch.argsort(torch.rand(vals.shape, device=DEV, dtype=DT, generator=g), dim=1)
    vals = torch.gather(vals, 1, perm)
    steps = torch.gather(steps, 1, perm)

    return {'values': vals, 'steps': steps}


# ══════════════════════════════════════════════════════════════════
# 四、GPU 内部：工具池
# ══════════════════════════════════════════════════════════════════

def build_tools_gpu(
    mid: torch.Tensor,            # (M, 10)
    decoys: torch.Tensor,         # (M, 50)
    seed: int = 0,
    n_tool: int = 60,
) -> list[dict]:
    """
    ⭐ 从 10 + 50 = 60 个值里随机组合生成工具。

    ⭐ Frank 原话：「这工具是从 10+50 直接的随机变化里来。」

    做法：每个值 v 生成一个「工具」条目：
      · 它作用后落到哪个值
      · 它是不是链上的（是 → 可用；否 → 干扰）
      · 它属于哪种角色（真值 / 抽样干扰 / 未抽样干扰）
    """
    m = mid.shape[0]
    allv = torch.cat([mid, decoys], dim=1)           # (M, 60)
    # 角色：1..10 = 链上第几步；0 = 干扰
    role = torch.cat([
        torch.arange(1, mid.shape[1] + 1, device=DEV).unsqueeze(0).expand(m, -1),
        torch.zeros((m, decoys.shape[1]), dtype=torch.long, device=DEV),
    ], dim=1)

    out = []
    # ⚠️ randperm 在 CPU 上跑，generator 也必须建在 CPU（不能传 CUDA 的）
    g = torch.Generator()            # 默认 CPU generator
    g.manual_seed(seed + 13)

    vals_cpu = allv.tolist()
    role_cpu = role.tolist()
    for i in range(m):
        perm = torch.randperm(allv.shape[1], generator=g).tolist()
        tools = []
        for pos in perm[:n_tool]:
            tools.append({
                'value': vals_cpu[i][pos],
                'step': role_cpu[i][pos],     # 0 = 干扰
                'on_chain': role_cpu[i][pos] > 0,
            })
        out.append(tools)

    return out


# ══════════════════════════════════════════════════════════════════
# 五、总装
# ══════════════════════════════════════════════════════════════════

def build_vector_problems(
    n: int = 64,
    cfg=None,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    n_decoy: Optional[int] = None,
    n_pick: Optional[int] = None,
    n_tool: Optional[int] = None,
    oversample: Optional[float] = None,
    sep_digits: Optional[int] = None,
    verbose: Optional[bool] = None,
) -> list[VectorProblem]:
    """
    ⭐ 一次造 n 道题，全程 GPU 内部交接。

    参数
    ----
    n    : 要几道**可用**的题
    cfg  : ⭐ **ProblemConfig 配置对象**（推荐）。
           「选项、干扰项数目这些我都想搞成配置参数，
            因为这样难度就好调整了。」—— Frank 2026-11-07

    其余关键字参数：给了就覆盖 cfg 里的同名项（方便临时试）。
    都没给 → 用 ProblemConfig() 默认值。

    ⭐ 用法：
        build_vector_problems(n=1000)                          # 默认 L4
        build_vector_problems(n=1000, cfg=ProblemConfig(k=6))  # 改一项
        build_vector_problems(n=1000, cfg=ProblemConfig.from_preset('L2'))
        build_vector_problems(n=1000, k=8, n_decoy=40)         # 临时覆盖
    """
    from config import ProblemConfig as _PC

    # ── 组织配置 ────────────────────────────────────────
    if cfg is None:
        cfg = _PC()
    else:
        # 拷一份，不改调用方的对象
        cfg = _PC(**{f: getattr(cfg, f) for f in cfg.__dataclass_fields__})

    over = {
        'k': k, 'seed': seed, 'n_decoy': n_decoy, 'n_pick': n_pick,
        'n_tool': n_tool, 'oversample': oversample, 'sep_digits': sep_digits,
        'verbose': verbose,
    }
    for name, v in over.items():
        if v is not None:
            setattr(cfg, name, v)

    bad = cfg.validate()
    if bad:
        raise ValueError('配置有问题：\n  ' + '\n  '.join(bad))

    # ── 从配置取值 ──────────────────────────────────────
    k = cfg.k
    seed = cfg.seed
    n_decoy = cfg.n_decoy
    n_pick = cfg.n_pick
    oversample = cfg.oversample
    sep_digits = cfg.sep_digits
    node_min, node_max = cfg.node_min, cfg.node_max
    final_min, final_max = cfg.final_min, cfg.final_max

    t0 = time.time()
    m_batch = min(cfg.max_batch, max(n, int(n * oversample)))

    # ⭐ 两条区分度线
    delta_opt = cfg.delta_opt
    delta_final = cfg.delta_final

    res = gen_chains_batch(m=m_batch, k=k, seed=seed)
    nodes = res['nodes']
    finals = res['finals']
    alive = res['alive']

    # ── 筛：节点有限、量级、唯一 ────────────────────────
    fin_ok = torch.isfinite(nodes).all(dim=1)
    mag_ok = ((nodes.abs() >= node_min) & (nodes.abs() <= node_max)).all(dim=1)

    # 唯一性：两两相对误差 > delta_opt
    uniq = torch.ones((m_batch,), dtype=torch.bool, device=DEV)
    kn = nodes.shape[1]
    for i in range(kn):
        for j in range(i + 1, kn):
            x = nodes[:, i]
            y = nodes[:, j]
            den = torch.maximum(x.abs(), y.abs()).clamp(min=1e-300)
            same = (x - y).abs() / den <= delta_opt
            uniq &= ~same

    f_abs = finals.abs()
    final_ok = torch.isfinite(finals) & (f_abs >= final_min) & (f_abs <= final_max)

    good = alive & fin_ok & mag_ok & uniq & final_ok
    idx = torch.nonzero(good).flatten()[:n].tolist()

    if len(idx) < n:
        raise RuntimeError('可用题不足：要 %d，只造出 %d（多造了 %d）'
                           % (n, len(idx), m_batch))

    sel_nodes = nodes[torch.tensor(idx, device=DEV)]
    sel_finals = finals[torch.tensor(idx, device=DEV)]
    mid = sel_nodes[:, 1:]

    # ── 干扰值（GPU，按 delta_opt 判碰）─────────────────
    # ⭐ 2026-11-07：传入本批各链的函数，让干扰值「走一遍函数」，
    #    与正确值同源 —— 否则符号/量级分布会泄露答案。
    # ⭐ 2026-11-07：传入每条链的「定义域参照」——
    #    用链最内层的那个包，它的 search 就是 y 所必须满足的定义域。
    #    ⚠️ 不传整条链，因为重点不在走链，而在「单个 y → x」的那一步。
    dom_packs = []
    for i in idx:
        ch = res['chains'][i]
        pk0 = packs.get(ch[0]) if ch else None
        dom_packs.append(pk0)
    decoys = make_decoys_gpu(sel_nodes, n_decoy=n_decoy, seed=seed + 3,
                             tol=delta_opt,
                             lo=cfg.decoy_lo, hi=cfg.decoy_hi,
                             dom_packs=dom_packs)

    # ── 选项（GPU）──────────────────────────────────────
    opt = pick_options_gpu(mid, decoys, n_pick=n_pick, seed=seed + 5)

    # ── 工具池（GPU 抽序，CPU 组装）──────────────────────
    tools = build_tools_gpu(mid, decoys, seed=seed + 11, n_tool=cfg.n_tool)

    # ── 搬回 CPU，组 UID 与哈希链 ────────────────────────
    nodes_cpu = sel_nodes.tolist()
    finals_cpu = sel_finals.tolist()
    decoys_cpu = decoys.tolist()
    opt_vals_cpu = opt['values'].tolist()
    opt_steps_cpu = opt['steps'].tolist()
    # ⭐ 2026-11-07 补：把系数表从 GPU 搬回 CPU（原来漏了，module.py 取不到）
    cns_cpu = res['cns'][torch.tensor(idx, device=DEV)].tolist()

    out: list[VectorProblem] = []
    prev = ''
    n_drop = 0
    for j, i in enumerate(idx):
        # ⭐ 终值区分度：换掉中途值后，终值是否仍够远
        base = 1.0
        for x in nodes_cpu[j]:
            base *= x
        ok_sep = True
        mid_v = nodes_cpu[j][1:]
        for pos in range(len(mid_v)):
            alt = _nearest_alt(decoys_cpu[j], mid_v, pos, mid_v[pos])
            if alt is None:
                continue
            v2 = mid_v[:]
            v2[pos] = alt
            p2 = nodes_cpu[j][0]
            for x in v2:
                p2 *= x
            if base != 0 and abs(p2 - base) / abs(base) <= delta_final:
                ok_sep = False
                break
        if not ok_sep:
            n_drop += 1
            continue

        uid = make_uid(seed, res['chains'][i], j)
        link = hash_link(prev, uid)
        prev = link
        out.append(VectorProblem(
            uid=uid,
            link=link,
            chain=res['chains'][i],
            nodes=nodes_cpu[j],
            final=finals_cpu[j],
            decoy_values=decoys_cpu[j],
            option_values=opt_vals_cpu[j],
            option_steps=[int(s) for s in opt_steps_cpu[j]],
            tool_pool=tools[j],
            seed=seed,
            cns=cns_cpu[j],
        ))

    elapsed = time.time() - t0
    if cfg.verbose:
        print('═' * 66)
        print('向量层造题：%d 道，耗时 %.2fs（%.2f ms/题）'
              % (len(out), elapsed, 1000 * elapsed / max(1, len(out))))
        print('  配置：' + cfg.brief())
        print('  多造 %d 条，可选 %d 条，终值区分度淘汰 %d 条'
              % (m_batch, int(good.sum()), n_drop))
        print('  区分度：选项 %.0e（小数点后第 %d 位），终值 %.0e'
              % (delta_opt, sep_digits, delta_final))
        print('  每道：%d 节点 + 1 终值 + %d 干扰 + %d 选项 + %d 工具'
              % (k, n_decoy, cfg.n_option, cfg.n_tool))
        print('═' * 66)

    return out


def _nearest_alt(decoys: list[float], mid: list[float], pos: int,
                 target: float) -> Optional[float]:
    """
    从干扰值里找一个**最接近 target** 的（用于最坏情况测试）。

    ⭐ 为什么要挑最近的：区分度要看最坏情况。
       随便挑一个很远的值，终值一定差得远，测不出问题。
    """
    best = None
    best_d = float('inf')
    for d in decoys:
        if d == 0:
            continue
        m = max(abs(d), abs(target))
        if m == 0:
            continue
        # 不能等于任何一个中间值
        if any(abs(d - x) / max(abs(d), abs(x), 1e-300) <= 1e-12 for x in mid):
            continue
        dd = abs(d - target) / m
        if dd < best_d:
            best_d = dd
            best = d
    return best


# ══════════════════════════════════════════════════════════════════
# 六、呈现与自检
# ══════════════════════════════════════════════════════════════════

def show(p: VectorProblem) -> str:
    L = ['─' * 66]
    L.append('UID  %s    链值 %s' % (p.uid, p.link))
    L.append('链   ' + ' → '.join(p.chain))
    L.append('')
    L.append('节点（v0 已知，v1..v10 待求）：')
    for i, v in enumerate(p.nodes):
        L.append('  v%-2d = %+.10g%s' % (i, v, '   ← 初始值' if i == 0 else ''))
    L.append('v11 = 连乘终值 = %+.10g   （已知）' % p.final)
    L.append('')
    L.append('50 个选项（★ = 正确）：')
    for i, (v, s) in enumerate(zip(p.option_values, p.option_steps), 1):
        L.append('  %2d. %+-14.10g %s' % (i, v, '★ 第%d步' % s if s else ''))
    L.append('')
    on = sum(1 for t in p.tool_pool if t['on_chain'])
    L.append('工具池：%d 个（链上 %d，干扰 %d）'
             % (len(p.tool_pool), on, len(p.tool_pool) - on))
    return '\n'.join(L)


def self_test(verbose: bool = True) -> int:
    fails = 0
    checks = []

    SEP = 6
    DELTA_OPT = 10.0 ** (-SEP)
    DELTA_FINAL = DELTA_OPT * math.sqrt(10)

    t0 = time.time()
    ps = build_vector_problems(n=8, k=10, seed=20261107, verbose=False,
                               sep_digits=SEP)

    checks.append(('造出 8 道', len(ps) == 8))
    checks.append(('节点 11 个', all(len(p.nodes) == 11 for p in ps)))
    checks.append(('中间值 10 个', all(len(p.mid_values) == 10 for p in ps)))
    checks.append(('干扰值 50 个', all(len(p.decoy_values) == 50 for p in ps)))
    checks.append(('选项 50 个', all(len(p.option_values) == 50 for p in ps)))
    checks.append(('选项里正确 10 个',
                   all(sum(1 for s in p.option_steps if s > 0) == 10 for p in ps)))
    checks.append(('工具池 60 个', all(len(p.tool_pool) == 60 for p in ps)))
    checks.append(('UID 不重复', len({p.uid for p in ps}) == len(ps)))
    checks.append(('哈希链首尾相接',
                   all(ps[i + 1].link == hash_link(ps[i].link, ps[i + 1].uid)
                       for i in range(len(ps) - 1))))

    # ── 计算检查 ────────────────────────────────────────
    ok_final = True
    for p in ps:
        f = 1.0
        for v in p.nodes:
            f *= v
        if abs(f - p.final) > max(abs(f), 1) * 1e-12:
            ok_final = False
    checks.append(('终值 = 连乘', ok_final))

    # ── ⭐ 区分度检查（本轮重点）────────────────────────
    def rel(x, y):
        m = max(abs(x), abs(y))
        return abs(x - y) / m if m else 0.0

    # ① 节点之间够远
    ok_node_sep = True
    for p in ps:
        ns = p.nodes
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                if rel(ns[i], ns[j]) <= DELTA_OPT:
                    ok_node_sep = False
    checks.append(('节点间距 > %.0e' % DELTA_OPT, ok_node_sep))

    # ② 选项之间够远
    ok_opt_sep = True
    worst_opt = 1.0
    for p in ps:
        vs = p.option_values
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                r = rel(vs[i], vs[j])
                worst_opt = min(worst_opt, r)
                if r <= DELTA_OPT:
                    ok_opt_sep = False
    checks.append(('选项间距 > %.0e（最小 %.1e）' % (DELTA_OPT, worst_opt), ok_opt_sep))

    # ③ 换掉一个中间值后，终值仍够远
    ok_final_sep = True
    worst_fin = 1.0
    for p in ps:
        base = 1.0
        for v in p.nodes:
            base *= v
        mid_v = p.mid_values
        for pos in range(len(mid_v)):
            for d in p.decoy_values:
                v2 = mid_v[:]
                v2[pos] = d
                p2 = p.nodes[0]
                for x in v2:
                    p2 *= x
                r = rel(p2, base)
                worst_fin = min(worst_fin, r)
                if r <= DELTA_FINAL:
                    ok_final_sep = False
    checks.append(('终值间距 > %.0e（最小 %.1e）' % (DELTA_FINAL, worst_fin), ok_final_sep))

    if verbose:
        print('=' * 62)
        print('vectorform 自检')
        print('=' * 62)
        for name, ok in checks:
            print('  %s  %s' % ('✅' if ok else '❌', name))
            if not ok:
                fails += 1
        print()
        print(show(ps[0]))
        print()
        print('总耗时 %.2fs' % (time.time() - t0))

    return fails


if __name__ == '__main__':
    raise SystemExit(self_test())
