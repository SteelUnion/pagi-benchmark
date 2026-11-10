"""
组内一致映射 —— 一道题内，编号 → 名词/描述 的映射必须唯一且固定

⭐ 2026-11-07 Frank 定：
    「我们对变量 x 和函数，还有要去求的 y 都做了编码，
      每一个选项内还打包了，
      那么这些变量在同一组里面必须保持一致。」

⭐⭐ 核心要求：
    **同一道题（一组）内，一个编号只能对应一个名词。**
    不管这个编号出现在：
        · 选项描述里
        · 工具描述里
        · 题面里
    都必须映射到**同一个词**。

⭐ 与「随机」的关系（这是关键区分）：
    ┌──────────────────────────────────────────────┐
    │ 组与组之间：完全随机（不同题用不同词）          │
    │ 组之内    ：完全固定（同一编号永远同一词）      │
    └──────────────────────────────────────────────┘

⚠️ 这一点不做对，会出现两个严重问题：
    ① **泄露**：同一个变量在不同工具里叫不同名字，
       模型可以靠「哪个名字重复出现」推断出链的结构
    ② **无解**：题面说「求（誓言）」，工具里只有「surprise」，
       模型无法把两者对应起来，题目直接不可解

⚠️ 2026-11-07 实测发现的旧 bug：
    原来用 `step` 索引名词表，而 50 个干扰工具的 step 都是 0，
    导致 103/120 次都取到同一个词（誓言 51 次、surprise 52 次）。
    **那不是一致性，是随机性塌缩。**
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, r'<BENCH>\generator')


# ══════════════════════════════════════════════════════════════════
# 一、组内映射表
# ══════════════════════════════════════════════════════════════════

@dataclass
class GroupMap:
    """
    一道题（一组）内的「编号 → 名词」映射。

    ⚠️ 一旦建好就**不再变化** —— 同一编号永远映射到同一个词。
    """
    group_id: str                              # 本组标识（题 UID）
    var_names: dict[str, str] = field(default_factory=dict)
    #   ↑ {变量编号: 名词}   如 {'A000': '誓言', 'X001': 'surprise'}

    var_langs: dict[str, str] = field(default_factory=dict)
    #   ↑ {变量编号: 语言码}

    used_words: set[str] = field(default_factory=set)
    #   ↑ 已用过的词（保证组内不重复）

    # ── 查询 ──────────────────────────────────────────────
    def name_of(self, pid: str) -> Optional[str]:
        """取某编号对应的名词。找不到返回 None。"""
        return self.var_names.get(pid)

    def lang_of(self, pid: str) -> str:
        return self.var_langs.get(pid, '')

    def n(self) -> int:
        return len(self.var_names)

    # ── 校验 ──────────────────────────────────────────────
    def is_consistent(self) -> tuple[bool, str]:
        """
        自检：映射是否满足「组内一致」。

        三条要求：
            ① 每个编号只有一个词（dict 天然保证）
            ② 不同编号的词互不相同（否则模型分不清是哪一路）
            ③ 没有空词
        """
        words = list(self.var_names.values())
        if any(not w for w in words):
            return False, '有空名词'
        if len(set(words)) != len(words):
            from collections import Counter
            dup = [w for w, c in Counter(words).items() if c > 1]
            return False, '名词重复：%s' % dup
        return True, ''

    def dump(self) -> str:
        """本组映射的文本形式（写日志用）。"""
        lines = ['组 %s 的变量映射（%d 个）' % (self.group_id, self.n())]
        for pid in sorted(self.var_names):
            lines.append('  %-6s → %-16s (%s)' % (
                pid, self.var_names[pid], self.var_langs.get(pid, '')))
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════
# 二、建组映射
# ══════════════════════════════════════════════════════════════════

def build_group_map(group_id: str,
                    var_pids: list[str],
                    seed: int = 0,
                    lang: Optional[str] = None,
                    avoid: Optional[set[str]] = None) -> GroupMap:
    """
    ⭐ 给一组（一道题）的变量分配名词。

    参数
    ----
    group_id : 本组标识（用题的 UID）
    var_pids : 变量编号列表，如 ['A000','X001',...,'X010']
    seed     : 随机种子（同 seed 同映射，保证可复现）
    lang     : 限定语言（None = 中英混用）
    avoid    : 要避开的词（如已经在函数描述里用过的）

    ⚠️ 一次挑够，**保证组内不重复**。
    """
    import lexicon as _lex

    gm = GroupMap(group_id=group_id)
    if not var_pids:
        return gm

    lex = _lex.get_lexicon()
    picked = lex.pick(len(var_pids) + 20, seed=seed,
                      langs=[lang] if lang else None)

    # ⚠️ 若传入 avoid，先过滤掉
    avoid = avoid or set()
    cand = [w for w in picked if w.text not in avoid]

    # ⚠️ 不够就再挑一批（换 seed）
    tries = 0
    while len(cand) < len(var_pids) and tries < 10:
        tries += 1
        more = lex.pick(len(var_pids) * 3, seed=seed + tries * 101,
                        langs=[lang] if lang else None)
        for w in more:
            if w.text not in avoid and all(w.text != c.text for c in cand):
                cand.append(w)
        if len(cand) >= len(var_pids):
            break

    # ⚠️ 仍不够 → 报错，不静默降级
    if len(cand) < len(var_pids):
        raise RuntimeError(
            '词库不足：需要 %d 个不同名词，只取到 %d 个'
            % (len(var_pids), len(cand)))

    # ── 按编号顺序分配（顺序固定 → 可复现）───────────────
    for pid, w in zip(sorted(var_pids), cand):
        gm.var_names[pid] = w.text
        gm.var_langs[pid] = w.lang
        gm.used_words.add(w.text)

    ok, why = gm.is_consistent()
    if not ok:
        raise RuntimeError('组映射不自洽：%s' % why)

    return gm


# ══════════════════════════════════════════════════════════════════
# 三、按编号取词（供各处调用）
# ══════════════════════════════════════════════════════════════════

def var_name(gm: GroupMap, pid: str, fallback: str = '') -> str:
    """
    ⭐ 取某编号的名词。这是**唯一入口** ——
       选项、工具、题面都必须经它取名，才能保证一致。
    """
    v = gm.name_of(pid)
    if v is None:
        raise KeyError('组 %s 里没有编号 %s 的映射（现有：%s）'
                       % (gm.group_id, pid, ', '.join(sorted(gm.var_names))[:60]))
    return v


def step_var(gm: GroupMap, step: int) -> tuple[str, str]:
    """
    ⭐ 取「第 step 步变换」对应的（输入名, 输出名）。

    ⚠️ step 是链上第几步（1..k）。
       第 step 步的输入是 V{step-1}，输出是 V{step}。
    ⭐ 这样**每个工具都能拿到自己那两位变量的名字**，
      而不是全部挤到同一个词上（旧的 bug）。
    """
    import module as _M
    # ⚠️ 2026-11-08 改名：A000(i=0) + X001..X010(i>=1)
    #    必须走 var_pid()，不能手写前缀
    in_pid = _M.var_pid(step - 1)
    out_pid = _M.var_pid(step)
    return var_name(gm, in_pid), var_name(gm, out_pid)


# ══════════════════════════════════════════════════════════════════
# 四、自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0
    checks = []

    # ── ① 基本建组 ──────────────────────────────────────
    import module as _M2
    pids = [_M2.var_pid(i) for i in range(11)]
    gm = build_group_map('TEST-GROUP', pids, seed=20261107)

    checks.append(('建出 11 个映射', gm.n() == 11))
    checks.append(('自洽', gm.is_consistent()[0]))
    checks.append(('每个变量都有词', all(gm.name_of(p) for p in pids)))
    checks.append(('11 个词互不相同',
                   len(set(gm.var_names.values())) == 11))

    # ── ② ⭐ 一致性：同一编号永远同一个词 ────────────────
    same = all(gm.name_of('X003') == gm.name_of('X003') for _ in range(100))
    checks.append(('同一编号取 100 次结果相同', same))

    # ── ③ ⭐ step_var 正确（这是修 bug 的关键）────────────
    ins, outs = [], []
    for s in range(1, 11):
        a, b = step_var(gm, s)
        ins.append(a)
        outs.append(b)
    checks.append(('10 步的输入名各不相同', len(set(ins)) == 10))
    checks.append(('10 步的输出名各不相同', len(set(outs)) == 10))
    # 相邻两步应首尾相接：第s步输出 == 第s+1步输入
    chain_ok = all(outs[i] == ins[i + 1] for i in range(9))
    checks.append(('相邻两步首尾相接（out[i]==in[i+1]）', chain_ok))

    # ── ④ 同 seed 可复现 ────────────────────────────────
    gm2 = build_group_map('TEST-GROUP', pids, seed=20261107)
    checks.append(('同 seed 同映射',
                   gm.var_names == gm2.var_names))

    # ── ⑤ 异 seed 不同 ──────────────────────────────────
    gm3 = build_group_map('TEST-GROUP2', pids, seed=999)
    checks.append(('异 seed 不同映射',
                   gm.var_names != gm3.var_names))

    # ── ⑥ 限定语言 ──────────────────────────────────────
    gm_zh = build_group_map('ZH', pids, seed=1, lang='zh')
    checks.append(('限定 zh 时全是中文',
                   all(gm_zh.lang_of(p) == 'zh' for p in pids)))

    # ── ⑦ 找不到编号要报错 ──────────────────────────────
    try:
        var_name(gm, 'Z999')
        checks.append(('取不存在的编号应报错', False))
    except KeyError:
        checks.append(('取不存在的编号应报错', True))

    # ── ⑧ ⭐ 用词分布检查（旧 bug 的回归测试）────────────
    #    在一道 10 层题的 10 个链上工具里，应该出现 10 种不同的输入名
    step_inputs = [step_var(gm, s)[0] for s in range(1, 11)]
    checks.append(('10 个工具用 10 种不同输入名',
                   len(set(step_inputs)) == 10))
    from collections import Counter
    c = Counter(step_inputs)
    checks.append(('没有某个词被过度使用（最多 1 次）',
                   max(c.values()) == 1))

    if verbose:
        print('=' * 62)
        print('groupmap 自检')
        print('=' * 62)
        for name, ok in checks:
            print('  %s  %s' % ('✅' if ok else '❌', name))
            if not ok:
                fails += 1
        print()
        print(gm.dump())
        print()
        print('10 步的（输入名 → 输出名）:')
        for s in range(1, 11):
            a, b = step_var(gm, s)
            print('  第%2d步  （%s） → （%s）' % (s, a, b))

    return fails


if __name__ == '__main__':
    raise SystemExit(self_test())
