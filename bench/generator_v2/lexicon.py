"""
词汇库加载与随机挑选 —— 自动扫描目录、检查格式、按语言编号

⭐ 2026-11-07 Frank 定：
    「首先给这几批词汇编号，用脚本就行，中文用 cn 加数字，英语用 en 加数字，
      然后要有一个随机挑选器，按照需要替换的变量数量，
      随机挑选相应数量的词汇出来替换，支持多个 txt，
      在多个 txt 中均匀随机，（未来可以加入其他语言），目前就是 2 个，
      我的建议是他会自动搜索目录下 txt，并检查格式，
      符合格式的认定为词汇并按照编号计数，来实现随机。
      不符合格式的不做认定，这样也好，告诉了别人添加字典要什么格式。」

⭐ 设计要点：
    ① **自动扫描目录**：不用手工登记，把 txt 丢进去就行
    ② **格式自检**：符合格式的才算词汇库，不合的不认
    ③ **按语言编号**：文件名前缀决定语言（cn / en / jp …）
    ④ **多文件均匀随机**：先均匀选文件，再在文件内均匀选词
    ⑤ **可扩展**：新增语言 = 新增一个符合格式的 txt

⭐ 词汇库格式（这是给"添加字典的人"看的约定）：

    ```
    文件名：<语言码>_<说明>.txt     如 nouns_zh.txt → 语言码 zh
    编码：  UTF-8（可带 BOM，自动处理）
    每行：  一个词，前后不能有空格
    空行：  忽略
    注释：  # 开头的行忽略
    最短：  至少 10 个有效词才认定为词汇库
    ```

    ⚠️ 语言码取文件名中第一个下划线前的部分：
       nouns_zh.txt → 语言码 zh → 编号 CN001…
       nouns_en.txt → 语言码 en → 编号 EN001…

    ⚠️ 同一语言多个文件时，编号连续（不重置）。
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

WORDLIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordlib')

# ⚠️ 语言码 → 编号前缀
LANG_PREFIX = {
    'zh': 'CN',
    'cn': 'CN',
    'en': 'EN',
    'jp': 'JP',
    'ja': 'JP',
    'fr': 'FR',
    'de': 'DE',
    'ru': 'RU',
    'es': 'ES',
    'ar': 'AR',
}

# ⚠️ 非法文件（虽然是 txt，但不是词汇库）
#    ⭐ 靠文件名特征排除，而不是靠格式碰运气
NON_WORDLIST_PATTERNS = [
    r'^exclude',          # exclude_extra.txt
    r'^readme',
    r'^license',
    r'_verify$',          # gsc_level_1_verify.txt（验证用，不是词库）
]

MIN_WORDS = 10             # 少于这个词数不认定为词汇库


# ══════════════════════════════════════════════════════════════════
# 一、词汇条目
# ══════════════════════════════════════════════════════════════════

@dataclass
class Word:
    """一个编号词条。"""
    wid: str          # 编号，如 CN0001 / EN0001
    text: str         # 词本身
    lang: str         # 语言码
    src: str          # 来源文件名


# ══════════════════════════════════════════════════════════════════
# 二、格式检查
# ══════════════════════════════════════════════════════════════════

def _lang_of(filename: str) -> Optional[str]:
    """
    从文件名判断语言码。

    ⚠️ 2026-11-07 实测修正：
       第一版取「第一个下划线前」的部分当语言码，
       结果 nouns_zh.txt → 'nouns'（错），gsc_level_1.txt → 'gsc'（错）。
       ⭐ 语言码可能在**任意一段**，所以逐段找「已登记的语言码」。

    规则：
        nouns_zh.txt → 'zh'
        nouns_en.txt → 'en'
        zh_words.txt → 'zh'
        words_ja.txt → 'ja'
        gsc_level_1.txt → None（没有任何一段是语言码）
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = re.split(r'[_\-\s]+', base)
    for p in parts:
        pl = p.lower()
        if pl in LANG_PREFIX:
            return pl
    # 整体就是一个语言码（如 zh.txt）
    if base.lower() in LANG_PREFIX:
        return base.lower()
    return None


def _is_non_wordlist(filename: str) -> bool:
    """文件名的形态看起来不像词汇库（如 exclude / readme / _verify）。"""
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    for pat in NON_WORDLIST_PATTERNS:
        if re.search(pat, base):
            return True
    return False


def load_wordlist(path: str) -> tuple[list[str], str]:
    """
    读一个 txt，返回 (有效词列表, 问题描述)。
    问题描述为空串表示格式合格。

    ⚠️ 校验规则（写进文档，告诉添加字典的人）：
       ① 必须是 UTF-8 编码
       ② 每行一个词，前后不能有空格
       ③ 空行和 # 开头的行忽略
       ④ 至少 MIN_WORDS 个有效词
    """
    if not os.path.isfile(path):
        return [], '文件不存在'

    # ── ① 编码 ──────────────────────────────────────────
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:   # utf-8-sig 自动去 BOM
            raw = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='gb18030') as f:
                raw = f.read()
            return [], '不是 UTF-8 编码（检测到 GB18030）'
        except Exception:
            return [], '编码无法识别'

    # ── ② 逐行读 ────────────────────────────────────────
    words: list[str] = []
    bad_lines: list[tuple[int, str]] = []
    for i, line in enumerate(raw.split('\n'), 1):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        # ⚠️ 行内有空白 → 不是「一个词一行」的格式
        if any(c.isspace() for c in s):
            bad_lines.append((i, s))
            continue
        # ⚠️ 过长的行也不是词（可能是段说明）
        if len(s) > 40:
            bad_lines.append((i, s[:40] + '…'))
            continue
        words.append(s)

    if bad_lines:
        return words, '有 %d 行格式不符（首例 第%d行: %s）' % (
            len(bad_lines), bad_lines[0][0], bad_lines[0][1])

    # ── ③ 数量 ──────────────────────────────────────────
    if len(words) < MIN_WORDS:
        return words, '有效词只有 %d 个（少于 %d）' % (len(words), MIN_WORDS)

    return words, ''


# ══════════════════════════════════════════════════════════════════
# 三、词库注册表
# ══════════════════════════════════════════════════════════════════

@dataclass
class WordTrack:
    """一种语言的全部词汇（可能来自多个文件）。"""
    lang: str
    prefix: str                     # 编号前缀，如 CN
    words: list[Word] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.words)


@dataclass
class Lexicon:
    """
    全部词库。

    ⚠️ 按语言分组。每种语言一条 track，编号在该语言内连续。
    """
    tracks: dict[str, WordTrack] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (文件, 原因)

    # ── 查询 ──────────────────────────────────────────────
    @property
    def langs(self) -> list[str]:
        return sorted(self.tracks)

    def total(self) -> int:
        return sum(t.n for t in self.tracks.values())

    def get(self, wid: str) -> Optional[Word]:
        for t in self.tracks.values():
            for w in t.words:
                if w.wid == wid:
                    return w
        return None

    def summary(self) -> str:
        parts = ['%s:%d' % (l, self.tracks[l].n) for l in self.langs]
        return '词库 %d 种语言，共 %d 词（%s）' % (
            len(self.tracks), self.total(), ' '.join(parts))

    # ── 随机挑选 ──────────────────────────────────────────
    def pick(self, n: int, seed: int = 0,
             langs: Optional[list[str]] = None) -> list[Word]:
        """
        ⭐ 随机挑 n 个词，**在多种语言之间均匀随机**。

        做法（两级均匀）：
            ① 先均匀选语言
            ② 再在该语言内均匀选词

        ⚠️ 为什么不用「全部词混在一起随机」：
           中文 3532 词、英文 3000 词，混在一起随机的话
           中文会被选中的概率略高（53.9% vs 46.1%）。
           ⭐ 两级均匀保证「每种语言被选中的次数大致相等」。

        ⚠️ 同一批挑选内**不重复**。
        """
        use = langs if langs else self.langs
        use = [l for l in use if l in self.tracks and self.tracks[l].n > 0]
        if not use:
            return []
        if n <= 0:
            return []

        rng = random.Random(seed)
        picked: list[Word] = []
        seen: set[str] = set()

        # ⚠️ 防死循环：若需要的数量超过总词数，最多给到总数
        cap = sum(self.tracks[l].n for l in use)
        n = min(n, cap)

        guard = 0
        while len(picked) < n and guard < n * 100:
            guard += 1
            lang = rng.choice(use)              # ① 均匀选语言
            t = self.tracks[lang]
            w = rng.choice(t.words)             # ② 语言内均匀选词
            if w.wid in seen:                   # 本批不重复
                continue
            seen.add(w.wid)
            picked.append(w)

        return picked


# ══════════════════════════════════════════════════════════════════
# 四、扫描目录并建库
# ══════════════════════════════════════════════════════════════════

def scan(directory: str = WORDLIB, verbose: bool = False) -> Lexicon:
    """
    ⭐ 扫描目录下所有 txt，检查格式，建立带编号的词库。

    ⚠️ 不符合格式的文件**跳过并记录原因**，不报错 ——
       这样别人加字典加错了，程序照常跑，只是那个文件不生效。
    """
    lex = Lexicon()
    if not os.path.isdir(directory):
        return lex

    files = sorted(f for f in os.listdir(directory)
                   if f.lower().endswith('.txt'))
    stats: dict[str, int] = {}     # 语言 → 已用编号数

    for fn in files:
        path = os.path.join(directory, fn)

        # ── ① 文件名形态排除 ────────────────────────────
        if _is_non_wordlist(fn):
            lex.skipped.append((fn, '文件名不像词汇库（exclude/readme/_verify 等）'))
            continue

        # ── ② 语言码 ────────────────────────────────────
        lang = _lang_of(fn)
        if lang is None:
            lex.skipped.append((fn, '文件名无法判断语言码（应形如 nouns_zh.txt）'))
            continue
        if lang not in LANG_PREFIX:
            lex.skipped.append((fn, '未登记的语言码 %r' % lang))
            continue
        prefix = LANG_PREFIX[lang]

        # ── ③ 格式检查 ──────────────────────────────────
        words, problem = load_wordlist(path)
        if problem:
            lex.skipped.append((fn, problem))
            continue
        if not words:
            lex.skipped.append((fn, '没有有效词'))
            continue

        # ── ④ 建 track + 编号 ───────────────────────────
        track = lex.tracks.get(lang)
        if track is None:
            track = WordTrack(lang=lang, prefix=prefix)
            lex.tracks[lang] = track
            stats[lang] = 0

        base = stats[lang]
        for i, w in enumerate(words, 1):
            track.words.append(Word(
                wid='%s%04d' % (prefix, base + i),
                text=w,
                lang=lang,
                src=fn,
            ))
        stats[lang] = base + len(words)
        track.sources.append(fn)

    if verbose:
        print(lex.summary())
        for lang in lex.langs:
            t = lex.tracks[lang]
            print('  %-4s %-6d 个词  来源: %s' % (
                lang, t.n, ', '.join(t.sources)))
            print('       编号: %s … %s' % (t.words[0].wid, t.words[-1].wid))
        if lex.skipped:
            print('  跳过的文件:')
            for fn, why in lex.skipped:
                print('    ✗ %-28s %s' % (fn, why))

    return lex


# ══════════════════════════════════════════════════════════════════
# 五、全局单例（懒加载）
# ══════════════════════════════════════════════════════════════════

_LEXICON: Optional[Lexicon] = None


def get_lexicon(reload: bool = False, directory: str = WORDLIB) -> Lexicon:
    """取全局词库（首次调用时扫描目录）。"""
    global _LEXICON
    if _LEXICON is None or reload:
        _LEXICON = scan(directory)
    return _LEXICON


def pick_words(n: int, seed: int = 0,
               langs: Optional[list[str]] = None) -> list[str]:
    """
    ⭐ 最简接口：挑 n 个词，返回词本身。

    ⚠️ 这是 `skins.py` 会调用的函数。
    """
    lex = get_lexicon()
    return [w.text for w in lex.pick(n, seed=seed, langs=langs)]


def pick_word_objs(n: int, seed: int = 0,
                   langs: Optional[list[str]] = None) -> list[Word]:
    """挑 n 个词，返回带编号的对象（日志用）。"""
    lex = get_lexicon()
    return lex.pick(n, seed=seed, langs=langs)


# ══════════════════════════════════════════════════════════════════
# 六、自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0
    checks = []

    lex = scan(WORDLIB)

    checks.append(('扫描到 2 种语言', len(lex.langs) == 2))
    checks.append(('包含中文', 'zh' in lex.langs))
    checks.append(('包含英文', 'en' in lex.langs))
    checks.append(('中文编号以 CN 开头',
                   all(w.wid.startswith('CN') for w in lex.tracks['zh'].words)))
    checks.append(('英文编号以 EN 开头',
                   all(w.wid.startswith('EN') for w in lex.tracks['en'].words)))
    checks.append(('编号全唯一',
                   len({w.wid for t in lex.tracks.values() for w in t.words})
                   == lex.total()))

    # 跳过的文件（exclude / verify 应该被跳过）
    skipped_names = [fn for fn, _ in lex.skipped]
    checks.append(('exclude_extra.txt 被跳过', 'exclude_extra.txt' in skipped_names))
    checks.append(('gsc_level_1.txt 被跳过', 'gsc_level_1.txt' in skipped_names))
    checks.append(('gsc_level_1_verify.txt 被跳过',
                   'gsc_level_1_verify.txt' in skipped_names))

    # 挑选
    pk = lex.pick(10, seed=1)
    checks.append(('挑 10 个', len(pk) == 10))
    checks.append(('挑选不重复', len({w.wid for w in pk}) == 10))
    checks.append(('同 seed 同结果',
                   [w.wid for w in lex.pick(10, seed=1)] == [w.wid for w in pk]))
    checks.append(('异 seed 不同结果',
                   [w.wid for w in lex.pick(10, seed=2)] != [w.wid for w in pk]))

    # 两级均匀：各语言次数应大致相等
    big = lex.pick(200, seed=42)
    from collections import Counter
    c = Counter(w.lang for w in big)
    checks.append(('两级均匀（各语言约 50%）',
                   abs(c['zh'] - c['en']) <= 30))

    # 限定语言
    only_zh = lex.pick(20, seed=3, langs=['zh'])
    checks.append(('可限定只挑中文', all(w.lang == 'zh' for w in only_zh)))

    # 超量请求
    over = lex.pick(999999, seed=4)
    checks.append(('超量请求不崩', len(over) <= lex.total()))

    # 格式检查：坏文件
    # ⚠️ 2026-11-07 修：原来用 nouns_xx.txt，但 'xx' 不是已登记语言码，
    #    文件先被「语言码未登记」拦下，根本没走到格式检查 → 测试假失败。
    #    ⭐ 改用已登记语言码（fr），才能真正测到格式检查。
    bad_dir = os.path.join(WORDLIB, '_selftest_bad')
    try:
        os.makedirs(bad_dir, exist_ok=True)
        with open(os.path.join(bad_dir, 'nouns_fr.txt'), 'w', encoding='utf-8') as f:
            f.write('bonjour\nhas space\nanother one\n')     # 有空格 → 不合格式
        bad = scan(bad_dir)
        checks.append(('坏格式被拒', len(bad.tracks) == 0))
        checks.append(('坏格式记录了原因',
                       any('格式不符' in why for _, why in bad.skipped)))
    finally:
        import shutil
        shutil.rmtree(bad_dir, ignore_errors=True)

    if verbose:
        print('=' * 62)
        print('lexicon 自检')
        print('=' * 62)
        for name, ok in checks:
            print('  %s  %s' % ('✅' if ok else '❌', name))
            if not ok:
                fails += 1
        print()
        print(lex.summary())
        print()
        print('编号示例:')
        for lang in lex.langs:
            t = lex.tracks[lang]
            print('  %-4s %s  %s …  %s  %s' % (
                lang, t.words[0].wid, t.words[0].text,
                t.words[-1].wid, t.words[-1].text))
        print()
        print('随机挑选 10 个（seed=20261107）:')
        for w in lex.pick(10, seed=20261107):
            print('  %-8s %s' % (w.wid, w.text))
        print()
        if lex.skipped:
            print('跳过的文件:')
            for fn, why in lex.skipped:
                print('  ✗ %-28s %s' % (fn, why))

    return fails


if __name__ == '__main__':
    raise SystemExit(self_test())
