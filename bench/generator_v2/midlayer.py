"""
midlayer —— 中间层：把选项底变成题面（换词 + 函数中英替换）

════════════════════════════════════════════════════════════════
【这一层在做什么】（弗兰克 2026-11-08）
════════════════════════════════════════════════════════════════

    「现在写中间层，中间层就显示一组 b 个选项，每个选项包含刚才那个
      格式的内容，并对接到我们的出题器。

      然后看看老代码里怎么拿字典做替换，怎么写格式的，
      看看把字典替换 xy，把函数的中英文替换接过来。」

    ⇒ 中间层 = 选项底 → 题面文本

        ① x 的值  → 换成一个词（如「香蕉」）
        ② y       → 换成一个词（如「铜铃」），只留编码不给值
        ③ 函数名  → 中英文都能换（exp → 「指数」/「exponential」）

    ⚠️ 弗兰克：「每个选项每一个词，词和词之间是不可以重复的，
       我们有六十组变量共计 120 个变量要命名，中英混编。」
       ⇒ 60 项 × 2 个词 = 120 个词，全局不重复。

════════════════════════════════════════════════════════════════
【老代码的做法（照搬）】
════════════════════════════════════════════════════════════════

    groupmap.build_group_map(group_id, var_pids, seed, lang, avoid)
        · 组内不重复（自己保证）
        · avoid 是外部传进来的禁用词（跨批去重）
        · 一次挑够 var_pids 那么多个

    lexicon.pick_words(n, seed, langs)
        · langs=None → 只有英文
        · langs=('zh','en') → 中英混编   ← 我们要这个

    skeleton.Skeleton 里有三套写法：
        .plain     ASCII，题面用这个       exp(%s)
        .latex     LaTeX
        .spoken    中文口语                该变量的 e 次方
        .zh_name   中文名                  指数

    ⚠️ 我上一版没传 langs，默认只出英文 —— 这是错的。

════════════════════════════════════════════════════════════════
【格式（弗兰克定的最终格式）】
════════════════════════════════════════════════════════════════

    「其实只要有 x 的值和 y 的表达式就可以了，
      函数的编号知道就行，不用拿出来。」

    ⇒ 每个选项：

        香蕉
          香蕉 = 3.14159
          Y-O001 = (铜铃( (香蕉)² ÷ 100 )) ÷ 100

    ⚠️ 函数编号不显示。
    ⚠️ Y 只留编码，不给值。
"""

from __future__ import annotations

import hashlib
import math
import sys

sys.path.insert(0, r'<BENCH>\generator_v2')

import lexicon
import optbase
import skeleton

__all__ = ['WordBank', 'build_wordbank', 'MidOption', 'build_midlayer',
           'render_midlayer', 'self_test']


# ══════════════════════════════════════════════════════════════════
# 一、词库：120 个词，全局不重复，中英混编
# ══════════════════════════════════════════════════════════════════

class WordBank:
    """
    ⭐ 一道题的全部词。

    ⚠️ 弗兰克：「词和词之间是不可以重复的。」
    """

    __slots__ = ('words', 'codes', 'n')

    def __init__(self):
        self.words: dict[str, str] = {}       # 编码 → 词
        self.codes: dict[str, str] = {}       # 词 → 编码
        self.n = 0

    def add(self, code: str, w: str) -> None:
        if w in self.codes:
            raise ValueError('词 %r 重复了（已给了 %s）' % (w, self.codes[w]))
        self.words[code] = w
        self.codes[w] = code
        self.n += 1

    def get(self, code: str) -> str:
        return self.words.get(code, '')

    def langs(self) -> dict:
        zh = sum(1 for w in self.words.values()
                 if any('\u4e00' <= c <= '\u9fff' for c in w))
        return {'zh': zh, 'en': self.n - zh}


def build_wordbank(pids: list[str], seed: int = 0,
                   langs: tuple = ('zh', 'en'),
                   avoid: set | None = None) -> WordBank:
    """
    ⭐ 给一批编码配词。

    ⚠️ 必须传 langs=('zh','en') —— lexicon 默认 langs=None 只出英文。
    ⚠️ avoid 是外部禁用词（函数名、题面关键词）。
    """
    avoid = set(avoid or ())
    need = len(pids) + 40                       # 多摇一些，过滤掉 avoid
    picked = lexicon.pick_words(need, seed=seed, langs=list(langs))

    bank = WordBank()
    it = iter(picked)
    for pid in pids:
        while True:
            try:
                w = next(it)
            except StopIteration:
                raise RuntimeError(
                    '词不够：要 %d 个，词库给完了' % len(pids))
            if w in avoid or w in bank.codes:
                continue
            bank.add(pid, w)
            break
    return bank


# ══════════════════════════════════════════════════════════════════
# 二、函数的中英文写法
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# 二、表达式模块（Term）
# ══════════════════════════════════════════════════════════════════
#
#   ⚠️⚠️ 2026-11-09 弗兰克：
#   「我们的组装是根据代码来的、模块化的，你的改动也要基于模块化和组装本身，
#    而不是在已经生成到一半的文本上改，你这样只会越改越麻烦变成屎山。」
#
#   ⇒ 表达式不用字符串拼，先用【结构】表示，最后一次性展开。
#     括号加不加由结构决定，不由字符串猜。
#
#   节点类型：
#       Var      变量（一个词）
#       Fun      函数调用：名字 + 一个子表达式
#       Bin      二元运算：左 + 运算符 + 右
#
#   ⚠️ 展开时的括号规则：
#       Var          → 不加括号
#       Fun          → 名字(子表达式)
#       Bin          → 左 op 右，要不要加括号看父节点
#                      （父是 Fun 或 Bin 且自己有运算符 → 加）

# ══════════════════════════════════════════════════════════════════
# 二、表达式模块
# ══════════════════════════════════════════════════════════════════
#
#   ⚠️⚠️ 2026-11-09 弗兰克：
#   「直接用我们的函数表和代码在底层转中间那层识别好，
#    然后根据识别好的模块去替换，而不是现在这串改不完的屎山。
#    但还是要注意括号问题，括号在双层嵌套时，
#    对长句也就是上一步运算的描述整体加括号。」
#
#   ⇒ 分两层：
#
#       ① 识别层：从 packs（函数总表，带编码）认出「这是哪个函数」，
#                 并准备好它的「词条」——中文名 / 英文名 / 参数槽位。
#                 不猜字符串，查表。
#
#       ② 展开层：按结构把词条填进表达式，括号由结构决定。
#
#   ⚠️ 括号规则（弗兰克定的）：
#       双层嵌套时，「上一步运算」作为一个整体描述，要整体加括号。
#       所以凡是「运算节点」嵌进「函数节点」作为一个参数，
#       一律整体包起来 —— 不靠猜，靠节点类型。

import packs as _packs


# ══════════════════════════════════════════════════════════════════
# 识别层：函数词条
# ══════════════════════════════════════════════════════════════════

# ⚠️ 中文短句名（比 skeleton.zh_name 更适合嵌进长式子）
ZH_SHORT = {
    'hypot':    '与一点五的平方和开方',
    'xlogy':    '二点五乘自然对数',
    'xlog1py':  '二点五乘一加自然对数',
    'expm1':    '指数减一',
    'exprel':   '指数减一再除',
    'exp2':     '二进制指数',
    'exp10':    '十进制指数',
    'log1p':    '对数加一',

    # ⚠️⚠️ 2026-11-10 弗兰克：
    #    「函数统一中文也可以，我觉得是部分函数没有英文翻译，
    #      我们老函数其实做了双语翻译……只要不给原函数名称，
    #      用描述没问题，让模型自己抽象。」
    #
    #    ⇒ 这 12 个函数的 skeleton 中文名里带着原函数名的英文字母
    #      （logit / softplus / sigmoid / zeta / W / i0 / i1），
    #      卷面印出来等于把原函数名给了模型。
    #    ⇒ 全部换成【中文数学译名】，不出现任何原函数名字符串。
    'logit':    '对数几率',                    # logit(x) = ln(x/(1-x))
    'expit':    '对数几率的反函数',            # sigmoid 的正式说法
    'softplus': '平滑正部',                    # softplus(x)=ln(1+e^x)
    'log_expit': '对数几率反函数的对数',       # ln(sigmoid(x))
    'zeta':     '赫尔维茨级数和',              # 原为「赫尔维茨 zeta 函数」
    'lambertw': '朗伯乘积对数',                # 原为「朗伯 W」
    'iv':       '第一类零阶修正柱函数',        # 原为「修正贝塞尔 I₀」
    'kv':       '第二类零阶修正柱函数',        # 原为「修正贝塞尔 K₀」
    'ive':      '第一类零阶缩放修正柱函数',
    'kve':      '第二类零阶缩放修正柱函数',
    'i0':       '第一类零阶柱函数',            # 原为「贝塞尔 i0」
    'i1':       '第一类一阶柱函数',            # 原为「贝塞尔 i1」
}


class FuncEntry:
    """
    ⭐ 一个函数的词条：识别一次，到处用。

    字段
    ----
    name     : 函数名（packs 里的名字）
    code     : 编码（E008 之类），证明是从总表里认出来的
    zh       : 中文名
    en       : 英文名
    slots    : 参数槽位模板。例如 'x(%s)'、'x(%s, 1.5)'、'(%s)^2'
               —— 已经把「名字里已说的固定参数」去掉后的样子
    has_name : 模板里有没有函数名（纯数学写法如 '(%s)^2' 就没有）
    """

    __slots__ = ('name', 'code', 'zh', 'en',
                 'slots_expr', 'slots_desc', 'has_name')

    def __init__(self, name, code, zh, en,
                 slots_expr, slots_desc, has_name):
        self.name = name
        self.code = code
        self.zh = zh
        self.en = en
        self.slots_expr = slots_expr   # 表达式写法：参数照原样
        self.slots_desc = slots_desc   # 描述写法：参数已在名字里，去掉
        self.has_name = has_name


# ⚠️ 名字里已经说了参数的函数 → 模板里的那个固定数要屏蔽
#    （例：'二阶指数积分' 说了阶数 2，就不能再写 expn(2, x)）
ZH_HIDES_PARAM: set[str] = {
    'expn',
    'eval_hermite', 'eval_laguerre', 'eval_chebyt',
    'eval_chebyu', 'eval_legendre',
    'jv', 'yv', 'iv', 'kv',
    'hypot', 'xlogy', 'xlog1py',
}


def _strip_fixed_param(plain: str) -> str:
    """⭐ 把模板里的固定数字参数去掉，只留一个 %s。"""
    import re
    out = re.sub(r'\(\s*[-+]?\d+\.?\d*\s*,\s*%s\s*\)', '(%s)', plain)
    out = re.sub(r'\(\s*%s\s*,\s*[-+]?\d+\.?\d*\s*\)', '(%s)', out)
    return out


_ENTRIES: dict[str, FuncEntry] = {}


def func_entry(name: str) -> FuncEntry | None:
    """
    ⭐ 从【函数总表】认出这个函数，返回它的词条。

    ⚠️ 识别靠 packs（有编码的正表），不靠猜字符串。
    ⚠️ 认不出来返回 None —— 调用方按「不是函数」处理。
    """
    if name in _ENTRIES:
        return _ENTRIES[name]

    pk = _packs.get(name)
    if pk is None:
        return None

    sk = skeleton.get_skeleton(name)
    plain = sk.plain if (sk is not None and sk.plain) else ''

    zh = ZH_SHORT.get(name) or (
        sk.zh_name if (sk is not None and sk.zh_name) else name)

    # 英文名：从模板里取（'exp(%s)' → 'exp'），没有模板就用函数名
    en = name

    # 模板 → 槽位式
    if not plain:
        slots, has_name = '%s', True
    else:
        # ⚠️ 2026-11-09 弗兰克：「把函数表达式和描述二者区分开，
        #    而不仅仅是语言区分。因为表达式确实要靠参数表达，而描述可能不用。」
        #
        #   ⇒ 用【描述】还是【表达式】由「中文名有没有说清参数」决定，
        #     不是由语言决定：
        #         名字已说参数 且 用中文 → 用描述，参数去掉
        #         其余                    → 用表达式，参数照常
        # ① 表达式写法：模板原样（参数保留）
        # ② 描述写法：若名字已说参数，把模板里的固定数去掉
        p_expr = plain
        p_desc = _strip_fixed_param(plain) if name in ZH_HIDES_PARAM else plain

        def _split(p2):
            if '(' in p2:
                head = p2.split('(')[0]
                hn = bool(head) and head != '%s'
                return (p2[len(head):] if hn else p2), hn
            return p2, False

        slots_expr, has_name = _split(p_expr)
        slots_desc, _hn2 = _split(p_desc)

    e = FuncEntry(name, getattr(pk, 'code', ''), zh, en,
                  slots_expr, slots_desc, has_name)
    _ENTRIES[name] = e
    return e


# ══════════════════════════════════════════════════════════════════
# 表达式节点
# ══════════════════════════════════════════════════════════════════

class Term:
    """
    ⭐ 表达式节点基类。

    ⚠️⚠️ 2026-11-09 弗兰克（最后一条，也是根子上的那条）：
    「你现在的设计好像还在渲染层管这个事，渲染是结果，我们不去调整结果，
      去调整算法和模块本身。」
    「当描述和模块什么的组合成对象的时候，也是可追踪的。
      我们要改的是括号针对这些内容不重复添加，
      应该在括号添加的算法层加一个针对括号添加的判断，
      即模块或者模块集只添加一次括号。」

    ⇒ 括号是【节点属性】，不是【字符串修补】。

        节点类型            渲染出来有没有自带边界
        Var（一个词）        有（就是一个词）
        Fun（名字(参数)）    有（括弧是函数调用的一部分）
        Bin（左 op 右）      没有 —— 所以它被嵌进别人时要包一层

    ⭐ 加括号的判断只看「这是不是 Bin 节点」，不看字符。
       这样每个模块/模块集最多被包一次。
    """
    """⭐ 表达式节点基类。"""
    __slots__ = ()

    def render(self, lang: str = 'zh') -> str:
        raise NotImplementedError

    def is_compound(self) -> bool:
        """⭐ 是不是「上一步运算」（需要整体加括号）。"""
        return False


class Var(Term):
    """⭐ 一个变量（一个词）。"""
    __slots__ = ('word',)

    def __init__(self, word: str):
        self.word = word

    def render(self, lang: str = 'zh') -> str:
        return self.word


def _balanced(t: str) -> bool:
    """⭐ 这个字符串的括号是否平衡（用来判「最外层是不是一整对」）。"""
    depth = 0
    for ch in t:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _slot_has_boundary(tmpl: str) -> bool:
    """⭐ 模板里的 %s 是不是已经被括号或逗号围住了。"""
    import re
    for mm in re.finditer(r'%s', tmpl):
        left = tmpl[:mm.start()].rstrip()
        right = tmpl[mm.end():].lstrip()
        if not (left.endswith('(') or left.endswith(',')):
            return False
        if not (right.startswith(')') or right.startswith(',')):
            return False
    return True


class Const(Term):
    """⭐ 一个常数。"""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def render(self, lang: str = 'zh') -> str:
        v = self.value
        return ('%g' % v) if isinstance(v, float) else str(v)


class Fun(Term):
    """⭐ 函数调用节点。词条来自识别层。"""
    __slots__ = ('name', 'arg')

    def __init__(self, name: str, arg: Term):
        self.name = name
        self.arg = arg

    def render(self, lang: str = 'zh') -> str:
        e = func_entry(self.name)
        arg_txt = self._render_arg(lang)

        if e is None:
            # 不是总表里的函数 → 原样写成 名字(参数)
            return '%s(%s)' % (self.name, arg_txt)

        # ⚠️ 弗兰克：「把函数表达式和描述二者区分开，而不仅仅是语言区分。」
        #   中文且名字已说参数 → 用描述版（参数去掉）
        #   其余                → 用表达式版（参数照常）
        _desc = (lang == 'zh') and (self.name in ZH_HIDES_PARAM)
        tmpl = e.slots_desc if _desc else e.slots_expr
        head = e.zh if lang == 'zh' else e.en
        if e.has_name:
            tmpl = head + tmpl
        n = tmpl.count('%s')
        if n == 0:
            return tmpl

        # ⚠️⚠️ 2026-11-09 弗兰克：「模块或者模块集只添加一次括号。」
        #   ⇒ 模板自己已经把参数包住了（如 '反余弦(%s)'），就不要再包。
        #     判据：模板里 %s 的位置，它的左边紧邻 '(' 或 ','，右边紧邻 ')' 或 ','。
        #     这跟「复不复合」无关，是「模板有没有现成的位置」。
        if _slot_has_boundary(tmpl):
            return tmpl % ((arg_txt,) * n)
        return tmpl % ((arg_txt,) * n)

    def _render_arg(self, lang: str) -> str:
        """
        ⭐ 参数写法。

        ⚠️⚠️ 2026-11-09 弗兰克：「模块或者模块集只添加一次括号。」

        ⇒ 判断顺序：
            ① 参数是运算节点（上一步是加减乘除）吗？
               不是 → 直接用（函数结果、变量都自带边界）
            ② 是 → 它的结果【自己】带括号吗？（顶层是 Bin 且有括号保护）
               带 → 直接用
               不带 → 包一层

        ⚠️ 这里不再看「模板有没有括号」，那是 Fun.render 的事。
           参数自己要不要包，只看参数自己。
        """
        txt = self.arg.render(lang)
        if not isinstance(self.arg, Bin):
            return txt
        # 运算节点：只有最外层没有括号时才包
        t = txt.strip()
        if t.startswith('(') and _balanced(t):
            return t
        return '(' + t + ')'


class Bin(Term):
    """⭐ 二元运算节点（本文只用到除以）。"""
    __slots__ = ('left', 'op', 'right')

    def __init__(self, left: Term, op: str, right: Term):
        self.left = left
        self.op = op
        self.right = right

    def is_compound(self) -> bool:
        return True

    def render(self, lang: str = 'zh') -> str:
        l = self.left.render(lang)
        if isinstance(self.left, Bin):
            l = '(' + l + ')'
        r = self.right.render(lang)
        if isinstance(self.right, Bin):
            r = '(' + r + ')'
        return '%s %s %s' % (l, self.op, r)


# ══════════════════════════════════════════════════════════════════
# 搭表达式：一个选项
# ══════════════════════════════════════════════════════════════════

def option_term(x_word: str, f1: str, f2: str,
                c1: float, c2: float) -> Term:
    """
    ⭐ 一个选项的表达式树。

        Y = f1( f2(X) ÷ c2 ) ÷ c1
    """
    t: Term = Var(x_word)
    t = Fun(f2, t)
    if c2 != 1.0:
        t = Bin(t, '÷', Const(c2))
    t = Fun(f1, t)
    if c1 != 1.0:
        t = Bin(t, '÷', Const(c1))
    return t


def func_zh(name: str) -> str:
    """⭐ 函数中文名（走识别层）。"""
    e = func_entry(name)
    return e.zh if e is not None else name


def func_en(name: str) -> str:
    """⭐ 函数英文名。"""
    e = func_entry(name)
    return e.en if e is not None else name


class MidOption:
    """⭐ 中间层的一个选项。"""
    __slots__ = ('idx', 'pid', 'x_word', 'y_code', 'x_value',
                 'f1', 'f2', 'c1', 'c2', 'text')

    def __init__(self, idx: int):
        self.idx = idx
        self.pid = ''
        self.x_word = ''
        self.y_code = ''
        self.x_value = float('nan')
        self.f1 = ''
        self.f2 = ''
        self.c1 = 1.0
        self.c2 = 1.0
        self.text = ''

    def render(self) -> str:
        return self.text


def build_midlayer(ops: list, seed: int = 0,
                   langs: tuple = ('zh', 'en'),
                   avoid: set | None = None) -> list:
    """
    ⭐ 选项底列表 → 中间层选项列表（词已换好，表达式已展开）。

    ⚠️ 120 个词：60 个给 x、60 个给 y，全局不重复。
    """
    pids: list[str] = []
    for op in ops:
        pids.append('XW-%s' % op.pid)
    for op in ops:
        pids.append('YW-%s' % op.pid)

    bad = set(avoid or ())
    for op in ops:
        bad.add(op.f1)
        bad.add(op.f2)

    bank = build_wordbank(pids, seed=seed, langs=langs, avoid=bad)

    out = []
    for op in ops:
        mo = MidOption(op.idx)
        mo.pid = op.pid
        mo.x_word = bank.get('XW-%s' % op.pid)
        mo.y_code = op.codes['Y']
        mo.x_value = op.x
        mo.f1, mo.f2 = op.f1, op.f2
        mo.c1, mo.c2 = op.c1, op.c2

        # ⚠️⚠️ 2026-11-10 弗兰克：
        #    「函数统一中文也可以……只要不给原函数名称，
        #      用描述没问题，让模型自己抽象。」
        #
        #    ⇒ 这里原来写的是 lang = 'zh' if x词含中文 else 'en'
        #      语言跟着 x 的词走 —— x 是英文词时整个式子就用英文名渲染，
        #      卷面直接印出 digamma(...)、logit(...)、ellipk(...)，
        #      等于把函数身份交给模型，90 个工具的难度就没了。
        #
        #    ⇒ 固定用中文。x 的词是中文还是英文不影响式子写法。
        lang = 'zh'

        # ⭐ 表达式树一次搭好，一次展开
        term = option_term(mo.x_word, op.f1, op.f2, op.c1, op.c2)
        body = term.render(lang)

        mo.text = '%s\n  %s = %.12g\n  %s = %s' % (
            mo.x_word, mo.x_word, mo.x_value, mo.y_code, body)
        out.append(mo)
    return out


def render_midlayer(mos: list[MidOption], header: str = '') -> str:
    L = []
    if header:
        L.append(header)
        L.append('')
    for mo in mos:
        L.append(mo.render())
        L.append('')
    return '\n'.join(L)


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
        print('midlayer 自检（中间层：换词 + 函数中英）')
        print('=' * 96)
        print('-' * 96)

    # ① 词库中英混编
    bank = build_wordbank(['P%d' % i for i in range(120)], seed=20261108)
    ls = bank.langs()
    ck('120 个词', bank.n == 120, '%d' % bank.n)
    ck('中英混编', ls['zh'] > 0 and ls['en'] > 0,
       '中文 %d / 英文 %d' % (ls['zh'], ls['en']))
    ck('词不重复', len(set(bank.words.values())) == 120)
    ck('同种子可复现',
       build_wordbank(['P%d' % i for i in range(120)],
                      seed=20261108).words == bank.words)

    # ② 函数中英名
    ck('exp 中文名', func_zh('exp') not in ('', 'exp'), func_zh('exp'))
    ck('pow2 中文名', func_zh('pow2') not in ('', 'pow2'), func_zh('pow2'))

    # ③ 造 60 个选项 → 中间层
    import torch
    from mathpack import DT, DEV
    g = torch.Generator(device=DEV)
    g.manual_seed(20261108)
    ys = torch.rand(60, device=DEV, dtype=DT, generator=g) * 8.0 + 1.0
    ops = optbase.build_options(ys)
    mos = build_midlayer(ops, seed=20261108)

    ck('中间层出 60 项', len(mos) == 60, '%d' % len(mos))
    ck('每项都有 x 的词', all(mo.x_word for mo in mos))
    ck('每项词互不相同',
       len({mo.x_word for mo in mos}) == 60)
    ck('x 的词和 y 的词分开管（共 120）',
       len({w for w in build_wordbank(
           ['XW-' + o.pid for o in ops] + ['YW-' + o.pid for o in ops],
           seed=20261108).words.values()}) == 120)

    # ④ 题面不含 y 的值、不含函数编号
    txt_all = render_midlayer(mos)
    bad_y = [mo.pid for mo, op in zip(mos, ops)
             if ('%.6g' % op.y) in mo.text]
    ck('题面里不含任何 y 的值', not bad_y, str(bad_y[:3]))
    bad_code = [mo.pid for mo, op in zip(mos, ops)
                if op.f1_code in mo.text or op.f2_code in mo.text]
    ck('题面里不出现函数编号', not bad_code, str(bad_code[:3]))
    ck('题面里出现 y 的编码',
       all(mo.y_code in mo.text for mo in mos))

    # ⑤ 题面式子能真算出 y（最重要的一条）
    #    ⚠️ 2026-11-10：原来这里手写只支持 pow2/exp/log 三个函数，
    #       随机 60 项一个都没命中，`验了 0 项` 反而把失败藏住了。
    #       改成走 packs 的真实现 —— 选项里出现什么函数都验得到。
    import math
    import packs as _packs

    def _eval(mo):
        """按题面的写法算 y：下层 → 除以 c2 → 上层 → 除以 c1。"""
        try:
            x = float(mo.x_value)
            p2 = _packs.get(mo.f2)
            p1 = _packs.get(mo.f1)
            import torch as _torch
            from mathpack import DT as _DT, DEV as _DEV
            t = _torch.tensor([x], device=_DEV, dtype=_DT)
            v = float(p2._fwd(t)[0]) / mo.c2
            t2 = _torch.tensor([v], device=_DEV, dtype=_DT)
            return float(p1._fwd(t2)[0]) / mo.c1
        except Exception:                               # noqa: BLE001
            return None

    n_eval = 0
    bad_eval = []
    for mo, op in zip(mos, ops):
        yv = _eval(mo)
        if yv is None or not math.isfinite(yv):
            continue
        n_eval += 1
        if abs(yv - op.y) / max(abs(op.y), 1e-300) > 1e-6:
            bad_eval.append((mo.pid, yv, op.y))
    ck('按题面式子能算出正确的 y', not bad_eval and n_eval > 0,
       '验了 %d 项，错的 %s' % (n_eval, bad_eval[:2]))
    # ⚠️ 覆盖率本身也要合格：60 项至少验到 50 项，
    #    不然「算得出」只是因为几乎没验。
    ck('题面验算覆盖 ≥50 项', n_eval >= 50, '验了 %d 项' % n_eval)

    if verbose:
        print('-' * 96)
        print(render_midlayer(mos[:4], header='【中间层样例（前 4 项）】'))
        print('-' * 96)
        print('词库分布：中文 %d / 英文 %d' % (ls['zh'], ls['en']))
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
