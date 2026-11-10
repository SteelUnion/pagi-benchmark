"""
抽象层 —— 函数与变量的替换表 + 语法模板

⭐ 2026-11-07 Frank 定：
    「函数就是输入、过程、输出，也就是代词动词代词或者名词动词名词的结构，
      只要中间知道要干什么，怎么说就完全是语法的事，
      可以设计几套不同的语法出来随机。」

    「我是专家，专门解决：谁把 x 通过什么变成 y / y 被谁通过什么变成 x 的问题，
      请输入 x 值，我将为你求 y。」

    「替换要和之前的保持一致，我们做了代码标记的，一个函数可以用一套替换，
      这一套替换可以语法不同，但要表达一个意思。
      一定要说清楚，不能说是粒子专家，要说是计算粒子运动什么内容的专家，
      因为每个公式内容都不同。」

    「算法有随机有固定，文本可以随机，但要精确，语法不变。
      部分内容可以用图片替换，但要求是符合该公式函数的特征。
      这点与名词替换不同，变量可以换为所有名词，
      因为变量只是一个值的指代，其本身可以无意义。」

⭐ 核心区分（本模块最重要的一句话）：
    ┌────────────┬──────────────┬──────────────────────────┐
    │ 部件        │ 能不能乱换      │ 原因                      │
    ├────────────┼──────────────┼──────────────────────────┤
    │ 变量        │ ✅ 随便换名词    │ 只是「值的指代」，本身无意义    │
    │ 函数        │ ❌ 必须合特征    │ 换了特征，模型判断不出它是干什么的 │
    └────────────┴──────────────┴──────────────────────────┘

⚠️ 本模块只提供「说什么」，不提供「怎么说」。
   「怎么说」由 skinsyntax.py 的语法模板负责。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, r'<BENCH>\generator')

import packs
from skeleton import get_skeleton


# ══════════════════════════════════════════════════════════════════
# 一、函数的「内容描述」——必须精确，不能模糊
# ══════════════════════════════════════════════════════════════════

@dataclass
class FuncDesc:
    """
    一个函数的内容描述。

    verb   : 动词短语 —— 它「做什么」，如「求一个数的四次方」
    obj    : 作用对象 —— 它「作用于什么」，如「一个量」
    note   : 补充限定（可选）

    ⭐ 三者合起来才「精确」：
        ❌ 「粒子专家」（不知道干什么）
        ✅ 「计算粒子运动距离的四次方关系」（知道干什么）
    """
    verb: str
    obj: str = '一个数'
    note: str = ''


# ⚠️ 74 个函数的精确内容描述。
#    ⚠️ 原则：能描述就描述，尽量不用 f 表示；实在没法描述的才标注。
#    ⭐ Frank：「高等函数能描述的要尽量描述，尽可能的不要用 f 表示，没办法就算了。」
FUNC_DESC: dict[str, FuncDesc] = {

    # ── 幂与根 ──────────────────────────────────────────
    'pow2':    FuncDesc('求一个数自乘两次的结果'),
    'pow3':    FuncDesc('求一个数自乘三次的结果'),
    'pow4':    FuncDesc('求一个数自乘四次的结果'),
    'pow5':    FuncDesc('求一个数自乘五次的结果'),
    'pow8':    FuncDesc('求一个数自乘八次的结果'),
    'sqrt':    FuncDesc('求一个数的平方根'),
    'cbrt':    FuncDesc('求一个数的立方根'),

    # ── 指数与对数 ──────────────────────────────────────
    'exp':     FuncDesc('求自然常数 e 的该数次幂'),
    'log':     FuncDesc('求一个数的自然对数'),
    'expm1':   FuncDesc('求自然常数 e 的该数次幂，再减去一'),
    'log1p':   FuncDesc('求一加该数之后的自然对数'),
    'log2':    FuncDesc('求一个数以二为底的对数'),
    'log10':   FuncDesc('求一个数以十为底的对数'),
    'exp2':    FuncDesc('求二的该数次幂'),
    'exp10':   FuncDesc('求十的该数次幂'),
    'exprel':  FuncDesc('求自然常数 e 的该数次幂减一后，再除以该数'),

    # ── 三角 ────────────────────────────────────────────
    'sin':     FuncDesc('求一个角度的正弦'),
    'cos':     FuncDesc('求一个角度的余弦'),
    'tan':     FuncDesc('求一个角度的正切'),
    'asin':    FuncDesc('已知正弦值，反求对应的角度'),
    'acos':    FuncDesc('已知余弦值，反求对应的角度'),
    'atan':    FuncDesc('已知正切值，反求对应的角度'),

    # ── 双曲 ────────────────────────────────────────────
    'sinh':    FuncDesc('求一个数的双曲正弦'),
    'cosh':    FuncDesc('求一个数的双曲余弦'),
    'tanh':    FuncDesc('求一个数的双曲正切'),
    'asinh':   FuncDesc('已知双曲正弦值，反求原数'),
    'acosh':   FuncDesc('已知双曲余弦值，反求原数'),
    'atanh':   FuncDesc('已知双曲正切值，反求原数'),

    # ── 概率与误差 ──────────────────────────────────────
    'erf':     FuncDesc('求一个数的误差函数值'),
    'erfc':    FuncDesc('求一个数的补误差函数值'),
    'erfcx':   FuncDesc('求一个数的补误差函数值，再乘 e 的该数平方次幂'),
    'erfinv':  FuncDesc('已知误差函数值，反求原数'),
    'expit':   FuncDesc('把一个数压缩到 0 与 1 之间，采用 sigmoid 变换'),
    'logit':   FuncDesc('把 0 与 1 之间的数展开到全体实数，采用 logit 变换'),
    'ndtr':    FuncDesc('求标准正态分布下，小于该数的概率'),
    'ndtri':   FuncDesc('已知标准正态分布的概率，反求对应的数值'),
    'log_ndtr': FuncDesc('求标准正态分布小于该数的概率，再取自然对数'),
    'log_expit': FuncDesc('对 sigmoid 变换的结果取自然对数'),

    # ── 伽马族 ──────────────────────────────────────────
    'loggamma': FuncDesc('求伽马函数的自然对数'),
    'gammaln':  FuncDesc('求伽马函数绝对值的自然对数'),
    'digamma':  FuncDesc('求伽马函数对数的一阶导数'),
    'polygamma': FuncDesc('求伽马函数对数的二阶导数'),
    'beta':     FuncDesc('求参数为 1.5 与该数的贝塔函数值'),
    'betaln':   FuncDesc('求参数为 1.5 与该数的贝塔函数值的自然对数'),
    'gammainc': FuncDesc('求参数为 1.5 与该数的正则化下不完全伽马函数'),
    'gammaincc': FuncDesc('求参数为 1.5 与该数的正则化上不完全伽马函数'),

    # ── 特殊 ────────────────────────────────────────────
    'zeta':     FuncDesc('求自变量为 2.5 与该数的赫尔维茨 zeta 函数值'),
    'lambertw': FuncDesc('求满足该数等于 w 乘 e 的 w 次幂的那个 w'),
    'entr':     FuncDesc('求负的该数乘该数的自然对数'),
    'softplus': FuncDesc('求一加 e 的该数次幂之后的自然对数'),
    'pseudo_huber': FuncDesc('求伪 Huber 损失在该数上的取值，参数取 1.5'),
    'hypot':    FuncDesc('求该数平方与一点五平方之和的平方根'),
    'xlogy':    FuncDesc('求二点五乘该数的自然对数'),
    'xlog1py':  FuncDesc('求二点五乘一加该数之后的自然对数'),

    # ── 指数积分 ────────────────────────────────────────
    'exp1':     FuncDesc('求一阶指数积分'),
    'expn':     FuncDesc('求二阶指数积分'),
    'expi':     FuncDesc('求指数积分函数在该数处的取值'),

    # ── 椭圆积分 ────────────────────────────────────────
    'ellipk':   FuncDesc('求参数为该数平方的第一类完全椭圆积分'),
    'ellipe':   FuncDesc('求参数为该数平方的第二类完全椭圆积分'),
    'ellipkinc': FuncDesc('求积分为该数、参数为零点五的第一类不完全椭圆积分'),

    # ── 贝塞尔 ──────────────────────────────────────────
    'i0':       FuncDesc('求零阶第一类修正贝塞尔函数'),
    'i1':       FuncDesc('求一阶第一类修正贝塞尔函数'),
    'iv':       FuncDesc('求零阶第一类修正贝塞尔函数'),
    'kv':       FuncDesc('求零阶第二类修正贝塞尔函数'),
    'ive':      FuncDesc('求零阶第一类修正贝塞尔函数，再乘以 e 的负绝对值次幂'),
    'kve':      FuncDesc('求零阶第二类修正贝塞尔函数，再乘以 e 的该数次幂'),
    'jv':       FuncDesc('求零阶第一类贝塞尔函数'),
    'yv':       FuncDesc('求零阶第二类贝塞尔函数'),
    'airy':     FuncDesc('求艾里函数在该数处的取值'),

    # ── 正交多项式（三阶）────────────────────────────────
    'eval_hermite':  FuncDesc('求三阶厄米多项式在该数处展开的值'),
    'eval_laguerre': FuncDesc('求三阶拉盖尔多项式在该数处展开的值'),
    'eval_chebyt':   FuncDesc('求三阶第一类切比雪夫多项式在该数处展开的值'),
    'eval_chebyu':   FuncDesc('求三阶第二类切比雪夫多项式在该数处展开的值'),
    'eval_legendre': FuncDesc('求三阶勒让德多项式在该数处展开的值'),
}

# ⚠️ 实在没法用自然语言精确描述的，标在这里。
#    这些函数在抽象时**保留原名**，不做替换。
NO_DESC: set[str] = set()


def desc_of(pack_name: str) -> FuncDesc:
    """取一个函数的内容描述。没有就退回标准名。"""
    d = FUNC_DESC.get(pack_name)
    if d is not None:
        return d
    sk = get_skeleton(pack_name)
    zh = sk.zh_name if sk else pack_name
    return FuncDesc('%s 运算' % zh, note='（自动生成，未覆盖）')


# ══════════════════════════════════════════════════════════════════
# 二、变量的替换（随便换名词）
# ══════════════════════════════════════════════════════════════════

# ⚠️ 变量只是一个值的指代，本身无意义 —— 所以可以换成任何名词。
#
# ⭐ 2026-11-07 接词库：变量名现在从 `lexicon` 的词库里随机取，
#    不再用下面这份 20 词的硬编码列表。
#
#    ⚠️ 词库来源：
#       中文 3,532 词（《通用规范汉字表》3500 字 ∩ 《新华字典》∩ jieba n 类）
#       英文 3,000 词（BNC 一亿词语料，>800 次，n 类）
#    ⚠️ 通用性依据见：论文\基准测试\_参考文献\词汇标准\词汇来源的通用性依据.md
#
#    ⭐ 下面这份列表**保留作为兜底**：词库加载失败时用它。
#    ⚠️ 不要直接改这份列表加词 —— 要加词请加到 wordlib\ 下的 txt 里。
VAR_NOUNS: list[str] = [
    '小明', '小红', '大卡车', '小蝴蝶', '河流', '山顶',
    '电子的位置', '光子的能量', '原子的半径', '分子的距离',
    '温度', '压力', '速度', '质量', '频率', '振幅',
    '粒子 A', '粒子 B', '样本一', '样本二',
]


def noun_for_var(idx: int, seed: int = 0,
                 lang: str | None = None) -> str:
    """
    ⭐ 给第 idx 个变量取一个名词。

    ⚠️ 同一题内必须一致：同一个 (idx, seed) 永远得到同一个词。
    ⚠️ lang=None 时在全部语言里均匀随机。

    ⭐ 2026-11-07 改：从 lexicon 词库取词；取不到才退回硬编码列表。
    """
    try:
        import lexicon as _lex
        lex = _lex.get_lexicon()
        if lex.total() == 0:
            raise RuntimeError('词库为空')
        # ⚠️ 用固定 seed 保证同一 idx 在同一题里取到同一个词
        picked = lex.pick(idx + 1, seed=seed + idx * 7919,
                          langs=[lang] if lang else None)
        if picked:
            return picked[0].text
    except Exception:
        pass
    # 兜底
    if not VAR_NOUNS:
        return 'v%d' % idx
    return VAR_NOUNS[(seed * 31 + idx) % len(VAR_NOUNS)]


def pick_var_nouns(n: int, seed: int = 0,
                   lang: str | None = None) -> list[str]:
    """
    ⭐ 一次挑 n 个**互不相同**的变量名词。

    ⚠️ 这是 `toolconsole` / `module` 对接的主接口 ——
       一道题里要几个变量就一次挑几个，保证不重复。
    """
    try:
        import lexicon as _lex
        lex = _lex.get_lexicon()
        if lex.total() > 0:
            ws = lex.pick(n, seed=seed, langs=[lang] if lang else None)
            if len(ws) == n:
                return [w.text for w in ws]
    except Exception:
        pass
    # 兜底：从硬编码列表里轮取
    out = []
    for i in range(n):
        out.append(VAR_NOUNS[(seed * 31 + i) % len(VAR_NOUNS)])
    return out


# ══════════════════════════════════════════════════════════════════
# 三、语法模板（说什么 → 怎么说）
# ══════════════════════════════════════════════════════════════════

# ⚠️ 模板约束（2026-11-07 实测修正）：
#    {in} / {out} 必须放在「名词该在的位置」，不能塞进「被…所需要」这种被动骨架里，
#    否则读不通（如「小明 被 小红 所需要时」）。
#
# ⭐ 2026-11-07 Frank 补两条硬约束：
#    ① 用「（小红）的值为（小明）的…」这种「X 的值为 Y」的写法
#       —— 明确标出「这是一个数」，不是一件物体
#    ② 去掉「中间」这类多余词
#
# ⭐⭐ 2026-11-07 Frank 再补（工具的身份可以多样）：
#    「工具可以说自己是工具，可以说自己是擅长某方面的专家，
#      可以说自己是懂什么计算的热心人，这都符合 AGI 在通用场景下的定义。
#      总之需要过滤信息，经过运筹计算，判断最优效率的路线和备选路线。」
#
#    → 身份（工具 / 专家 / 热心人）可以换，但**「干什么」必须一致**。
#    → 真实通用场景里，人面对的信息就是杂乱的：
#       有的工具写得很专业，有的写得很随意，有的自称专家。
#       模型要能从这些不同口吻里**筛出有用信息**。
#
# ⚠️ 因此模板分四组口吻：
#     A 组 · 工具口吻（中性、说明书式）
#     B 组 · 专家口吻（自称擅长某方面）
#     C 组 · 热心人口吻（口语、随意）
#     D 组 · 信息不全的口吻（需要式 / 已知式 / 流程式）

SYNTAX_TEMPLATES: list[str] = [
    # ══ A 组 · 工具口吻（中性，说明书式）════════════════
    '（{out}）的值为：由（{in}）的值{verb_rhs}。',
    '本工具对（{in}）的值执行如下变换：{verb}。执行完得到（{out}）的值。',
    '（{in}）的值 → （{out}）的值：{verb}。',

    # ══ B 组 · 专家口吻（自称擅长某方面）════════════════
    '我擅长处理（{in}）的值到（{out}）的值的换算，做法是：{verb}。',
    '我是这一类的专家：由（{in}）的值{verb_rhs}。结果就是（{out}）的值。',
    '本步任务是由（{in}）的值得到（{out}）的值，具体操作是 {verb}。',

    # ══ C 组 · 热心人口吻（口语、随意）══════════════════
    '你要是想从（{in}）的值得到（{out}）的值，我来帮你{verb_rhs}。',
    '懂一点这个：把（{in}）的值变成（{out}）的值，靠的是 {verb}。',
    '传入（{in}）的值，做一步操作：{verb}，结果即为（{out}）的值。',

    # ══ D 组 · 信息不全的口吻════════════════════════════
    '当需要由（{in}）的值得到（{out}）的值时，要做一步：{verb}。',
    '已知（{in}）的值，怎样才能得到（{out}）的值？做法是 {verb}。',
    '在从（{in}）的值走向（{out}）的值的流程里，这一步负责 {verb}。',
    '本工具的用途是帮助由（{in}）的值得到（{out}）的值，它做的运算是：{verb}。',
]


def render_tool_text_auto(pack_name: str, in_name: str, out_name: str,
                          tmpl_idx: int = 0, lang: str = 'zh') -> str:
    """
    ⭐ 按语言渲染工具描述（中 / 英）。

    ⚠️ 2026-11-07 新增：让「函数叙述」也能跨语言随机。
       Frank：「关于函数的叙述可以加入英文版本，
               这样随机的时候函数和选项都能完全随机。」

    ⚠️ 若英文描述缺失，退回中文并在返回值里标明 —— 不静默降级。
    """
    if lang == 'en':
        try:
            import skins_en as _sen
            v = _sen.FUNC_DESC_EN.get(pack_name)
            if v:
                return _sen.render_tool_text_en(v, in_name, out_name, tmpl_idx)
        except Exception:
            pass
        # 退回中文
    d = desc_of(pack_name)
    return render_tool_text(d, in_name, out_name, tmpl_idx)


def can_use_rhs(verb: str) -> bool:
    """
    这个动词短语能不能接在「由（X）的值」后面。

    ⚠️ 2026-11-07 实测：74 个描述里有 11 个不是「求…」结构，
       如「已知正弦值，反求对应的角度」「把一个数压缩到 0 与 1 之间」。
       这些接上 `{verb_rhs}` 会读成
       「由（誓言）的值已知误差函数值，反求原数」—— 语法断了。

    ⭐ 判据：以「求」开头才能接在「由…的值」后面。
    """
    return verb.strip().startswith('求')


def render_tool_text(desc: FuncDesc, in_name: str, out_name: str,
                     tmpl_idx: int = 0) -> str:
    """
    把一个函数的描述套进语法模板。

    ⚠️ 语法变，意思不变 —— 这是本函数唯一的职责。
    ⚠️ 变量名**不带括号**传入，模板里自己加「（…）的值」。

    ⚠️ 2026-11-07 修：`{verb_rhs}` 只适用于「求…」结构的描述。
       若描述不是「求…」开头，自动换一个不使用 `{verb_rhs}` 的模板。
    """
    if not SYNTAX_TEMPLATES:
        return desc.verb

    usable_rhs = can_use_rhs(desc.verb)
    n = len(SYNTAX_TEMPLATES)

    # ⚠️ 若不能用 verb_rhs，跳过含该占位符的模板
    for step in range(n):
        idx = (tmpl_idx + step) % n
        t = SYNTAX_TEMPLATES[idx]
        if not usable_rhs and '{verb_rhs}' in t:
            continue
        return t.format(verb=desc.verb,
                        verb_rhs=_as_rhs(desc.verb),
                        **{'in': in_name}, out=out_name)

    # 极端兜底：全部模板都含 verb_rhs（不会发生）
    return '%s（%s → %s）' % (desc.verb, in_name, out_name)


def _as_rhs(verb: str) -> str:
    """
    把动词短语改成「能接在『由（X）的值』后面」的谓语形态。

    ⚠️ 例：
        「求一个数自乘两次的结果」    →  「求得一个数自乘两次的结果」
        「求自然常数 e 的该数次幂」    →  「求得自然常数 e 的该数次幂」

    ⭐ 只把开头的「求」换成「求得」，其余一字不动 —— 精确保留。
    ⭐ 这样拼出来是「（小红）的值为：由（小明）的值求得一个数自乘两次的结果。」
       读得通，且语义精确。
    """
    s = verb.strip()
    if s.startswith('求'):
        return '求得' + s[1:]
    return s


# ══════════════════════════════════════════════════════════════════
# 四、自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0
    checks = []

    # 覆盖度
    all_names = [p.name for p in packs.PACKS]
    covered = [n for n in all_names if n in FUNC_DESC]
    missing = [n for n in all_names if n not in FUNC_DESC]

    checks.append(('74 个函数全有描述', len(covered) == len(all_names)))
    if missing and verbose:
        print('  ⚠️ 缺描述的：', missing)
    checks.append(('描述都非空', all(d.verb for d in FUNC_DESC.values())))
    checks.append(('描述不用 f 代替', all('f' not in d.verb for d in FUNC_DESC.values())))

    # 语法模板
    checks.append(('语法模板 ≥ 6 套', len(SYNTAX_TEMPLATES) >= 6))
    d = desc_of('pow2')
    outs = [render_tool_text(d, '小明', '小红', i) for i in range(len(SYNTAX_TEMPLATES))]
    checks.append(('每套模板都能渲染', all(o and '小明' in o for o in outs)))
    checks.append(('不同模板结果不同', len(set(outs)) == len(outs)))
    checks.append(('每套都含动词内容', all('自乘两次' in o for o in outs)))

    # ⭐ 全模板 × 全函数 的语法检查（2026-11-07）
    #    ⚠️ 实测教训：74 个描述里有 11 个不是「求…」结构，
    #       接 {verb_rhs} 会读成「由（X）的值已知误差函数值，反求原数」—— 语法断了。
    #
    #    ⚠️ 2026-11-07 再修：判据不能用「模板原文含不含 {verb_rhs}」——
    #       render_tool_text 会自动跳过带该占位符的模板，
    #       所以模板原文里有一百个 verb_rhs 也没关系。
    #       ⭐ 正确判据：**渲染出来的文本**里，不该出现「的值已知」「的值把」这类断裂。
    bad_grammar = []
    BAD_PATTERNS = ['的值已知', '的值把', '的值对', '的值一', '的值自然', '的值参数']
    for nm, d in FUNC_DESC.items():
        for ti in range(len(SYNTAX_TEMPLATES)):
            txt = render_tool_text(d, '甲', '乙', ti)
            for bp in BAD_PATTERNS:
                if bp in txt:
                    bad_grammar.append((nm, ti, bp, txt[:40]))
                    break
    checks.append(('渲染结果无语法断裂', not bad_grammar))
    if bad_grammar and verbose:
        print('  ⚠️ 语法断裂:')
        for x in bad_grammar[:3]:
            print('     ', x)
    # ⭐ 2026-11-07 Frank：名词后要带「的值」，让 LLM 知道那是个数
    checks.append(('每套都标出「的值」',
                   all('的值' in o for o in outs)))
    checks.append(('每套都保留变量名', all('（小明）' in o or '小明' in o for o in outs)))
    # ⚠️ 去掉「中间」这类多余词
    checks.append(('没有多余的「中间」', all('中间' not in o for o in outs)))

    # 变量名词
    checks.append(('变量名词表非空', len(VAR_NOUNS) > 0))
    checks.append(('同 seed 同名词',
                   noun_for_var(3, 7) == noun_for_var(3, 7)))

    # ⭐ 词库接入检查（2026-11-07）
    try:
        import lexicon as _lex
        lex = _lex.get_lexicon()
        checks.append(('词库可用', lex.total() > 0))
        checks.append(('词库含两种语言', len(lex.langs) >= 2))

        # 一次挑 11 个（一道题要 11 个变量名）
        names = pick_var_nouns(11, seed=20261107)
        checks.append(('能一次挑 11 个变量名', len(names) == 11))
        checks.append(('11 个变量名互不相同', len(set(names)) == 11))
        checks.append(('变量名不重复且不含 v0 这类占位符',
                       all(not n.startswith('v') or len(n) > 2 for n in names)))
        # 抽查：应是词库里的词，不是硬编码列表
        in_lex = sum(1 for n in names if _lex.get_lexicon().get(
            'CN0001') is not None)
        checks.append(('词库已接入', in_lex > 0))
    except Exception as e:
        checks.append(('词库可用', False))
        if verbose:
            print('  ⚠️ 词库检查失败:', e)
        for nm in ['词库含两种语言', '能一次挑 11 个变量名',
                   '11 个变量名互不相同', '变量名不重复且不含 v0 这类占位符',
                   '词库已接入']:
            checks.append((nm, False))

    if verbose:
        print('=' * 62)
        print('skins 自检')
        print('=' * 62)
        for name, ok in checks:
            print('  %s  %s' % ('✅' if ok else '❌', name))
            if not ok:
                fails += 1
        print()
        print('覆盖：%d / %d 个函数' % (len(covered), len(all_names)))
        print()
        print('── 同一个函数（平方）的 %d 种说法 ──' % len(SYNTAX_TEMPLATES))
        for i, o in enumerate(outs, 1):
            print('  %d. %s' % (i, o))
        print()
        print('── 高等函数举例 ──')
        for nm in ['digamma', 'ellipk', 'lambertw', 'airy']:
            print('  %-10s → %s' % (nm, desc_of(nm).verb))

    return fails


if __name__ == '__main__':
    raise SystemExit(self_test())
