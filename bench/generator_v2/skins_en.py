"""
英文函数描述 + 英文语法模板 —— 让函数陈述也能随机跨语言

⭐ 2026-11-07 Frank 定：
    「我认为关于函数的叙述可以加入英文版本，
      这样随机的时候函数和选项都能完全随机。
      你看看能不能实现，能的话要做准，
      函数这玩意差一个字母一个空格都差老远了。」

⭐⭐ 为什么要单独一个文件而不是塞进 skins.py：
    ① 文本量大（74 条描述 × 2 语言 + 两套模板）
    ② **可核对性**：英文版本要对得上数学定义，
       单独成文便于逐条审校，不必翻过 1000 行代码
    ③ **可替换性**：将来加日语/法语，照这个文件再加一份

⚠️ 翻译纪律（Frank 的要求落实为三条）：
    ① **逐条对译，不许意译**：
       「求一个数自乘两次的结果」→ "multiply a number by itself twice"
       ❌ 不能译成 "square a number"（漏了「两次」这个结构信息）
    ② **参数必须写全**：
       「求参数为该数平方的第一类完全椭圆积分」
       → "first-kind complete elliptic integral with parameter equal to the square of the input"
       ⚠️ 漏掉 "parameter equal to the square of the input" 就变了另一个函数
    ③ **固定参数必须显式**：
       「自变量为 2.5 与该数的赫尔维茨 zeta 函数」
       → "Hurwitz zeta function of the input as second argument, with first argument fixed at 2.5"

⚠️ 英文模板的占位符与中文一致：
    {verb}  动词短语全形
    {in}    输入变量名
    {out}   输出变量名
    ⚠️ 英文没有「{verb_rhs}」这一说 —— 英文语法结构不同，
       所有模板都用 {verb} 全形，不需要削词。
"""
from __future__ import annotations


# ══════════════════════════════════════════════════════════════════
# 一、74 个函数的英文描述
# ══════════════════════════════════════════════════════════════════

# ⚠️ 对照 skins.FUNC_DESC 逐条翻译，键名完全一致。
# ⭐ 维护规则：改中文描述时必须同步改这里，否则 skins 的自检会报错。
FUNC_DESC_EN: dict[str, str] = {

    # ── 幂与根 ──────────────────────────────────────────
    'pow2':    'multiply a number by itself twice',
    'pow3':    'multiply a number by itself three times',
    'pow4':    'multiply a number by itself four times',
    'pow5':    'multiply a number by itself five times',
    'pow8':    'multiply a number by itself eight times',
    'sqrt':    'take the square root of a number',
    'cbrt':    'take the cube root of a number',

    # ── 指数与对数 ──────────────────────────────────────
    'exp':     'raise the constant e to the power of the input',
    'log':     'take the natural logarithm of a number',
    'expm1':   'raise the constant e to the power of the input, then subtract one',
    'log1p':   'add one to the input, then take the natural logarithm',
    'log2':    'take the base-two logarithm of a number',
    'log10':   'take the base-ten logarithm of a number',
    'exp2':    'raise two to the power of the input',
    'exp10':   'raise ten to the power of the input',
    'exprel':  'raise the constant e to the power of the input, subtract one, then divide by the input',

    # ── 三角 ────────────────────────────────────────────
    'sin':     'take the sine of an angle',
    'cos':     'take the cosine of an angle',
    'tan':     'take the tangent of an angle',
    'asin':    'given a sine value, recover the corresponding angle',
    'acos':    'given a cosine value, recover the corresponding angle',
    'atan':    'given a tangent value, recover the corresponding angle',

    # ── 双曲 ────────────────────────────────────────────
    'sinh':    'take the hyperbolic sine of a number',
    'cosh':    'take the hyperbolic cosine of a number',
    'tanh':    'take the hyperbolic tangent of a number',
    'asinh':   'given a hyperbolic sine value, recover the original number',
    'acosh':   'given a hyperbolic cosine value, recover the original number',
    'atanh':   'given a hyperbolic tangent value, recover the original number',

    # ── 概率与误差 ──────────────────────────────────────
    'erf':     'take the error function value of a number',
    'erfc':    'take the complementary error function value of a number',
    'erfcx':   'take the complementary error function value of a number, then multiply by e raised to that number squared',
    'erfinv':  'given an error function value, recover the original number',
    'expit':   'squeeze a number into the interval between zero and one, by a sigmoid transform',
    'logit':   'spread a number between zero and one onto the whole real line, by a logit transform',
    'ndtr':    'take the probability that a standard normal variable falls below the input',
    'ndtri':   'given a standard normal probability, recover the corresponding value',
    'log_ndtr': 'take the probability that a standard normal variable falls below the input, then take its natural logarithm',
    'log_expit': 'take the natural logarithm of the result of a sigmoid transform',

    # ── 伽马族 ──────────────────────────────────────────
    'loggamma': 'take the natural logarithm of the gamma function',
    'gammaln':  'take the natural logarithm of the absolute value of the gamma function',
    'digamma':  'take the first derivative of the logarithm of the gamma function',
    'polygamma': 'take the second derivative of the logarithm of the gamma function',
    'beta':     'take the beta function with first argument fixed at 1.5 and second argument equal to the input',
    'betaln':   'take the natural logarithm of the beta function with first argument fixed at 1.5 and second argument equal to the input',
    'gammainc': 'take the regularized lower incomplete gamma function with parameter fixed at 1.5 and the input as the other argument',
    'gammaincc': 'take the regularized upper incomplete gamma function with parameter fixed at 1.5 and the input as the other argument',

    # ── 特殊 ────────────────────────────────────────────
    'zeta':     'take the Hurwitz zeta function with first argument fixed at 2.5 and second argument equal to the input',
    'lambertw': 'find that w satisfying the equation: the input equals w times e raised to the power of w',
    'entr':     'take the negative of the input multiplied by the natural logarithm of the input',
    'softplus': 'add one to e raised to the power of the input, then take the natural logarithm',
    'pseudo_huber': 'take the pseudo-Huber loss at the input, with the parameter fixed at 1.5',
    'hypot':    'take the square root of the sum of the square of the input and 1.5 squared',
    'xlogy':    'take 2.5 multiplied by the natural logarithm of the input',
    'xlog1py':  'take 2.5 multiplied by the natural logarithm of one plus the input',

    # ── 指数积分 ────────────────────────────────────────
    'exp1':     'take the first-order exponential integral',
    'expn':     'take the second-order exponential integral',
    'expi':     'take the exponential integral function at the input',

    # ── 椭圆积分 ────────────────────────────────────────
    'ellipk':   'take the first-kind complete elliptic integral with parameter equal to the square of the input',
    'ellipe':   'take the second-kind complete elliptic integral with parameter equal to the square of the input',
    'ellipkinc': 'take the first-kind incomplete elliptic integral with the input as the amplitude and the parameter fixed at 0.5',

    # ── 贝塞尔 ──────────────────────────────────────────
    'i0':       'take the zeroth-order modified Bessel function of the first kind',
    'i1':       'take the first-order modified Bessel function of the first kind',
    'iv':       'take the zeroth-order modified Bessel function of the first kind',
    'kv':       'take the zeroth-order modified Bessel function of the second kind',
    'ive':      'take the zeroth-order modified Bessel function of the first kind, then multiply by e raised to the negative absolute value of the input',
    'kve':      'take the zeroth-order modified Bessel function of the second kind, then multiply by e raised to the power of the input',
    'jv':       'take the zeroth-order Bessel function of the first kind',
    'yv':       'take the zeroth-order Bessel function of the second kind',
    'airy':     'take the Airy function at the input',

    # ── 正交多项式（三阶）────────────────────────────────
    'eval_hermite':  'expand the third-order Hermite polynomial at the input',
    'eval_laguerre': 'expand the third-order Laguerre polynomial at the input',
    'eval_chebyt':   'expand the third-order first-kind Chebyshev polynomial at the input',
    'eval_chebyu':   'expand the third-order second-kind Chebyshev polynomial at the input',
    'eval_legendre': 'expand the third-order Legendre polynomial at the input',
}


# ══════════════════════════════════════════════════════════════════
# 二、英文语法模板（10 套，对应中文的四组口吻）
# ══════════════════════════════════════════════════════════════════

# ⚠️ 英文不需要 {verb_rhs} —— 英文句子结构不同，
#    所有模板都直接嵌 {verb} 全形。
#
# ⚠️ 与中文一样分四组口吻（工具 / 专家 / 热心人 / 信息不全），
#    这样「身份多样」的设计在中英两侧一致。
SYNTAX_TEMPLATES_EN: list[str] = [
    # ══ A 组 · 工具口吻（中性，说明书式）════════════════
    'The value of ({out}) is obtained by the following step on the value of ({in}): {verb}.',
    'This tool performs the following transform on the value of ({in}): {verb}. The result is the value of ({out}).',
    'Value of ({in}) → value of ({out}): {verb}.',

    # ══ B 组 · 专家口吻（自称擅长某方面）════════════════
    'I specialise in converting the value of ({in}) into the value of ({out}); the step is to {verb}.',
    'I am an expert in this kind of operation: starting from the value of ({in}), {verb}. The result is the value of ({out}).',
    'The task of this step is to go from the value of ({in}) to the value of ({out}); concretely, {verb}.',

    # ══ C 组 · 热心人口吻（口语、随意）══════════════════
    'If you want to get the value of ({out}) from the value of ({in}), I can help you {verb}.',
    'I know a bit about this: to turn the value of ({in}) into the value of ({out}), {verb}.',
    'Feed in the value of ({in}), do one operation: {verb}. The result is the value of ({out}).',

    # ══ D 组 · 信息不全的口吻════════════════════════════
    'When the value of ({out}) is needed from the value of ({in}), one step is required: {verb}.',
    'Given the value of ({in}), how do we obtain the value of ({out})? The way is to {verb}.',
    'In the flow from the value of ({in}) towards the value of ({out}), this step is responsible for the operation: {verb}.',
    'This tool is meant to help obtain the value of ({out}) from the value of ({in}); the operation it performs is: {verb}.',
]


# ══════════════════════════════════════════════════════════════════
# 三、渲染（英文侧）
# ══════════════════════════════════════════════════════════════════

def render_tool_text_en(verb_en: str, in_name: str, out_name: str,
                        tmpl_idx: int = 0) -> str:
    """
    把英文动词短语套进英文模板。

    ⚠️ 与中文不同，英文**不需要**判断能不能接 verb_rhs ——
      英文所有模板都用 {verb} 全形，任何动词短语都能嵌进去。
    """
    if not SYNTAX_TEMPLATES_EN:
        return verb_en
    t = SYNTAX_TEMPLATES_EN[tmpl_idx % len(SYNTAX_TEMPLATES_EN)]
    return t.format(verb=verb_en, **{'in': in_name}, out=out_name)


# ══════════════════════════════════════════════════════════════════
# 四、自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0
    checks = []
    import sys as _sys
    _sys.path.insert(0, r'<BENCH>\generator')
    import skins

    # ── ① 覆盖度：中英键名必须完全一致 ──────────────────
    zh_keys = set(skins.FUNC_DESC)
    en_keys = set(FUNC_DESC_EN)
    checks.append(('英文描述覆盖全部 74 个函数', len(en_keys) == len(zh_keys)))
    checks.append(('中英键名完全一致', zh_keys == en_keys))
    if zh_keys != en_keys and verbose:
        print('  ⚠️ 中文有英文无:', sorted(zh_keys - en_keys)[:5])
        print('  ⚠️ 英文有中文无:', sorted(en_keys - zh_keys)[:5])

    # ── ② 非空、全小写开头、无中文 ──────────────────────
    checks.append(('英文描述都非空',
                   all(v.strip() for v in FUNC_DESC_EN.values())))
    no_cjk = []
    for k, v in FUNC_DESC_EN.items():
        if any('\u4e00' <= ch <= '\u9fff' for ch in v):
            no_cjk.append(k)
    checks.append(('英文描述不含中文字符', not no_cjk))
    if no_cjk and verbose:
        print('  ⚠️ 含中文的:', no_cjk[:5])

    # ── ③ ⭐ 固定参数必须显式写出（Frank：「差一个字母差老远」）──
    must_have_param = {
        'beta': '1.5', 'betaln': '1.5', 'gammainc': '1.5', 'gammaincc': '1.5',
        'zeta': '2.5', 'xlogy': '2.5', 'xlog1py': '2.5',
        'pseudo_huber': '1.5', 'hypot': '1.5', 'ellipkinc': '0.5',
    }
    miss_param = []
    for k, p in must_have_param.items():
        if p not in FUNC_DESC_EN.get(k, ''):
            miss_param.append((k, p))
    checks.append(('固定参数在英文描述里显式写出', not miss_param))
    if miss_param and verbose:
        print('  ⚠️ 缺参数:', miss_param)

    # ── ④ 模板 ──────────────────────────────────────────
    checks.append(('英文模板 ≥ 10 套', len(SYNTAX_TEMPLATES_EN) >= 10))
    outs = [render_tool_text_en('multiply a number by itself twice',
                                'cat', 'river', i)
            for i in range(len(SYNTAX_TEMPLATES_EN))]
    checks.append(('每套英文模板都能渲染', all(o and 'cat' in o for o in outs)))
    checks.append(('英文模板结果各不相同', len(set(outs)) == len(outs)))
    checks.append(('每套都含动词内容',
                   all('multiply a number by itself twice' in o for o in outs)))

    # ── ⑤ 全函数 × 全模板 无渲染异常 ────────────────────
    bad = []
    for k, v in FUNC_DESC_EN.items():
        for ti in range(len(SYNTAX_TEMPLATES_EN)):
            try:
                t = render_tool_text_en(v, 'alpha', 'beta', ti)
                if not t or 'alpha' not in t or v not in t:
                    bad.append((k, ti))
            except Exception:
                bad.append((k, ti))
    checks.append(('74 函数 × 10 模板 全部渲染正常', not bad))
    if bad and verbose:
        print('  ⚠️ 渲染异常:', bad[:5])

    if verbose:
        print('=' * 62)
        print('skins_en 自检')
        print('=' * 62)
        for name, ok in checks:
            print('  %s  %s' % ('✅' if ok else '❌', name))
            if not ok:
                fails += 1
        print()
        print('英文描述数：%d' % len(FUNC_DESC_EN))
        print()
        print('── 同一个函数（pow2）的 %d 种英文说法 ──' % len(SYNTAX_TEMPLATES_EN))
        for i, o in enumerate(outs, 1):
            print('  %2d. %s' % (i, o))
        print()
        print('── 固定参数抽查（这是最易错的地方）──')
        for k in ['beta', 'zeta', 'ellipkinc', 'hypot', 'pseudo_huber']:
            print('  %-12s %s' % (k, FUNC_DESC_EN[k]))

    return fails


if __name__ == '__main__':
    raise SystemExit(self_test())
