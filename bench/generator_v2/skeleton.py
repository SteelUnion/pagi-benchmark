"""
骨架层（Skeleton）—— 把每个函数「写出来」，注入文本框给 LLM 认

════════════════════════════════════════════════════════════════
【为什么要有这一层】（Frank 2026-11-06 21:00 定）
════════════════════════════════════════════════════════════════

    原话：「先要生成每种函数对应的骨架，每种函数要有对应的骨架，
          能注入到文本框，让 llm 识别。
          连函数我们都表达不出来，后面的功能无从谈起，
          现在是题目生成的地基施工。」

    ❌ 改之前：MathPack 只有 name（'ive'）+ note（'I₀缩放'）。
       **没有一个字段是「这个函数长什么样」**。
       74 个工具里，只有 exprel 的 note 塞了半句 '(e^x−1)/x'。

    ❌ 所以连这一步都做不到：
           把一个函数写进题面 → 塞进文本框 → LLM 看着认

    ⭐ 这一层补的就是这个：

        ① 把函数写进题面（文本）
        ② 把函数渲染成图（图片）
        ③ 让 LLM 从写法里认出这是什么

════════════════════════════════════════════════════════════════
【骨架是什么】
════════════════════════════════════════════════════════════════

    骨架 = 一个函数的「可书写形式」，三套写法：

        latex   ← 写题面、渲染 PNG（论文与图片都用它）
        plain   ← 纯 ASCII，塞进 jsonl 日志、命令行
        spoken  ← 中文口语（「把函数过程换成图片或文本表达」用这个）

    ⚠️ 三者必须是**同一个数学对象**，不能各有各的写法。

════════════════════════════════════════════════════════════════
【⚠️ 固定参数必须写进骨架】
════════════════════════════════════════════════════════════════

    registry 里的函数**不是**通用形式，很多带写死的参数。骨架必须如实写出，
    否则题面写的和实际算的不是一个东西：

        ive      → i0e(x)         缩放版！不是 I₀(x)
        kve      → k0e(x)         缩放版
        iv/kv/jv/yv → 全是 **0 阶**，不是 ν 阶
        i0/i1    → 未缩放 i0 / i1
        polygamma → ψ⁽²⁾          **2 阶**导，不是 ψ⁽¹⁾
        zeta     → ζ(2.5, x)      **Hurwitz ζ**，不是 Riemann ζ(x)
        xlogy    → 2.5·log x      系数 2.5
        xlog1py  → 2.5·log(1+x)
        beta     → B(1.5, x)
        betaln   → ln B(1.5, x)
        gammainc → P(1.5, x)
        gammaincc→ Q(1.5, x)
        hypot    → √(x² + 1.5²)
        pseudo_huber → √(2.25(1−e^{−(x/1.5)²}))
        expn     → **E₂**，不是 E₁
        ellipkinc→ 不完全第一类，模数 **0.5**
        eval_*   → 全部 **3 阶**多项式

    ⚠️ 2026-11-06 第一版骨架表按「通用形式」写，9 处写错，
       被 render_all() 的覆盖率检查抓出来 —— 这条检查现在保留。

════════════════════════════════════════════════════════════════
【怎么用】
════════════════════════════════════════════════════════════════

    from skeleton import get_skeleton, render_all

    sk = get_skeleton('ive')
    sk.tex('y')      # 'I_0(y)e^{-|y|}'  ← LaTeX
    sk.asc('y')      # 'i0e(y)'          ← ASCII
    sk.say('y')      # 'y 的第零阶第一类修正贝塞尔函数，带指数缩放'

    render_all()     # 全量检查
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ['Skeleton', 'SKELETONS', 'BY_NAME', 'get_skeleton',
           'render_all', 'coverage']


# ================================================================
# 一、骨架本体
# ================================================================


@dataclass
class Skeleton:
    """一个函数的「可书写形式」，三套写法指向同一个数学对象。"""

    name: str                      # 和 MathPack.name 对齐
    kind: str                      # ELEMENTARY / HIGHER
    domain: str                    # 学科领域

    latex: str = ''                # LaTeX：写题面、渲染 PNG
    plain: str = ''                # ASCII：日志、命令行
    spoken: str = ''               # 中文口语

    zh_name: str = ''              # 中文全名
    role: str = ''                 # 'outer' / 'inner' / ''
    fixed: dict = field(default_factory=dict)   # 写死的参数（如实记录）

    # ⭐ 逆运算的 LaTeX 写法（2026-11-06 新增）
    #    可以是解析形式（log 的逆就是 e^x），也可以留空走通用记号 f^{-1}(x)。
    #    ⚠️ 不能靠「给正向 LaTeX 加 ^{-1}」：
    #       'e^{-|x|}I_0(x)' 加^{-1} → 像整个乘积的倒数（错）
    #       't1^4' 加^{-1} → t1^4^{-1} 双重上标，LaTeX 编译报错
    inv_latex: str = ''
    inv_plain: str = ''

    # ------------------------------------------------------------
    def _fill(self, tpl: str, var: str) -> str:
        """
        把 %s 换成变量符号。

        ⚠️ 2026-11-06 修：一个模板里可能有**多个** %s
           （exprel 的 '(exp(%s)-1)/(%s)' 有两个），
           原来只传 var 一个 → TypeError: not enough arguments。
        """
        if not tpl:
            return ''
        n = tpl.count('%s')
        if n == 0:
            return tpl
        return tpl % ((var,) * n)

    def tex(self, var: str = 'y') -> str:
        return self._fill(self.latex, var)

    def asc(self, var: str = 'y') -> str:
        return self._fill(self.plain, var)

    def say(self, var: str = 'y') -> str:
        return self._fill(self.spoken, var)

    # ------------------------------------------------------------
    # ⭐ 逆运算写法（2026-11-06 新增）
    # ------------------------------------------------------------

    def tex_inverse(self, var: str = 'x') -> str:
        """
        逆运算的 LaTeX。

        ⭐ 优先用解析形式（log⁻¹ = e^x，pow4⁻¹ = √x），
           没有解析形式就退回通用记号 f^{-1}(x)。
        ⚠️ 绝不「给正向表达式加 ^{-1}」——
           见 Skeleton.inv_latex 上面的注释。
        """
        if self.inv_latex:
            return self._fill(self.inv_latex, var)
        base = self.tex(var)
        return r'\operatorname{inv}\left(%s\right)' % base

    def asc_inverse(self, var: str = 'x') -> str:
        if self.inv_plain:
            return self._fill(self.inv_plain, var)
        return 'inv(%s)' % self.asc(var)

    def missing(self) -> list[str]:
        """哪几套写法还缺着（地基没打完）。"""
        return [k for k in ('latex', 'plain', 'spoken')
                if not getattr(self, k)]


# ================================================================
# 二、74 个骨架 —— 逐条对照 registry.py 的真实定义
# ================================================================


def _E(name, domain, latex, plain, spoken, zh='', role='', fixed=None,
       inv=None):
    """
    造一个初等骨架。

    inv = (inv_latex, inv_plain) —— 逆运算怎么写。
    ⚠️ 不给 inv 就走通用记号 operatorname{inv}(...)，
       绝不「给正向表达式加 ^{−1}」（会出歧义和双重上标）。
    """
    il, ip = inv if inv else ('', '')
    return Skeleton(name=name, kind='ELEMENTARY', domain=domain,
                    latex=latex, plain=plain, spoken=spoken,
                    zh_name=zh or name, role=role, fixed=fixed or {},
                    inv_latex=il, inv_plain=ip)


def _H(name, domain, latex, plain, spoken, zh='', role='', fixed=None,
       inv=None):
    il, ip = inv if inv else ('', '')
    return Skeleton(name=name, kind='HIGHER', domain=domain,
                    latex=latex, plain=plain, spoken=spoken,
                    zh_name=zh or name, role=role, fixed=fixed or {},
                    inv_latex=il, inv_plain=ip)


SKELETONS: list[Skeleton] = [

    # ============================================================
    # 初等函数（53 个）
    # ============================================================

    # ---------- 幂与根（互为逆）----------
    _E('pow2', '通用', r'%s^2', '(%s)^2', '%s 的平方', '平方',
       inv=(r'\sqrt{%s}', 'sqrt(%s)')),
    _E('pow3', '通用', r'%s^3', '(%s)^3', '%s 的立方', '立方',
       inv=(r'\sqrt[3]{%s}', 'cbrt(%s)')),
    _E('pow4', '通用', r'%s^4', '(%s)^4', '%s 的四次方', '四次方',
       inv=(r'\sqrt[4]{%s}', '(%s)^0.25')),
    _E('pow5', '通用', r'%s^5', '(%s)^5', '%s 的五次方', '五次方',
       inv=(r'\sqrt[5]{%s}', '(%s)^0.2')),
    _E('pow8', '通用', r'%s^8', '(%s)^8', '%s 的八次方', '八次方',
       inv=(r'\sqrt[8]{%s}', '(%s)^0.125')),
    _E('sqrt', '通用', r'\sqrt{%s}', 'sqrt(%s)', '%s 的平方根', '平方根',
       inv=(r'%s^2', '(%s)^2')),
    _E('cbrt', '通用', r'\sqrt[3]{%s}', 'cbrt(%s)', '%s 的立方根', '立方根',
       inv=(r'%s^3', '(%s)^3')),

    # ---------- 指数对数（互为逆）----------
    _E('exp', '通用', r'e^{%s}', 'exp(%s)', 'e 的 %s 次幂', '指数',
       inv=(r'\ln %s', 'log(%s)')),
    _E('expm1', '通用', r'e^{%s}-1', 'expm1(%s)', 'e 的 %s 次幂减一', 'e 的该变量次方减一',
       inv=(r'\ln(1+%s)', 'log1p(%s)')),
    _E('exprel', '通用', r'\dfrac{e^{%s}-1}{%s}',
       '(expm1(%s))/(%s)', 'e 的 %s 次幂减一后除以 %s', 'e 的该变量次方减一再除以该变量'),
    _E('log', '通用', r'\ln %s', 'log(%s)', '%s 的自然对数', '自然对数',
       inv=(r'e^{%s}', 'exp(%s)')),
    _E('log1p', '通用', r'\ln(1+%s)', 'log1p(%s)', '一加 %s 的自然对数', '一加该变量的自然对数',
       inv=(r'e^{%s}-1', 'expm1(%s)')),
    _E('log2', '通用', r'\log_2 %s', 'log2(%s)', '以二为底 %s 的对数', '以二为底的对数',
       inv=(r'2^{%s}', 'exp2(%s)')),
    _E('log10', '通用', r'\log_{10} %s', 'log10(%s)', '以十为底 %s 的对数', '以十为底的对数',
       inv=(r'10^{%s}', 'exp10(%s)')),
    _E('exp2', '通用', r'2^{%s}', 'exp2(%s)', '二的 %s 次幂', '二的该变量次方',
       inv=(r'\log_2 %s', 'log2(%s)')),
    _E('exp10', '通用', r'10^{%s}', 'exp10(%s)', '十的 %s 次幂', '十的该变量次方',
       inv=(r'\log_{10} %s', 'log10(%s)')),
    _E('softplus', '通用', r'\ln\left(1+e^{%s}\right)', 'softplus(%s)',
       '一加 e 的 %s 次幂再取自然对数', 'softplus'),
    _E('loggamma', '核工程', r'\ln\Gamma(%s)', 'loggamma(%s)',
       '%s 的伽马函数取自然对数', '对数伽马'),
    _E('gammaln', '核工程', r'\ln\left|\Gamma(%s)\right|', 'gammaln(%s)',
       '%s 的伽马函数绝对值取自然对数', '对数伽马绝对值'),

    # ---------- 三角双曲 ----------
    _E('sin', '通用', r'\sin %s', 'sin(%s)', '%s 的正弦', '正弦', 'inner',
       inv=(r'\arcsin %s', 'asin(%s)')),
    _E('cos', '通用', r'\cos %s', 'cos(%s)', '%s 的余弦', '余弦', 'inner',
       inv=(r'\arccos %s', 'acos(%s)')),
    _E('tan', '通用', r'\tan %s', 'tan(%s)', '%s 的正切', '正切',
       inv=(r'\arctan %s', 'atan(%s)')),
    _E('atan', '通用', r'\arctan %s', 'atan(%s)', '%s 的反正切', '反正切',
       inv=(r'\tan %s', 'tan(%s)')),
    _E('asin', '通用', r'\arcsin %s', 'asin(%s)', '%s 的反正弦', '反正弦',
       inv=(r'\sin %s', 'sin(%s)')),
    _E('acos', '通用', r'\arccos %s', 'acos(%s)', '%s 的反余弦', '反余弦',
       inv=(r'\cos %s', 'cos(%s)')),
    _E('sinh', '通用', r'\sinh %s', 'sinh(%s)', '%s 的双曲正弦', '双曲正弦',
       inv=(r'\operatorname{asinh} %s', 'asinh(%s)')),
    _E('cosh', '通用', r'\cosh %s', 'cosh(%s)', '%s 的双曲余弦', '双曲余弦',
       inv=(r'\operatorname{acosh} %s', 'acosh(%s)')),
    _E('tanh', '通用', r'\tanh %s', 'tanh(%s)', '%s 的双曲正切', '双曲正切',
       inv=(r'\operatorname{atanh} %s', 'atanh(%s)')),
    _E('asinh', '通用', r'\operatorname{asinh} %s', 'asinh(%s)',
       '%s 的反双曲正弦', '反双曲正弦',
       inv=(r'\sinh %s', 'sinh(%s)')),
    _E('acosh', '通用', r'\operatorname{acosh} %s', 'acosh(%s)',
       '%s 的反双曲余弦', '反双曲余弦',
       inv=(r'\cosh %s', 'cosh(%s)')),
    _E('atanh', '通用', r'\operatorname{atanh} %s', 'atanh(%s)',
       '%s 的反双曲正切', '反双曲正切',
       inv=(r'\tanh %s', 'tanh(%s)')),
    # ⚠️ hypot 第二参数写死 1.5
    _E('hypot', '通用', r'\sqrt{%s^2+1.5^2}', 'hypot(%s, 1.5)',
       '%s 的平方与一点五的平方之和再开方',
       '该变量平方与一点五平方之和的开方', fixed={'b': 1.5}),

    # ---------- 误差函数与统计 ----------
    _E('erf', '热传导', r'\operatorname{erf} %s', 'erf(%s)',
       '%s 的误差函数', '误差函数',
       inv=(r'\operatorname{erfinv} %s', 'erfinv(%s)')),
    _E('erfc', '热传导', r'\operatorname{erfc} %s', 'erfc(%s)',
       '%s 的补误差函数', '补误差函数'),
    _E('erfinv', '热传导', r'\operatorname{erfinv} %s', 'erfinv(%s)',
       '%s 的误差函数反函数', '误差函数的逆函数',
       inv=(r'\operatorname{erf} %s', 'erf(%s)')),
    _E('erfcx', '热传导', r'\operatorname{erfcx} %s', 'erfcx(%s)',
       '%s 的缩放补误差函数', '缩放补误差函数'),
    _E('expit', '信息论', r'\sigma(%s)', 'expit(%s)',
       '%s 的 sigmoid 函数', 'sigmoid',
       inv=(r'\ln\dfrac{%s}{1-%s}', 'logit(%s)')),
    _E('logit', '信息论', r'\ln\dfrac{%s}{1-%s}', 'logit(%s)',
       '%s 除以一减 %s 再取自然对数', 'logit',
       inv=(r'\sigma(%s)', 'expit(%s)')),
    _E('ndtr', '统计', r'\Phi(%s)', 'ndtr(%s)',
       '%s 的标准正态分布累积分布函数', '标准正态分布累积分布函数',
       inv=(r'\Phi^{-1}(%s)', 'ndtri(%s)')),
    _E('ndtri', '统计', r'\Phi^{-1}(%s)', 'ndtri(%s)',
       '%s 的标准正态分布分位数函数', '标准正态分布分位数函数',
       inv=(r'\Phi(%s)', 'ndtr(%s)')),
    _E('log_ndtr', '统计', r'\ln\Phi(%s)', 'log_ndtr(%s)',
       '%s 的标准正态累积分布函数取自然对数', '标准正态累积分布函数取自然对数'),
    _E('log_expit', '信息论', r'\ln\sigma(%s)', 'log_expit(%s)',
       '%s 的 sigmoid 函数取自然对数', '对数 sigmoid'),

    # ---------- Γ / 数论 / 信息论 ----------
    _E('digamma', '数论', r'\psi(%s)', 'digamma(%s)',
       '%s 的双伽马函数', '双伽马函数'),
    # ⚠️ polygamma 是 **2 阶**导
    _E('polygamma', '数论', r'\psi^{(2)}(%s)', 'polygamma(2, %s)',
       '%s 的二阶多伽马函数', '多伽马函数', fixed={'n': 2}),
    # ⚠️ zeta 是 Hurwitz ζ(2.5, x)，不是 Riemann ζ(x)
    _E('zeta', '数论', r'\zeta(2.5, %s)', 'zeta(2.5, %s)',
       '赫尔维茨 zeta 函数，参数 2.5 与 %s', '赫尔维茨 zeta 函数', fixed={'s': 2.5}),
    # ⚠️ xlogy / xlog1py 第一参数写死 2.5
    _E('xlogy', '信息论', r'2.5\ln %s', 'xlogy(2.5, %s)',
       '二点五乘 %s 的自然对数', '二点五乘该变量的自然对数', fixed={'x': 2.5}),
    _E('xlog1py', '信息论', r'2.5\ln(1+%s)', 'xlog1py(2.5, %s)',
       '二点五乘一加 %s 的自然对数', '二点五乘一加该变量的自然对数', fixed={'x': 2.5}),
    # ⚠️ beta / betaln 第一参数写死 1.5
    _E('beta', '热传导', r'B(1.5, %s)', 'beta(1.5, %s)',
       '贝塔函数，参数一点五与 %s', '贝塔函数', fixed={'a': 1.5}),
    _E('betaln', '热传导', r'\ln B(1.5, %s)', 'betaln(1.5, %s)',
       '贝塔函数取自然对数，参数一点五与 %s', '贝塔函数取自然对数', fixed={'a': 1.5}),
    # ⚠️ gammainc / gammaincc 第一参数写死 1.5
    _E('gammainc', '核工程', r'P(1.5, %s)', 'gammainc(1.5, %s)',
       '正则化下不完全伽马函数，参数一点五与 %s', '正则化下不完全伽马函数',
       fixed={'a': 1.5}),
    _E('gammaincc', '核工程', r'Q(1.5, %s)', 'gammaincc(1.5, %s)',
       '正则化上不完全伽马函数，参数一点五与 %s', '正则化上不完全伽马函数',
       fixed={'a': 1.5}),
    _E('entr', '信息论', r'-%s\ln %s', 'entr(%s)',
       '负的 %s 乘 %s 的自然对数', '负的该变量乘该变量的自然对数'),

    # ---------- 损失型 ----------
    # ⚠️ pseudo_huber: √(2.25(1−e^{−(x/1.5)²}))
    _E('pseudo_huber', '信息论',
       r'\sqrt{2.25\left(1-e^{-\left(%s/1.5\right)^2}\right)}', 'pseudo_huber(%s)',
       '二点二五乘以一减 e 的负的 %s 除以一点五的平方次幂，再开方',
       '伪 Huber', fixed={'delta': 1.5}),

    # ============================================================
    # 高等函数（21 个）
    # ============================================================

    # ---------- 贝塞尔族（全部 0 阶）----------
    _H('jv', '流体力学', r'J_0(%s)', 'jv(0, %s)',
       '%s 的第零阶第一类贝塞尔函数', '第一类贝塞尔 J₀', fixed={'nu': 0}),
    _H('yv', '流体力学', r'Y_0(%s)', 'yv(0, %s)',
       '%s 的第零阶第二类贝塞尔函数', '第二类贝塞尔 Y₀', fixed={'nu': 0}),
    # ⚠️ iv / kv 是**未缩放**的 i0 / k0
    _H('iv', '流体力学', r'I_0(%s)', 'iv(0, %s)',
       '%s 的第零阶第一类修正贝塞尔函数', '修正贝塞尔 I₀', fixed={'nu': 0}),
    _H('kv', '流体力学', r'K_0(%s)', 'kv(0, %s)',
       '%s 的第零阶第二类修正贝塞尔函数', '修正贝塞尔 K₀', fixed={'nu': 0}),
    # ⚠️ ive / kve 是**缩放版**（名字里的 e 就是这个意思）
    _H('ive', '流体力学', r'e^{-|%s|}I_0(%s)', 'i0e(%s)',
       '%s 的第零阶第一类修正贝塞尔函数，带指数缩放', '缩放修正贝塞尔 I₀',
       fixed={'nu': 0, 'scaled': True}),
    _H('kve', '流体力学', r'e^{%s}K_0(%s)', 'k0e(%s)',
       '%s 的第零阶第二类修正贝塞尔函数，带指数缩放', '缩放修正贝塞尔 K₀',
       fixed={'nu': 0, 'scaled': True}),
    _H('i0', '流体力学', r'I_0(%s)', 'i0(%s)',
       '%s 的第零阶第一类修正贝塞尔函数（未缩放）', '贝塞尔 i0'),
    _H('i1', '流体力学', r'I_1(%s)', 'i1(%s)',
       '%s 的第一阶第一类修正贝塞尔函数', '贝塞尔 i1'),

    # ---------- 艾里 ----------
    _H('airy', '光学', r'\operatorname{Ai}(%s)', 'airy_ai(%s)',
       '%s 的艾里函数 Ai', '艾里函数'),

    # ---------- 正交多项式（全部 3 阶）----------
    _H('eval_hermite', '量子力学', r'H_3(%s)', 'hermite(3, %s)',
       '%s 的三阶厄米多项式', '三阶厄米多项式', fixed={'n': 3}),
    _H('eval_laguerre', '量子力学', r'L_3(%s)', 'laguerre(3, %s)',
       '%s 的三阶拉盖尔多项式', '三阶拉盖尔多项式', fixed={'n': 3}),
    _H('eval_chebyt', '数值分析', r'T_3(%s)', 'chebyt(3, %s)',
       '%s 的三阶第一类切比雪夫多项式', '三阶第一类切比雪夫多项式', fixed={'n': 3}),
    _H('eval_chebyu', '数值分析', r'U_3(%s)', 'chebyu(3, %s)',
       '%s 的三阶第二类切比雪夫多项式', '三阶第二类切比雪夫多项式', fixed={'n': 3}),
    _H('eval_legendre', '量子力学', r'P_3(%s)', 'legendre(3, %s)',
       '%s 的三阶勒让德多项式', '三阶勒让德多项式', fixed={'n': 3}),

    # ---------- 椭圆积分 ----------
    _H('ellipk', '材料力学', r'K(%s)', 'ellipk(%s)',
       '%s 的第一类完全椭圆积分', '第一类完全椭圆积分'),
    _H('ellipe', '材料力学', r'E(%s)', 'ellipe(%s)',
       '%s 的第二类完全椭圆积分', '第二类完全椭圆积分'),
    # ⚠️ ellipkinc 模数写死 0.5
    _H('ellipkinc', '材料力学', r'F(\phi, 0.5)\big|_{m=0.5}',
       'ellipkinc(%s, 0.5)', '%s 的第一类不完全椭圆积分，模数零点五',
       '第一类不完全椭圆积分', fixed={'m': 0.5}),

    # ---------- 指数积分 ----------
    _H('exp1', '核工程', r'E_1(%s)', 'exp1(%s)',
       '%s 的一阶指数积分', '一阶指数积分'),
    # ⚠️ expn 是 **E₂**，不是 E₁
    _H('expn', '核工程', r'E_2(%s)', 'expn(2, %s)',
       '%s 的二阶指数积分', '二阶指数积分', fixed={'n': 2}),
    _H('expi', '核工程', r'\operatorname{Ei}(%s)', 'expi(%s)',
       '%s 的指数积分函数 Ei', '指数积分函数'),

    # ---------- 化工 ----------
    _H('lambertw', '化工', r'W(%s)', 'lambertw(%s)',
       '%s 的朗伯 W 函数', '朗伯 W',
       inv=(r'%s\,e^{%s}', '(%s)*exp(%s)')),
]


# ================================================================
# 三、建索引
# ================================================================

BY_NAME: dict[str, Skeleton] = {s.name: s for s in SKELETONS}


def get_skeleton(name: str) -> Skeleton:
    """按函数名取骨架。"""
    if name not in BY_NAME:
        raise KeyError(f'没有 {name} 的骨架')
    return BY_NAME[name]


# ================================================================
# 四、覆盖检查 —— 地基打没打完
# ================================================================


def coverage() -> dict:
    """
    ⭐ 两个方向都查：
        ① 每个 MathPack 有没有骨架
        ② 每个骨架是不是三套写法都齐
    """
    import packs

    have = set(BY_NAME)
    need = set(p.name for p in packs.PACKS)
    incomplete = [(s.name, s.missing()) for s in SKELETONS if s.missing()]

    return {
        'packs': len(need),
        'skeletons': len(SKELETONS),
        'missing': sorted(need - have),   # 有包没骨架 → 地基缺块
        'extra': sorted(have - need),     # 有骨架没包 → 悬空引用
        'incomplete': incomplete,         # 写法不全
    }


def render_all(verbose: bool = True) -> int:
    """把全部骨架渲染一遍，返回问题数量。0 = 地基打完。"""
    import packs

    cov = coverage()
    if verbose:
        print('=' * 110)
        print('骨架层检查（每个函数能不能写出来）')
        print('=' * 110)
        print(f'{"":<2} {"包名":<14} {"类型":<5} {"LaTeX":<32} {"ASCII":<22} {"中文"}')
        print('-' * 110)

    for p in packs.PACKS:
        s = BY_NAME.get(p.name)
        if not verbose:
            continue
        if s is None:
            print(f'❌ {p.name:<14} {p.kind[:4]:<5} —— 没有骨架 ——')
        else:
            flag = '✅' if not s.missing() else '🟡'
            print(f'{flag} {p.name:<14} {p.kind[:4]:<5} '
                  f'{s.tex("y"):<32} {s.asc("y"):<22} {s.say("y")}')

    n_bad = len(cov['missing']) + len(cov['extra']) + len(cov['incomplete'])
    if verbose:
        print('-' * 110)
        print(f'数理包 {cov["packs"]} 个，骨架 {cov["skeletons"]} 个')
        if cov['missing']:
            print(f'⚠️ 有包没骨架（{len(cov["missing"])}）：'
                  + ', '.join(cov['missing']))
        if cov['extra']:
            print(f'⚠️ 有骨架没包（{len(cov["extra"])}）：'
                  + ', '.join(cov['extra']))
        if cov['incomplete']:
            print(f'⚠️ 写法不全（{len(cov["incomplete"])}）：')
            for n, m in cov['incomplete']:
                print(f'    {n}: 缺 {", ".join(m)}')
        if n_bad == 0:
            print('⭐ 地基已打完：每个函数都能写出来')
        else:
            print(f'❌ 还有 {n_bad} 处问题')
    return n_bad


if __name__ == '__main__':
    raise SystemExit(0 if render_all() == 0 else 1)
