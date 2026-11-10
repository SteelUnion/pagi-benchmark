"""
题目模块化 —— 把一道题拆成「变量 / 函数 / 系数」三类可独立寻址的部件

⭐ 2026-11-07 Frank 定：
    「先模块化，怎么换是第二步的事情。
      把变量和函数还有系数拆开来，同时打上独立代码和标记，
      我的想法是，我们的中间量针对函数要识别出编号这是第几个编号函数，
      这是独立于哈希值存在的，因为函数的抽象化可能要包含特征，每个函数不一样。」

⭐ 三层设计：
    ┌──────────────────────────────────────────────┐
    │ 中间态中间层（本模块）：所有部件用【标准名 + 独立编号】     │
    │   · 变量   V###  ← 位置决定，链上第几个点                 │
    │   · 函数   E###/H###  ← 复用 packs.py 已有的编号         │
    │   · 系数   C###  ← 按 (层, 样本) 编号                    │
    │ ⚠️ 这一层不做任何替换。                           │
    └──────────────────────────────────────────────┘
              ↓ 【第二步：替换层，本模块不做】
    ┌──────────────────────────────────────────────┐
    │ 最终态：部件被替换成词库里的词 / 图片                     │
    └──────────────────────────────────────────────┘

⚠️ 编号独立于哈希：
    编号是「这个部件是什么」，哈希是「这道题有没有被改」。
    两者不能混。改题面文字不该让编号变，改数学结构才该让哈希变。

⚠️ 本模块只做拆分与标记，不做计算、不做替换。
    计算在 vectorform.py，替换是第二步。
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, r'<BENCH>\generator')

import packs
from skeleton import get_skeleton


# ══════════════════════════════════════════════════════════════════
# 一、部件类型与编号规则
# ══════════════════════════════════════════════════════════════════

VAR = 'VAR'      # 变量：v0..v10
FUNC = 'FUNC'    # 函数：链上每一层的工具
COEF = 'COEF'    # 系数：每层的 cn
VALUE = 'VALUE'  # 数值选项：50 个候选值

# ⚠️ 编号前缀说明（写进论文时用这张表）
#
# ⭐⭐⭐ 2026-11-08 Frank 定：变量必须按「来源」分两种前缀
#
#    之前的错误：起点和中间自变量都叫 V###
#        A000（起点，预指派）→ V000
#        X001..X010（每节算出的自变量）→ V001..V010
#        ⚠️ 两类东西混在一个前缀里，看代码时根本分不清
#           「这个是给的」还是「这个是每节函数算出来的」。
#           —— 这个歧义实际导致过多次理解错误。
#
#    ⭐ 现在分开：
#        A###  预指派（Assigned）
#              造题时定死的输入，只有一个：A000。
#              它是**题面唯一给出数值**的量 —— 出黑白数字图。
#              ⚠️ 不经函数生成，不是随机的。
#
#        X###  链节自变量（eXogenous，每节函数的输入）
#              X001..X010，每节一个。
#              每个都是**上一步的函数输出**，模型必须自己推。
#              ⚠️ 要拿两次抽象才给模型：
#                  ① 文本抽象 → 词典名词（誓言 / 洋芋 …）
#                  ② 图片抽象 → 「誓言 = -9」这样的图
#                 两维等权，这就是「多模态抽象维数 = 2」。
#              ⚠️ 数值本身绝不出现在题面文字里。
#
#    ⭐ 为什么 A 与 X 分开而不是编号连续：
#        连续编号会让「A000 与 X001 有没有关系」变成要记的事；
#        前缀不同则一眼分清，不需要记。
#
#    ⚠️ 函数（E###/H###）与系数（C##）的编号**不受影响** ——
#       它们独立于变量，编码也是独立的。
PREFIX = {
    VAR: 'A/X',    # A000（预指派）+ X001..X010（链节自变量）
    FUNC: '',      # 直接复用 packs 的 code：E001 / H003
    COEF: 'C',     # C00 .. C09    按层编号（层内再按样本，但本模块单题）
    VALUE: 'N',    # N00 .. N49    按呈现顺序编号
}

# ⚠️ 两种变量的前缀常量 —— 别在代码里写裸字符串
P_ASSIGNED = 'A'   # 预指派：A000
P_EXOGEN = 'X'     # 链节自变量：X001..X010


def var_pid(i: int) -> str:
    """
    ⭐ 第 i 个链上值的编号（i = 0..k）。

    i == 0        → 'A000'   预指派，题面给数值
    i >= 1        → 'X%03d'  链节自变量，要两次抽象

    ⚠️ 全模块只走这一个函数生成变量编号 —— 不要在别处手写。
    """
    if i == 0:
        return '%s%03d' % (P_ASSIGNED, 0)
    return '%s%03d' % (P_EXOGEN, i)


@dataclass
class Part:
    """
    一个可独立寻址的部件。

    ⚠️ 这是「中间态」的最小单位。替换层拿到的就是一串 Part。
    """
    pid: str                  # 独立编号，如 'X003' / 'E001' / 'C02' / 'N17'
    ptype: str                # VAR / FUNC / COEF / VALUE
    std_name: str             # 标准名（中文），替换前用这个
    std_text: str             # 标准写法（题面里直接用的形式）
    # ── 标记（供替换层判断"这个能不能换、怎么换"）──
    replaceable: bool = True  # 能不能替换
    frozen: bool = False      # 冻结：绝不替换
    feature: str = ''         # 函数特征（只对 FUNC 有值）
    layer: int = -1           # 所在层（-1 = 不适用）
    role: str = ''            # 角色：LITERAL / INVERSE / CHAIN / DECOY / CORRECT
    meta: dict = field(default_factory=dict)   # 其它附属信息

    def key(self) -> str:
        """替换键：同一 key 的部件在最终态里必须替换成同一个东西。"""
        return '%s|%s' % (self.ptype, self.pid)


@dataclass
class Module:
    """
    一道题的模块化中间态。

    parts   : 全部部件
    links   : 部件之间的连接关系（谁接谁）
    """
    uid: str
    parts: list[Part] = field(default_factory=list)
    links: list[tuple[str, str, str]] = field(default_factory=list)
    #           ↑ (from_pid, op, to_pid)   如 ('X001','E001','X002')

    # ── 查询 ──────────────────────────────────────────────
    def by_type(self, t: str) -> list[Part]:
        return [p for p in self.parts if p.ptype == t]

    def by_pid(self, pid: str) -> Optional[Part]:
        for p in self.parts:
            if p.pid == pid:
                return p
        return None

    def replaceable(self) -> list[Part]:
        """可以替换的部件（替换层的输入）。"""
        return [p for p in self.parts if p.replaceable and not p.frozen]

    def frozen(self) -> list[Part]:
        return [p for p in self.parts if p.frozen]

    def signature(self) -> str:
        """
        ⭐ 结构签名：只由「编号 + 连接关系」算出，**不含任何文字**。

        ⚠️ 用途：验证替换没改结构。
           替换前后 signature 必须一致 —— 这是第二步的验收判据。
        """
        s = ';'.join('%s:%s' % (p.pid, p.ptype) for p in sorted(self.parts, key=lambda x: x.pid))
        s += '||' + ';'.join('%s>%s>%s' % l for l in sorted(self.links))
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]

    def stats(self) -> dict:
        d: dict = {}
        for p in self.parts:
            d[p.ptype] = d.get(p.ptype, 0) + 1
        d['可替换'] = len(self.replaceable())
        d['冻结'] = len(self.frozen())
        return d


# ══════════════════════════════════════════════════════════════════
# 二、函数特征：每个函数不一样，替换时要看这个
# ══════════════════════════════════════════════════════════════════

def function_feature(pack) -> str:
    """
    ⭐ 生成一个函数的特征标签。

    ⚠️ Frank：「因为函数的抽象化可能要包含特征，每个函数不一样。」
    ⚠️ 这是给替换层判断「这个函数名能不能替换」用的。

    当前给出四维特征：
        kind:   elementary / higher
        inv:    named（有具名逆）/ symbolic（只能写 f⁻¹）
        sign:   POS / NEG / BOTH（能接受什么符号的输入）
        fam:    函数族（pow / exp / log / trig / special / ...）
    """
    kind = 'elementary' if pack.kind == 'ELEMENTARY' else 'higher'

    sk = get_skeleton(pack.name)
    inv = 'named' if (sk is not None and sk.inv_latex) else 'symbolic'

    lo, hi = pack.search
    if lo >= 0:
        sign = 'POS'
    elif hi <= 0:
        sign = 'NEG'
    else:
        sign = 'BOTH'

    return '%s|inv=%s|%s|%s' % (kind, inv, sign, _family(pack.name))


def _family(name: str) -> str:
    """按名字粗分函数族。⚠️ 只用于标记，不参与计算。"""
    n = name.lower()
    rules = [
        ('pow', 'pow'), ('sqrt', 'pow'), ('cbrt', 'pow'),
        ('exp', 'exp'), ('log', 'log'),
        ('sin', 'trig'), ('cos', 'trig'), ('tan', 'trig'),
        ('asin', 'itrig'), ('acos', 'itrig'), ('atan', 'itrig'),
        ('sinh', 'hyp'), ('cosh', 'hyp'), ('tanh', 'hyp'),
        ('asinh', 'ihyp'), ('acosh', 'ihyp'), ('atanh', 'ihyp'),
        ('erf', 'erf'), ('ndtr', 'cdf'), ('ndtri', 'cdf'),
        ('gamma', 'gamma'), ('lgamma', 'gamma'), ('digamma', 'gamma'),
        ('beta', 'beta'), ('i0', 'bessel'), ('i1', 'bessel'),
        ('iv', 'bessel'), ('kv', 'bessel'), ('ive', 'bessel'), ('kve', 'bessel'),
        ('jv', 'bessel'), ('yv', 'bessel'),
        ('ellip', 'ellip'),
        ('zeta', 'special'), ('lambertw', 'special'), ('expi', 'special'),
        ('expn', 'special'), ('airy', 'special'),
        ('eval_', 'poly'), ('hypot', 'special'),
        ('logit', 'logit'), ('expit', 'logit'),
        ('softplus', 'smooth'), ('exprel', 'smooth'), ('entr', 'entropy'),
        ('xlogy', 'xlog'), ('xlog1py', 'xlog'),
        ('gammainc', 'gammainc'), ('gammaincc', 'gammainc'),
        ('pseudo_huber', 'smooth'),
    ]
    for key, fam in rules:
        if key in n:
            return fam
    return 'other'


def _fill_spoken(tmpl: str, arg: str = 'x') -> str:
    """
    把 spoken 模板里的 %s 全部填成 arg。

    ⚠️ 有些模板有多个 %s（如 hypot 的「%s 的平方与 b 的平方之和再开方」），
       直接用 `tmpl % arg` 会报 not enough arguments。
    ⚠️ 也不做无脑 replace —— 万一模板里本来就有字母 x 会被误替换。
       这里只按 %s 逐个替换，数量对得上才用 %。
    """
    n = tmpl.count('%s')
    if n <= 0:
        return tmpl
    try:
        return tmpl % tuple([arg] * n)
    except Exception:
        # 兜底：按出现顺序逐个替换
        out = tmpl
        for _ in range(n):
            out = out.replace('%s', arg, 1)
        return out


# ══════════════════════════════════════════════════════════════════
# 三、从一个 Problem 拆成 Module
# ══════════════════════════════════════════════════════════════════

def build_module(prob) -> Module:
    """
    ⭐ 把 vectorform.VectorProblem 拆成模块化中间态。

    参数
    ----
    prob : VectorProblem（含 chain / nodes / final / cns / option_values 等）

    ⚠️ 本函数**不做替换**，只做拆分与编号。
    """
    uid = getattr(prob, 'uid', 'UNKNOWN')
    mod = Module(uid=uid)

    nodes = list(prob.nodes)
    chain = list(prob.chain)
    cns = list(getattr(prob, 'cns', []) or [])

    # ── ① 变量 A000（预指派）+ X001..X010（链节自变量）────
    #    A000 = 起点，题面给数值；X001..X010 = 每节函数算出的自变量
    for i, val in enumerate(nodes):
        pid = var_pid(i)
        if i == 0:
            role = 'LITERAL'          # 已知：起点（预指派）
        elif i == len(nodes) - 1:
            role = 'CHAIN'            # 终点（也是待求之一）
        else:
            role = 'CHAIN'            # 中间待求
        mod.parts.append(Part(
            pid=pid,
            ptype=VAR,
            std_name=('a%d' % i) if i == 0 else ('x%d' % i),
            std_text=('a%d' % i) if i == 0 else ('x%d' % i),
            replaceable=True,
            layer=-1,
            role=role,
            meta={
                'value': val,
                'known': i == 0,
                # ⚠️ 这两个标记是给替换层用的 ——
                #     预指派的给数值图，链节自变量的给两次抽象
                'assigned': i == 0,
                'exogen': i > 0,
            },
        ))

    # ── ② 函数 FUNC（复用 packs 的 code 作为独立编号）────
    for i, nm in enumerate(chain):
        pk = packs.get(nm)
        sk = get_skeleton(nm)
        pid = pk.code                       # E001 / H003 …（独立于哈希）
        feat = function_feature(pk)
        # ⚠️ 初等函数名可换，高等函数名冻结（换了会失真）
        is_elem = (pk.kind == 'ELEMENTARY')
        mod.parts.append(Part(
            pid=pid,
            ptype=FUNC,
            std_name=(sk.zh_name if sk else nm),
            std_text=_fill_spoken(sk.spoken) if (sk and sk.spoken) else nm,
            replaceable=is_elem,
            frozen=not is_elem,
            feature=feat,
            layer=i,
            role='CHAIN',
            meta={
                'pack': nm,
                'latex': sk.latex if sk else '',
                'plain': sk.plain if sk else '',
                'spoken': sk.spoken if sk else '',
                'inv_latex': sk.inv_latex if sk else '',
                'kind': pk.kind,
            },
        ))

    # ── ③ 系数 COEF ────────────────────────────────────
    for i, cn in enumerate(cns):
        pid = 'C%02d' % i
        mod.parts.append(Part(
            pid=pid,
            ptype=COEF,
            std_name='cn%d' % i,
            std_text=('%g' % cn) if cn else '',
            replaceable=True,
            layer=i,
            role='CHAIN',
            meta={'cn': cn},
        ))

    # ── ④ 选项值 VALUE（按呈现顺序编号）───────────────
    opt_vals = list(getattr(prob, 'option_values', []) or [])
    opt_steps = list(getattr(prob, 'option_steps', []) or [])
    for i, v in enumerate(opt_vals):
        step = opt_steps[i] if i < len(opt_steps) else 0
        pid = 'N%02d' % i
        mod.parts.append(Part(
            pid=pid,
            ptype=VALUE,
            std_name='n%d' % i,
            std_text='%.12g' % v,
            replaceable=False,          # 数值本身不替换，只换呈现
            frozen=False,
            role='CORRECT' if step > 0 else 'DECOY',
            meta={'value': v, 'step': int(step)},
        ))

    # ── ⑤ 连接关系 ─────────────────────────────────────
    #    var(i) --FUNC[i]--> var(i+1) ，中间的乘 cn 记为 COEF
    #    ⚠️ A000 -f0-> X001 -f1-> X002 … （变量编号走 var_pid）
    for i, nm in enumerate(chain):
        pk = packs.get(nm)
        fpid = pk.code
        mod.links.append((var_pid(i), fpid, var_pid(i + 1)))
        if i < len(cns):
            mod.links.append(('C%02d' % i, 'MUL', var_pid(i + 1)))

    return mod


# ══════════════════════════════════════════════════════════════════
# 四、呈现：中间态长什么样
# ══════════════════════════════════════════════════════════════════

def show_module(mod: Module) -> str:
    L = ['═' * 70]
    L.append('题目模块 · UID %s' % mod.uid)
    st = mod.stats()
    L.append('部件统计：' + '  '.join('%s=%d' % (k, v) for k, v in sorted(st.items())))
    L.append('结构签名：%s' % mod.signature())
    L.append('═' * 70)

    L.append('')
    L.append('【变量 VAR】')
    for p in mod.by_type(VAR):
        L.append('  %-6s %-6s 值=%-16.10g %s' % (
            p.pid, p.std_name, p.meta.get('value', 0),
            '（已知起点）' if p.meta.get('known') else ''))

    L.append('')
    L.append('【函数 FUNC】')
    for p in mod.by_type(FUNC):
        L.append('  %-6s 第%-2d层 %-14s %s' % (
            p.pid, p.layer, p.std_name,
            '🔒冻结' if p.frozen else '可替换'))
        L.append('         特征: %s' % p.feature)

    L.append('')
    L.append('【系数 COEF】')
    for p in mod.by_type(COEF):
        L.append('  %-6s 第%-2d层 cn=%-12g' % (p.pid, p.layer, p.meta.get('cn', 0) or 0))

    L.append('')
    L.append('【选项 VALUE】')
    vals = mod.by_type(VALUE)
    L.append('  共 %d 个（正确 %d，干扰 %d）' % (
        len(vals),
        sum(1 for p in vals if p.role == 'CORRECT'),
        sum(1 for p in vals if p.role == 'DECOY')))

    L.append('')
    L.append('【连接】')
    for a, op, b in mod.links[:6]:
        L.append('  %s --%s--> %s' % (a, op, b))
    if len(mod.links) > 6:
        L.append('  … 共 %d 条' % len(mod.links))

    return '\n'.join(L)


# ══════════════════════════════════════════════════════════════════
# 五、自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0
    checks = []

    import vectorform
    ps = vectorform.build_vector_problems(n=3, k=10, seed=20261107, verbose=False)
    mods = [build_module(p) for p in ps]
    m = mods[0]

    checks.append(('拆出 3 个模块', len(mods) == 3))
    checks.append(('变量 11 个', len(m.by_type(VAR)) == 11))
    checks.append(('函数 10 个', len(m.by_type(FUNC)) == 10))
    checks.append(('系数 10 个', len(m.by_type(COEF)) == 10))
    checks.append(('选项 50 个', len(m.by_type(VALUE)) == 50))
    checks.append(('选项里正确 10 个',
                   sum(1 for p in m.by_type(VALUE) if p.role == 'CORRECT') == 10))

    # 编号唯一
    pids = [p.pid for p in m.parts]
    checks.append(('编号全部唯一', len(pids) == len(set(pids))))

    # 函数编号复用 packs 的 code
    func_pids = [p.pid for p in m.by_type(FUNC)]
    checks.append(('函数编号是 E###/H### 形式',
                   all(p.startswith(('E', 'H')) and len(p) == 4 for p in func_pids)))

    # 高等函数冻结、初等可换
    fparts = m.by_type(FUNC)
    ok_freeze = all(
        (p.meta['kind'] == 'ELEMENTARY') == (not p.frozen) for p in fparts)
    checks.append(('初等可换、高等冻结', ok_freeze))

    # 特征非空
    checks.append(('每个函数都有特征', all(p.feature for p in fparts)))

    # 结构签名只依赖编号与连接，不含文字
    sig1 = m.signature()
    # 改掉一个部件的文字，签名不应变
    m2 = build_module(ps[0])
    for p in m2.parts:
        if p.ptype == FUNC:
            p.std_name = '完全不同的名字'
            p.std_text = 'XXXXXXXX'
            break
    checks.append(('改文字不影响结构签名', m2.signature() == sig1))

    # 同 seed 同签名
    m3 = build_module(ps[0])
    checks.append(('同数据同签名', m3.signature() == sig1))

    # 不同题签名不同
    m4 = build_module(ps[1])
    checks.append(('不同题签名不同', m4.signature() != sig1))

    # 连接数 = 10 条 FUNC + 10 条 COEF
    checks.append(('连接 20 条', len(m.links) == 20))

    if verbose:
        print('=' * 62)
        print('module 自检')
        print('=' * 62)
        for name, ok in checks:
            print('  %s  %s' % ('✅' if ok else '❌', name))
            if not ok:
                fails += 1
        print()
        print(show_module(m))
        print()

        # 打印一个高等函数的特征样例
        for p in m.by_type(FUNC):
            if p.frozen:
                print('高等函数特征样例：')
                print('  %s  第%d层  %s' % (p.pid, p.layer, p.std_name))
                print('  特征: %s' % p.feature)
                break

    return fails


if __name__ == '__main__':
    raise SystemExit(self_test())
