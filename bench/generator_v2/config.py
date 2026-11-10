"""
题目配置 —— 所有难度参数集中在一处

⭐ 2026-11-07 Frank 定：
    「选项、干扰项数目这些我都想搞成配置参数，因为这样难度就好调整了。」

⭐ 为什么集中：
    难度不是靠改代码调的，是靠改配置调的。
    改完之后，配置本身要写进日志（保存清单 §二 的「复现必需参数」）。

⚠️ 每一项都标明「它影响什么难度」。改之前先看那一列。
⚠️ 默认值 = 当前论文里定下的数（除非注明是新增可调项）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


# ══════════════════════════════════════════════════════════════════
# 难度等级预设（overview 用）
# ══════════════════════════════════════════════════════════════════

DIFFICULTY_PRESETS = {
    # 名    k   干扰   入选项  区分度
    # ⚠️ 2026-11-07：已删除 n_tool 键 —— 工具数是因变量，见 ProblemConfig.n_tool
    'L1': dict(k=4,   n_decoy=20, n_pick=10, sep_digits=4),
    'L2': dict(k=6,   n_decoy=30, n_pick=20, sep_digits=5),
    'L3': dict(k=8,   n_decoy=40, n_pick=30, sep_digits=6),
    'L4': dict(k=10,  n_decoy=50, n_pick=40, sep_digits=6),
    'L5': dict(k=12,  n_decoy=80, n_pick=60, sep_digits=7),
}


@dataclass
class ProblemConfig:
    """
    一道题的完整配置。

    ⭐ 分组：链 / 数值层 / 选项层 / 工具层 / 量级 / 复现
    """

    # ── ① 链 ──────────────────────────────────────────
    k: int = 10
    """复合次数。10 → 11 个节点 + 1 个连乘终值 = 第 12 个数。
       影响难度：越长越难，也越容易撞车。"""

    # ── ② 数值层 ──────────────────────────────────────
    n_decoy: int = 50
    """干扰值个数。
       影响难度：干扰值越多，工具池越大、可选面越宽。"""

    # ── ③ 选项层 ──────────────────────────────────────
    n_pick: int = 40
    """从 n_decoy 个干扰值里抽几个进选项。
       选项总数 = k（正确的中间值）+ n_pick。
       ⚠️ 若 n_pick > n_decoy 会报错（不能抽超过库存）。"""

    sep_digits: int = 6
    """区分度：小数点后第几位才算不同。
       影响难度：位数越大越严，链被淘汰得越多。
       ⚠️ 终值判据 = 本值 × √k（由它推出，不另设）。"""

    min_gap_rel: float = 0.0
    """最小相对间距（额外要求）。
       0 = 只按 sep_digits 判；
       >0 = 任意两个节点必须差这么多（如 0.01 = 1%）。
       ⚠️ 2026-11-07 提出但未启用，默认 0 保持现状。"""

    # ── ④ 工具层 ──────────────────────────────────────
    # ⚠️ 2026-11-07 Frank 定：**没有 n_tool 这个参数。**
    #
    #   「那个才是总工具量吧，因为你的"工具"说白了他不就是正运算那部分的函数吗，
    #     你正运算的函数不是一开始就定了吗，n_tools 看起来像个因变量而不是自变量啊？
    #     你不可能取60个解不了题的工具出来吧？」
    #
    #   → 工具数 = k + n_decoy，**纯因变量**，不单独设。
    #   → 原 n_tool 会把已算出的工具截断，可能把链上必需的截掉，造出解不了的题。
    #   → 想改工具数，就改 k 或 n_decoy。

    # ── ⑤ 量级约束 ────────────────────────────────────
    node_min: float = 1e-30
    """每个节点 |v| 的下界。出界即淘汰。"""

    node_max: float = 1e30
    """每个节点 |v| 的上界。"""

    final_min: float = 1e-50
    """连乘终值 |v11| 的下界。Frank：小数点前后 50 位以内。"""

    final_max: float = 1e50
    """连乘终值的上界。"""

    decoy_lo: float = 1e-3
    """干扰值的取值下界（会被每条链自身的量级再收窄）。"""

    decoy_hi: float = 1e3
    """干扰值的取值上界。"""

    # ── ⑥ 复现 ────────────────────────────────────────
    seed: int = 0
    """随机种子。⚠️ 必须写进日志，否则复现不了。"""

    oversample: float = 1.3
    """多造多少倍。造 100 道就先生成 130 条再筛。
       ⚠️ 越高越不容易造不出来，代价是慢。"""

    max_batch: int = 200000
    """单批上限（防止内存爆）。
       ⚠️ 实测 32768 条只占少量显存，这个上限很保守。"""

    # ── ⑦ 输出 ────────────────────────────────────────
    verbose: bool = True

    # ══════════════════════════════════════════════════
    # 派生量（改配置自动跟着变，不用手填）
    # ══════════════════════════════════════════════════

    @property
    def n_nodes(self) -> int:
        """节点数 = k + 1（v0 到 vk）。"""
        return self.k + 1

    @property
    def n_mid(self) -> int:
        """中间值个数 = k（v1 到 vk，也是正确答案的个数）。"""
        return self.k

    @property
    def n_option(self) -> int:
        """选项总数 = 中间值 + 抽入的干扰。默认 10 + 40 = 50。"""
        return self.n_mid + self.n_pick

    @property
    def n_tool(self) -> int:
        """
        工具总数 = k + n_decoy。

        ⚠️ 2026-11-07 Frank 定：**这是因变量，不是自变量，不可单独设。**
           「那个才是总工具量吧，因为你的"工具"说白了他不就是正运算那部分的函数吗，
             你正运算的函数不是一开始就定了吗，n_tools 看起来像个因变量而不是自变量啊？
             你不可能取60个解不了题的工具出来吧？」

        ⭐ 为什么不能当参数：原来把它做成参数，会在工具池里截断，
           可能把链上必需的工具截掉，造出解不了的题。
        ⭐ 想改工具数 → 改 k 或 n_decoy。
        """
        return self.k + self.n_decoy

    @property
    def delta_opt(self) -> float:
        """选项之间的最小相对差 = 10^(-sep_digits)。"""
        return 10.0 ** (-self.sep_digits)

    @property
    def delta_final(self) -> float:
        """终值之间的最小相对差 = delta_opt × √k（k 段误差累加）。"""
        import math
        return self.delta_opt * math.sqrt(self.k)

    # ══════════════════════════════════════════════════
    # 校验
    # ══════════════════════════════════════════════════

    def validate(self) -> list[str]:
        """返回问题列表（空 = 没问题）。"""
        bad = []

        if self.k < 1:
            bad.append('k 必须 ≥ 1，现在是 %d' % self.k)
        if self.n_decoy < 1:
            bad.append('n_decoy 必须 ≥ 1，现在是 %d' % self.n_decoy)
        if self.n_pick < 0:
            bad.append('n_pick 不能为负')
        if self.n_pick > self.n_decoy:
            bad.append('n_pick(%d) 不能大于 n_decoy(%d) —— 不能抽超过库存'
                       % (self.n_pick, self.n_decoy))
        # ⚠️ n_tool 已改为因变量（= k + n_decoy），恒满足下限与上限，
        #    故此处不再校验 n_tool。保留这两条注释是为了说明「为什么不需要校验」。
        #    原校验：n_tool < n_mid  → 已不可能（n_tool = k + n_decoy ≥ k）
        #    原校验：n_tool > n_decoy + n_mid → 已不可能（恒等）
        if self.sep_digits < 1:
            bad.append('sep_digits 必须 ≥ 1')
        if self.sep_digits > 15:
            bad.append('sep_digits 超过 15 时 float64 表示不了')
        if not (0 < self.node_min < self.node_max):
            bad.append('node_min 必须 < node_max 且都为正')
        if not (0 < self.final_min < self.final_max):
            bad.append('final_min 必须 < final_max 且都为正')
        if self.oversample < 1.0:
            bad.append('oversample 必须 ≥ 1.0')
        if self.max_batch < self.k:
            bad.append('max_batch 太小')

        return bad

    # ══════════════════════════════════════════════════
    # 序列化（写进日志用）
    # ══════════════════════════════════════════════════

    def to_dict(self) -> dict:
        d = asdict(self)
        d['_derived'] = {
            'n_nodes': self.n_nodes,
            'n_mid': self.n_mid,
            'n_option': self.n_option,
            'n_tool': self.n_tool,          # ⚠️ 因变量，写进派生量
            'delta_opt': self.delta_opt,
            'delta_final': self.delta_final,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def brief(self) -> str:
        """一行摘要，写在日志抬头。"""
        return ('k=%d 节点%d 选项%d(%d+%d) 干扰%d 工具%d 区分度%d位 种子%d'
                % (self.k, self.n_nodes, self.n_option, self.n_mid, self.n_pick,
                   self.n_decoy, self.n_tool, self.sep_digits, self.seed))

    @classmethod
    def from_preset(cls, name: str, **override) -> 'ProblemConfig':
        """从难度预设建配置。"""
        if name not in DIFFICULTY_PRESETS:
            raise KeyError('没有这个预设：%s（可选 %s）'
                           % (name, ', '.join(DIFFICULTY_PRESETS)))
        base = dict(DIFFICULTY_PRESETS[name])
        base.update(override)
        return cls(**base)


# ══════════════════════════════════════════════════════════════════
# 值的命名规则 —— 预指派 vs 链节自变量
# ══════════════════════════════════════════════════════════════════

# ⭐⭐⭐ 2026-11-08 Frank 定：这两类值必须分开命名，不能都叫 V###
#
#    「你之所以会搞反，是因为之前的命名没有区分预指派的链条v和
#      两次函数运算造出来的自变量，如果大家都叫v那根本没法区分，
#      我建议把自变量的命名规则改一下……
#      但是必须要在核心配置文件里面，说明预指派的值跟单个链节包中
#      自变量值命名的特征以及如何区分。」
#
# ⚠️ 这个歧义不是抽象的洁癖 —— 它实际导致过多次理解错误：
#    我以为「自变量」指整个链的起点，于是把图片该绑哪个值搞反了。

VALUE_KINDS = {
    # ── 预指派 ───────────────────────────────────────
    'assigned': {
        'prefix': 'A',              # A000（只有一个）
        'pattern': 'A%03d',         # A000
        'example': 'A000',
        'source': '造题时定死的输入',
        # ⚠️ 关键区分特征：它不是任何函数算出来的，也不是随机的
        'comes_from': 'nothing',    # 无上游
        'count': '1 个（只有起点）',
        'in_problem_text': '✅ 直接给数值',
        'how_to_present': '黑白数字图（白底黑字 / 黑底白字）',
        'abstraction_dims': 0,      # 不做抽象
        'role': 'LITERAL',
    },
    # ── 链节自变量 ───────────────────────────────────
    'exogen': {
        'prefix': 'X',              # X001 .. X010
        'pattern': 'X%03d',         # X001
        'example': 'X001',
        # ⚠️ 2026-11-08 重构后含义变了：
        #    老：上一节函数 f 的输出 × c × cn（嵌套复合的产物）
        #    新：由本节 y 用反函数反推出来，是该节的【输入】
        'source': 'yₙ 用反函数两次复合反向求出（y → ÷cn → ÷c → f⁻¹）',
        'comes_from': 'inverse_of_y',
        'count': 'k 个（每节一个，k = 链长）',
        'in_problem_text': '❌ 绝不给数值',
        'how_to_present': '文本抽象 + 图片抽象（两维等权）',
        'abstraction_dims': 2,
        'role': 'CHAIN',
    },
    # ── ⭐ 因变量（2026-11-08 新增）──────────────────
    'dependent': {
        'prefix': 'Y',              # Y001 .. Y010
        'pattern': 'Y%03d',         # Y001
        'example': 'Y001',
        # ⚠️ 弗兰克：「y 才是最开始的数，x 是逆推的」
        'source': '最初的数，造题时直接生成；每节配自己的随机函数链（m 层）',
        'comes_from': 'generated',
        'count': 'k 个真 + n_pick 个假（共 60 个候选）',
        'in_problem_text': '❌ 绝不给数值',
        'how_to_present': '文本抽象 + 图片抽象（两维等权）',
        'abstraction_dims': 2,
        'role': 'CHAIN',
    },
    # ── 链尾值（2026-11-08 新增）─────────────────────
    'chain_tail': {
        'prefix': 'Z',
        'pattern': 'Z',             # 只有一个，不带编号
        'example': 'Z',
        'source': 'A000 × Y001 × … × Y010 的连乘结果',
        'comes_from': 'chain_product',
        'count': '1 个',
        'in_problem_text': '✅ 直接给数值',
        'how_to_present': '黑白数字图',
        'abstraction_dims': 0,
        'role': 'LITERAL',
    },
}

# ⭐ 多模态抽象维数 —— Frank 2026-11-08 定义
#
#    「我们对于不同维度的抽象比如文字和图片，要等比例划分，
#      在核心配置可以登记，应该有一个集合，记录内部有多少元素，
#      我们目前应该是文字和图片2维度吧？我定义这叫多模态抽象维数，
#      设置为2，意味着针对自变量……他既要变化一次文本，
#      也要变化一次图片（告知替换文本表示的具体值）」
#
# ⚠️ 等比例 = 每个维度权重相同 = 1/维数。
#    权重不等就不能叫「维数」，只能叫「加权组合」。
MODALITY_DIMS = ('text', 'image')

MODALITY_WEIGHTS = {
    d: 1.0 / len(MODALITY_DIMS) for d in MODALITY_DIMS
}


def modality_n() -> int:
    """多模态抽象维数。"""
    return len(MODALITY_DIMS)


def check_naming(pid: str) -> str:
    """
    ⭐ 判断一个变量编号属于哪一类：'assigned' / 'exogen' / 'y' / 'chain_tail' / 报错。

    ⚠️ 任何按编号分支的代码都应该调这个，而不是自己判断首字母。
    """
    if not isinstance(pid, str) or len(pid) < 4:
        raise ValueError('变量编号格式不对：%r（应为 A000 / X001 这类）' % (pid,))
    head, digits = pid[0], pid[1:]
    if not digits.isdigit():
        raise ValueError('变量编号后半段应为数字：%r' % (pid,))
    if head == VALUE_KINDS['assigned']['prefix']:
        return 'assigned'
    if head == VALUE_KINDS['exogen']['prefix']:
        return 'exogen'
    if head == VALUE_KINDS['dependent']['prefix']:
        return 'y'
    if head == VALUE_KINDS['chain_tail']['prefix']:
        return 'chain_tail'
    raise ValueError('未知的变量前缀：%r（只认 A / X / Y / Z）' % (pid,))


# ══════════════════════════════════════════════════════════════════
# 十一、⭐ 2026-11-08 重构：链节内嵌套 + 外部连乘
# ══════════════════════════════════════════════════════════════════
#
# 【弗兰克 2026-11-08 定】
#
#   「我们现在算法变了，不是暴力叠加嵌套了，而是链节内嵌套加外部连乘，
#     误差出现的来源不同了。」
#
#   「对每个链节都打包，像之前一样，函数复用之前的打包代码。
#     y 和 x 也按照之前的 uid 打包，
#     这样参与打包的选项变量有 10*2+50*2 总计 120 个变量，
#     分别是 60 个 x 和 y，其他的东西不变。」
#
#   「反正现在就两层，就这么写吧，你到核心配置里面再记一下，
#     我们只是为了验证测试这么写的，**函数理论上可以无限嵌套，
#     只需要改变写法，但那不改变本测试的框架，只会改变代码实现方式**。」
#
# ⚠️ 这一段是「重构后的结构」的唯一权威登记处。
#    别去代码里翻，代码实现可以换，这里的定义不能悄悄改。
# ══════════════════════════════════════════════════════════════════

# ── 链节内外两层结构 ─────────────────────────────────
#
#        链节 L001（内嵌套 m 层）        链节外（连乘，无系数）
#     ┌──────────────────────────┐
#     │  v₀ = X001               │
#     │  v₁ = cn₁·c₁·g₁(v₀)      │
#     │  v₂ = cn₂·c₂·g₂(v₁)      │      A000 × Y001 × … × Y010 = Z
#     │  Y001 = v₂               │
#     └──────────────────────────┘
#
#   ⚠️ 系数只在【内】层，外层的连乘是纯乘法 ——
#      一旦外层加系数，∏Y 的定义被污染，2047 组合的判据失效。

# ⭐ 每节嵌套层数
#
#   ⚠️ 2026-11-08 弗兰克：「反正现在就两层，就这么写吧」
#   ⚠️ 这个 2 是【默认值】，不是理论上限。改成可调：
#      Frank 2026-11-08：「包括之前的单链节嵌套次数，目前用的是双层，
#                          这个值建议可调。」
#      ⇒ 配置项 layers_per_link 见下层 ProblemConfig；这里只是默认值。
LAYERS_PER_LINK = 2

# ⚠️ 理论说明（写进论文时的原话）
LAYERS_NOTE = (
    '函数理论上可以无限嵌套，只需要改变写法（见 LINK_NOTATION），'
    '但那不改变本测试的框架，只会改变代码实现方式。'
)
LAYERS_PER_LINK_MAX_THEORY = None    # None = 无理论上限
LAYERS_PER_LINK_RANGE = (1, 6)       # ⭐ 允许调的范围（超过 6 层括号深度暴增）

# ⭐ 链节每层用哪个函数（2026-11-08 弗兰克：放到配置文件，外面可控制）
#
#   弗兰克：「把函数调用什么，这个看看能不能放到配置文件。
#            然后到时候外面可以控制用哪些函数。」
#
#   ⚠️ 长度必须 == LAYERS_PER_LINK。canvas.Canvas 从这里读，不在代码里写死。
LINK_FUNCS = ['exp', 'pow2']

# ══════════════════════════════════════════════════════════════════
# ⭐ 难度开关（2026-11-09）
# ══════════════════════════════════════════════════════════════════
#
#   弗兰克：「留下各种配置开关，切换难度。我们有好几个维度难度可以切换。
#            保留给其他难度切换配置的框架，后面回来写。
#            先只搞图片的基础进阶切换。」
#
#   ⚠️ 与上面的 DIFFICULTY_PRESETS 不冲突：
#      PRESETS 是【整组预设】（L1..L4），这里是【逐维度开关】。
#      PRESETS 管链长、工具数等；这里只管呈现方式。

DIFFICULTY = {
    # ── 图片维度（已实现三档）────────────────────
    'image': 'advanced',
    #   'basic'    图片上直接写数字（基础难度）
    #   'advanced' 科学计数法 + 逐位画图案（进阶难度）
    #   'none'     不给图，x 的值直接印在题面上

    # ── 语言维度（框架已留，待实现）──────────────
    'lang': 'mixed',
    #   'zh' / 'en' / 'mixed'

    # ── 函数呈现维度（框架已留，待实现）──────────
    'func_render': 'both',
    #   'expression' / 'desc' / 'both'

    # ── 工具档位（已实现两档，2026-11-09）────────
    'tool_level': 'advanced',
    #   'basic'    一个选项 = 一个计算器
    #              输入 x → 输出 f1(f2(x)/c2)/c1，直接得 Y
    #
    #   'advanced' 一个选项 = 两个单层工具（共 120 个），编号打乱
    #              T-xxx 输入 x → 输出 f2(x)      （无系数）
    #              T-yyy 输入 v → 输出 f1(v)      （无系数）
    #              模型要自己：配对 → 串联 → 自己乘除 c1/c2
    #
    #   ⚠️ 进阶档刻意不给系数，测的是「记不记得住、会不会自己缩放」。

    # ── 提示强度维度（框架已留，待实现）──────────
    'hint': 'full',
    #   'none' / 'full'
}

# ⚠️ 已实现的档位（没实现的档位调用会报错，不静默降级）
DIFFICULTY_IMPLEMENTED = {
    'image': ('basic', 'advanced', 'none'),
    'lang': ('mixed',),
    'func_render': ('both',),
    'tool_level': ('basic', 'advanced'),
    'hint': ('full',),
}


def difficulty(key: str) -> str:
    """⭐ 读一个难度档位。未实现的档位抛错，不静默降级。"""
    v = DIFFICULTY.get(key, '')
    ok = DIFFICULTY_IMPLEMENTED.get(key, ())
    if v not in ok:
        raise NotImplementedError(
            '难度档位 %s=%r 尚未实现（已实现：%s）' % (key, v, list(ok)))
    return v


def check_difficulty() -> list:
    """⭐ 校验全部难度开关。返回问题列表（空 = 全通过）。"""
    bad = []
    for k, v in DIFFICULTY.items():
        ok = DIFFICULTY_IMPLEMENTED.get(k, ())
        if v not in ok:
            bad.append('%s=%r 未实现（可用 %s）' % (k, v, list(ok)))
    return bad


# ⭐ 尾数保留几位有效数字（进阶图案难度用）
#
#   实测：x 截断到 N 位有效数字后，反推出的 y 相对误差
#       N=10 → 3.11e-08   N=12 → 1.67e-09
#       N=14 → 4.72e-11   N=16 → 1.15e-10
#   ⚠️ 选项往返容差 1e-9，要求再严一级 → 1e-10。N=14 满足。
SCI_DIGITS = 14

# ⭐ 链节记号的写法
#
#   「一层一命名」：每层一个定义式，括号只剩 1 层。
#   ⚠️ 依据：文献说人不易处理超过约 2 层嵌套。
#      m ≤ 2 时可以整体写成一行；m ≥ 3 必须拆成一排定义式。
LINK_NOTATION = {
    'inline_max': LAYERS_PER_LINK,   # ≤ 这个层数就整体写一行
    'style': '一层一命名',
    'bracket_levels': ('(', '[', '{'),
    'why': '人不易处理超过约 2 层嵌套；不同括号可表示不同层级',
}

# ── 链条：A000 是链头，中间连乘 Y，结尾 Z ───────────────
#
#    A000  链头      题面给出（已知条件）
#    Y###  链身      k 个，每节一个，待求
#    Z     链尾值    题面给出（已知条件）
#
#    A000 × Y001 × Y002 × … × Y010 = Z
CHAIN_HEAD = 'A000'
CHAIN_MID = 'Y%03d'          # Y001 .. Y010
CHAIN_TAIL = 'Z'
CHAIN_TAIL_PID = 'Z'         # ⚠️ 链尾只有一个，不带编号

# ── 60 个 y 的构造 ───────────────────────────────────
#
#   「构造了 60 个 y，其中 10 个真 y，50 个假 y，
#     但每个 y 都有对应的随机函数和 x 值」
#
#   ⚠️ 不区分真假，60 个走同一条构造流水线 ——
#      每个 y 都配自己的随机函数（m 层）+ 自己的 x + 自己的 c/cn
N_TRUE_Y = 10                 # = k
N_FAKE_Y = 50                 # = n_decoy
N_OPTIONS_WITH_LINK = 120     # 60 个 y + 60 个 x

# ── 干扰项必须避开的集合（弗兰克：「所有跟 z 相关的值都排除了」）──
#
#   ⚠️ 不是 21 个。乘法没有顺序，任意几个数相乘都是一个合法组合：
#      11 个元素 → 2^11 − 1 = 2047 个乘积
#
#   弗兰克：「乘法是没有顺序的，他们之间的所有组合都要排除，
#             这也是为什么我把计算放到 gpu，因为 cpu 做不完。」
DECOY_AVOID_COMBINATIONS = True   # 排除全部 2047 个组合乘积，不是只排 11 个节点
DECOY_AVOID_COMBOS_COUNT = 2 ** 11 - 1   # 2047


def n_link_values(k: int = None) -> int:
    """一个链节产出的变量数：k 个 Y + k 个 X。"""
    k = N_TRUE_Y if k is None else k
    return 2 * k


def n_packed_vars(n_pick: int = None) -> int:
    """
    参与打包的选项变量总数。

    ⭐ 弗兰克：「参与打包的选项变量有 10*2+50*2 总计 120 个变量」
    """
    pick = N_FAKE_Y if n_pick is None else n_pick
    return 2 * (N_TRUE_Y + pick)


# ══════════════════════════════════════════════════════════════════
# ⭐ 功能池：哪些函数可以入链节
# ══════════════════════════════════════════════════════════════════
#
# 弗兰克 2026-11-08：
#   「然后把使用的函数也做进配置文件，这样以后可以让测试者自己配置。」
#
# ⚠️ 注意与 `inverse_trusted` 的关系：
#    `inverse_trusted=False` 的 15 个包（sin/cos/airy/yv/entr/…）
#    【无论怎么配置都不能入链节】—— 因为它们的反函数不唯一，
#    链节的一一对应会崩。见 linkpack.chain_pool()。
#
#    ⇒ 本配置只能在【那 59 个可信包之内】再收窄，不能越过它放宽。

# 默认：空 = 用全部可信包（59 个）
FUNC_POOL_MODE = 'inverse_trusted'   # 'inverse_trusted' | 'explicit' | 'family'
FUNC_POOL_EXPLICIT: tuple = ()       # mode='explicit' 时用这个白名单
FUNC_POOL_EXCLUDE: tuple = ()        # 无论哪种模式，这些都不要


def func_pool(mode: str = None, explicit=None, exclude=None) -> list[str]:
    """
    ⭐ 可入链节的函数池（配置化）。

    参数
    ----
    mode     : 'inverse_trusted'（默认，全部可信包）
               'explicit'（只用 explicit 里列的）
               'family'（暂未实现，保留）
    explicit : mode='explicit' 时的白名单
    exclude  : 无论哪种模式都排除的名字

    ⚠️ 无论如何，`inverse_trusted=False` 的包一定被挡掉 —— 这是硬约束。
    """
    import packs as _packs

    mode = FUNC_POOL_MODE if mode is None else mode
    explicit = FUNC_POOL_EXPLICIT if explicit is None else explicit
    exclude = FUNC_POOL_EXCLUDE if exclude is None else exclude

    trusted = [p.name for p in _packs.PACKS if p.inverse_trusted]

    if mode == 'inverse_trusted':
        pool = list(trusted)
    elif mode == 'explicit':
        want = set(explicit)
        pool = [n for n in trusted if n in want]      # ⚠️ 先过可信，再取白名单
    elif mode == 'family':
        pool = list(trusted)                          # 占位，暂等同默认
    else:
        raise ValueError('未知的 func_pool mode：%r' % (mode,))

    ex = set(exclude)
    return [n for n in pool if n not in ex]


# ══════════════════════════════════════════════════════════════════
# ⭐ 链条组装的可调参数（弗兰克 2026-11-08）
# ══════════════════════════════════════════════════════════════════
#
# 「现在的10个选项和50个干扰项建议做成核心配置文件可调的模式」
# 「然后把使用的函数也做进配置文件，这样以后可以让测试者自己配置。」

# ── 选项与干扰项 ────────────────────────────────────
#
#   10 个真 y  = 链条上 Y001..Y010
#   50 个假 y  = 散落的链节，避开 2047 个组合
#   ⇒ 60 个 y 候选，配上各自的 x，共 120 个变量
CHAIN_N_TRUE = 10
CHAIN_N_FAKE = 50

# 干扰项必须避开的组合乘积（2^11 − 1 = 2047）
CHAIN_AVOID_COMBOS = True

# ⭐ GPU 批大小（2026-11-08 实测定的）
#
#   弗兰克：「先改并发量……你根据该进程占用决定。」
#
#   ⚠️⚠️ 2026-11-08 二次调整：cn 候选表加了 2^m 细档（122 → 434 个），
#      张量尺寸涨 3.6 倍 ⇒ 原来 2048 的峰值从 5.96 GB 涨到 21.12 GB，
#      超 16GB 显存。
#
#   实测对照（RTX 5060 Ti 16GB）：
#      cn 候选 122 个时：M=2048 → 5.96 GB
#      cn 候选 434 个时：M=2048 → 21.12 GB  ❌ OOM
#      ⇒ 同比例缩：2048 × 122/434 ≈ 576，取 512 留余量
#
#   ⚠️ 512 个 y ≈ 8 道题（每题 60 个 y）。超过就切块。
GPU_BATCH_Y = 1024

# ── 真 y 的取值区间（连乘要落在合理量级）─────────────
#
# ⚠️ 2026-11-08 弗兰克：「把生成的数量级也定位在核心配置文件里，
#                        以后能找到在哪调就行。我不认为现在要改这个值。」
#
# ⭐ 所以【现在就定在这里，不改】。要调的时候改下面两行。
#
# ⚠️ 为什么跨度是 4 个数量级（实测）：
#   1e-2 .. 1e2 跨度 = 4 级。
#   实测真 y 跨度 3.56 级、假 y 跨度 3.86 级 —— 不是生成出错，
#   是区间本身就这么宽。要收窄就改 CHAIN_Y_DECADES。
CHAIN_Y_LO = 1e-2
CHAIN_Y_HI = 1e2

# ⭐ 数量级跨度（十年数）—— 直接可调，不用算 log
#
#   ⚠️ CHAIN_Y_LO / HI 与 CHAIN_Y_DECADES 必须一致：
#      DECADES = log10(HI) - log10(LO)
#      ⇒ 改区间就同步改这个数，或者只改这个数让它算区间（见 y_range_of）
CHAIN_Y_DECADES = 4

# ── 链头 / 链尾的取值区间 ────────────────────────────
CHAIN_HEAD_LO = 1.0
CHAIN_HEAD_HI = 100.0
CHAIN_HEAD_DECADES = 2

# ── 连乘终值的可接受量级（太小/太大都会溢出或退化）──
CHAIN_Z_MIN = 1e-30
CHAIN_Z_MAX = 1e30


def y_range_of(lo: float = None, decades: float = None) -> tuple:
    """
    ⭐ 按「下界 + 数量级跨度」算出取值区间。

    ⚠️ 单一真源：区间由 CHAIN_Y_LO 与 CHAIN_Y_DECADES 算出，
       不要在别处写死 1e2。
    """
    lo = CHAIN_Y_LO if lo is None else lo
    d = CHAIN_Y_DECADES if decades is None else decades
    return (lo, lo * (10.0 ** d))


def n_y_candidates(n_true: int = None, n_fake: int = None) -> int:
    """y 候选总数 = 真 + 假。"""
    t = CHAIN_N_TRUE if n_true is None else n_true
    f = CHAIN_N_FAKE if n_fake is None else n_fake
    return t + f


def link_pid(i: int) -> str:
    """链节编号：i=1 → 'L001'。⚠️ 从 1 开始。"""
    return 'L%03d' % i


def y_pid(i: int) -> str:
    """因变量编号：i=1 → 'Y001'。"""
    return 'Y%03d' % i


def x_pid(i: int) -> str:
    """自变量编号：i=1 → 'X001'。"""
    return 'X%03d' % i


# ══════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0
    checks = []

    c = ProblemConfig()
    checks.append(('默认配置无问题', c.validate() == []))
    checks.append(('默认：11 个节点', c.n_nodes == 11))
    checks.append(('默认：10 个中间值', c.n_mid == 10))
    checks.append(('默认：50 个选项', c.n_option == 50))
    checks.append(('默认：区分度 1e-06', abs(c.delta_opt - 1e-6) < 1e-18))

    # 派生量随配置变
    c2 = ProblemConfig(k=6, n_decoy=30, n_pick=20)
    checks.append(('改 k → 节点数跟着变', c2.n_nodes == 7))
    checks.append(('改 n_pick → 选项数跟着变', c2.n_option == 26))
    checks.append(('n_tool 是因变量 = k+n_decoy', c2.n_tool == 36))

    # 校验能抓到错
    checks.append(('n_pick > n_decoy 被抓到',
                   any('不能大于' in x for x in ProblemConfig(n_decoy=10, n_pick=20).validate())))
    checks.append(('sep_digits 过大被抓到',
                   any('float64' in x for x in ProblemConfig(sep_digits=20).validate())))

    # 预设
    ok_preset = True
    for name in DIFFICULTY_PRESETS:
        pc = ProblemConfig.from_preset(name)
        if pc.validate():
            ok_preset = False
    checks.append(('五个预设都合法', ok_preset))

    checks.append(('可序列化成 JSON', len(c.to_json()) > 100))

    # ── 值的命名规则（2026-11-08 新增）────────────────────
    checks.append(('多模态抽象维数 = 2', modality_n() == 2))
    checks.append(('维度是文字与图片',
                   set(MODALITY_DIMS) == {'text', 'image'}))
    checks.append(('两维等权（各 1/2）',
                   all(abs(w - 0.5) < 1e-12 for w in MODALITY_WEIGHTS.values())))
    checks.append(('权重之和为 1',
                   abs(sum(MODALITY_WEIGHTS.values()) - 1.0) < 1e-12))

    # 两类值的区分
    checks.append(('A000 判为预指派', check_naming('A000') == 'assigned'))
    checks.append(('X001 判为链节自变量', check_naming('X001') == 'exogen'))
    checks.append(('X010 判为链节自变量', check_naming('X010') == 'exogen'))
    # ⚠️ 2026-11-08 改：Z001 现在合法了（Z 是链尾前缀）。
    #    原来的用例想表达的是「带编号的链尾不合法」——
    #    改测 Z 本身没有编号，Z001 这种带编号的形式属于编号但它不算「链尾唯一」。
    #    ⭐ 保留这个区分：Z 是唯一的一个，不带编号；CHAIN_TAIL_PID = 'Z'。
    for bad in ('V000', 'A00', 'X00a', '', 'Q001', 'W001'):
        try:
            check_naming(bad)
            checks.append(('非法编号 %r 被拒' % bad, False))
        except (ValueError, TypeError):
            checks.append(('非法编号 %r 被拒' % bad, True))

    # ⭐ 预指派不抽象、链节自变量抽象两次
    checks.append(('预指派不做抽象',
                   VALUE_KINDS['assigned']['abstraction_dims'] == 0))
    checks.append(('链节自变量抽象两次',
                   VALUE_KINDS['exogen']['abstraction_dims'] == 2))
    checks.append(('链节自变量的维数 == 多模态维数',
                   VALUE_KINDS['exogen']['abstraction_dims'] == modality_n()))
    checks.append(('预指派直接给数值',
                   VALUE_KINDS['assigned']['in_problem_text'].startswith('✅')))
    checks.append(('链节自变量不给数值',
                   VALUE_KINDS['exogen']['in_problem_text'].startswith('❌')))
    checks.append(('两类前缀不同',
                   VALUE_KINDS['assigned']['prefix']
                   != VALUE_KINDS['exogen']['prefix']))

    # ── 2026-11-08 重构：链节内嵌套 + 外部连乘 ────────────
    checks.append(('Y 判为因变量', check_naming('Y001') == 'y'))
    checks.append(('Z 判为链尾', check_naming('Z999') == 'chain_tail'))
    checks.append(('因变量抽象两次',
                   VALUE_KINDS['dependent']['abstraction_dims'] == modality_n()))
    checks.append(('因变量不给数值',
                   VALUE_KINDS['dependent']['in_problem_text'].startswith('❌')))
    checks.append(('链尾直接给数值',
                   VALUE_KINDS['chain_tail']['in_problem_text'].startswith('✅')))

    # 四个前缀互不相同
    heads = [VALUE_KINDS[k]['prefix'] for k in VALUE_KINDS]
    checks.append(('四个前缀互不相同（%s）' % ''.join(heads),
                   len(set(heads)) == len(heads)))

    # 链节层数
    checks.append(('当前每节 %d 层' % LAYERS_PER_LINK, LAYERS_PER_LINK == 2))
    checks.append(('层数无理论上限（记录在案）', LAYERS_PER_LINK_MAX_THEORY is None))
    checks.append(('写法说明非空', len(LAYERS_NOTE) > 20))
    checks.append(('写法：≤%d 层可写一行' % LINK_NOTATION['inline_max'],
                   LINK_NOTATION['inline_max'] == LAYERS_PER_LINK))

    # 变量数
    checks.append(('每节出 20 个变量（10 Y + 10 X）', n_link_values() == 20))
    checks.append(('参与打包 120 个（60 y + 60 x）', n_packed_vars() == 120))
    checks.append(('n_packed_vars = 10*2 + 50*2', n_packed_vars() == 10 * 2 + 50 * 2))

    # 编号工厂
    checks.append(('链节编号 L001', link_pid(1) == 'L001'))
    checks.append(('因变量编号 Y010', y_pid(10) == 'Y010'))
    checks.append(('自变量编号 X001', x_pid(1) == 'X001'))
    checks.append(('同 idx 后缀一致',
                   link_pid(7)[1:] == y_pid(7)[1:] == x_pid(7)[1:] == '007'))

    # 干扰项排除组合
    checks.append(('排除全部组合（不是只排节点）', DECOY_AVOID_COMBINATIONS is True))
    checks.append(('组合数 = 2^11-1 = 2047', DECOY_AVOID_COMBOS_COUNT == 2047))

    if verbose:
        print('=' * 62)
        print('config 自检')
        print('=' * 62)
        for name, ok in checks:
            print('  %s  %s' % ('✅' if ok else '❌', name))
            if not ok:
                fails += 1
        print()
        print('默认配置摘要：', c.brief())
        print()
        print('难度预设：')
        for name, pr in DIFFICULTY_PRESETS.items():
            pc = ProblemConfig.from_preset(name)
            print('  %-4s k=%-3d 干扰%-3d 入选项%-3d 工具%-3d 区分度%d位'
                  % (name, pc.k, pc.n_decoy, pc.n_pick,
                     pc.n_tool, pc.sep_digits))
        print()

    return fails


if __name__ == '__main__':
    raise SystemExit(self_test())
