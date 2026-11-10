"""
工具控制台 —— 60 个计算器，可进入单个、可调用、可记录、可抽象

⭐ 2026-11-07 Frank 定：
    「工具就是工具，没什么干扰工具，那是干扰项，工具就是计算器。
      是一个控制台，参照你的学术手册浏览器用的工具，但不要复用，
      只看逻辑和代码，意图是提供交互给 LLM 并记录 LLM 的所有行为。」

    「建议是 60 个项，每个项都是 x 和对应的嵌套函数计算，
      那么这个工具就是嵌套函数本身，可以输入 x 值得到 y 值。」

    「先给工具也做成模块化的，因为工具首先要集成列表在控制台，
      可以进入单个工具，然后要有输出输入值，
      每个工具的界面要有首页键回到首页。
      然后工具里面的显示内容可以替换，这样工具也可以被抽象，
      比如用来计算粒子的无规则运动这样的抽象描述等等。」

⭐ 设计要点（参照 workbench.mjs 的逻辑，不复用其代码）：
    ① 所有调用走 **一个总入口** → 天然能记录全部行为
    ② 有「列表页」与「单个工具页」两级界面
    ③ 每个工具页有「回到首页」的操作
    ④ 界面显示内容（文字）与数学内核**分离** —— 后者才可抽象替换
    ⑤ 60 个工具全部是真计算器，没有真假之分

⚠️ 本模块不生成工具（那是 vectorform 的事），
   只负责「把 60 个工具组织成可交互的控制台」。
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

sys.path.insert(0, r'<BENCH>\generator')

import torch
import packs
from mathpack import DT, DEV
from skeleton import get_skeleton


# ══════════════════════════════════════════════════════════════════
# 一、抽象层：显示内容可替换
# ══════════════════════════════════════════════════════════════════

@dataclass
class ToolSkin:
    """
    一个工具的「皮」——显示给模型看的所有文字。

    ⚠️ 与数学内核（.pack）完全分离。
       换皮不换核 = 同一道题换个说法，难度可调。
    """
    title: str = ''          # 工具标题，如「四次方计算器」
    desc: str = ''           # 能力描述，如「我处理粒子的无规则运动」
    input_label: str = 'x'   # 输入框标签
    output_label: str = 'y'  # 输出标签
    home_label: str = '返回首页'

    def is_abstracted(self) -> bool:
        """是否已被抽象（不再是标准名）。"""
        std = {'title', 'desc'}
        return bool(self.title) and bool(self.desc)


def std_skin(pack) -> ToolSkin:
    """标准皮：用 skeleton 里的标准中文名。"""
    sk = get_skeleton(pack.name)
    zh = sk.zh_name if sk else pack.name
    return ToolSkin(
        title='%s 计算器' % zh,
        desc='输入一个数，计算它的%s。' % zh,
    )


def abstract_skin(pack, in_name: str = '输入量', out_name: str = '输出量',
                  tmpl_idx: int = 0, seed: int = 0,
                  lang: str = 'zh') -> ToolSkin:
    """
    ⭐ 抽象皮：从中间层（skins 的描述表）派生，套用随机语法模板。

    ⚠️ 2026-11-07 Frank：
        「函数内部显示的内容可以替换，这样工具也可以被抽象。」
        「由中间层随机抽象化推到控制台界面。」

    ⚠️ 与 std_skin 的区别：
        std_skin    → title 是「平方 计算器」，desc 是「计算它的平方」
                       ⭐ 直接暴露数学名，抽象层失效
        abstract_skin → title 是中性编号，desc 是精确的专家描述
                       ⭐ 说清「做什么」，但不说「叫什么函数」

    ⚠️ title 用中性名 —— 不能是函数中文名，
       否则模型看标题就知道是什么函数，描述写得再好也没用。

    ⚠️ 但也不能 60 个全叫同一个名字（那样太假，一眼看出是程序生成）。
       → 用一组中性名，按工具编号轮换。

    ⚠️ 2026-11-07 加 lang 参数：支持中 / English 两套描述与模板。
       Frank：「关于函数的叙述可以加入英文版本，
               这样随机的时候函数和选项都能完全随机。」
    """
    import skins as _skins
    text = _skins.render_tool_text_auto(
        pack.name, in_name, out_name, tmpl_idx=tmpl_idx, lang=lang)
    # ⭐ 标题中性化：一组中性名轮换，不含任何函数信息
    if lang == 'en':
        title = _NEUTRAL_TITLES_EN[(seed * 7 + tmpl_idx) % len(_NEUTRAL_TITLES_EN)]
    else:
        title = _NEUTRAL_TITLES[(seed * 7 + tmpl_idx) % len(_NEUTRAL_TITLES)]
    return ToolSkin(
        title=title,
        desc=text,
        input_label=in_name,
        output_label=out_name,
    )


# ⚠️ 中性标题池：只表达「这是一个计算工具」，不含任何函数线索
_NEUTRAL_TITLES = [
    '数值变换工具',
    '单位换算器',
    '量值处理器',
    '数值转换台',
    '参数演算器',
    '数值加工台',
    '量值换算器',
    '数据变换器',
    '数值推演工具',
    '量值处理台',
]

_NEUTRAL_TITLES_EN = [
    'value transformer',
    'unit converter',
    'quantity processor',
    'value convertor',
    'parameter evaluator',
    'value workbench',
    'quantity converter',
    'data transformer',
    'value derivation tool',
    'quantity workbench',
]


# ══════════════════════════════════════════════════════════════════
# 二、单个工具
# ══════════════════════════════════════════════════════════════════

@dataclass
class Tool:
    """
    一个工具 = 一个计算器。

    tid     : 在控制台里的编号（T01..T60）
    pack    : 数学内核（真的能算）
    skin    : 显示内容（可替换）
    role    : 在本题里的角色（chain = 链上 / off = 不在链上）
                ⚠️ 角色**不给模型看**，只用于我们自己校验
    variant : 同一函数被复用时，用来区分「第几次出现」的标记
    """
    tid: str
    pack: object
    skin: ToolSkin
    role: str                # 'chain' / 'off'
    step: int = 0            # 链上第几步（0 = 不在链上）
    variant: int = 1         # 同函数第几次出现（1 = 首次；>1 = 复用）

    def call(self, x: float) -> dict:
        """
        调用这个计算器：输入 x → 输出 y。

        ⚠️ 走的是数理包自己的正运算（含包系数 c），与造题时完全一致。
        ⚠️ 返回值里**不含** role / step —— 不给模型任何提示。
        """
        try:
            t = torch.tensor([float(x)], device=DEV, dtype=DT)
            f = self.pack._fwd(t) * self.pack.c
            v = f[0]
            val = float(v)
            if not math.isfinite(val):
                return {'ok': False, 'error': '定义域外', 'x': x, 'y': None}
            return {'ok': True, 'x': float(x), 'y': val}
        except Exception as e:
            return {'ok': False, 'error': '计算失败', 'detail': str(e), 'x': x, 'y': None}

    def card(self) -> str:
        """列表页上的这一行。⚠️ 不显示 role。"""
        return '[%s] %s' % (self.tid, self.skin.title)

    def page(self) -> str:
        """单个工具页的渲染。"""
        L = ['┌' + '─' * 56 + '┐']
        L.append('│ %-54s │' % ('[%s]  %s   (%s)' % (self.tid, self.skin.title, self.skin.home_label)))
        L.append('├' + '─' * 56 + '┤')
        # desc 可能较长，折行
        for line in _wrap(self.skin.desc, 50):
            L.append('│ %-54s │' % line)
        L.append('├' + '─' * 56 + '┤')
        L.append('│  输入  %-46s │' % ('%s = ?' % self.skin.input_label))
        L.append('│  输出  %-46s │' % ('%s = ?' % self.skin.output_label))
        L.append('└' + '─' * 56 + '┘')
        return '\n'.join(L)


# ══════════════════════════════════════════════════════════════════
# 二·五、新题型的工具：一个选项 = 一个计算器
# ══════════════════════════════════════════════════════════════════
#
#   ⚠️ 弗兰克 2026-11-09：
#   「60 个选项配的 60 个工具，也就是说每个 y 都可以求出来，
#    只需要把工具用好。输入值可以换，输出会变成新的值，
#    控制台必须记录每一步操作交互，并记录每次输入和输出的值。」
#
#   ⇒ 一个工具 = 一个选项的式子
#         y = f1( f2(x) ÷ c2 ) ÷ c1
#     模型输入一个值 → 得到 y。
#     ⚠️ 与出题时的算法【完全一致】（出题是反抗 x，这里是正算 y）。

class OptionTool(Tool):
    """
    ⭐ 从出卷器的一个选项建的计算器。

    ⚠️ 继承旧 Tool，只换 call() 的算法与构造方式。
    ⚠️ 返回值同样不含 role / step。
    """

    __slots__ = ('f1', 'f2', 'c1', 'c2', '_p1', '_p2', 'pid', 'expr')

    def call(self, x: float) -> dict:
        """
        ⭐ 输入 x → 输出 y。与出题时的正向算法一致。

            v = f2(x) ÷ c2
            y = f1(v) ÷ c1
        """
        try:
            t = torch.tensor([float(x)], device=DEV, dtype=DT)
            v = self._p2._fwd(t) / self.c2
            y = self._p1._fwd(v) / self.c1
            val = float(y[0])
            if not math.isfinite(val):
                return {'ok': False, 'error': '定义域外', 'x': x, 'y': None}
            return {'ok': True, 'x': float(x), 'y': val}
        except Exception as e:                          # noqa: BLE001
            return {'ok': False, 'error': '计算失败',
                    'detail': str(e), 'x': x, 'y': None}


def tool_from_option(op, mo, title: str = '', desc: str = '') -> OptionTool:
    """
    ⭐ 从出卷器的一个选项造一个计算器。

    op : optbase.Option
    mo : midlayer.MidOption（取变量名与题面式子）
    """
    import packs as _packs
    p1 = _packs.get(op.f1)
    p2 = _packs.get(op.f2)
    skin = ToolSkin(
        title=title or ('计算器 %s' % op.pid),
        desc=desc or ('输入一个数，算出 %s 的值' % op.pid),
        input_label=mo.x_word if mo is not None else 'x',
        output_label=op.codes['Y'] if hasattr(op, 'codes') else 'y',
        home_label='首页',
    )
    t = OptionTool(
        tid=op.pid, pack=p1, skin=skin, role='off',
        step=0, variant=1,
    )
    t.f1, t.f2 = op.f1, op.f2
    t.c1, t.c2 = op.c1, op.c2
    t._p1, t._p2 = p1, p2
    t.pid = op.pid
    t.expr = (mo.text.split('\n')[-1].strip() if mo is not None else '')
    return t


def tools_from_exam(pp) -> list:
    """
    ⭐ 按卷子的工具档位建计算器。

    ⚠️ 弗兰克 2026-11-09：
    「激进点，一起改，在图那里也设置工具难度，基础和进阶，
      新写一套进阶难度的，也跟基础一样，从前面获取数据，
      只是没有系数，然后每步拆开，打乱排序，
      让 llm 自己输入输出数值算。」

    basic    → 60 个计算器（两层 + 系数，直接出 y）
    advanced → 120 个单层工具（无系数，编号打乱）

    ⚠️ 两条路都不生成新东西，参数全部来自卷子。
    """
    level = getattr(pp, 'tool_level', None)
    # ⚠️ 卷子文件里可能带 tool_level；load_paper 会把它挂到 pp 上
    if level is None:
        level = 'basic'

    if level == 'advanced':
        import toolsplus
        return toolsplus.build_split_tools(pp, seed=getattr(pp, 'seed', 0))

    return _tools_from_exam_basic(pp)


def _tools_from_exam_basic(pp) -> list:
    """
    ⭐ 从卷子里的 60 组参数，建 60 个计算器。

    ⚠️ 这是「答卷器只提供工具」的实现。
       它复用生成器的函数表（packs）和两层复合算法（OptionTool.call），
       但参数全部来自卷子 —— 不生成任何新东西。

    参数 pp : examfile.load_paper() 返回的对象
    """
    import packs as _packs
    out = []
    for i, op in enumerate(pp.ops):
        mo = pp.mos[i] if i < len(pp.mos) else None
        p1 = _packs.get(op.f1)
        p2 = _packs.get(op.f2)
        skin = ToolSkin(
            title='计算器 %s' % op.pid,
            desc='输入一个数，算出 %s 的值' % op.pid,
            input_label=(mo.x_word if mo is not None else 'x'),
            output_label='Y-%s' % op.pid,
            home_label='首页',
        )
        t = OptionTool(tid=op.pid, pack=p1, skin=skin,
                       role='off', step=0, variant=1)
        t.f1, t.f2 = op.f1, op.f2
        t.c1, t.c2 = op.c1, op.c2
        t._p1, t._p2 = p1, p2
        t.pid = op.pid
        t.expr = (mo.text.split('\n')[-1].strip()
                  if mo is not None else '')
        out.append(t)
    return out


def _wrap(s: str, n: int) -> list[str]:
    """按长度折行（中文按 2 算）。"""
    out = []
    cur = ''
    width = 0
    for ch in s:
        w = 2 if ord(ch) > 127 else 1
        if width + w > n:
            out.append(cur)
            cur, width = ch, w
        else:
            cur += ch
            width += w
    if cur:
        out.append(cur)
    return out or ['']


# ══════════════════════════════════════════════════════════════════
# 三、调用记录
# ══════════════════════════════════════════════════════════════════

@dataclass
class CallRecord:
    seq: int             # 第几次调用
    t: float             # 相对开局的秒数
    action: str          # 'list' / 'open' / 'call' / 'home'
    tid: str = ''
    x: Optional[float] = None
    y: Optional[float] = None
    # ⚠️ 2026-11-10：进阶档的系数由模型自己给（做乘法）。
    #    调用记录里必须留下它 —— 判「模型会不会自己配系数」
    #    全靠这个字段，不然只看到 x 和 y。
    factor: Optional[float] = None
    ok: bool = True
    note: str = ''


# ══════════════════════════════════════════════════════════════════
# 四、控制台
# ══════════════════════════════════════════════════════════════════

class ToolConsole:
    """
    工具控制台 —— 所有交互的唯一入口。

    ⭐ 设计参照 workbench.mjs 的 act() 思路：
       所有动作都经过 _log()，因此**没有任何交互能绕过记录**。

    ⚠️ 有「列表页」和「单个工具页」两级，模拟真实工具面板。
    """

    def __init__(self, tools: list[Tool], home_label: str = '首页'):
        self.tools = {t.tid: t for t in tools}
        self.order = [t.tid for t in tools]
        self.current: Optional[str] = None      # 当前打开的工具（None = 列表页）
        self.home_label = home_label
        self.records: list[CallRecord] = []
        self.t0 = time.time()
        self.n_call = 0                          # 真实调用次数（不含导航）
        self.n_nav = 0                           # 导航次数

    # ── 记录（唯一入口）──────────────────────────────────
    def _log(self, action: str, **kw) -> CallRecord:
        r = CallRecord(
            seq=len(self.records) + 1,
            t=round(time.time() - self.t0, 4),
            action=action, **kw)
        self.records.append(r)
        return r

    # ── 列表页 ──────────────────────────────────────────
    def home(self) -> str:
        """回到首页（工具列表）。"""
        self.current = None
        self.n_nav += 1
        self._log('home')
        return self.render_home()

    def render_home(self) -> str:
        """渲染列表页。⚠️ 只显示编号与标题，不显示任何角色信息。"""
        L = ['┌' + '─' * 58 + '┐']
        L.append('│ %-56s │' % ('工具控制台   （共 %d 个工具）' % len(self.tools)))
        L.append('├' + '─' * 58 + '┤')
        L.append('│ %-56s │' % ('用法：open <编号> 进入  /  call <编号> <输入> 直接计算'))
        L.append('└' + '─' * 58 + '┘')
        L.append('')
        # 三列排
        cards = [self.tools[t].card() for t in self.order]
        for i in range(0, len(cards), 3):
            row = cards[i:i + 3]
            L.append('  ' + ''.join('%-24s' % c for c in row).rstrip())
        return '\n'.join(L)

    # ── 进入单个工具 ────────────────────────────────────
    def open(self, tid: str) -> str:
        """进入某个工具页。"""
        t = self.tools.get(tid)
        if t is None:
            self._log('open', tid=tid, ok=False, note='不存在的编号')
            return '⚠️ 没有编号为 %s 的工具。输入 home 回到列表。' % tid
        self.current = tid
        self.n_nav += 1
        self._log('open', tid=tid)
        return t.page()

    # ── 调用 ────────────────────────────────────────────
    def call(self, tid: str, x: float) -> str:
        """
        ⭐ 唯一的计算入口。任何计算都必须经这里 → 不可能漏记。

        ⚠️ 返回值只含 x 和 y，**不含 role / step**。
        """
        t = self.tools.get(tid)
        if t is None:
            self._log('call', tid=tid, ok=False, note='不存在的编号')
            return '⚠️ 没有编号为 %s 的工具。' % tid

        try:
            xf = float(x)
        except (TypeError, ValueError):
            self._log('call', tid=tid, ok=False, note='输入不是数字')
            return '⚠️ 输入必须是数字。'

        self.n_call += 1
        res = t.call(xf)

        if res['ok']:
            self._log('call', tid=tid, x=xf, y=res['y'])
            return ('[%s] %s\n  输入 %s = %s\n  输出 %s = %s'
                    % (tid, t.skin.title,
                       t.skin.input_label, _fmt(xf),
                       t.skin.output_label, _fmt(res['y'])))
        self._log('call', tid=tid, x=xf, ok=False, note=res.get('error', '失败'))
        return ('[%s] %s\n  输入 %s = %s\n  ⚠️ %s'
                % (tid, t.skin.title, t.skin.input_label, _fmt(xf),
                   res.get('error', '计算失败')))

    # ── 换皮（抽象层）───────────────────────────────────
    def reskin(self, tid: str, skin: ToolSkin) -> bool:
        """给某个工具换显示内容。⚠️ 只换文字，数学内核不动。"""
        t = self.tools.get(tid)
        if t is None:
            return False
        t.skin = skin
        self._log('reskin', tid=tid)
        return True

    # ── 记录导出 ────────────────────────────────────────
    def dump(self) -> str:
        """把全部交互记录导出成文本（给日志用）。"""
        L = ['═' * 66]
        L.append('工具控制台 · 交互记录')
        L.append('共 %d 条（真实计算 %d 次，导航 %d 次）'
                 % (len(self.records), self.n_call, self.n_nav))
        L.append('═' * 66)
        for r in self.records:
            if r.action == 'call':
                L.append('%4d %8.3fs  call  %s  x=%-14s y=%-14s %s'
                         % (r.seq, r.t, r.tid,
                            _fmt(r.x) if r.x is not None else '-',
                            _fmt(r.y) if r.y is not None else '-',
                            r.note))
            else:
                L.append('%4d %8.3fs  %-5s %s %s'
                         % (r.seq, r.t, r.action, r.tid, r.note))
        return '\n'.join(L)

    def digest(self) -> str:
        """交互记录的摘要（用于哈希链/防篡改）。"""
        s = '|'.join('%d:%s:%s:%s' % (r.seq, r.action, r.tid,
                                       '' if r.y is None else '%.17g' % r.y)
                     for r in self.records)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════
# 四·五、答卷器：控制台的最外层
# ══════════════════════════════════════════════════════════════════
#
#   ⚠️ 弗兰克 2026-11-09：
#   「试卷和选项在控制台应该是收敛的，因为已经注入上下文了，
#    展开是为了让 llm 能复查。首页应该是试卷的展开入口和
#    60 个工具的展开入口。点进去可以看试卷内容，
#    最上面有返回交互，随时可以返回。
#    每个页面除了首页以外都可以返回。」
#
#   「答案应该是把十个正确选项的编号，输入到出卷器的答卷器部分，
#    在那里可以填十个选项，然后点击交卷」
#
#   ⇒ 页面结构：
#
#        首页 home
#         ├── paper    试卷（展开复查）
#         ├── tools    60 个工具列表
#         │    └── tool:<pid>   单个工具页
#         └── answer   答卷（填 10 个编号）
#
#   ⚠️ 除首页外每页都有「返回」。


class ExamConsole:
    """
    ⭐ 答卷器：控制台最外层。

    所有交互都经过 _log()，没有任何操作能绕过记录。
    """

    N_ANSWER = 10

    def __init__(self, paper, tools: list, title: str = '基准测试'):
        self.paper = paper
        self.tools = {t.tid: t for t in tools}
        self.order = [t.tid for t in tools]
        self.title = title
        self.page = 'home'
        self.cur_tool: Optional[str] = None
        self.slots: list[str] = []          # 答卷的 10 个格子
        self.records: list = []
        self.t0 = time.time()
        self.n_call = 0
        self.n_nav = 0
        self.submitted = False
        self.submit_result: Optional[dict] = None
        # ⚠️ 问卷编号 —— 门控按它做标记，记录里必须带
        self.exam_id = getattr(paper, 'exam_id', '')

    # ── 记录 ────────────────────────────────────
    def _log(self, action: str, **kw):
        r = CallRecord(seq=len(self.records) + 1,
                       t=round(time.time() - self.t0, 4),
                       action=action, **kw)
        self.records.append(r)
        return r

    # ── 首页 ────────────────────────────────────
    def home(self) -> str:
        self.page, self.cur_tool = 'home', None
        self.n_nav += 1
        self._log('home')
        W = 60
        L = ['┌' + '─' * W + '┐']
        for line in _wrap('%s · 控制台' % self.title, W - 2):
            L.append('│ %-*s │' % (W - 2, line))
        L.append('├' + '─' * W + '┤')
        for line in _wrap('这里有三个入口。除首页外的任何页面都能返回这里。', W - 2):
            L.append('│ %-*s │' % (W - 2, line))
        L.append('├' + '─' * W + '┤')
        L.append('│ %-*s │' % (W - 2, '  [1] paper   展开试卷（题目与 60 个选项）'))
        L.append('│ %-*s │' % (W - 2, '  [2] tools   60 个计算器'))
        L.append('│ %-*s │' % (W - 2, '  [3] answer  答卷（填 10 个编号）'))
        L.append('├' + '─' * W + '┤')
        L.append('│ %-*s │' % (W - 2,
                 '用法：paper / tools / answer / back'))
        L.append('└' + '─' * W + '┘')
        if self.slots:
            L.append('')
            L.append('答卷已填：%s' % ', '.join(self.slots))
        return '\n'.join(L)

    # ── 试卷页 ──────────────────────────────────
    def show_paper(self) -> str:
        """
        ⭐ 显示试卷 —— 60 个【选项】，不是 90 个工具。

        ⚠️⚠️ 2026-11-10 修了一个严重的错：
        「问题在于试卷有没有注入上下文……检查一下他目前要怎么读题」

            实测：模型把 T-077、T-089 这些工具编号填进答案格，
            并说「60 options (T-001..T-090, they say 60 but list 90)」。

            原来怀疑是模型抽象能力不足 —— 不对，是我们喂错了。

            这一段的旧代码：
                L.append('选项：')
                for tid in self.order:          ← self.order 是 90 个【工具】
                    t = self.tools[tid]
                    L.append('      %s  %s' % (tid, t.expr))

            ⇒ 把 90 个工具标成「选项」返回给模型。
              模型分不清，是因为我们告诉它工具就是选项。

        ⇒ 现在：选项从 self.paper.ops 取（60 个，O 开头），
                工具只在 list_tools 里列（90 个，T 开头）。
                两者在卷面上绝不混称。
        """
        self.page, self.cur_tool = 'paper', None
        self.n_nav += 1
        self._log('paper')
        pp = self.paper
        ops = list(getattr(pp, 'ops', []) or [])
        mos = list(getattr(pp, 'mos', []) or [])

        L = []
        L.append('=' * 62)
        L.append('试卷')
        L.append('=' * 62)
        L.append('  A000 = %.12g' % pp.a000)
        L.append('  Z    = %.12g' % pp.z)
        L.append('')
        L.append('  求：A000 × [ ] × [ ] × … × [ ] = Z')
        L.append('      （10 个空，各填一个【选项】代码）')
        L.append('')
        L.append('=' * 62)
        L.append('选项（共 %d 个，代码形如 O001）' % len(ops))
        L.append('=' * 62)

        # ⚠️ pp.ops / pp.mos 本身已经是展示顺序（与 pp.order 一致），
        #    直接按数组顺序走即可。pp.order 里装的是 pid 字符串。
        for k, (op, mo) in enumerate(zip(ops, mos)):
            L.append('')
            L.append('  %s' % op.pid)
            L.append('    x 的名称：%s' % (mo.x_word if mo else ''))
            L.append('    图：figs/%03d.png' % (k + 1))
            body = (mo.text or '').split('\n')[-1].strip() if mo else ''
            L.append('    %s' % body)

        L.append('')
        L.append('=' * 62)
        L.append('计算工具另见 list_tools（共 %d 个，代码形如 T-001）'
                 % len(self.tools))
        L.append('=' * 62)
        return '\n'.join(L)

    # ── 工具列表 ────────────────────────────────
    def show_tools(self) -> str:
        self.page, self.cur_tool = 'tools', None
        self.n_nav += 1
        self._log('tools')
        L = ['┌' + '─' * 60 + '┐']
        L.append('│ %-60s │' % '[tools] 计算器（共 %d 个）' % len(self.tools))
        L.append('├' + '─' * 60 + '┤')
        # ⚠️ 不写用法 —— 怎么调，模型自己看工具 schema
        cards = ['[%s]' % t for t in self.order]
        for i in range(0, len(cards), 8):
            L.append('│ %-60s │' % ('  ' + ' '.join('%-7s' % c for c in cards[i:i+8])))
        L.append('└' + '─' * 60 + '┘')
        return '\n'.join(L)

    # ── 单个工具页 ──────────────────────────────
    def open_tool(self, tid: str) -> str:
        t = self.tools.get(tid)
        if t is None:
            self._log('open', tid=tid, ok=False, note='不存在的编号')
            return ('⚠️ 没有编号为 %s 的项。'
                    '用 list_tools 看全部编号。' % tid)
        self.page, self.cur_tool = 'tool', tid
        self.n_nav += 1
        self._log('open', tid=tid)
        L = ['┌' + '─' * 60 + '┐']
        L.append('│ %-60s │' % ('[%s]' % tid))
        L.append('├' + '─' * 60 + '┤')
        # ⚠️ 进阶档的描述较长（抽象描述），必须折行
        for line in _wrap(t.expr or '(无说明)', 58):
            L.append('│ %-60s │' % line)
        L.append('├' + '─' * 60 + '┤')
        L.append('│ %-60s │' % ('  输入  %s = ?' % t.skin.input_label))
        L.append('│ %-60s │' % ('  输出  %s = ?' % t.skin.output_label))
        L.append('├' + '─' * 60 + '┤')
        L.append('│ %-60s │' % ('  算：calc(pid="%s", x=<数值>)' % tid))
        L.append('└' + '─' * 60 + '┘')
        return '\n'.join(L)

    # ── 计算 ────────────────────────────────────
    def calc(self, tid: str, x, factor=None) -> str:
        """
        ⭐ 用一个工具算：输出 = f(x) × factor。

        ⚠️ 弗兰克 2026-11-10：
           「那我们的工具里，就不该用除法，该用乘法。」

           卷面写的是 f(x) ÷ c。要除 0.125，factor 就填 8。
           系数自己调整。

           factor 不填按 1 算，就是纯 f(x)。

        ⚠️ 基础档（60 个计算器）自带两层和系数，
           factor 对它无意义 —— 传了就忽略并说清楚。
        """
        t = self.tools.get(tid)
        if t is None:
            self._log('call', tid=tid, ok=False, note='不存在的编号')
            return '⚠️ 没有编号为 %s 的项。' % tid
        try:
            xf = float(x)
        except (TypeError, ValueError):
            self._log('call', tid=tid, ok=False, note='输入不是数字')
            return '⚠️ 输入必须是数字。'

        # ⚠️ 进阶档工具（LayerTool）收 factor；基础档工具不吃这个参数
        try:
            import toolsplus as _tp
            adv = isinstance(t, _tp.LayerTool)
        except Exception:                               # noqa: BLE001
            adv = hasattr(t, 'kind')
        if adv:
            self.n_call += 1
            res = t.call(xf, factor)
        else:
            self.n_call += 1
            res = t.call(xf)
            if factor not in (None, '', 1, 1.0):
                return ('[%s]\n  输入 %s = %s\n  输出 %s = %s\n'
                        '  （本项固定两层与系数，不接受额外系数）'
                        % (tid, t.skin.input_label, _fmt(xf),
                           t.skin.output_label, _fmt(res.get('y'))))

        if res['ok']:
            self._log('call', tid=tid, x=xf, y=res['y'],
                      factor=res.get('factor'))
            fv = res.get('factor', 1.0)
            tail = ('' if fv == 1.0 else '\n  系数  = %s' % _fmt(fv))
            return ('[%s]\n  输入 %s = %s%s\n  输出 %s = %s'
                    % (tid, t.skin.input_label, _fmt(xf), tail,
                       t.skin.output_label, _fmt(res['y'])))
        self._log('call', tid=tid, x=xf, ok=False,
                  note=res.get('error', '失败'))
        return '[%s]\n  ⚠️ %s' % (tid, res.get('error', '计算失败'))

    # ── 答卷页 ──────────────────────────────────
    def show_answer(self) -> str:
        self.page, self.cur_tool = 'answer', None
        self.n_nav += 1
        self._log('answer')
        L = ['┌' + '─' * 60 + '┐']
        L.append('│ %-60s │' % '[answer] 答卷')
        L.append('├' + '─' * 60 + '┤')
        for line in _wrap('把 %d 个选项编号填进格子，然后 submit() 交卷。' % self.N_ANSWER, 58):
            L.append('│ %-60s │' % line)
        L.append('├' + '─' * 60 + '┤')
        for k in range(self.N_ANSWER):
            v = self.slots[k] if k < len(self.slots) else '空'
            L.append('│ %-60s │' % ('  格子 %2d : %s' % (k + 1, v)))
        L.append('├' + '─' * 60 + '┤')
        L.append('│ %-60s │' % '  fill(pid)  追加一格 / show_answer() / submit()')
        L.append('└' + '─' * 60 + '┘')
        return '\n'.join(L)

    def fill(self, tid: str) -> str:
        """
        ⭐ 往答卷里填一格。

        ⚠️⚠️ 2026-11-09 实测：进阶档有 120 个工具，
           `self.tools` 里全是 T-xxx，于是 T-009 被当成有效编号
           填进了答卷。答卷只能填【选项】，不能填工具。

        ⇒ 校验用 self.paper 的选项集合，不用 self.tools。
        """
        if len(self.slots) >= self.N_ANSWER:
            self._log('fill', tid=tid, ok=False, note='格子已满')
            return ('⚠️ %d 格已满。用 show_answer() 查看已填内容。'
                    % self.N_ANSWER)

        # ⚠️ 合法的答卷项 = 卷子里的选项 pid（O001…），不是工具编号
        valid = {op.pid for op in getattr(self.paper, 'ops', [])}
        if tid not in valid:
            self._log('fill', tid=tid, ok=False, note='不是选项编号')
            # 中性报错：不点破两套编号的区别
            return '⚠️ 无效编号：%s' % tid
        if tid in self.slots:
            self._log('fill', tid=tid, ok=False, note='重复填同一项')
            return '⚠️ %s 已经在答卷里了。' % tid
        self.slots.append(tid)
        self._log('fill', tid=tid)
        return '已填入第 %d 格：%s' % (len(self.slots), tid)

    def set_slot(self, k: int, tid: str) -> str:
        if not (1 <= k <= self.N_ANSWER):
            self._log('set', tid=tid, ok=False, note='格号越界')
            return '⚠️ 格号必须在 1..%d。' % self.N_ANSWER
        if tid not in self.tools:
            self._log('set', tid=tid, ok=False, note='不存在的编号')
            return '⚠️ 没有编号为 %s 的项。' % tid
        while len(self.slots) < k:
            self.slots.append('')
        self.slots[k - 1] = tid
        self._log('set', tid=tid)
        return '第 %d 格已设为 %s' % (k, tid)

    def clear(self) -> str:
        self.slots = []
        self._log('clear')
        return '答卷已清空。'

    def submit(self) -> str:
        """
        ⭐ 交卷。判分走 paper.check()。

        ⚠️ 只告诉模型对/不对与连乘误差，不告诉它哪个错了。
        """
        filled = [s for s in self.slots if s]
        if len(filled) != self.N_ANSWER:
            self._log('submit', ok=False, note='未填满')
            return ('⚠️ 需要正好 %d 个编号，现在填了 %d 个。'
                    % (self.N_ANSWER, len(filled)))
        self.submitted = True
        r = self.paper.check(filled)
        self.submit_result = r
        self._log('submit', ok=bool(r['ok']),
                  note='err=%.3e' % r['err'])
        if r['ok']:
            return '✅ 交卷成功。10 个编号的连乘等于 Z。'
        return ('❌ 交卷未通过。\n  连乘 = %.12g\n  Z    = %.12g\n'
                '  相对误差 = %.3e\n（不告诉你哪个错。可以回 tools 再算。）'
                % (r['product'], r['z'], r['err']))

    # ── 返回 ────────────────────────────────────
    def back(self) -> str:
        self._log('back')
        if self.page == 'home':
            return self.home()
        if self.page in ('paper', 'tools', 'answer'):
            return self.home()
        if self.page == 'tool':
            return self.show_tools()
        return self.home()

    # ── 导出 ────────────────────────────────────
    def dump(self) -> str:
        L = ['═' * 66]
        L.append('答卷器 · 交互记录')
        if self.exam_id:
            L.append('问卷编号：%s' % self.exam_id)
        L.append('共 %d 条（计算 %d 次，导航 %d 次）；交卷 %s'
                 % (len(self.records), self.n_call, self.n_nav,
                    '是' if self.submitted else '否'))
        L.append('═' * 66)
        for r in self.records:
            if r.action == 'call':
                L.append('%4d %8.3fs  calc  %s  x=%-16s y=%-16s %s'
                         % (r.seq, r.t, r.tid,
                            _fmt(r.x) if r.x is not None else '-',
                            _fmt(r.y) if r.y is not None else '-', r.note))
            else:
                L.append('%4d %8.3fs  %-7s %s %s'
                         % (r.seq, r.t, r.action, r.tid, r.note))
        return '\n'.join(L)

    def digest(self) -> str:
        s = '|'.join('%d:%s:%s:%s' % (r.seq, r.action, r.tid,
                                       '' if r.y is None else '%.17g' % r.y)
                     for r in self.records)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]


def _fmt(v) -> str:
    """
    ⭐ 工具输出的数字格式。

    ⚠️⚠️ 2026-11-10 弗兰克：「看看还有啥。」

        实测发现：这里原来是 '%.10g'（10 位有效数字），
        而卷面 x 有 14 位有效数字（config.SCI_DIGITS）。

        ⇒ 输入比输出精确，模型拿 10 位的中间值往下算，
          误差被放大到 2.0e-07 —— 最后 Z 对不上，会冤枉模型。

    ⇒ 改成 '%.17g'：float64 无损往返的标准位数，
      模型拿到的就是工具真正算出来的值。
    """
    if v is None:
        return '-'
    try:
        return '%.17g' % float(v)
    except Exception:
        return str(v)


# ══════════════════════════════════════════════════════════════════
# 五、从一道题建控制台
# ══════════════════════════════════════════════════════════════════

def _family(name: str) -> str:
    """
    按名字分函数族。⚠️ 只用于「打散选工具」，不参与计算。

    ⭐ 2026-11-07 Frank：
       「工具允许重复就好了呗，这个无所谓的，但我怕重复度太高，
         简化工具数量，能不能最大化打散的情况，允许有限重复？」

       → 做法：按族轮流取，同一族用满 max_per_family 次就换下一族。
    """
    n = name.lower()
    rules = [
        ('pow', 'pow'), ('sqrt', 'pow'), ('cbrt', 'pow'),
        ('exp10', 'exp'), ('exp2', 'exp'), ('expi', 'expint'), ('expn', 'expint'),
        ('expm1', 'exp'), ('exp', 'exp'),
        ('log', 'log'),
        ('asin', 'trig'), ('acos', 'trig'), ('atan', 'trig'),
        ('sinh', 'hyp'), ('cosh', 'hyp'), ('tanh', 'hyp'),
        ('asinh', 'ihyp'), ('acosh', 'ihyp'), ('atanh', 'ihyp'),
        ('sin', 'trig'), ('cos', 'trig'), ('tan', 'trig'),
        ('gamma', 'gamma'), ('lgamma', 'gamma'), ('digamma', 'gamma'),
        ('beta', 'beta'), ('betaln', 'beta'),
        ('i0', 'bessel'), ('i1', 'bessel'), ('iv', 'bessel'), ('kv', 'bessel'),
        ('ive', 'bessel'), ('kve', 'bessel'),
        ('erf', 'erf'),
        ('ndtr', 'normal'), ('ndtri', 'normal'),
        ('ellip', 'ellip'),
        ('zeta', 'special'), ('lambertw', 'special'), ('airy', 'special'),
        ('hypot', 'geom'),
        ('logit', 'logit'), ('expit', 'logit'),
        ('softplus', 'smooth'), ('exprel', 'smooth'), ('pseudo', 'smooth'),
        ('entr', 'entropy'),
        ('xlog', 'xlog'), ('xlog1py', 'xlog'),
        ('gammainc', 'gammainc'), ('gammaincc', 'gammainc'),
    ]
    for key, fam in rules:
        if key in n:
            return fam
    return 'other'


def pick_tools_spread(
    chain: list[str],
    n_total: int = 60,
    max_per_family: int = 2,
    seed: int = 0,
) -> list[tuple[object, int]]:
    """
    ⭐ 最大化打散地选工具：按函数族轮流取，同族最多用 max_per_family 次。

    返回 [(pack, variant), ...]，length = n_total。

    ⚠️ 链上那 10 个必须全在，且各算一次。
    ⚠️ 若素材不够（族用满仍凑不够），把 max_per_family 自动放宽并记录。
    """
    import random
    rng = random.Random(seed)

    out: list[tuple[object, int]] = []
    used = Counter()          # 族 -> 出现次数
    used_pack = Counter()     # 包名 -> 出现次数

    # ── ① 链上 10 个必须全在 ────────────────────────────
    for nm in chain:
        pk = packs.get(nm)
        if pk is None or not pk.inverse_trusted:
            continue
        out.append((pk, 1))
        used[_family(nm)] += 1
        used_pack[nm] += 1
        if len(out) >= n_total:
            return out[:n_total]

    # ── ② 其余：按族轮流取，打散 ────────────────────────
    #
    # ⚠️ 2026-11-07 实测发现一个算术必然：
    #    可逆包共 59 个，13 个族。链上取走 10 个后剩 49 个要补进 50 个工具。
    #    若每族只许 2 次，13×2 = 26，远远不够 50。
    #    → 必须让「大族多取、小族少取」，即按族内包数按比例分配配额。
    #
    # ⭐ 这就是 Frank 说的「最大化打散 + 允许有限重复」：
    #    打散优先（先保证族覆盖），配额按比例放宽（保证数量）。

    pool = [p for p in packs.PACKS
            if p.inverse_trusted and p.name not in set(chain)]

    # 按族分组
    by_fam: dict[str, list] = {}
    for p in pool:
        by_fam.setdefault(_family(p.name), []).append(p)
    for f in by_fam:
        rng.shuffle(by_fam[f])

    # ⭐ 按族内可提供的数量排序（大族在前），族内顺序随机
    fam_order = sorted(by_fam.keys(), key=lambda f: -len(by_fam[f]))
    quota = {f: max_per_family for f in fam_order}

    # ⭐ 动态放宽：若配额总和不够，按族大小按比例加
    def _total_quota():
        return sum(min(quota[f], len(by_fam[f])) for f in fam_order)

    need = n_total - len(out)
    rounds = 0
    while _total_quota() < need and rounds < 20:
        rounds += 1
        for f in fam_order:
            quota[f] += 1
        need = n_total - len(out)
        if _total_quota() >= n_total - 10:
            break

    used_f = Counter()
    used_pack = Counter()
    for t in out:
        pass  # 链上的已计入 used/used_pack，此处不再重复

    limit = max_per_family
    guard = 0
    while len(out) < n_total and guard < 10000:
        guard += 1
        picked_any = False
        # 按「已取次数 / 配额」升序，优先补最欠的族
        for f in sorted(fam_order, key=lambda x: used_f[x] / max(1, quota[x])):
            if len(out) >= n_total:
                break
            if used_f[f] >= quota[f]:
                continue
            lst = by_fam[f]
            if not lst:
                continue
            cand = None
            for p in lst:
                if used_pack[p.name] == 0:
                    cand = p
                    break
            if cand is None:
                break          # 本族已无可新包，跳过（不重复用同包）
            lst.remove(cand)
            v = used_pack[cand.name] + 1
            out.append((cand, v))
            used_f[f] += 1
            used_pack[cand.name] += 1
            picked_any = True
        if not picked_any:
            break

    # ── ③ 仍不够 → 允许同包复用 ────────────────────────
    if len(out) < n_total:
        all_rev = [p for p in packs.PACKS if p.inverse_trusted]
        i = 0
        while len(out) < n_total and all_rev:
            p = all_rev[i % len(all_rev)]
            i += 1
            if used_pack[p.name] >= 2:
                continue
            v = used_pack[p.name] + 1
            out.append((p, v))
            used_pack[p.name] += 1

    return out[:n_total]


def build_console(prob, seed: int = 0, n_tool: int = 60,
                  max_per_family: int = 2,
                  abstract: bool = True,
                  module=None) -> ToolConsole:
    """
    ⭐ 从一道 VectorProblem 建工具控制台。

    ⚠️ n_tool 个工具 = 链上 10 个（用链上函数）+ 其余（按族打散选）。
    ⚠️ 全部是真计算器，没有真假之分。
    ⚠️ 编号最后统一打乱 —— 不让「链上的排在前面」泄露。

    参数
    ----
    abstract : ⭐ 是否启用抽象皮（默认 True）
               True  → 用 skins 的精确描述 + 随机语法，标题中性化
               False → 用 skeleton 的标准名（调试用，会泄露答案）
    module   : ⭐ **预留接口**：module.Module（中间态）
               给了就从中间态取词（变量名与题目一致）；
               没给就用默认名。
               ⚠️ 这是「中间层 → 控制台」的对接点，
                 将来系统提示词注入器也从这里取。
    """
    import random
    rng = random.Random(seed)

    chain = list(prob.chain)
    picked = pick_tools_spread(chain, n_total=n_tool,
                               max_per_family=max_per_family, seed=seed)

    # ⭐ 从中间态取变量名（若给了）
    #    ⚠️ 中间态里的 std_name 是 v0/v1 这种内部名 —— 不能直接给模型看。
    #       必须换成名词（Frank：变量可以换为所有名词）。
    #
    # ⭐⭐ 2026-11-07 改（修一致性 bug）：
    #    原来用 step 索引名词列表，而 50 个干扰工具的 step 都是 0，
    #    导致 103/120 次取到同一个词（实测：誓言 51 次、surprise 52 次）。
    #    **那不是一致性，是随机性塌缩。**
    #
    #    现在改用 groupmap：
    #      ① 先给这一组的 11 个变量编号（V000..V010）分配名词
    #      ② 每个工具按自己的 step 取「输入名 / 输出名」
    #      ③ 同一编号在任何工具里都是同一个词
    gmap = None
    if module is not None:
        try:
            import module as _mod_mod
            import groupmap as _gmap
            parts = module.by_type(_mod_mod.VAR)
            pids = [p.pid for p in parts]
            gmap = _gmap.build_group_map(
                group_id=getattr(prob, 'uid', 'GROUP'),
                var_pids=pids,
                seed=seed)
        except Exception:
            gmap = None

    def _names_for(tool_step: int, j: int = 0) -> tuple[str, str]:
        """
        取这个工具的（输入名, 输出名）。

        ⚠️ tool_step >= 1（链上工具）：用 groupmap 的 step_var，保证首尾相接
        ⚠️ tool_step == 0（干扰工具）：它不在链上，没有固定位置，
           → 从本组已用的 11 个名词里随机取两个**不同的**，
             这样它说的话仍然属于同一套词汇，不会引入组外新词。
        """
        if gmap is not None:
            try:
                import groupmap as _gmap
                if tool_step >= 1:
                    return _gmap.step_var(gmap, tool_step)
                # 干扰工具：从本组词里随机取两个不同的
                import random as _rnd
                r = _rnd.Random(seed * 100003 + j)
                pool = list(gmap.var_names.values())
                if len(pool) >= 2:
                    a, b = r.sample(pool, 2)
                    return a, b
            except Exception:
                pass
        return '输入量', '输出量'

    tools: list[Tool] = []
    chain_set = set(chain)
    for j, (pk, variant) in enumerate(picked):
        # ⚠️ 链上包的第一份算「链上」，后续复用算「off」
        on_chain = (pk.name in chain_set and
                    sum(1 for t in tools if getattr(t.pack, 'name', '') == pk.name) == 0)
        step = 0
        if on_chain:
            step = chain.index(pk.name) + 1

        # ⭐ 皮：标准 or 抽象
        if abstract:
            # ⚠️ 2026-11-07 改：按 step 取「这一步自己的」输入/输出名，
            #    不再让所有干扰工具都取到同一个词。
            in_nm, out_nm = _names_for(step, j)

            # ⭐ 描述语言跟随变量语言（输入名是英文 → 用英文描述）
            #    ⚠️ 这样一条描述内部语言统一，不会出现
            #       「loop 的值为：由 manifestation 的值求得一个数的误差函数值」这种混杂。
            use_lang = 'zh'
            if gmap is not None:
                try:
                    import re as _re2
                    if _re2.search(r'[A-Za-z]', in_nm):
                        use_lang = 'en'
                except Exception:
                    pass

            skin = abstract_skin(pk, in_nm, out_nm,
                                 tmpl_idx=rng.randrange(10), seed=seed + j,
                                 lang=use_lang)
        else:
            skin = std_skin(pk)

        tools.append(Tool(
            tid='T%02d' % (len(tools) + 1),
            pack=pk,
            skin=skin,
            role='chain' if on_chain else 'off',
            step=step,
            variant=variant,
        ))

    # ── 打乱编号 ────────────────────────────────────────
    rng.shuffle(tools)
    for i, t in enumerate(tools):
        t.tid = 'T%02d' % (i + 1)

    return ToolConsole(tools)


# ══════════════════════════════════════════════════════════════════
# 六、自检
# ══════════════════════════════════════════════════════════════════

def self_test(verbose: bool = True) -> int:
    fails = 0
    checks = []

    import vectorform
    ps = vectorform.build_vector_problems(n=1, k=10, seed=20261107, verbose=False)
    p = ps[0]
    con = build_console(p, seed=20261107)

    checks.append(('工具数 = 60', len(con.tools) == 60))
    checks.append(('编号全唯一', len(set(con.order)) == 60))
    checks.append(('链上工具 10 个',
                   sum(1 for t in con.tools.values() if t.role == 'chain') == 10))
    checks.append(('非链上工具 50 个',
                   sum(1 for t in con.tools.values() if t.role == 'off') == 50))

    # ⭐ 打散度检查（2026-11-07 Frank：「最大化打散，允许有限重复」）
    #
    # ⚠️ 判据要分两层：
    #    链上那 10 个是「必须全在」的，它们是什么族由链决定，不能挑；
    #    打散只对「补充的 50 个」有意义。
    #    所以先看补充部分的打散度，再看总体。
    chain_tools = [t for t in con.tools.values() if t.role == 'chain']
    off_tools = [t for t in con.tools.values() if t.role == 'off']
    fam_off = Counter(_family(t.pack.name) for t in off_tools)
    fam_all = Counter(_family(t.pack.name) for t in con.tools.values())
    name_cnt = Counter(t.pack.name for t in con.tools.values())

    checks.append(('补充部分：覆盖全部可用族',
                   len(fam_off) >= min(13, len(set(_family(p.name) for p in packs.PACKS if p.inverse_trusted)))))
    checks.append(('补充部分：同族不超过族内包数',
                   all(v <= len([p for p in packs.PACKS
                                 if p.inverse_trusted and _family(p.name) == f])
                       for f, v in fam_off.items())))
    checks.append(('同函数最多出现 2 次', max(name_cnt.values()) <= 2))
    checks.append(('全体：族数 ≥ 10', len(fam_all) >= 10))

    # 列表页不泄露角色
    home = con.render_home()
    checks.append(('列表页不含「链上」字样', '链上' not in home and 'chain' not in home))
    checks.append(('列表页含全部编号',
                   all(t.tid in home for t in con.tools.values())))

    # 进入单个工具
    tid0 = con.order[0]
    page = con.open(tid0)
    checks.append(('工具页有首页键', '首页' in page))
    checks.append(('工具页有输入输出', '输入' in page and '输出' in page))

    # 调用
    t = con.tools[tid0]
    n_before = len(con.records)
    r = con.call(tid0, 2.0)
    checks.append(('调用有结果', '输出' in r))
    checks.append(('调用被记录', len(con.records) == n_before + 1))

    # 调用结果与包的正运算一致
    res = t.call(2.0)
    expect = float(t.pack._fwd(torch.tensor([2.0], device=DEV, dtype=DT))[0]) * t.pack.c
    checks.append(('结果与数理包一致', abs(res['y'] - expect) < 1e-12))

    # 返回值不泄露角色
    checks.append(('返回值不含角色字段', 'role' not in res and 'step' not in res))

    # 回首页
    con.home()
    checks.append(('回首页后 current 为空', con.current is None))

    # 换皮
    con.reskin(tid0, ToolSkin(title='粒子运动计算器',
                              desc='我计算粒子无规则运动的位移。'))
    page2 = con.open(tid0)
    checks.append(('换皮生效', '粒子运动计算器' in page2))
    checks.append(('换皮后内核未变', con.tools[tid0].pack is t.pack))

    # 记录摘要稳定
    d1 = con.digest()
    d2 = con.digest()
    checks.append(('摘要可复现', d1 == d2))

    # 不存在的工具
    r = con.call('T99', 1.0)
    checks.append(('不存在的编号有提示', '没有编号' in r))

    # 非数字输入
    r = con.call(tid0, 'abc')
    checks.append(('非数字输入有提示', '必须是数字' in r))

    # ⭐ 抽象皮检查（2026-11-07 接 skins 后新增）
    home2 = con.render_home()
    all_desc = ' '.join(t.skin.desc for t in con.tools.values())
    all_title = ' '.join(t.skin.title for t in con.tools.values())
    from skeleton import get_skeleton as _gs
    # 取所有函数的数学中文名
    zh_names = []
    for t in con.tools.values():
        sk = _gs(t.pack.name)
        if sk and sk.zh_name:
            zh_names.append(sk.zh_name)
    # ⚠️ 抽象皮的标题里不能出现任何函数中文名
    leaked_title = [z for z in zh_names if z in all_title]
    checks.append(('抽象皮标题不泄露函数名', not leaked_title))
    if leaked_title and verbose:
        print('   ⚠️ 标题泄露：', leaked_title[:5])
    checks.append(('抽象皮描述非空', all(t.skin.desc for t in con.tools.values())))
    # ⭐ 标题不能 60 个全一样（那样一眼看出程序生成）
    titles = {t.skin.title for t in con.tools.values()}
    checks.append(('标题不止一种', len(titles) > 1))

    # 中间态对接（预留接口）
    try:
        import module as _m
        import skins as _skins
        mod = _m.build_module(p)
        con2 = build_console(p, seed=20261107, abstract=True, module=mod)
        checks.append(('可从中间态建控制台', len(con2.tools) == 60))

        # ⚠️ 2026-11-07 修：不能用 std_name（v0/v1）比 ——
        #    变量已按 Frank 的要求换成名词了，内部名不该出现在题面里。
        nouns = {_skins.noun_for_var(k, 20261107) for k in range(11)}
        used = any(any(n in t.skin.desc for n in nouns if n)
                   for t in con2.tools.values())
        checks.append(('描述里用了名词（非 v0/v1）', used))
        # ⭐ 内部名绝不能泄露给模型
        leaked_var = [t.tid for t in con2.tools.values()
                      if any(('（%s）' % ('v%d' % k)) in t.skin.desc for k in range(11))]
        checks.append(('描述里没有 v0/v1 内部名', not leaked_var))
        # ⚠️ 2026-11-07 修：加入英文描述后，判据要双语。
        #    中文说「（小明）的值」，英文说 "the value of (cat)"。
        def _has_value_marker(s: str) -> bool:
            return ('的值' in s) or ('value of' in s.lower())

        checks.append(('描述里都标出数值（的值 / value of）',
                       all(_has_value_marker(t.skin.desc)
                           for t in con2.tools.values())))

        # ⭐⭐ 2026-11-07 新增：组内一致性检查（修 bug 的回归测试）
        #    ⚠️ 旧 bug：50 个干扰工具 step 都是 0，全取到同一个词，
        #       实测 誓言 51 次 / surprise 52 次。
        import re as _re
        from collections import Counter as _C
        name_cnt = _C()
        for t in con2.tools.values():
            for m in _re.finditer(r'（([^）]+)）', t.skin.desc):
                name_cnt[m.group(1)] += 1
        checks.append(('描述里出现的名词 ≥ 5 种', len(name_cnt) >= 5))
        # ⚠️ 任何一个词都不该被用过半（120 次提及里不该某个词占 50 次）
        worst = max(name_cnt.values()) if name_cnt else 0
        checks.append(('没有某个名词被过度使用（<20 次）', worst < 20))
        if verbose and worst >= 20:
            print('   ⚠️ 最常用的名词用了 %d 次: %s'
                  % (worst, name_cnt.most_common(3)))
    except Exception as e:
        checks.append(('可从中间态建控制台', False))
        if verbose:
            print('   ⚠️ 中间态对接失败：', e)
        checks.append(('描述里用了名词（非 v0/v1）', False))
        checks.append(('描述里没有 v0/v1 内部名', False))
        checks.append(('描述里都带「的值」', False))

    if verbose:
        print('=' * 62)
        print('toolconsole 自检')
        print('=' * 62)
        for name, ok in checks:
            print('  %s  %s' % ('✅' if ok else '❌', name))
            if not ok:
                fails += 1
        print()
        print('── 列表页（前 12 个）──')
        home_lines = home.split('\n')
        print('\n'.join(home_lines[:8]))
        print('   ...')
        print()
        print('── 单个工具页 ──')
        print(page)
        print()
        print('── 一次调用 ──')
        print(r if '输出' in (r := con.call(tid0, 2.0)) else r)

    return fails


if __name__ == '__main__':
    raise SystemExit(self_test())
