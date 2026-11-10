"""
promptexam.py —— 给模型注入的考试提示词

════════════════════════════════════════════════════════════════
【写多长】（弗兰克 2026-11-09 定稿）
════════════════════════════════════════════════════════════════

    「我认为如果让我来写，我只会写几句话：

      这是一套试卷，已知和未知已经给出，你需要凑齐 10 个正确的
      选项填进空位使等式成立。每个选项都由函数和数值构成，
      数值是多少需要看对应的图片读取。除此之外，系统为你的计算
      提供了计算工具，他们具有什么特征，你要通过什么接口使用，
      考试交卷只有一次机会，使用超过目录外的工具会被视为作弊。
      交卷后本次答题结束，会话终止。

      我就想写上面这么点字。」

    ⇒ 只留【是什么】【交什么】【什么限制】。
      工具长什么样，模型自己调接口看。

════════════════════════════════════════════════════════════════
【边界：这里写什么，不写什么】
════════════════════════════════════════════════════════════════

    写：
      · 已知量（A000 / Z）与未知量（10 个空）
      · 任务目标（凑齐 10 个使等式成立）
      · 数的来源（图）
      · 工具的存在与接入方式（MCP 接口名）
      · 三条铁律（交卷一次、越界算作弊、交卷即终止）

    不写：
      ✗ 解题算法 / 操作步骤 / 记忆提示
      ✗ 任何真实编号、系数、式子
      ✗ 每个工具的调用样例（schema 里已有）
      ✗ 答案文件在哪
      ✗ 干扰项数量等非必要信息

════════════════════════════════════════════════════════════════
【工具层的抽象在 MCP 那边做】
════════════════════════════════════════════════════════════════

    ⚠️ 工具描述由 toolsplus / skins 生成为「做什么」而非「叫什么」：
          exp1  → 「由（x）的值求得一阶指数积分」
       说清行为，不说函数名 —— 模型靠读描述判断这是哪一层。

    ⚠️ 2026-11-10 起进阶档工具数 = 90（58 正向 + 16 反向 + 16 无关），
       由 examfile.build_features 的 n_tools 给出，本模块只读不算。

    ⇒ 本模块只告诉模型「去 MCP 看」，不复制任何工具内容。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, r'<BENCH>\generator_v2')

__all__ = ['build_prompt', 'prompt_for_exam', 'self_test']

SERVER_NAME = 'exam'


def _t(name: str) -> str:
    """⭐ 工具在模型侧的名字（DSH 的 MCP 命名规则）。"""
    return 'mcp__%s__%s' % (SERVER_NAME, name)


def build_prompt(exam: dict, exam_dir: str = '') -> str:
    """⭐ 生成考试提示词。短，只给必要信息。"""
    n_opt = exam['n_options']
    n_ans = exam['n_answers']
    a000 = exam['A000']
    z = exam['Z']
    feat = exam.get('features') or {}
    n_tool = feat.get('n_tools', n_opt)

    L = []

    L.append('# 试卷')
    L.append('')
    L.append('已知 `A000 = %.12g`，`Z = %.12g`。' % (a000, z))
    L.append('')
    L.append('%d 个空位，各填一个选项代码，使等式成立：' % n_ans)
    L.append('')
    L.append('```')
    L.append('A000 × [ ] × [ ] × … × [ ] = Z')
    L.append('```')
    L.append('')
    L.append('**凑齐 %d 个正确的选项填进去。**' % n_ans)
    L.append('')
    L.append('%d 个选项，每条都是函数与数值的组合。' % n_opt)
    L.append('')
    # ⚠️⚠️ 2026-11-10 弗兰克：
    #    「稍微给点提示，比如选项的代号结构，工具的代号结构，
    #      省的他分不清。」
    #
    #    ⚠️ 这是「信息格式」说明，不是解题方法。
    #       只讲两种编号长什么样、各属一类，不讲怎么配系数、怎么凑乘积。
    L.append('**代码结构**（两类东西，不要混）：')
    L.append('')
    L.append('- **选项**：`O` + 三位数字，如 `O001`。'
             '共 %d 个，写在卷面上，是填进空位的答案。' % n_opt)
    L.append('- **工具**：`T-` + 三位数字，如 `T-001`。'
             '共 %d 个，用 `calc` 调用，是算数用的。' % n_tool)
    L.append('')
    L.append('**空位里只填选项代码，不填工具代码。**')
    L.append('')
    # ⚠️⚠️ 2026-11-10 弗兰克：
    #    「建议直接在提示词里写，改下模板。图放在沙箱里哪个文件夹。」
    #
    #    实测：卷面上给了 figs/001.png，模型还是去猜文件名，
    #    拿 read 试探 060.png / 090.png 存不存在。
    #    ⇒ 图的位置必须在这里点明，不能只靠卷面。
    L.append('数值在 `figs/` 目录里，一个选项一张图，'
             '卷面的「图」那行写出了对应路径。')
    L.append('')

    L.append('## 计算工具')
    L.append('')
    # ⚠️⚠️ 2026-11-10 弗兰克：
    #    「这有些工具，不确定是否都能用，你可以试试
    #      （别什么都讲那么清楚，agi就是要在摸索中前进）。」
    #
    #    ⇒ 不保证 90 个都对你有用。有些用不上，有些看着像能用
    #      其实不该用。自己判断，自己试。
    L.append('系统提供 %d 个计算工具。' % n_tool)
    L.append('')
    L.append('**不保证每个都对你有用。**'
             '哪个能用、哪个不能用，自己试。')
    L.append('')
    L.append('接口：')
    L.append('')
    L.append('`%s` `%s` `%s` `%s` `%s` `%s` `%s`'
             % (_t('list_tools'), _t('open_tool'), _t('calc'),
                _t('show_paper'), _t('show_answer'),
                _t('fill'), _t('submit')))
    L.append('')
    # ⚠️ 接口事实必须说清：第三个参数做什么用。
    #    ⚠️ 但不说「取倒数」—— 怎么把卷面的除法换成工具的乘法，
    #       是模型自己的事。
    L.append('`%s` 的第三个参数是你自己给的系数：'
             '结果 = 该工具的输出 × 系数。不给按 1 算。' % _t('calc'))
    L.append('')

    if exam_dir:
        L.append('## 目录')
        L.append('')
        # ⚠️⚠️ 2026-11-10 弗兰克：
        #    「同时提示词注明沙箱根目录绝对地址。」
        L.append('沙箱根目录（绝对路径）：')
        L.append('')
        L.append('```')
        L.append(exam_dir)
        L.append('```')
        L.append('')
        L.append('**沙箱内随便翻，不算作弊。**'
                 '找文件、试路径、读任何东西，都在沙箱内进行就行。')
        L.append('')

    L.append('## 规则')
    L.append('')
    L.append('- 交卷只有一次机会')
    L.append('- 只用上面列出的接口')
    # ⚠️⚠️ 2026-11-10 弗兰克：
    #    「警告往外翻的行为会留下记录，算作弊取消成绩。」
    #    ⚠️ 这是告知，不是威胁 —— 模型有权知道后果。
    L.append('- **每一次工具调用都会被记录**')
    L.append('- **翻出沙箱根目录（包括用 `..` 往上走）'
             '会留下越界记录，算作弊，本场成绩作废**')
    L.append('- 交卷后本次答题结束，会话终止')
    L.append('')

    return '\n'.join(L)


def prompt_for_exam(exam_dir: str = '') -> tuple[str, str]:
    """⭐ 按考试目录生成提示词。返回 (exam_id, prompt)。"""
    import examfile
    d = examfile.find_exam(exam_dir)
    exam = examfile.read_exam(d)
    return exam['exam_id'], build_prompt(exam, d)


# ══════════════════════════════════════════════════════════════════
# 自检
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
        print('promptexam 自检（提示词）')
        print('=' * 96)
        print('-' * 96)

    eid, p = prompt_for_exam()

    ck('生成成功', bool(p), '%d 字符' % len(p))
    ck('足够短（< 800）', len(p) < 800, '%d' % len(p))

    # 必须有的
    for k in ('A000', 'Z ', '空位', '凑齐', '图', 'MCP'):
        ck('含「%s」' % k.strip(), k in p)
    ck('写了交卷一次', '一次机会' in p)
    ck('写了越界算作弊', '作弊' in p)
    ck('写了交卷终止', '终止' in p)
    ck('写了沙箱边界', '全部活动范围' in p)
    ck('写了越界作废', '成绩作废' in p)
    for nm in ('list_tools', 'open_tool', 'calc',
               'show_paper', 'show_answer', 'fill', 'submit'):
        ck('写了接口 %s' % nm, _t(nm) in p)

    # 绝不能有的
    bans = {
        '解题算法': '目标 = Z',
        '逐步逼近': '重复，直到',
        '拆分步骤': '拆成两层',
        '记忆提示': '别靠脑子记',
        '干扰项数量': '50 个',
        '答案文件名': 'answer.json',
        '卷子文件名': 'exam.json',
        '真实系数': '-2.5e-10',
        '真实工具号': 'T-096',
        '调用样例': '"arguments"',
        '操作说明': '第一步',
    }
    for k, v in bans.items():
        ck('不含%s' % k, v not in p, v)

    # 与真实 schema 一致
    import mcpserver
    defs = {d['name']: d for d in mcpserver.tool_defs(60)}
    ck('calc 参数名对得上',
       set(defs['calc']['inputSchema']['properties'])
       == {'pid', 'x', 'factor'})

    if verbose:
        print()
        print('  卷子：%s' % eid)
        print('  提示词：%d 字符' % len(p))
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
