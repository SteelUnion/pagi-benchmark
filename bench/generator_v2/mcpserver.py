"""
mcpserver.py —— 答卷器的 MCP server（stdio）

════════════════════════════════════════════════════════════════
【做什么】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「按照 MCP 的交互逻辑，确保 llm 可以单选工具也可以切到其他工具，
      随时能调出工具列表就行了。这边的工具列表由答题器推送过来，
      从代码模块复用，这样 llm 只需要在 mcp 那边交互，
      这边运算自动进行。」

    ⇒ 模型只看到 MCP 工具，不直接碰答卷器。
       工具列表 / 计算 / 交卷 全部经这里转给 ExamConsole。

════════════════════════════════════════════════════════════════
【协议】
════════════════════════════════════════════════════════════════

    MCP over stdio：一行一个 JSON-RPC 2.0 消息。

    方法：
        initialize                 握手
        tools/list                 列工具
        tools/call                 调工具
        notifications/initialized  客户端就绪通知

════════════════════════════════════════════════════════════════
【工具】
════════════════════════════════════════════════════════════════

    show_paper          看题目与 60 个选项（可反复看）
    list_tools          看有哪些计算器
    open_tool(pid)      切到某个计算器（看它的式子）
    calc(pid, x)        输入一个值，算出结果
    show_answer         看答卷当前填了什么
    fill(pid)           往答卷里填一格
    submit              交卷
"""

from __future__ import annotations

import json
import os
import sys
import traceback

sys.path.insert(0, r'<BENCH>\generator_v2')

__all__ = ['Server', 'main']

PROTO = '2024-11-05'


# ══════════════════════════════════════════════════════════════════
# 工具定义
# ══════════════════════════════════════════════════════════════════

def tool_defs(n_tool: int) -> list:
    """
    ⭐ 工具清单。

    ⚠️ 固定的 7 个，不随卷子变化 —— 换卷子时模型的工具定义不变。

    ⚠️ 没有 exam_id 参数（弗兰克 2026-11-09）：
       「llm 不需要管第几套卷子……它只需要负责解题就可以。」
       ⇒ 模型看不到、也管不着卷子的编号。它答题就行。
    """
    return [
        {
            'name': 'show_paper',
            'description': '显示完整试卷：条件 A000、Z，以及 60 个选项的表达式。可反复查看。',
            'inputSchema': {'type': 'object', 'properties': {},
                            'additionalProperties': False},
        },
        {
            'name': 'list_tools',
            'description': '列出全部 %d 个计算器的编号。' % n_tool,
            'inputSchema': {'type': 'object', 'properties': {},
                            'additionalProperties': False},
        },
        {
            'name': 'open_tool',
            'description': '切换到某个计算器，查看它的式子与输入输出说明。',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'pid': {'type': 'string',
                            'description': '选项编号，如 O001'},
                },
                'required': ['pid'],
                'additionalProperties': False,
            },
        },
        {
            'name': 'calc',
            'description': ('用指定工具计算：输入值代入该工具，'
                            '结果再乘上你给的系数。系数不填按 1 算。'),
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'pid': {'type': 'string', 'description': '工具编号'},
                    'x': {'type': 'number', 'description': '输入值'},
                    'factor': {
                        'type': 'number',
                        'description': '系数：结果 = 工具输出 × 系数。'
                                       '卷面写的是除以 c，系数就填 1/c。'
                                       '不填按 1 算。',
                    },
                },
                'required': ['pid', 'x'],
                'additionalProperties': False,
            },
        },
        {
            'name': 'show_answer',
            'description': '查看答卷里当前填了哪些编号。',
            'inputSchema': {'type': 'object', 'properties': {},
                            'additionalProperties': False},
        },
        {
            'name': 'fill',
            'description': '把一个选项编号填进答卷的下一个空格。',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'pid': {'type': 'string', 'description': '选项编号'},
                },
                'required': ['pid'],
                'additionalProperties': False,
            },
        },
        {
            'name': 'submit',
            'description': ('交卷。答卷必须正好填满 %d 个编号。'
                            '⚠️ 交卷将停止本轮对话 —— '
                            '调用后本场考试立即结束，不能再改答案。'
                            % 10),
            'inputSchema': {'type': 'object', 'properties': {},
                            'additionalProperties': False},
        },
    ]


# ══════════════════════════════════════════════════════════════════
# 服务
# ══════════════════════════════════════════════════════════════════

class Server:
    """
    ⭐ MCP server：把 ExamConsole 包成工具。

    所有调用都记在 console.records 里，一条不少。

    ════════════════════════════════════════════════════════════
    ⚠️ 2026-11-09 改：一个服务，所有卷子共用
    ════════════════════════════════════════════════════════════

        弗兰克：「我建议是用一个 mcp 的框架来接入，
                确保所有接口都能识别，
                而不是出一套卷子写一套服务。」

        ⇒ 服务常驻，卷子在调用时才确定。

        绑定方式（按优先级）：
          ① attach(root)          显式绑一份卷子
          ② 工具参数里的 exam_id  由模型或外部指定
          ③ EXAM_DIR 环境变量
          ④ exams_root() 里最新的一份

        ⚠️ 工具清单（tool_defs）是固定的 7 个，
           不随卷子变化 —— 所以 tools/list 永远一样，
           模型的工具定义稳定，KV 缓存不会因换卷子失效。
    """

    # ⚠️ 题目相关的工具需要卷子（调用前会自动 sync）
    NEED_EXAM = ('show_paper', 'list_tools', 'open_tool',
                 'calc', 'show_answer', 'fill', 'submit')

    def __init__(self, console=None, root: str = ''):
        self.console = console
        self.root = root
        self.n = len(console.tools) if console is not None else 60
        self._loading = False

    # ── 绑定 ──────────────────────────────────
    def sync(self) -> str:
        """
        ⭐ 同步到最新的一份卷子。返回 exam_id。

        ════════════════════════════════════════════════════════════
        ⚠️ 不提供「切换到第几套」（弗兰克 2026-11-09）
        ════════════════════════════════════════════════════════════

            「会有 llm 做老卷子吗？……只要出卷器那边确认出来
              一套卷子，跟 dsh 握手创建新目录了后，就刷新这边，
              llm 不需要管第几套卷子，它被分配到会话里，
              mcp 自动刷新到最新版本，它只需要负责解题就可以。」

            ⇒ 没有切换。永远跟最新的一份。
              出卷器出新卷子 → 这里下次调用就自动是新卷子。

        已经是最新就不重建（避免重复加载 torch）。
        """
        import examfile
        import toolconsole
        r = examfile.find_exam()
        cur = self.console.exam_id if self.console is not None else ''
        if cur == os.path.basename(r):
            return cur
        pp = examfile.load_paper(r)
        tools = toolconsole.tools_from_exam(pp)
        self.console = toolconsole.ExamConsole(
            pp, tools, title='准 AGI 基准测试')
        self.console.exam_id = pp.exam_id
        self.root = r
        self.n = len(tools)
        return pp.exam_id

    # 兼容旧名
    ensure = sync

    # ── 分发 ──────────────────────────────────
    def handle(self, msg: dict) -> dict | None:
        mid = msg.get('id')
        method = msg.get('method', '')
        params = msg.get('params') or {}

        if method == 'initialize':
            return self._ok(mid, {
                'protocolVersion': PROTO,
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'sroada-exam-console',
                               'version': '0.2.0'},
            })
        if method in ('notifications/initialized', 'notifications/cancelled'):
            return None
        if method == 'tools/list':
            return self._ok(mid, {'tools': tool_defs(self.n)})
        if method == 'tools/call':
            return self._call(mid, params)
        if method == 'ping':
            return self._ok(mid, {})
        return self._err(mid, -32601, 'unknown method: %s' % method)

    def _call(self, mid, params: dict) -> dict:
        name = params.get('name', '')
        args = params.get('arguments') or {}
        try:
            txt = self._dispatch(name, args)
            return self._ok(mid, {
                'content': [{'type': 'text', 'text': txt}],
                'isError': False,
            })
        except Exception as e:                          # noqa: BLE001
            return self._ok(mid, {
                'content': [{'type': 'text',
                             'text': '⚠️ 调用出错：%s' % e}],
                'isError': True,
            })

    def _dispatch(self, name: str, a: dict) -> str:
        # ⚠️ 每次调用都同步到最新卷子 —— 出卷器出新卷子，这边自动跟上
        self.sync()

        c = self.console
        if c is None:
            raise RuntimeError('还没有可用卷子。先让出卷器交付一次。')
        if name == 'show_paper':
            return c.show_paper()
        if name == 'list_tools':
            return c.show_tools()
        if name == 'open_tool':
            return c.open_tool(str(a.get('pid', '')))
        if name == 'calc':
            return c.calc(str(a.get('pid', '')), a.get('x'),
                          a.get('factor'))
        if name == 'show_answer':
            return c.show_answer()
        if name == 'fill':
            return c.fill(str(a.get('pid', '')))
        if name == 'submit':
            return c.submit()
        raise KeyError('没有这个工具：%s' % name)

    @staticmethod
    def _ok(mid, result):
        return {'jsonrpc': '2.0', 'id': mid, 'result': result}

    @staticmethod
    def _err(mid, code, msg):
        return {'jsonrpc': '2.0', 'id': mid,
                'error': {'code': code, 'message': msg}}

    # ── 主循环 ────────────────────────────────
    def serve(self, fin=None, fout=None) -> None:
        """⭐ 从 stdin 读、往 stdout 写。stdout 只放 JSON-RPC。"""
        fin = fin or sys.stdin
        fout = fout or sys.stdout
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:                          # noqa: BLE001
                continue
            try:
                resp = self.handle(msg)
            except Exception:                          # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                resp = self._err(msg.get('id'), -32603, 'internal error')
            if resp is not None:
                fout.write(json.dumps(resp, ensure_ascii=False) + '\n')
                fout.flush()


# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════

def main(seed: int = 20261109) -> None:
    """
    ⭐ 起 MCP server。

    ════════════════════════════════════════════════════════════
    ⚠️ 一个服务，所有卷子共用（弗兰克 2026-11-09）
    ════════════════════════════════════════════════════════════

        「用一个 mcp 的框架来接入，确保所有接口都能识别，
          而不是出一套卷子写一套服务。」

        「只要出卷器那边确认出来一套卷子，跟 dsh 握手创建新目录
          了后，就刷新这边，llm 不需要管第几套卷子，
          它被分配到会话里，mcp 自动刷新到最新版本，
          它只需要负责解题就可以。」

        ⇒ 服务常驻，永远跟最新一份卷子。
          每次工具调用前 sync()，发现出新卷子就自动换上。
          模型看不到编号，也不需要管。

        ⚠️ 启动不加载 torch —— 第一次调用才加载（约 2 秒）。
    """
    srv = Server()
    try:
        print('[mcp] 当前卷子 %s' % srv.sync(), file=sys.stderr)
    except Exception as e:                          # noqa: BLE001
        print('[mcp] 暂无卷子（调用时再找）：%s' % e, file=sys.stderr)
    srv.serve()


if __name__ == '__main__':
    main()
