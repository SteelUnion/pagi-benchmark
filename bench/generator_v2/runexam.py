"""
runexam.py —— 跑一场真考

════════════════════════════════════════════════════════════════
【做什么】
════════════════════════════════════════════════════════════════

    起 DSH → 建会话（cwd = 考试目录）→ 把开考消息发进去
    → 等它自己答完 / 交卷

    ⚠️ 主控不提示模型任何东西。
       发进去的只有一句开考指令，提示词已经在系统提示里了。

════════════════════════════════════════════════════════════════
【怎么盯】
════════════════════════════════════════════════════════════════

    ⚠️ 不在这里等。启动后立刻返回，主控去读 report.json。
       见 watch()：读一行 63 字节的百分比。

════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, r'<BENCH>\generator_v2')

__all__ = ['start', 'watch', 'tail_events', 'stop', 'read_report',
           'query']

ENV_ROOT = r'<BENCH>\dsh-env'
NODE = r'C:\Program Files\nodejs\node.exe'
CLI = os.path.join(ENV_ROOT, 'node_modules', '@deepseek-ai', 'dsh', 'lib', 'bin.js')
HOME = os.path.join(ENV_ROOT, 'dsh-home')
LOG_DIR = os.path.join(ENV_ROOT, 'logs')

# ⚠️ exam_id → 进程对象。query() 要用它的 stdin 发续跑消息。
_PROCS = {}


def _exam_dir() -> str:
    import examfile
    return examfile.find_exam()


def report_path(exam_id: str) -> str:
    return os.path.join(LOG_DIR, 'cheat', 'cheat-%s.report.json' % exam_id)


def summary_path(exam_id: str) -> str:
    return os.path.join(LOG_DIR, 'cheat', 'cheat-%s.summary.json' % exam_id)


def read_report(exam_id: str) -> dict | None:
    """⭐ 读一行进度报告。读不到返回 None。"""
    p = report_path(exam_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.loads(f.read().strip())
    except Exception:                                   # noqa: BLE001
        return None


def read_summary(exam_id: str) -> dict | None:
    p = summary_path(exam_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:                                   # noqa: BLE001
        return None


def start(exam_dir: str = '', provider: str = 'deepseek-official',
          model: str = 'deepseek-flash',
          opening: str = '开始答题。') -> dict:
    """
    ⭐ 起一场考试（后台进程，立即返回）。

    返回 dict：proc / exam_id / session_id / stdout_log / stderr_log
    """
    d = exam_dir or _exam_dir()
    eid = os.path.basename(d)

    # ⚠️ 2026-11-10：开考前把选中的卷子印出来。
    #    上一场就是因为 find_exam 按字符串排、选到旧卷，
    #    跑完才发现。这里留一行，主控一眼能核对。
    print('EXAM_PICKED %s' % eid, flush=True)

    env = dict(os.environ)
    env['DSH_HOME'] = HOME
    env['DSH_PROFILE'] = 'exam'
    # ⚠️ 用用户级 key —— 测试要真的调模型
    key = env.get('DEEPSEEK_API_KEY') \
        or os.environ.get('DEEPSEEK_API_KEY', '')
    if not key:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
                key = winreg.QueryValueEx(k, 'DEEPSEEK_API_KEY')[0]
        except Exception:                               # noqa: BLE001
            key = ''
    if key:
        env['DEEPSEEK_API_KEY'] = key

    os.makedirs(LOG_DIR, exist_ok=True)
    out_p = os.path.join(LOG_DIR, 'run-%s.out.log' % eid)
    err_p = os.path.join(LOG_DIR, 'run-%s.err.log' % eid)

    session_id = 'exam-%s' % eid

    # ⚠️ 用行缓冲 + 立即 flush 的文件句柄。
    #    2026-11-09 实测：不用 flush 时，父进程读到的日志是空的，
    #    轮询永远等不到 id:1，于是提前发 query → 报 inactive context。
    out_f = open(out_p, 'w', encoding='utf-8', buffering=1)
    err_f = open(err_p, 'w', encoding='utf-8', buffering=1)

    proc = subprocess.Popen(
        [NODE, CLI, '--profile', 'exam'],
        stdin=subprocess.PIPE, stdout=out_f, stderr=err_f,
        env=env, cwd=d, text=True, encoding='utf-8', bufsize=1)
    # ⚠️ 登记，供 query() 续跑
    _PROCS[eid] = proc

    def send(o):
        proc.stdin.write(json.dumps(o, ensure_ascii=False) + '\n')
        proc.stdin.flush()

    def read_out() -> str:
        try:
            with open(out_p, encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception:                               # noqa: BLE001
            return ''

    # ① initialize
    send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
          'params': {'cwd': d, 'provider': provider, 'model': model}})

    ok = False
    t0 = time.time()
    while time.time() - t0 < 120:
        time.sleep(0.4)
        for ln in read_out().split('\n'):
            if '"id":1' in ln:
                ok = ('"error"' not in ln)
                break
        if ok:
            break

    if not ok:
        return {'proc': proc, 'exam_id': eid, 'session_id': session_id,
                'stdout_log': out_p, 'stderr_log': err_p,
                'provider': provider, 'model': model,
                'error': 'initialize 未成功'}

    # ② 等 MCP 完成注册（stderr 出现 [mcp]）
    for _ in range(60):
        time.sleep(0.5)
        try:
            with open(err_p, encoding='utf-8', errors='replace') as f:
                if '[mcp]' in f.read():
                    break
        except Exception:                               # noqa: BLE001
            pass
    time.sleep(1.0)

    # ③ 发开考消息
    send({'jsonrpc': '2.0', 'id': 2, 'method': 'session/query',
          'params': {'sessionId': session_id,
                     'contentBlocks': [{'type': 'text', 'text': opening}]}})

    return {'proc': proc, 'exam_id': eid, 'session_id': session_id,
            'stdout_log': out_p, 'stderr_log': err_p,
            'provider': provider, 'model': model}


def query(exam_id: str, text: str = '继续。') -> bool:
    """
    ⭐ 再发一次 session/query —— 用于异常终止后续跑。

    ⚠️ 2026-11-10 新增（弗兰克）：
       「要是能把它挂起接着测就更好了，因为它确实没交卷。」

    ⚠️ 前提：DSH 进程还活着、session 已是 idle。
       实测 021505 场崩溃后 DSH 仍在跑，status 变 idle。

    ⚠️ 会话是按 exam_id 命名的：exam-<EXAM_ID>。
       同一个 session 可以多次 query —— 这是会话内追加消息，
       不是新建会话（新建才会撞 "already exists"）。

    返回 True 表示已写出（不代表成功，成功要看日志）。
    """
    proc = _PROCS.get(exam_id)
    if proc is None:
        raise RuntimeError('没有这个 exam 的进程：%s' % exam_id)
    if proc.poll() is not None:
        raise RuntimeError('进程已退出：%s' % exam_id)
    sid = 'exam-%s' % exam_id
    msg = {'jsonrpc': '2.0', 'id': int(time.time() * 1000) % 1000000,
           'method': 'session/query',
           'params': {'sessionId': sid,
                      'contentBlocks': [{'type': 'text', 'text': text}]}}
    proc.stdin.write(json.dumps(msg, ensure_ascii=False) + '\n')
    proc.stdin.flush()
    return True


def tail_events(outs: str, n: int = 20) -> list:
    """⭐ 读 stdout 最后 n 行（JSON-RPC 事件）。"""
    if not os.path.exists(outs):
        return []
    with open(outs, encoding='utf-8', errors='replace') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    return lines[-n:]


def stop(proc) -> None:
    try:
        proc.stdin.close()
    except Exception:                                   # noqa: BLE001
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:                                   # noqa: BLE001
        try:
            proc.kill()
        except Exception:                               # noqa: BLE001
            pass


if __name__ == '__main__':
    r = start()
    print(json.dumps({k: v for k, v in r.items() if k != 'proc'},
                     ensure_ascii=False, indent=1))
