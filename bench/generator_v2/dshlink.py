"""
dshlink.py —— 出卷器 ↔ 考试专用 DSH 的最小信道

════════════════════════════════════════════════════════════════
【这一层做什么】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「模型先不上，只要能输入进去就行，每条消息在 dsh 上应该有限制。」

    ⇒ 本模块只做一件事：把【题目输入】送进 DSH 的 API 入口，
       验证信道通不通。不调模型，不判分。

════════════════════════════════════════════════════════════════
【通道形态】
════════════════════════════════════════════════════════════════

    dsh --profile exam  是 JSON-RPC over stdio（不是 HTTP，没有端口）。

        stdin  ← 我们写 JSON-RPC 请求
        stdout → dsh 回 JSON-RPC 响应与通知

    ⚠️ stdout 只承载 JSON-RPC 帧，不能混日志。
    ⚠️ 消息长度限制要在发送前检查。

════════════════════════════════════════════════════════════════
【协议方法】（查 dsh-sdk-jsonrpc-server README）
════════════════════════════════════════════════════════════════

    initialize           初始化，指定 provider / model
    session/query        发提示词（高层，支持内容块数组）
    session/prompt       发提示词（低层，排队输入）
    session/wait         等一轮结束
    shutdown             关闭

    ⚠️ session/query 是官方 SDK run() 用的入口，支持内容块数组。
       题面走它。

════════════════════════════════════════════════════════════════
【消息长度】
════════════════════════════════════════════════════════════════

    ⚠️ 一条 JSON-RPC 帧是一行。行太长可能被截断或触发上限。

    弗兰克：「每条消息在 dsh 上应该有限制。」

    ⇒ 本模块先按 MAX_FRAME_BYTES 检查，超了就报错（不静默截断）。
       实测出来的上限写进 MANIFEST 的 notes。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, r'<BENCH>\generator_v2')

__all__ = ['DSH', 'MAX_FRAME_BYTES', 'check_frame', 'self_test']

# ⚠️ 单帧字节上限。先给一个保守值，实测后再改。
MAX_FRAME_BYTES = 256 * 1024

ENV_ROOT = r'<BENCH>\dsh-env'
NODE = r'C:\Program Files\nodejs\node.exe'
CLI_JS = os.path.join(ENV_ROOT, 'node_modules',
                      '@deepseek-ai', 'dsh', 'lib', 'bin.js')
DSH_HOME = os.path.join(ENV_ROOT, 'dsh-home')


# ══════════════════════════════════════════════════════════════════
# 一、帧检查
# ══════════════════════════════════════════════════════════════════

def check_frame(obj) -> tuple[bool, int, str]:
    """
    ⭐ 检查一帧能不能发。

    返回 (ok, n_bytes, why)
    """
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        return (False, 0, '序列化失败：%s' % e)
    n = len(s.encode('utf-8'))
    if n > MAX_FRAME_BYTES:
        return (False, n, '帧 %d 字节，超上限 %d' % (n, MAX_FRAME_BYTES))
    # ⚠️ json.dumps 会把 \n 转义成 \\n，所以序列化结果里不可能有裸换行。
    #    这一条不需要检查（原来查了但永远不触发）。
    return (True, n, '')


# ══════════════════════════════════════════════════════════════════
# 二、DSH 连接
# ══════════════════════════════════════════════════════════════════

class DSH:
    """
    ⭐ 一个考试用的 DSH 进程（JSON-RPC over stdio）。

    ⚠️ 独立 DSH_HOME（dsh-env\\dsh-home），不碰本机的 .dsh
    ⚠️ 不启动模型时，发 initialize 可能就够验证信道了
    """

    def __init__(self, profile: str = 'exam', cwd: str = '',
                 verbose: bool = False):
        self.profile = profile
        self.cwd = cwd or os.getcwd()
        self.verbose = verbose
        self.proc = None
        self._buf = []
        self._lines = []
        self._lock = threading.Lock()
        self._reader = None

    # ── 启动 ──────────────────────────────────────
    def start(self) -> None:
        env = dict(os.environ)
        env['DSH_HOME'] = DSH_HOME
        env['DSH_PROFILE'] = self.profile
        # ⚠️ 考试环境不带 API key。
        #    子进程会继承用户级 DEEPSEEK_API_KEY —— 不清掉的话，
        #    「只测信道」会变成真的调模型、真的花钱。
        #    要测模型时由调用方显式传 key 进来（见 start_with_key）。
        for k in ('DEEPSEEK_API_KEY', 'DEEPSEEK_API_BASE',
                  'DSH_API_KEY'):
            env.pop(k, None)
        self.proc = subprocess.Popen(
            [NODE, CLI_JS, '--profile', self.profile],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, cwd=self.cwd,
            text=True, encoding='utf-8', bufsize=1)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        if self.verbose:
            print('[dsh] 已启动 pid=%d' % self.proc.pid)

    def _read_loop(self) -> None:
        for line in self.proc.stdout:
            line = line.rstrip('\n')
            if not line:
                continue
            with self._lock:
                self._lines.append(line)
            if self.verbose:
                print('[dsh<-] %s' % line[:200])

    # ── 发送 ──────────────────────────────────────
    def send(self, obj) -> dict:
        ok, n, why = check_frame(obj)
        if not ok:
            return {'ok': False, 'why': why, 'bytes': n}
        s = json.dumps(obj, ensure_ascii=False)
        try:
            self.proc.stdin.write(s + '\n')
            self.proc.stdin.flush()
        except Exception as e:                        # noqa: BLE001
            return {'ok': False, 'why': '写入失败：%s' % e, 'bytes': n}
        return {'ok': True, 'bytes': n, 'frame': s}

    # ── 收 ────────────────────────────────────────
    def recv(self, timeout: float = 10.0,
             want_id=None) -> dict | None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._lines:
                    line = self._lines.pop(0)
                else:
                    line = None
            if line is None:
                time.sleep(0.05)
                continue
            try:
                obj = json.loads(line)
            except Exception:                          # noqa: BLE001
                continue
            if want_id is not None and obj.get('id') != want_id:
                # 不是我们要的响应，留着（通知也留）
                with self._lock:
                    self._lines.append(line)
                time.sleep(0.05)
                continue
            return obj
        return None

    def drain(self, timeout: float = 2.0) -> list:
        t0 = time.time()
        out = []
        while time.time() - t0 < timeout:
            with self._lock:
                if not self._lines:
                    break
                out.append(self._lines.pop(0))
            time.sleep(0.02)
        return out

    # ── 关 ────────────────────────────────────────
    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception:                              # noqa: BLE001
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:                              # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:                          # noqa: BLE001
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop()


# ══════════════════════════════════════════════════════════════════
# 三、自检
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
        print('dshlink 自检（信道）')
        print('=' * 96)
        print('-' * 96)

    # ① 帧检查
    ok, n, why = check_frame({'a': 1})
    ck('小帧能过', ok and n < 100, '%d 字节' % n)

    big = {'x': 'a' * (MAX_FRAME_BYTES + 10)}
    ok2, n2, why2 = check_frame(big)
    ck('超大帧被拦', not ok2, why2[:40])

    # ② 环境文件
    ck('node 存在', os.path.exists(NODE))
    ck('dsh CLI 存在', os.path.exists(CLI_JS))
    ck('dsh-home 存在', os.path.isdir(DSH_HOME))
    ck('exam profile 存在',
       os.path.isdir(os.path.join(DSH_HOME, 'profiles', 'exam')))

    if verbose:
        print()
        print('  帧上限：%d 字节' % MAX_FRAME_BYTES)
        print('  DSH_HOME：%s' % DSH_HOME)
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
