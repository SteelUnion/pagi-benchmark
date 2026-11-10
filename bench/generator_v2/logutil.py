"""
logutil —— 统一日志：所有测试的中间过程和结果，每次一个文件

════════════════════════════════════════════════════════════════
【为什么要有这个】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「改一下代码，默认把所有测试的中间过程和结果写进日志。
      每次一个文件。」

    ⚠️ 我被骂了几轮才明白：不存过程 = 每次你想看细节我都得重跑。
       跑一次二十分钟，你就得再等二十分钟。

    ⇒ 从现在起，任何测试：
         ① 开头写一行 head（时间、种子、参数）
         ② 每个样本写一行（全部中间量 + 成败 + 原因）
         ③ 结尾写一行 summary

════════════════════════════════════════════════════════════════
【用法】
════════════════════════════════════════════════════════════════

    import logutil
    lg = logutil.open_log("test_name")        # 自动建 trace/<名字>_<时间>.jsonl
    lg.head({"seed": 123, "pool": 58})
    lg.rec({"f1":..., "ok": True, ...})
    lg.summary({"n": 1000, "ok": 900})
    lg.close()                                # 或者 with 语句
"""

from __future__ import annotations

import json
import os
import time

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace")


class Log:
    """⭐ 一次测试一个文件。每行一条 JSON。"""

    __slots__ = ("path", "_f", "n", "t0")

    def __init__(self, name: str):
        os.makedirs(TRACE_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(TRACE_DIR, "%s_%s.jsonl" % (name, ts))
        self._f = open(self.path, "w", encoding="utf-8")
        self.n = 0
        self.t0 = time.time()

    def head(self, **kw):
        rec = {"type": "head", "time": time.time(), "name": self.path}
        rec.update(kw)
        self._w(rec)

    def rec(self, **kw):
        """⭐ 一条样本记录。写完全部中间量，不要只写结果。"""
        kw.setdefault("t", round(time.time() - self.t0, 4))
        kw.setdefault("i", self.n)
        self.n += 1
        self._w(kw)

    def summary(self, **kw):
        rec = {"type": "summary", "n": self.n,
               "elapsed": round(time.time() - self.t0, 3)}
        rec.update(kw)
        self._w(rec)

    def _w(self, rec):
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def open_log(name: str) -> Log:
    """⭐ 开一个日志文件。名字自己定（如 'opttest'、'enum'）。"""
    return Log(name)


# ================================================================
# 自检
# ================================================================

def self_test(verbose: bool = True) -> int:
    fails = 0

    def ck(n, c, e=""):
        nonlocal fails
        if verbose:
            print(("  ✅ " if c else "  ❌ ") + n + (("   " + e) if e else ""))
        if not c:
            fails += 1

    if verbose:
        print("=" * 96)
        print("logutil 自检")
        print("=" * 96)

    with open_log("_selftest") as lg:
        lg.head(seed=1, note="自检")
        for i in range(5):
            lg.rec(a=i, b=i * 2, ok=(i % 2 == 0))
        lg.summary(n=5)
        path = lg.path

    ck("文件已建", os.path.exists(path), path)
    lines = open(path, encoding="utf-8").read().strip().split("\n")
    ck("行数 = 1 head + 5 rec + 1 summary = 7", len(lines) == 7,
       "%d 行" % len(lines))
    ck("第一条是 head", json.loads(lines[0])["type"] == "head")
    ck("最后一条是 summary", json.loads(lines[-1])["type"] == "summary")
    ck("中间是样本记录", json.loads(lines[1]).get("a") == 0)

    if verbose:
        print("-" * 96)
        print("失败 %d 项" % fails)
    return fails


if __name__ == "__main__":
    raise SystemExit(0 if self_test() == 0 else 1)
