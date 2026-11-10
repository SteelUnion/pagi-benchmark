"""hold.py —— 起考试并保持父进程不死。

⚠️ 2026-11-09 诊断：
    runexam.start() 返回后 Python 进程退出
    → DSH 子进程收到 EOF 跟着退出
    → 主控看到的是它临死前写的 error

所以考试必须由一个【活着的父进程】持有。

════════════════════════════════════════════════════════════════
【2026-11-10 新增：异常终止后续跑】（弗兰克）
════════════════════════════════════════════════════════════════

    「要是能把它挂起接着测就更好了，因为它确实没交卷。」

    ⚠️ 背景：021505 场模型在参数里写了 "x": 2**10（不是合法 JSON），
       整个 turn 报 MALFORMED_RESPONSE 终止。DSH 进程没死，
       session.status 变成 idle，但 hold.py 一直在等进程退出，
       于是干等到超时。

    ⇒ 现在：检测到「turn 结束但不是交卷」时，可以再发一次 query 续跑。

    ⚠️⚠️ 但这事儿必须由参数控制，不能默认开：

       续跑 = 帮模型维持 ⇒ 会把「挂起指标」污染成 0

       所以：
         --resume 0   不续（默认）。指标最干净。
         --resume N   最多续 N 次，且每次都在日志里标出来。

    ⚠️ 续跑记录由门控写进 cheat jsonl，类型是 'error'（不是 'suspended'），
       两者分开统计，不会混淆。
"""
import sys, os, time, json, argparse

sys.path.insert(0, r'<BENCH>\generator_v2')
import runexam


def _last_turn_state(eid: str) -> str:
    """
    ⭐ 读会话日志，返回最后一次 turn/end 的 kind。

    返回 'completed' / 'error' / 'aborted' / '' （还没结束）
    """
    p = os.path.join(runexam.LOG_DIR, 'run-%s.out.log' % eid)
    if not os.path.exists(p):
        return ''
    kind = ''
    try:
        with open(p, encoding='utf-8', errors='replace') as f:
            for ln in f:
                ln = ln.strip()
                if '"turn/end"' not in ln and 'turn/end' not in ln:
                    continue
                try:
                    o = json.loads(ln)
                except Exception:                       # noqa: BLE001
                    continue
                e = (o.get('params') or {}).get('event') or {}
                if e.get('type') == 'turn/end':
                    kind = ((e.get('data') or {}).get('reason') or {}).get('kind') or ''
    except Exception:                                   # noqa: BLE001
        return ''
    return kind


def _did_submit(eid: str) -> bool:
    """⭐ 这场交过卷吗（看会话日志里有没有 mcp__exam__submit 调用）。"""
    p = os.path.join(runexam.LOG_DIR, 'run-%s.out.log' % eid)
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding='utf-8', errors='replace') as f:
            for ln in f:
                if 'mcp__exam__submit' in ln and '"tool/call"' in ln:
                    return True
    except Exception:                                   # noqa: BLE001
        return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', type=int, default=0,
                    help='异常终止后最多续跑几次（默认 0 = 不续）')
    ap.add_argument('--resume-prompt', default='继续。',
                    help='续跑时发的话。越少越好，别帮它想。')
    a = ap.parse_args()

    r = runexam.start()
    eid = r['exam_id']
    print('STARTED ' + eid, flush=True)
    if a.resume:
        print('RESUME_ON max=%d' % a.resume, flush=True)

    stop_file = os.path.join(runexam.LOG_DIR, 'STOP')
    if os.path.exists(stop_file):
        os.remove(stop_file)

    n_resume = 0
    last_state = ''
    idle_since = None

    while True:
        time.sleep(2)

        if os.path.exists(stop_file):
            print('STOP requested', flush=True)
            break

        if r['proc'].poll() is not None:
            print('proc exited code=%s' % r['proc'].returncode, flush=True)
            break

        st = _last_turn_state(eid)

        # ⚠️ 状态刚变成「非正常结束」且没交卷 ⇒ 可能可以续
        if st and st != 'completed' and st != last_state:
            last_state = st
            if st == 'error':
                print('TURN_ENDED error', flush=True)
                idle_since = time.time()

        # ⚠️ error 之后等 3 秒，确认它真的不动了
        if idle_since and time.time() - idle_since > 3:
            if n_resume < a.resume and not _did_submit(eid):
                n_resume += 1
                idle_since = None
                print('RESUME %d' % n_resume, flush=True)
                try:
                    runexam.query(eid, a.resume_prompt)
                except Exception as ex:                 # noqa: BLE001
                    print('RESUME_FAILED %s' % ex, flush=True)
                    idle_since = None
            else:
                idle_since = None
                if not _did_submit(eid):
                    print('NO_SUBMIT_DONE', flush=True)
                    break

    runexam.stop(r['proc'])
    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
