"""
handoff.py —— 出卷器 → DSH 的单向交付

════════════════════════════════════════════════════════════════
【做什么】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「dsh 这边应该跟出卷器握手单向提交一下目录，出卷器根据本次
      出卷的时间戳，自动为那边打开新会话，然后需要的图像全部
      按照要求自动化生成到那边目录里。」

    ⇒ 单向：出卷器 → DSH，不反向。
       交付内容 = 一个考试目录，结构固定，DSH 那边按路径找。

════════════════════════════════════════════════════════════════
【交付目录结构】
════════════════════════════════════════════════════════════════

    dsh-env\\exams\\<问卷编号>\\
        exam.json          考试描述（给 DSH 的握手文件）
        paper.md           题面
        figs\\              60 张图（DSH 沙箱里能读到）
        answer.json        答案（⚠️ 不交给模型，仅判分用）

    ⚠️ 问卷编号就是本次出卷的时间戳，唯一。
    ⚠️ 每次交付都是新目录，绝不覆盖旧的。

════════════════════════════════════════════════════════════════
【握手文件 exam.json】
════════════════════════════════════════════════════════════════

    DSH 侧读这个文件就知道：
        · 这是哪次考试（exam_id）
        · 题目在哪（paper.md）
        · 图在哪（figs/）
        · 几个选项、答案几格
        · 会话应该用哪个 id
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, r'<BENCH>\generator_v2')

import pack as _pack

__all__ = ['exams_root', 'deliver', 'verify_delivery',
           'install_prompt', 'self_test']

# ⚠️ DSH 侧的考试目录根（在 dsh-env 里，跟 DSH_HOME 同一棵树）
ENV_ROOT = r'<BENCH>\dsh-env'

# ⚠️ exam profile 的 patch 文件 —— 提示词写进它的标记区
PATCH_FILE = os.path.join(ENV_ROOT, 'dsh-home', 'profiles', 'exam',
                          'cordis.patch.yml')

PROMPT_BEGIN = '      PROMPT-BEGIN'
PROMPT_END = '      PROMPT-END'


def install_prompt(txt: str, patch_file: str = '') -> bool:
    """
    ⭐ 把考试提示词写进 profile 的 patch，注入成系统提示词。

    ⚠️ 为什么不用 !!js 读文件：patch 里的 !!js 在 ESM 里跑，
       没有 require，读不了文件（实测 ReferenceError: require is not defined）。
       ⇒ 直接改写 YAML 里 system-prompt 的 personaPrefix，最稳。

    ⚠️ 只替换 personaPrefix 块的内容，文件其它部分一概不动。
    """
    p = patch_file or PATCH_FILE
    if not os.path.exists(p):
        return False
    with open(p, encoding='utf-8') as f:
        lines = f.read().split('\n')

    # 找 system-prompt 条目下的 personaPrefix
    try:
        i = next(k for k, ln in enumerate(lines)
                 if ln.strip().startswith('personaPrefix:'))
    except StopIteration:
        return False

    # 块内容 = 从下一行起，所有以 6 空格以上缩进开头（或空行）的行
    indent = ' ' * 6
    j = i + 1
    while j < len(lines):
        ln = lines[j]
        if ln.strip() and not ln.startswith(indent):
            break
        if ln.strip() and ln.startswith(' ' * 4) and \
                not ln.startswith(indent):
            break
        j += 1
    # 去掉块尾的空行（保留下一行的缩进结构）
    while j > i + 1 and not lines[j - 1].strip():
        j -= 1

    block = [indent + ln if ln.strip() else ''
             for ln in txt.rstrip().split('\n')]
    # ⚠️ 去掉结尾的注释行/其它键，只替换块内容
    out = lines[:i + 1] + block + lines[j:]
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    return True


def exams_root() -> str:
    """⭐ 考试交付根目录。"""
    return os.path.join(ENV_ROOT, 'exams')


# ══════════════════════════════════════════════════════════════════
# 一、交付
# ══════════════════════════════════════════════════════════════════

def _abs_fig_paths(paper_txt: str, dst: str) -> str:
    """
    ⭐ 把卷面里的相对图路径换成绝对路径。

    ⚠️ 卷面生成时写的是 `图：figs/001.png`（那时还不知道交付到哪），
       交付时才知道 dst，在这里替换成 `<dst>\\figs\\001.png`。

    ⚠️ 只替换「图：」那一行的路径，别的行不动。
    """
    import re

    def _sub(m):
        fn = m.group(1)
        return '图：%s' % os.path.join(dst, 'figs', fn)

    return re.sub(r'图：figs/([0-9A-Za-z_\-]+\.png)', _sub, paper_txt)


def deliver(src_root: str, dst_root: str = '') -> dict:
    """
    ⭐ 把一个封包单向交付到考试目录（DSH 的会话 cwd）。

    ⚠️ 方向（弗兰克 2026-11-09）：
        「出卷器来注入卷子吧，答题器只提供工具，
          出卷器可以按固定格式写一个具备特征的试卷内容到目录，
          这样门控可以特定放行固定格式操作，无关操作全部拦截。」

        ⇒ 出卷器 → 考试目录。目录就是 DSH 的沙箱根。
          门控按「固定格式」放行：paper.md / exam.json / figs\\*.png

    参数
    ----
    src_root : 出卷器封包目录（packs\\EXAM-...）
    dst_root : 考试根目录（默认 dsh-env\\exams）

    返回 dict：exam_id、dst、session_id、files
    """
    src_root = os.path.abspath(src_root)
    if not os.path.isdir(src_root):
        raise FileNotFoundError('封包目录不存在：%s' % src_root)

    exam_id = os.path.basename(src_root)
    dst = os.path.join(dst_root or exams_root(), exam_id)
    if os.path.isdir(dst):
        raise FileExistsError('已交付过同一编号：%s' % dst)

    os.makedirs(dst, exist_ok=True)

    # ⚠️⚠️ 2026-11-09 改（弗兰克）：
    #   「出题器和答题器应该独立于 dsh 沙箱，这是我们的基准测试层，
    #     dsh 是用来握手做测试的平台，你把答案放他沙箱里他怎么可能不看呢？
    #     这是个超级 bug，我建议把该搬的搬出来，沙箱里只放答题器自身，
    #     就是说他可以调用沙箱内的工具，但只能看到我们展示的题目和工具，
    #     其他东西都不能放沙箱里。」
    #
    #   ⇒ 沙箱目录（模型能看）只放：
    #         paper.md          卷面
    #         exam_public.json  公开清单（无答案）
    #         figs/             图
    #
    #     密钥目录（模型看不到）放：
    #         exam_secret.json  answers + y + 系数
    #
    #   ⚠️ 原来的 exam.json / answer.json 含答案，它在沙箱里，
    #      实测模型 read 了 exam.json，直接抄走 answers。

    # ── ① 卷面（模型读这个）─────────────────
    #
    # ⚠️⚠️ 2026-11-10 弗兰克：
    #    「图片地址提示改文绝对地址，同时在提示词注明
    #      沙箱根目录绝对地址。」
    #
    #    相对路径模型还是会试错，直接换成绝对路径。
    #    卷面里写的是 figs/001.png，这里替换成 <dst>\figs\001.png。
    _paper_src = os.path.join(src_root, 'paper.md')
    with open(_paper_src, encoding='utf-8') as f:
        _ptxt = f.read()
    _ptxt = _abs_fig_paths(_ptxt, dst)
    with open(os.path.join(dst, 'paper.md'), 'w', encoding='utf-8') as f:
        f.write(_ptxt)

    # ── ② 密钥文件（沙箱外）─────────────────
    import examfile as _ef
    sec_dir = os.path.dirname(_ef.secret_path(exam_id))
    # ⚠️ 封包里那份含答案的卷子，文件名是 SECRET_NAME
    sec_src = os.path.join(src_root, _ef.SECRET_NAME)
    if not os.path.exists(sec_src):
        sec_src = os.path.join(src_root, _ef.FILE_NAME)   # 兼容旧封包
    if os.path.exists(sec_src):
        os.makedirs(sec_dir, exist_ok=True)
        shutil.copy2(sec_src, os.path.join(sec_dir, _ef.SECRET_NAME))
        # ③ 公开清单（沙箱内，无答案）
        _ef.write_public(_ef.read_exam(sec_src), dst)

    # ── ③ 图（60 张，模型靠 readImage 看）──────
    src_figs = os.path.join(src_root, 'figs')
    dst_figs = os.path.join(dst, 'figs')
    os.makedirs(dst_figs, exist_ok=True)
    n_fig = 0
    if os.path.isdir(src_figs):
        for fn in sorted(os.listdir(src_figs)):
            if fn.endswith('.png'):
                shutil.copy2(os.path.join(src_figs, fn),
                             os.path.join(dst_figs, fn))
                n_fig += 1

    # ── ④ 答案：⚠️【不进沙箱】 ────────────────
    #
    #   弗兰克 2026-11-09：答案放沙箱里，模型必然看。
    #   ⇒ 只写进密钥目录，沙箱里一个字都不留。
    ans_src = os.path.join(src_root, 'answer.json')
    if os.path.exists(ans_src):
        os.makedirs(sec_dir, exist_ok=True)
        shutil.copy2(ans_src, os.path.join(sec_dir, 'answer.json'))

    # ── ⑤ 在公开清单里补交付信息 ─────────────
    man_src = os.path.join(src_root, 'manifest.json')
    man = json.load(open(man_src, encoding='utf-8')) \
        if os.path.exists(man_src) else {}

    ep = os.path.join(dst, _ef.PUBLIC_NAME)
    if os.path.exists(ep):
        d = json.load(open(ep, encoding='utf-8'))
        d['session_id'] = 'exam-%s' % exam_id
        d['paths'] = {'paper': 'paper.md', 'figs': 'figs',
                      'prompt': 'prompt.md'}
        d['n_figs'] = n_fig
        d['delivered'] = True
        d['note'] = '公开清单，不含答案。'
        with open(ep, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=1)

    # ── ⑥ 考试提示词 ──────────────────────────
    #
    #   弗兰克 2026-11-09：「我们要给 llm 注入一段提示词……」
    #
    #   ⚠️ 写两份：
    #       ① 考试目录里一份（留档，模型也能看到）
    #       ② exams\prompt.md（DSH profile 启动时读这份注入系统提示词）
    prompt_src = os.path.join(src_root, 'prompt.md')
    if os.path.exists(prompt_src):
        import promptexam
        d2 = json.load(open(ep, encoding='utf-8')) if os.path.exists(ep) \
            else {}
        txt = promptexam.build_prompt(d2, dst) if d2 else \
            open(prompt_src, encoding='utf-8').read()
        with open(os.path.join(dst, 'prompt.md'), 'w',
                  encoding='utf-8') as f:
            f.write(txt)
        # ② DSH 侧读的那份（考试根目录下）
        root_base = dst_root or exams_root()
        live = os.path.join(root_base, 'prompt.md')
        try:
            with open(live, 'w', encoding='utf-8') as f:
                f.write(txt)
        except OSError:
            pass   # ⚠️ 自定义临时目录时可以不写

        # ③ 写进 profile，注入成系统提示词
        #    ⚠️ 只在正式考试根目录下做，测试目录不动 profile
        if not dst_root:
            install_prompt(txt)

    return {'exam_id': exam_id, 'dst': dst,
            'session_id': 'exam-%s' % exam_id,
            'n_figs': n_fig, 'seed': man.get('seed')}


# ══════════════════════════════════════════════════════════════════
# 二、验证交付
# ══════════════════════════════════════════════════════════════════

def verify_delivery(dst: str) -> dict:
    """
    ⭐ 校验一次交付。

    ⚠️ 除了「该有的在」，还要验「不该有的不在」 ——
       答案文件绝不能出现在沙箱里。
    """
    problems = []
    if not os.path.isdir(dst):
        return {'ok': False, 'problems': ['目录不存在'], 'info': {}}

    import examfile as _ef
    ep = os.path.join(dst, _ef.PUBLIC_NAME)
    if not os.path.exists(ep):
        return {'ok': False,
                'problems': ['%s 缺失' % _ef.PUBLIC_NAME], 'info': {}}
    exam = json.load(open(ep, encoding='utf-8'))

    for fn in ('paper.md', _ef.PUBLIC_NAME):
        if not os.path.exists(os.path.join(dst, fn)):
            problems.append('%s 缺失' % fn)

    # ⚠️⚠️ 沙箱里绝不能出现答案类文件
    for bad in ('exam.json', 'answer.json', 'options.json',
                'manifest.json', _ef.SECRET_NAME):
        if os.path.exists(os.path.join(dst, bad)):
            problems.append('⚠️ 答案类文件出现在沙箱里：%s' % bad)

    # ⚠️ 公开清单里绝不能含答案
    if 'answers' in exam:
        problems.append('⚠️ 公开清单里含 answers')
    for it in exam.get('items', []):
        if 'y' in it:
            problems.append('⚠️ 公开清单里含 y 值')
            break
        if 'c1' in it or 'c2' in it:
            problems.append('⚠️ 公开清单里含系数')
            break

    figs_dir = os.path.join(dst, 'figs')
    n = len([f for f in os.listdir(figs_dir)
             if f.endswith('.png')]) if os.path.isdir(figs_dir) else 0
    if n != exam.get('n_figs', 0):
        problems.append('图数不符：目录 %d，清单 %d'
                        % (n, exam.get('n_figs', 0)))
    if n == 0:
        problems.append('一张图都没有')

    for k in ('exam_id', 'seed', 'A000', 'Z', 'features'):
        if k not in exam:
            problems.append('公开清单缺字段：%s' % k)

    # ⚠️ 密钥文件必须在沙箱外存在
    eid = os.path.basename(dst.rstrip('\\/'))
    if not os.path.exists(_ef.secret_path(eid)):
        problems.append('密钥文件不存在（答卷器会读不到）：%s'
                        % _ef.secret_path(eid))

    return {'ok': not problems, 'problems': problems,
            'info': {'exam_id': exam.get('exam_id'),
                     'session_id': exam.get('session_id'),
                     'n_figs': n}}


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
        print('handoff 自检（单向交付）')
        print('=' * 96)
        print('-' * 96)

    ck('交付根目录路径', exams_root().endswith('exams'))

    # 找最近一个封包
    here = os.path.dirname(os.path.abspath(__file__))
    packs_dir = os.path.join(here, 'packs')
    cands = [d for d in os.listdir(packs_dir)
             if d.startswith('EXAM-') and
             os.path.isdir(os.path.join(packs_dir, d))]
    if not cands:
        ck('有可用封包（先跑一次 pack）', False)
        return fails
    # ⚠️ 按编号里的时间戳取最新（examfile.exam_sort_key），
    #    不按字符串 —— 否则种子大的旧包会被当成「最新」。
    import examfile as _ef
    src = os.path.join(packs_dir, max(cands, key=_ef.exam_sort_key))

    tmp_dst = os.path.join(here, 'packs', '_handoff_test')
    if os.path.isdir(tmp_dst):
        shutil.rmtree(tmp_dst, ignore_errors=True)

    r = deliver(src, tmp_dst)
    ck('交付成功', os.path.isdir(r['dst']), r['exam_id'])
    ck('图数 60', r['n_figs'] == 60, '%d 张' % r['n_figs'])
    ck('有 session_id', bool(r['session_id']), r['session_id'])

    v = verify_delivery(r['dst'])
    ck('交付校验通过', v['ok'], str(v['problems'][:2]))

    # ⚠️ _try_deliver 返回的是异常的【类型】，不是实例
    got = _try_deliver(src, tmp_dst)
    ck('重复交付会拒绝', got is not None and issubclass(got, FileExistsError),
       got.__name__ if got else '没抛错')

    if verbose:
        print()
        print('  源封包：%s' % src)
        print('  交付到：%s' % r['dst'])
        print('  会话 id：%s' % r['session_id'])
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


def _try_deliver(src, dst):
    try:
        deliver(src, dst)
        return None
    except Exception as e:                              # noqa: BLE001
        return type(e)


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
