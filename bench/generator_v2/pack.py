"""
pack.py —— 封包：把一道题的全部产物打成一个文件夹

════════════════════════════════════════════════════════════════
【封包里有什么】
════════════════════════════════════════════════════════════════

    pack_<种子>_<时间>/
        paper.md          题面（模型看的）
        answer.json       答案 + 判分信息（不发给模型）
        manifest.json     清单：种子、难度档位、每项指纹、图的 SHA-256
        options.json      60 项的完整结构（供程序读取）
        figs/
            fig_O001_basic.png ...
            fig_O060_basic.png

════════════════════════════════════════════════════════════════
【为什么要有 manifest】
════════════════════════════════════════════════════════════════

    弗兰克：「封包就是把目前的要用的东西都打包，
             同时确认每次测试生成的种子这些数据会被保存。」

    ⇒ 任何一次测试都要能凭 manifest 复现：
         · 种子
         · 难度档位（image / lang / func_render / hint）
         · 函数池状态（ban 名单）
         · 每一项的指纹（sha256）
         · 每张图的指纹（sha256）
         · 逻辑禁用的档位

    ⚠️ 保存清单的三条铁律（论文\基准测试\保存清单.md）：
        ① 能重算的不用存，不能重算的必须存
        ② 数字必须带条件
        ③ 错误必须留痕，不删
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, r'<BENCH>\generator_v2')

import assemble
import config
import examfile
_ef_name = examfile.SECRET_NAME
import figs
import optbase

__all__ = ['pack_paper', 'verify_pack', 'self_test']


# ══════════════════════════════════════════════════════════════════
# 一、封包
# ══════════════════════════════════════════════════════════════════

def exam_id(seed: int = 0) -> str:
    """
    ⭐ 问卷编号：每次考试唯一。

        格式  EXAM-<种子>-<YYMMDD-HHMMSS>
        例    EXAM-20261109-261109-174410

    ⚠️ 弗兰克 2026-11-09：「每次考试要编号，试卷，题目，选项之间
       统一一次编号，重新生成试卷会刷新所有内容。」
    """
    import time as _t
    return 'EXAM-%d-%s' % (int(seed), _t.strftime('%y%m%d-%H%M%S'))


def _sha(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def pack_paper(pp: assemble.Paper, out_root: str = '',
               stamp: str = '') -> str:
    """
    ⭐ 把一道题的全部产物打进一个文件夹。

    返回文件夹路径。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if not out_root:
        out_root = os.path.join(here, 'packs')
    # ⚠️ 目录名 = 问卷编号（唯一，重新生成就换编号）
    name = exam_id(pp.seed) if not stamp else \
        'EXAM-%d-%s' % (int(pp.seed), stamp)
    root = os.path.join(out_root, name)
    if os.path.isdir(root):
        raise FileExistsError('问卷编号重复：%s' % root)
    os.makedirs(root, exist_ok=True)

    # ⚠️ 编号写回考卷 —— 卷子文件与后续交付都靠它
    pp.exam_id = name

    # ── ① 题面 ─────────────────────────────────────
    paper_path = os.path.join(root, 'paper.md')
    with open(paper_path, 'w', encoding='utf-8') as f:
        f.write(assemble.render_paper(pp, show_answer=False))

    # ── ② 答案 ─────────────────────────────────────
    ymap = {op.pid: op.y for op in pp.ops}
    prod = pp.a000
    for p in pp.answers:
        prod *= ymap[p]
    ans = {
        'seed': pp.seed,
        'A000': pp.a000,
        'Z': pp.z,
        'answers': sorted(pp.answers),
        'products': {p: ymap[p] for p in sorted(pp.answers)},
        'product_all': prod,
        'err': abs(prod - pp.z) / abs(pp.z) if pp.z else float('inf'),
        'group_true': pp.group_true,
        'sim_orig': pp.sim_orig,
        'n_try': pp.n_try,
    }
    ans_path = os.path.join(root, 'answer.json')
    with open(ans_path, 'w', encoding='utf-8') as f:
        json.dump(ans, f, ensure_ascii=False, indent=1)

    # ── ③ 60 项的完整结构 ───────────────────────────
    opts = []
    for i, op in enumerate(pp.ops):
        mo = pp.mos[i]
        opts.append({
            'pid': op.pid,
            'is_answer': op.pid in pp.answers,
            'x_word': mo.x_word,
            'x': op.x,
            'y': op.y,
            'y_code': op.codes['Y'],
            'f1': op.f1, 'f1_code': op.f1_code,
            'f2': op.f2, 'f2_code': op.f2_code,
            'c1': op.c1, 'c2': op.c2,
            'formula_zh': mo.text.split('\n')[-1].strip(),
            'err': op.err,
            'ok': op.ok,
            'digest': op.digest,
            # ⚠️ 2026-11-10：图按展示顺序命名，这里取实际生成出来的名字
            #    （pp.figs 里记着 path，兜底才退回旧命名法）
            'fig': (os.path.basename(pp.figs[i]['path'])
                    if i < len(pp.figs) and pp.figs[i]
                    else figs.fig_name(op.pid,
                                       pp.levels.get('image', 'basic'))),
        })
    opt_path = os.path.join(root, 'options.json')
    with open(opt_path, 'w', encoding='utf-8') as f:
        json.dump(opts, f, ensure_ascii=False, indent=1)

    # ── ④ 图 ───────────────────────────────────────
    fig_root = os.path.join(root, 'figs')
    os.makedirs(fig_root, exist_ok=True)
    fig_meta = []
    for op, fg in zip(pp.ops, pp.figs):
        if fg is None:
            continue
        dst = os.path.join(fig_root, os.path.basename(fg['path']))
        with open(fg['path'], 'rb') as src, open(dst, 'wb') as out:
            out.write(src.read())
        fig_meta.append({
            'pid': op.pid,
            'file': os.path.basename(dst),
            'sha256': _sha(dst),
            'value': fg.get('value'),
        })

    # ── ⑤ 卷子文件（含答案，答卷器/判分用）────────
    #
    # ⚠️ 封包里留一份完整卷（含答案）—— 这是出卷器的存档，
    #    它在 packs\ 下，不在沙箱里，模型看不到。
    exam_path = examfile.write_exam(pp, root)

    # ── ⑥ 考试提示词（模型据此知道怎么解题）────────
    prompt_path = os.path.join(root, 'prompt.md')
    import promptexam
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(promptexam.build_prompt(
            examfile.read_exam(exam_path), root))

    # ── ⑦ 清单 ─────────────────────────────────────
    manifest = {
        'exam_id': name,
        'seed': pp.seed,
        'created': name,
        'levels': pp.levels,
        'func_pool': len(optbase.func_pool()),
        'ban_funcs': sorted(optbase.BAN_FUNCS),
        'ban_as_f2': sorted(optbase.BAN_AS_F2),
        'ban_pairs': sorted([list(x) for x in optbase.BAN_PAIRS]),
        'n_options': len(pp.ops),
        'n_answers': len(pp.answers),
        'digits_sci': config.SCI_DIGITS,
        'format': examfile.FORMAT,
        'files': {
            'paper.md': _sha(paper_path),
            'answer.json': _sha(ans_path),
            'options.json': _sha(opt_path),
            _ef_name: _sha(exam_path),
            'prompt.md': _sha(prompt_path),
        },
        'figs': fig_meta,
    }
    man_path = os.path.join(root, 'manifest.json')
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    return root


# ══════════════════════════════════════════════════════════════════
# 二、验证封包
# ══════════════════════════════════════════════════════════════════

def verify_pack(root: str) -> dict:
    """
    ⭐ 校验一个封包：文件在不在、指纹对不对、连乘等不等 Z。

    返回 dict: {'ok': bool, 'problems': [...], 'info': {...}}
    """
    problems = []
    man_path = os.path.join(root, 'manifest.json')
    if not os.path.exists(man_path):
        return {'ok': False, 'problems': ['manifest.json 不存在'], 'info': {}}
    man = json.load(open(man_path, encoding='utf-8'))

    for fn, sha in man['files'].items():
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            problems.append('%s 缺失' % fn)
        elif _sha(p) != sha:
            problems.append('%s 指纹不符' % fn)

    for fg in man['figs']:
        p = os.path.join(root, 'figs', fg['file'])
        if not os.path.exists(p):
            problems.append('图 %s 缺失' % fg['file'])
        elif _sha(p) != fg['sha256']:
            problems.append('图 %s 指纹不符' % fg['file'])

    ans = json.load(open(os.path.join(root, 'answer.json'),
                         encoding='utf-8'))
    if abs(ans['err']) > 1e-9:
        problems.append('连乘误差 %.3e 超容差' % ans['err'])

    return {
        'ok': not problems,
        'problems': problems,
        'info': {'seed': man['seed'], 'levels': man['levels'],
                 'n_options': man['n_options'],
                 'n_figs': len(man['figs']),
                 'err': ans['err']},
    }


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
        print('pack 自检（封包）')
        print('=' * 96)
        print('-' * 96)

    pp = assemble.assemble(seed=20261109)

    # ⚠️ 2026-11-10：自检要能反复跑。
    #    上次的 _test 目录若留着，pack_paper 会按「编号重复」拒掉 ——
    #    那是对的生产行为，不该为了自检去放松它。
    #    ⇒ 自检自己先清掉自己的测试目录。
    _test_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'packs', '_test')
    _prev = os.path.join(_test_root, 'EXAM-20261109-test')
    if os.path.isdir(_prev):
        import shutil
        shutil.rmtree(_prev, ignore_errors=True)

    root = pack_paper(pp, out_root=_test_root, stamp='test')

    ck('封包目录存在', os.path.isdir(root), root)
    for fn in ('paper.md', 'answer.json', 'options.json',
               'manifest.json', _ef_name, 'prompt.md'):
        ck('%s 存在' % fn, os.path.exists(os.path.join(root, fn)))
    ck('figs 目录存在', os.path.isdir(os.path.join(root, 'figs')))

    n_fig = len([f for f in os.listdir(os.path.join(root, 'figs'))
                 if f.endswith('.png')])
    ck('60 张图', n_fig == 60, '%d 张' % n_fig)

    r = verify_pack(root)
    ck('封包校验通过', r['ok'], str(r['problems'][:3]))
    ck('连乘误差 0', r['info'].get('err', 1) < 1e-9,
       '%.1e' % r['info'].get('err', 1))

    # 清单里有种子和难度
    man = json.load(open(os.path.join(root, 'manifest.json'),
                         encoding='utf-8'))
    ck('清单有种子', man['seed'] == 20261109)
    ck('清单有难度档位', 'image' in man['levels'])
    ck('清单有 ban 名单', isinstance(man['ban_funcs'], list))
    ck('清单有图指纹', len(man['figs']) == 60
       and all(len(f['sha256']) == 64 for f in man['figs']))

    # 卷子文件能被答卷器读回，且参数与出卷时一致
    d = examfile.read_exam(os.path.join(root, _ef_name))
    ck('卷子文件可读', d['exam_id'] == man['exam_id'], d['exam_id'])
    ck('卷子 60 项', len(d['items']) == 60)
    ck('卷子参数与出题一致',
       all(d['items'][i]['f1'] == pp.ops[i].f1 and
           d['items'][i]['c2'] == pp.ops[i].c2
           for i in range(60)))

    if verbose:
        print()
        print('  封包：%s' % root)
        print('  大小：%d 文件'
              % sum(len(fs) for _, _, fs in os.walk(root)))
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
