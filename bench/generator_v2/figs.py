"""
figs.py —— 题目配图（按难度档位出图）

════════════════════════════════════════════════════════════════
【这一层做什么】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「题目就是保留 x 的变量名，但不输出 x 的值，
      x 应该有 1-60 编号，图片根据编号决定，
      然后题面函数有函数自身和中英文描述两版。」

    ⇒ 题面上不出现 x 的数值，只出现：
         · x 的词（变量名）
         · x 的编号（决定它是哪张图）
       数值只能从图里读出来。

════════════════════════════════════════════════════════════════
【两档难度】
════════════════════════════════════════════════════════════════

    image = 'basic'
        图上直接写数字（读得出，不需要数）
        文件名 fig_O001_basic.png

    image = 'advanced'
        科学计数法 + 逐位画图案（模型要数）
        文件名 fig_O001_advanced.png

    image = 'none'
        不给图，x 的值直接印在题面上

════════════════════════════════════════════════════════════════
【编号规则】
════════════════════════════════════════════════════════════════

    选项代码 O001..O060 已经是编号。
    图片文件名直接用它 —— 模型按编号找图。

    ⚠️ 选项编号不随打乱变化：
       打乱改的是【展示顺序】，选项代码始终跟着那一项本身。
       所以 O001 的图永远叫 fig_O001_*.png。
"""

from __future__ import annotations

import hashlib
import os
import random
import sys

sys.path.insert(0, r'<BENCH>\generator_v2')

import config
import glyphpack
import glyphpack2

__all__ = ['fig_name', 'draw_basic', 'draw_advanced', 'draw_one',
           'draw_all', 'self_test']

FIG_DIR_NAME = 'figs'


# ══════════════════════════════════════════════════════════════════
# 一、文件名
# ══════════════════════════════════════════════════════════════════

def fig_name(pid: str, level: str | None = None) -> str:
    """
    ⭐ 一项的图文件名。

    ⚠️ 用选项代码（O001），不用展示位置 —— 打乱后代码不变。
    """
    lv = level or config.difficulty('image')
    return 'fig_%s_%s.png' % (pid, lv)


def fig_name_by_index(k: int) -> str:
    """
    ⭐ 2026-11-10 起的新取名：按【展示顺序】编号。

    ⚠️ 弗兰克：「可以，但我们打乱了正确选项的顺序，
       图片的顺序也要跟着动。看现在图片怎么生成，那个逻辑很重要。」

        「1用展示顺序吧，别用pid了。」

    ⇒ 卷面第 k 行（从 1 数）的图就叫 001.png、002.png … 060.png。
      文件名不再带 pid，也不再带难度档位 ——
      模型只能靠「第几行」找到图，读不出任何关于选项本身的信息。

    ⚠️ 打乱只影响 pid → 行的映射；
       出图时按 pp.order 走一遍，序号自然跟展示顺序一致。
    """
    return '%03d.png' % int(k)


# ══════════════════════════════════════════════════════════════════
# 二、基础档：图上直接写数字
# ══════════════════════════════════════════════════════════════════

def _fixed_len_num(v: float) -> str:
    """
    ⭐ 把数字规范成【固定长度】的字符串。

    ⚠️ 弗兰克：「把数字格式按照最长的长度固定下来。」

    ⚠️ 统一用科学计数法：
        x 的跨度实测 33 个数量级（2.4e-32 ~ 52.7）
        普通写法长度差太多，字号没法固定。

        长度构成：负号(1) + 整数位(1) + 小数点(1) + 小数位(N-1)
                  + 'e' (1) + 指数符号(1) + 指数位(1~3)
        前段固定 = 1+1+1+(N-1)+1+1 = N+4

        本基准的指数范围：[−33, +2]，最多 2 位指数（含符号）
        ⇒ 总长固定 = N + 4 + 2 = N + 6
    """
    import math
    if v == 0 or not math.isfinite(v):
        mant = '0.' + '0' * (config.SCI_DIGITS - 1)
        return '+' + mant + 'e+00'

    a = abs(float(v))
    e = int(math.floor(math.log10(a)))
    m = a / (10.0 ** e)

    # 尾数取 N 位有效数字
    s = '%.*e' % (config.SCI_DIGITS - 1, m)
    mant = s.split('e')[0]

    esign = '-' if e < 0 else '+'
    eabs = str(abs(e))
    # 指数固定 2 位
    eabs = eabs.rjust(2, '0')

    # 正负号：正数用空位（普通空格在等宽字体里占位）
    sgn = '\u2212' if v < 0 else '+'
    return sgn + mant + 'e' + esign + eabs


def draw_basic(v: float, word: str, out_path: str,
               sign: int = 1) -> dict:
    """
    ⭐ 基础难度：图上直接写数字。

    弗兰克：「图片的基础难度就是图片上直接写数字那种。」
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 720, 1280
    img = Image.new('L', (W, H), 255)
    dr = ImageDraw.Draw(img)

    # ⚠️ 弗兰克：「把数字格式按照最长的长度固定下来。」
    #
    #   ⇒ 先把数字规范化成【固定长度】的字符串：
    #        · 保留 14 位有效数字
    #        · 统一转成科学计数法形式  如 '2.9640700431600e+00'
    #        · 长度固定，字号就能固定
    #
    #   ⚠️ 为什么统一用科学计数法：
    #       x 的跨度实测 33 个数量级（2.4e-32 ~ 52.7）
    #       用普通写法长度差太多，字号没法固定
    txt = _fixed_len_num(v)
    n_char = len(txt)

    # 字号：按【最长情况】固定。最长是 '-9.9999999999999e-999' 共 20 字符，
    #       但指数实际最多 2 位，所以按 19 字符留余量。
    MAX_CHARS = 20          # _fixed_len_num 的输出恒为 20 字符
    MAX_W = W - 80
    # 每字符约 0.6 倍字号 → 字号 = 可用宽度 / (字符数 × 0.6)
    fs = int(min(96, MAX_W / (MAX_CHARS * 0.62)))
    try:
        big = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', fs)
    except OSError:
        big = ImageFont.load_default()
    try:
        mid = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 26)
    except OSError:
        mid = ImageFont.load_default()

    dr.text((40, 200), '值 value', fill=0, font=mid)
    dr.text((40, 250), txt, fill=0, font=big)

    # 画一条参考线标出「数字左边界」，每张图都一样
    dr.line([(40, 380), (40, 420)], fill=160, width=2)

    y = H - 200
    dr.text((40, y), '这个数是「%s」的值。' % word, fill=0, font=mid) if word else None
    if word:
        y += 44
    dr.text((40, y), '（数字已直接给出，无需数数）', fill=0, font=mid)
    y += 36
    dr.text((40, y), '（本图数字统一为 %d 位有效数字，科学计数法）'
            % config.SCI_DIGITS, fill=0, font=mid)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    img.save(out_path)
    with open(out_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {'path': out_path, 'sha256': sha, 'value': float(v),
            'level': 'basic', 'word': word}


# ══════════════════════════════════════════════════════════════════
# 三、进阶档：科学计数法 + 逐位画图案
# ══════════════════════════════════════════════════════════════════

def draw_advanced(v: float, glyph: dict, word: str,
                  out_path: str) -> dict:
    """
    ⭐ 进阶难度：科学计数法 + 逐位画图案（复用 glyphpack2）。

    ⚠️ 尾数位数由 config.SCI_DIGITS 决定（默认 14）。
    """
    return glyphpack2.draw_sci(v, glyph, out_path, word=word,
                               sig=config.SCI_DIGITS)


# ══════════════════════════════════════════════════════════════════
# 四、按档位出图
# ══════════════════════════════════════════════════════════════════

def draw_one(op, mo, out_dir: str, pack: dict, rng: random.Random,
             level: str | None = None, idx: int | None = None) -> dict | None:
    """
    ⭐ 给一个选项出图。

    参数
        idx : 展示序号（从 1 数）。给了就按序号命名（001.png），
              不给就退回旧的 pid 命名。
    返回 dict（含路径与 SHA-256），level='none' 时返回 None。
    """
    lv = level or config.difficulty('image')
    if lv == 'none':
        return None

    fn = fig_name_by_index(idx) if idx else fig_name(op.pid, lv)
    path = os.path.join(out_dir, fn)
    if lv == 'basic':
        return draw_basic(op.x, mo.x_word, path)
    if lv == 'advanced':
        g = glyphpack.pick_glyph(pack, rng, 1)[0]
        return draw_advanced(op.x, g, mo.x_word, path)
    raise NotImplementedError('图片难度 %r 未实现' % lv)


def draw_all(ops: list, mos: list, out_dir: str,
             seed: int = 20261109,
             level: str | None = None,
             order: list | None = None) -> list:
    """
    ⭐ 给全部选项出图。

    ⚠️ 图案随机抽，但同一种子可复现。

    ⚠️ 2026-11-10：加了 order。
       给了 order（展示顺序，pp.order），就按【第几行】给图命名 ——
       第 k 行的图是 001.png、002.png …
       ⇒ 文件名与展示顺序绑定，与 pid 脱钩。

       不给 order 就按旧法（pid 命名），保持旧调用方能跑。
    """
    lv = level or config.difficulty('image')
    if lv == 'none':
        return [None] * len(ops)

    pack = glyphpack.build_pack(verbose=False) if lv == 'advanced' else {}
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    out = []
    for op, mo in zip(ops, mos):
        out.append(draw_one(op, mo, out_dir, pack, rng, lv))

    # ⚠️ 按展示顺序重命名：第 k 行 → 00k.png
    if order:
        for k, i in enumerate(order, start=1):
            old = out[i]
            if old is None:
                continue
            src = old['path']
            dst = os.path.join(out_dir, fig_name_by_index(k))
            if os.path.abspath(src) != os.path.abspath(dst):
                os.replace(src, dst)
            old['path'] = dst
            old['fig_no'] = k
    return out


# ══════════════════════════════════════════════════════════════════
# 五、自检
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
        print('figs 自检（按难度出图）')
        print('=' * 96)
        print('-' * 96)

    # ① 难度开关
    ck('难度校验通过', config.check_difficulty() == [],
       str(config.check_difficulty()))
    ck('image 当前档', config.difficulty('image') == 'advanced',
       config.difficulty('image'))
    ck('SCI_DIGITS = 14', config.SCI_DIGITS == 14)

    # ② 文件名
    ck('文件名用选项代码', fig_name('O001', 'basic') == 'fig_O001_basic.png')
    ck('进阶文件名', fig_name('O060', 'advanced') == 'fig_O060_advanced.png')

    # ③ 出图
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, FIG_DIR_NAME)

    r1 = draw_basic(2.438e-32, '香蕉',
                    os.path.join(outdir, '_test_basic.png'))
    ck('基础档出图', os.path.exists(r1['path']))
    ck('基础档有 SHA', len(r1['sha256']) == 64)

    pack = glyphpack.build_pack(verbose=False)
    g = glyphpack.pick_glyph(pack, random.Random(1), 1)[0]
    r2 = draw_advanced(2.438e-32, g, '香蕉',
                       os.path.join(outdir, '_test_advanced.png'))
    ck('进阶档出图', os.path.exists(r2['path']))
    ck('进阶档有 exp', r2['exp'] == -32)
    ck('进阶档 14 位', len(r2['digits']) == 14)

    # ④ none 档返回 None
    class _Op:
        pid = 'O001'
        x = 1.0
    class _Mo:
        x_word = '香蕉'
    ck('none 档不出图', draw_one(_Op(), _Mo(), outdir, {}, None,
                                 level='none') is None)

    if verbose:
        print()
        print('  基础档：%s' % r1['path'])
        print('  进阶档：%s' % r2['path'])
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
