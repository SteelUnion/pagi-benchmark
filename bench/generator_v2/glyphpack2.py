"""
glyphpack2 —— 图案告知（科学计数法版）

════════════════════════════════════════════════════════════════
【为什么要重做】（弗兰克 2026-11-09）
════════════════════════════════════════════════════════════════

    「这种很多无效值的情况，可以改成科学计数法 10 的次方 * 具体值。
      换一种制图方法，或者把所有的都改成科学计数法，
      只对真实值位数做精度限制。10 的几次方我们可以直接给，
      具体有效值，就请 llm 自己数，这样我们需要展示的位数更少了。」

    ⚠️ 旧做法的问题：
        x 的跨度实测 33 个数量级（最小 2.4e-32，最大 52.7）
        按「整数位 + 小数位」逐位画，2.4e-32 需要 31 排全是 0
        模型看到一长串 0，而且 12 位格式化还会把它压成 0.000000000000

    ⇒ 改成：一个数 = 尾数 × 10^指数
        指数 + 正负号  →  直接印在图上（不用数）
        尾数           →  画成图案（模型自己数）

════════════════════════════════════════════════════════════════
【位数怎么定】（实测）
════════════════════════════════════════════════════════════════

    实测：x 截断到 N 位有效数字后，反推出的 y 相对误差

        N=10 → y 最大 3.11e-08
        N=12 → y 最大 1.67e-09
        N=14 → y 最大 4.72e-11      ← 选这个
        N=16 → y 最大 1.15e-10      （浮点舍入反而略差）

    ⚠️ 选项的往返容差是 1e-9。
       弗兰克：「比这个精度要求再严一级」→ 1e-10。
       N=14 给出 4.72e-11，满足。

    ⇒ 尾数保留 14 位有效数字，归一化到 [1, 10)
        = 1 位整数 + 13 位小数 = 14 排

════════════════════════════════════════════════════════════════
【图长什么样】
════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────┐
    │  值 = − × 10^-32                  │   ← 正负号 + 指数，直接给
    │                                  │
    │  ones 个位         ■■■■          │
    │  tenths 十分位     ■■■■          │
    │  ...                             │
    │  hundred-trillionths 万亿分位  ■  │
    │                                  │
    │  这个数是「香蕉」的值              │   ← 提示
    │  每排图案个数 = 该位的数字          │
    └──────────────────────────────────┘

    ⚠️ 图上不出现任何内部编码（word=/glyph=/fig= 那套）。

════════════════════════════════════════════════════════════════
【复用什么】
════════════════════════════════════════════════════════════════

    glyphlib/ 的 Unicode 图案库（534 个符号）
    字体与 SHA-256 记录
    → 制图流程重写，图案来源不变
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

sys.path.insert(0, r'<BENCH>\generator_v2')

import glyphpack
from mathpack import DT, DEV  # noqa: F401

__all__ = ['SIG_DIGITS', 'split_sci', 'draw_sci', 'self_test']

# ⚠️ 尾数保留的有效数字位数（实测 14 位 → y 误差 4.72e-11）
SIG_DIGITS = 14

# 每排最多画几个图案
MAX_PER_ROW = 9

# 画布尺寸（竖屏，弗兰克 2026-11-09 定）
CANVAS_W = 720
CANVAS_H = 1280

# 小数位的中文名（1..13）
_FRAC_ZH = {
    1: '十分位', 2: '百分位', 3: '千分位', 4: '万分位',
    5: '十万分位', 6: '百万分位', 7: '千万分位',
    8: '亿分位', 9: '十亿分位', 10: '百亿分位',
    11: '千亿分位', 12: '万亿分位', 13: '十万亿分位',
}

_FRAC_EN = {
    1: 'tenths', 2: 'hundredths', 3: 'thousandths',
    4: 'ten-thousandths', 5: 'hundred-thousandths',
    6: 'millionths', 7: 'ten-millionths',
    8: 'hundred-millionths', 9: 'billionths',
    10: 'ten-billionths', 11: 'hundred-billionths',
    12: 'trillionths', 13: 'ten-trillionths',
}


# ══════════════════════════════════════════════════════════════════
# 一、拆成科学计数法
# ══════════════════════════════════════════════════════════════════

def split_sci(v: float, sig: int = SIG_DIGITS) -> dict:
    """
    ⭐ 把一个数拆成「符号 + 尾数各位 + 指数」。

    返回 dict:
        sign    : +1 / -1
        exp     : 10 的几次方（尾数归一化到 [1,10)）
        digits  : 尾数的各位数字，长度 = sig（1 位整数 + sig-1 位小数）
        zero    : 这个数是不是 0

    ⚠️ 归一化：0.02438 → 尾数 2.438, exp = -2
             19.1    → 尾数 1.91,  exp = 1
    """
    if v == 0 or not math.isfinite(v):
        return {'sign': 1, 'exp': 0,
                'digits': [0] * sig, 'zero': True}

    sign = -1 if v < 0 else 1
    a = abs(float(v))
    e = int(math.floor(math.log10(a)))

    # 尾数归一化到 [1, 10)
    m = a / (10.0 ** e)
    if m >= 10.0:
        m /= 10.0
        e += 1
    elif m < 1.0:
        m *= 10.0
        e -= 1

    # 取 sig 位有效数字
    s = '%.*e' % (sig - 1, m)                  # 形如 '2.4380000000000e+00'
    mant = s.split('e')[0].replace('.', '')
    digits = [int(ch) for ch in mant][:sig]
    while len(digits) < sig:
        digits.append(0)
    return {'sign': sign, 'exp': e, 'digits': digits, 'zero': False}


# ══════════════════════════════════════════════════════════════════
# 二、画图
# ══════════════════════════════════════════════════════════════════

def draw_sci(v: float, glyph: dict, out_path: str,
             word: str = '', sig: int = SIG_DIGITS,
             cell: int = 40, pad: int = 20) -> dict:
    """
    ⭐ 把一个数画成科学计数法的图。

    ⚠️ 图上出现：
        · 正负号 + 10 的指数（直接印）
        · 尾数的每一位（画成图案，模型数）
        · 每一排的中英文位名
        · 一行提示：「这个数是「词」的值；每排图案个数 = 该位数字」
    ⚠️ 图上【不出现】任何内部编码。
    """
    from PIL import Image, ImageDraw, ImageFont

    sp = split_sci(v, sig)
    digits = sp['digits']

    img = Image.new('L', (CANVAS_W, CANVAS_H), 255)
    dr = ImageDraw.Draw(img)

    # 字体
    sym_font = r'C:\Windows\Fonts\seguisym.ttf'
    zh_font = r'C:\Windows\Fonts\msyh.ttc'
    try:
        gf = ImageFont.truetype(sym_font, cell)
    except OSError:
        gf = ImageFont.load_default()
    try:
        lf = ImageFont.truetype(zh_font, 16)
    except OSError:
        lf = ImageFont.load_default()
    try:
        sf = ImageFont.truetype(zh_font, 22)
    except OSError:
        sf = ImageFont.load_default()

    ch = chr(int(glyph['key'].replace('U+', ''), 16))

    y = pad

    # ── ① 正负号 + 尾数占位 + 指数 ──────────────────
    #
    # ⚠️ 弗兰克：「尾数应该写到括号里，被空格框起来。」
    #    ⇒ 尾数这一格留空，明确告诉读者「这里要靠数图得到」。
    sign_txt = '−' if sp['sign'] < 0 else '+'
    if sp['zero']:
        head = '值 = 0'
    else:
        head = '值 = %s  ( 尾数见图 )  × 10^%d' % (sign_txt, sp['exp'])
    dr.text((pad, y), head, fill=0, font=sf)
    y += 34
    dr.text((pad, y),
            '（正负号与 10 的指数已给出；下方每一排的图案个数 = 该位的数字）',
            fill=0, font=lf)
    y += 34

    # ⚠️ 图案起始列：按【最长位名】的宽度定
    #
    #    弗兰克 2026-11-09：
    #    「按照所做的位数总数，定好，依据最长的决定图案那行缩进多少。
    #      每套图上面位数都应该一致。」
    labels = [_row_label(k, sig) for k in range(sig)]
    maxw = 0
    for lb in labels:
        try:
            w = int(dr.textlength(lb, font=lf))
        except AttributeError:
            w = len(lb) * 12
        maxw = max(maxw, w)
    MARK_X = pad + maxw + 24
    # ⚠️ 一条竖直分隔线，标出「图案从这一列开始」
    dr.line([(MARK_X - 12, y - 6), (MARK_X - 12, CANVAS_H - 110)],
            fill=180, width=1)

    # ── ② 尾数各位 ───────────────────────────────
    # ⚠️ 弗兰克：「确保我们每次生成的图都展示了所有的位数，即便他们是空值。」
    #    ⇒ 固定画 sig 排：1 位整数 + (sig-1) 位小数
    #      某一位是 0 也要画出那一排，标个「—」表示本排 0 个
    n_rows = sig
    avail = CANVAS_H - y - 120 - 26
    row_h = max(26, avail // n_rows)

    for k, d in enumerate(digits):
        label = _row_label(k, sig)
        # 位名：靠左，固定从 pad 开始
        dr.text((pad, y + 4), label, fill=0, font=lf)
        # 图案：一律从 MARK_X 开始，与位名长度无关
        x0 = MARK_X
        for t in range(min(d, MAX_PER_ROW)):
            dr.text((x0 + t * (cell + 6), y - 6), ch, fill=0, font=gf)
        if d == 0:
            dr.text((x0, y + 4), '—', fill=120, font=lf)

        # ⚠️ 个位那排下面，画小数点那一行
        #    弗兰克：「小数点还是应该保留，那行。」
        if k == 0:
            y += row_h
            dp_y = y - row_h // 2
            dr.text((MARK_X - 4, dp_y), '·', fill=0, font=sf)
            dr.text((MARK_X + 14, dp_y),
                    '小数点 decimal point（上方整数，下方小数）',
                    fill=0, font=lf)
            y += 26
            continue
        y += row_h

    # ── ③ 提示 ───────────────────────────────────
    y = CANVAS_H - 90
    if word:
        dr.text((pad, y), '这个数是「%s」的值。' % word, fill=0, font=sf)
        y += 32
    dr.text((pad, y),
            '本图固定展示 %d 位（1 位整数 + %d 位小数），'
            '空位画「—」。每排图案个数 = 该位的数字。'
            % (sig, sig - 1), fill=0, font=lf)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    img.save(out_path)

    with open(out_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    return {
        'path': out_path,
        'sha256': sha,
        'value': float(v),
        'sign': sp['sign'],
        'exp': sp['exp'],
        'digits': digits,
        'sig': sig,
        'glyph': glyph['key'],
        'glyph_name': glyph['name'],
        'word': word,
        'zero': sp['zero'],
    }


def _row_label(k: int, sig: int) -> str:
    """⭐ 第 k 排的位名（中英并列）。k=0 是个位。"""
    if k == 0:
        return 'ones 个位'
    n = k
    zh = _FRAC_ZH.get(n, '小数第%d位' % n)
    en = _FRAC_EN.get(n, '10^-%d' % n)
    return '%s %s' % (en, zh)


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
        print('glyphpack2 自检（科学计数法制图）')
        print('=' * 96)
        print('-' * 96)

    # ① 拆分数
    sp = split_sci(2.438e-32)
    ck('2.438e-32 → exp=-32', sp['exp'] == -32, 'exp=%d' % sp['exp'])
    ck('尾数首位是 2', sp['digits'][0] == 2, str(sp['digits'][:4]))

    sp2 = split_sci(19.1)
    ck('19.1 → exp=1', sp2['exp'] == 1, 'exp=%d' % sp2['exp'])
    ck('尾数首位是 1', sp2['digits'][0] == 1)

    sp3 = split_sci(-0.8497)
    ck('-0.8497 → 负号', sp3['sign'] == -1)
    ck('-0.8497 → exp=-1', sp3['exp'] == -1, 'exp=%d' % sp3['exp'])

    sp4 = split_sci(0.0)
    ck('0 → zero', sp4['zero'] is True)

    # ② 还原
    for v in (2.438e-32, 19.1, -0.8497, 1.0, 6.179e-20, -5.266e1):
        s = split_sci(v)
        if s['zero']:
            continue
        mant = s['digits'][0] + sum(
            d * 10.0 ** (-(i + 1)) for i, d in enumerate(s['digits'][1:]))
        back = s['sign'] * mant * (10.0 ** s['exp'])
        rel = abs(back - v) / abs(v)
        ck('还原 %.6g（误差 %.1e）' % (v, rel), rel < 1e-13)

    # ③ 画图
    pack = glyphpack.build_pack()
    import random
    rng = random.Random(20261109)
    gl = glyphpack.pick_glyph(pack, rng, 2)

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'glyphlib')
    r1 = draw_sci(2.438e-32, gl[0],
                  os.path.join(outdir, '_sci_tiny.png'),
                  word='香蕉')
    ck('画出 2.438e-32', os.path.exists(r1['path']))
    ck('图里有指数', r1['exp'] == -32)

    r2 = draw_sci(-0.8497, gl[1],
                  os.path.join(outdir, '_sci_small.png'),
                  word='铜铃')
    ck('画出 -0.8497', os.path.exists(r2['path']))
    ck('图里有负号', r2['sign'] == -1)

    r3 = draw_sci(19.1, gl[0],
                  os.path.join(outdir, '_sci_big.png'), word='钻石')
    ck('画出 19.1', os.path.exists(r3['path']))

    if verbose:
        print()
        print('  样例图：')
        print('    %s  （2.438e-32，14 排）' % r1['path'])
        print('    %s  （-0.8497）' % r2['path'])
        print('    %s  （19.1）' % r3['path'])
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
