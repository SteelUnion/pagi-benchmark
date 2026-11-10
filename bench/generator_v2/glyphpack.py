"""
glyphpack —— 基准图案：批量抽出、缓存、按值画图

════════════════════════════════════════════════════════════════
【这一层在做什么】（弗兰克 2026-11-08）
════════════════════════════════════════════════════════════════

    「从字符库批量抛出单位图案后，每个图做图的时候，
      随机抽取一个独立图案，来抽象数字概念。」

    「我建议我们跑的时候先把基准图案生成好，省的你老是问字符库问题。」

    数值 → 文字替换("香蕉") → 图片告知(数图案 = 36)

    图里画的全是同一个符号，但【用哪个符号是随机抽的】，
    所以模型不能靠认符号猜数，必须真去数。

════════════════════════════════════════════════════════════════
【多位数字怎么画】（弗兰克 2026-11-08）
════════════════════════════════════════════════════════════════

    「就一排一个数字，一排九架飞机代表这个数字的第几位数，
      一排几根柱子代表第几位数的数值，有几位数就用几排图片表示，
      底下注明是十进制数字。」

    「最高位是8第一排就8个元素，第几排数字是几就用几个元素，
      底下写清楚，哪一排对应第几位数字。」

    ⇒ 最终规则：【这一位是几，就画几个图案】

        845536（6 位）
            第 1 排（第 6 位）：8 个图案
            第 2 排（第 5 位）：4 个图案
            第 3 排（第 4 位）：5 个图案
            第 4 排（第 3 位）：5 个图案
            第 5 排（第 2 位）：3 个图案
            第 6 排（第 1 位）：6 个图案

    「小数点前小数点后都独立分，小数点我们给出来，具体值让 llm 自己去数。」

    ⇒ 小数点单独印在图里；前后的每一位各占一排。

════════════════════════════════════════════════════════════════
【来源与许可】
════════════════════════════════════════════════════════════════

    Unicode 16.0.0 官方数据，Unicode License V3（免费、无限制、可商用）
    依据：论文\\基准测试\\_参考文献\\图形标准\\图形来源的通用性依据.md

    ⚠️ Unicode 只规定码位与名称，不规定画法；字形由字体实现。
       所以出图必须记录字体路径与 SHA-256（见 glyphlib/README.md）。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys

sys.path.insert(0, r'<BENCH>\generator_v2')

GLYPH_DIR = r'<BENCH>\generator_v2\glyphlib'
INDEX_JSON = os.path.join(GLYPH_DIR, 'glyph_index.json')
# ⚠️ 基准图案缓存放这里，一次生成，反复用
PACK_JSON = os.path.join(GLYPH_DIR, 'glyph_pack.json')

# ⭐ 一页图上最多画几个图案（再密模型数不清）
MAX_PER_ROW = 9


def _load_index() -> list[dict]:
    with open(INDEX_JSON, encoding='utf-8') as f:
        return json.load(f)['entries']


def _usable(e: dict) -> bool:
    """
    ⭐ 这个符号能不能当计数单位？

    ⚠️ 要求：
        ① 单组件（n_comp == 1）—— 多组件的图会让人数错
        ② 有实体填充（fill > 0.05）—— 全空心的线稿在缩小时会糊
        ③ 不是标点/变音符这类不占位的
    """
    try:
        if int(e.get('n_comp', 0)) != 1:
            return False
        if float(e.get('fill', 0.0)) < 0.05:
            return False
        name = str(e.get('name', '')).upper()
        bad = ('COMBINING', 'VARIATION SELECTOR', 'ZERO WIDTH',
               'SOFT HYPHEN', 'MODIFIER LETTER')
        if any(b in name for b in bad):
            return False
        return True
    except (TypeError, ValueError):
        return False


def build_pack(force: bool = False, verbose: bool = True) -> dict:
    """
    ⭐ 一次性把可用图案筛出来存成缓存。

    弗兰克：「先把基准图案生成好，省的你老是问字符库问题。」
    """
    if os.path.exists(PACK_JSON) and not force:
        with open(PACK_JSON, encoding='utf-8') as f:
            return json.load(f)

    entries = _load_index()
    good = [e for e in entries if _usable(e)]

    # 按区块分组，保证随机抽的时候各区块都有机会
    by_block: dict[str, list[dict]] = {}
    for e in good:
        by_block.setdefault(e['block'], []).append(e)

    pack = {
        'source': 'Unicode 16.0.0',
        'license': 'Unicode License V3',
        'n_total': len(entries),
        'n_usable': len(good),
        'blocks': {k: len(v) for k, v in by_block.items()},
        'font': good[0]['font_file'] if good else '',
        'font_sha256': good[0]['font_sha256'] if good else '',
        'glyphs': [{'key': e['key'], 'name': e['name'],
                    'block': e['block'], 'fill': e['fill']}
                   for e in good],
    }
    os.makedirs(GLYPH_DIR, exist_ok=True)
    with open(PACK_JSON, 'w', encoding='utf-8') as f:
        json.dump(pack, f, ensure_ascii=False, indent=1)

    if verbose:
        print('=' * 96)
        print('基准图案缓存')
        print('=' * 96)
        print('来源 %s · 许可 %s' % (pack['source'], pack['license']))
        print('总条目 %d → 可用 %d' % (pack['n_total'], pack['n_usable']))
        print('字体 %s' % pack['font'])
        print('字体 SHA-256 %s' % pack['font_sha256'])
        print('区块分布：')
        for k, v in pack['blocks'].items():
            print('  %-40s %d' % (k, v))
        print('缓存写到 %s' % PACK_JSON)
    return pack


def pick_glyph(pack: dict, rng: random.Random | None = None,
               n: int = 1) -> list[dict]:
    """⭐ 随机抽 n 个互不相同的图案。"""
    rng = rng or random.Random()
    gl = pack['glyphs']
    if n > len(gl):
        raise ValueError('要 %d 个图案，库里只有 %d 个' % (n, len(gl)))
    return rng.sample(gl, n)


# ══════════════════════════════════════════════════════════════════
# 画图
# ══════════════════════════════════════════════════════════════════

def _split_number(v: float, max_frac: int = 12) -> dict:
    """
    ⭐ 把一个数拆成「整数部分各位 + 小数部分各位」。

    ⚠️⚠️ 2026-11-08 踩过：原来写 `('%.12f' % v).rstrip('0')`
        → 845536.0 变成 '845536.000000000000' → rstrip('0') 把
          【整数末尾的 6 也去掉了】→ 变 '845536.' → 再补 .0
          → 多出一排「小数第1位=0」。
    ⚠️ 弗兰克：「小数点后有数字分，没有就不分。」
        ⇒ 先把整数部分和小数部分分开，再各自处理；
          小数部分全是 0（或者本来就没有）→ frac = []，不占排。
    """
    neg = v < 0
    s = '%.12f' % abs(float(v))
    if '.' in s:
        ip, fp = s.split('.')
    else:
        ip, fp = s, ''
    fp = fp.rstrip('0')[:max_frac]          # ⚠️ 只对小数部分去尾零
    return {
        'sign': -1 if neg else 1,
        'int': [int(c) for c in (ip or '0')],
        'frac': [int(c) for c in fp],       # 没有小数 → 空列表
    }


def draw_number(v: float, glyph: dict, out_path: str,
                cell: int = 64, pad: int = 16,
                label: bool = True,
                word_name: str = '', word_code: str = '',
                fig_code: str = '') -> dict:
    """
    ⭐ 把一个数画成图：这一位是几，就画几个图案。

    ⚠️ 弗兰克：「小数点前小数点后都独立分，小数点我们给出来，
       具体值让 llm 自己去数。」

    布局：
        [符号]  第 1 排（第 N 位）：██
                第 2 排（第 N-1 位）：█████
                ⋮
        [小数点]
                第 a 排（十分位）：█
                第 b 排（百分位）：████
                ⋮
        底下注明：十进制

    返回 dict（含文件路径、SHA-256、每个数字用的图案）。
    """
    from PIL import Image, ImageDraw, ImageFont

    parts = _split_number(v)
    digits_int = parts['int']
    digits_frac = parts['frac']
    n_int = len(digits_int)
    n_frac = len(digits_frac)

    # 行数 = 整数位 + 小数位（+ 符号行如果有负号）
    rows = []
    for i, d in enumerate(digits_int):
        place = n_int - i                      # 第几位（从个位=1 数起）
        place_name = {1: '个位', 2: '十位', 3: '百位', 4: '千位',
                      5: '万位', 6: '十万位', 7: '百万位',
                      8: '千万位', 9: '亿位'}.get(place, '第%d位' % place)
        place_en = {1: 'ones', 2: 'tens', 3: 'hundreds', 4: 'thousands',
                    5: 'ten thousands', 6: 'hundred thousands',
                    7: 'millions', 8: 'ten millions', 9: 'billions'
                    }.get(place, '10^%d' % (place - 1))
        rows.append(('int', place, place_name, d,
                     '%s（%s）' % (place_en, place_name)))

    for j, d in enumerate(digits_frac):
        place = j + 1
        place_name = {1: '十分位', 2: '百分位', 3: '千分位',
                      4: '万分位', 5: '十万分位', 6: '百万分位',
                      7: '千万分位', 8: '亿分位', 9: '十亿分位',
                      10: '百亿分位', 11: '千亿分位',
                      12: '万亿分位'}.get(place, '小数第%d位' % place)
        place_en = {1: 'tenths', 2: 'hundredths', 3: 'thousandths',
                    4: 'ten-thousandths', 5: 'hundred-thousandths',
                    6: 'millionths', 7: 'ten-millionths',
                    8: 'hundred-millionths',
                    9: 'billionths', 10: 'ten-billionths',
                    11: 'hundred-billionths',
                    12: 'trillionths'}.get(place, '10^%d' % (-place))
        rows.append(('frac', place, place_name, d,
                     '%s（%s）' % (place_en, place_name)))

    # 每排最多 MAX_PER_ROW 个，超了就这一排折行
    def _n_lines(d):
        return max(1, (d + MAX_PER_ROW - 1) // MAX_PER_ROW)

    per_row = max(_n_lines(r[3]) for r in rows) if rows else 1
    row_h = per_row * (cell + 6) + 10
    label_h = 80 if label else 0
    # ⚠️ 右侧要放得下「第 N 排 = 第 N 位（十万位）」这类标注，
    #    原来只留 200 px，被裁掉了。
    W = pad * 2 + MAX_PER_ROW * (cell + 6) + 380
    H = (pad * 2 + row_h * len(rows) + label_h
         + (60 if parts['sign'] < 0 else 0)
         + (30 if digits_frac else 0))

    img = Image.new('L', (W, H), 255)
    dr = ImageDraw.Draw(img)

    font_path = os.environ.get('GLYPH_FONT', r'C:\Windows\Fonts\seguisym.ttf')
    label_font_path = r'C:\Windows\Fonts\msyh.ttc'
    try:
        gf = ImageFont.truetype(font_path, cell - 12)
    except OSError:
        gf = ImageFont.load_default()
    try:
        lf = ImageFont.truetype(label_font_path, 18)
    except OSError:
        lf = ImageFont.load_default()

    ch = chr(int(glyph['key'].replace('U+', ''), 16))

    y = pad
    if parts['sign'] < 0:
        dr.text((pad, y), '−', fill=0, font=gf)
        y += 40

    for (kind, place, pname, d, text) in rows:
        nl = _n_lines(d)
        for line in range(nl):
            cnt = min(MAX_PER_ROW, d - line * MAX_PER_ROW)
            for k in range(cnt):
                dr.text((pad + k * (cell + 6), y + line * (cell + 6)),
                        ch, fill=0, font=gf)
        dr.text((pad + MAX_PER_ROW * (cell + 6) + 12, y + 8),
                text if d > 0 else (text + '  ← 本排 0 个（空）'),
                fill=0, font=lf)
        y += row_h

        # ⚠️ 小数点画在【整数部分最后一行之后、小数部分之前】。
        #    原来把它放到了循环外面，跑到图最底下去了。
        if kind == 'int' and place == 1 and digits_frac:
            dr.text((pad, y - 6), '.', fill=0, font=gf)
            dr.text((pad + cell + 10, y + 6),
                    '← 小数点（decimal point），其后为小数部分',
                    fill=0, font=lf)
            y += 36

    if label:
        # ⚠️ 底部提示（弗兰克 2026-11-08）：
        #   「数字为小数点前共 位数，小数点后 位数，
        #    这个数字代表香蕉的值（香蕉是随机到的变量，动态生成）」
        word = word_name or '（未命名变量）'
        dr.text((pad, y + 18),
                '小数点前共 %d 位，小数点后 %d 位。这个数字代表「%s」的值。'
                % (n_int, n_frac, word), fill=0, font=lf)
        # ⚠️ 编码（弗兰克：文字随机的时候要有代码，图案也要有代码）
        dr.text((pad, y + 46),
                'word=%s  glyph=%s  fig=%s'
                % (word_code, glyph['key'], fig_code), fill=0, font=lf)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    img.save(out_path)

    with open(out_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    return {
        'path': out_path,
        'sha256': sha,
        'value': float(v),
        'glyph': glyph['key'],
        'glyph_name': glyph['name'],
        'rows': [(t, d) for (_k, _p, _n, d, t) in rows],
        'n_rows': len(rows),
        'font': font_path,
    }


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

    pack = build_pack()
    ck('基准图案有缓存', os.path.exists(PACK_JSON))
    ck('可用图案 > 400', pack['n_usable'] > 400,
       '%d / %d' % (pack['n_usable'], pack['n_total']))
    ck('记录了字体 SHA-256', len(pack['font_sha256']) == 64)

    rng = random.Random(20261108)
    gl = pick_glyph(pack, rng, 2)
    ck('随机抽两个图案', len(gl) == 2 and gl[0]['key'] != gl[1]['key'],
       '%s %s' % (gl[0]['key'], gl[1]['key']))

    # 拆数字
    d = _split_number(845536.0)
    ck('845536 拆成 6 位整数',
       d['int'] == [8, 4, 5, 5, 3, 6] and d['frac'] == [],
       str(d['int']))
    d2 = _split_number(7.99289410981)
    ck('7.99289410981 整数部分是 7', d2['int'] == [7])
    ck('小数部分第一位是 9', d2['frac'][0] == 9)

    # 画图
    out = os.path.join(os.path.dirname(PACK_JSON), '_test_845536.png')
    r = draw_number(845536.0, gl[0], out)
    ck('画出 845536', os.path.exists(out),
       '每排个数 %s' % [c for _t, c in r['rows']])
    ck('每排个数 = 各位数字', [c for _t, c in r['rows']] == [8, 4, 5, 5, 3, 6])
    ck('图有 SHA-256', len(r['sha256']) == 64)

    out2 = os.path.join(os.path.dirname(PACK_JSON), '_test_decimal.png')
    r2 = draw_number(7.99289410981, gl[1], out2)
    ck('画出小数 7.99289410981', os.path.exists(out2),
       '%d 排' % r2['n_rows'])

    if verbose:
        print('样例图：')
        print('  %s  （8,4,5,5,3,6）' % out)
        print('  %s  （7 + 11 位小数）' % out2)
        print('-' * 96)
        print('失败 %d 项' % fails)
    return fails


if __name__ == '__main__':
    raise SystemExit(0 if self_test() == 0 else 1)
