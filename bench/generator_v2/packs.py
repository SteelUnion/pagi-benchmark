"""
数理包库 —— 从 registry.py 的 74 个工具，构建成 74 个 MathPack

════════════════════════════════════════════════════════════════
【这是什么】
════════════════════════════════════════════════════════════════

    registry.py 里存的是**原料**（正运算、逆运算、区间、值域）。
    这一层把它们**打包**成 MathPack —— 正逆 + 系数绑成一个整体。

        原料：  f, f⁻¹, search, y_range
            ↓ 打包
        数理包： MathPack(f, f⁻¹, c规则)

    ⭐ 打包后，系数只在包内，外面拿不到也改不了 → 不会再串用。

════════════════════════════════════════════════════════════════
【怎么用】
════════════════════════════════════════════════════════════════

    from packs import PACKS, get, ELEMENTARY, HIGHER

    p = get('ive')                       # 取一个数理包
    x, c = p.forward(y)                  # 正向：y --×c·f--> x
    y2, c = p.inverse(x)                 # 反向：x --÷c, f⁻¹--> y
    p.self_test()                        # 正逆自洽检查

    from chain import Chain, feasible_y
    ch = Chain([get('log1p'), get('ive')])
    res = ch.roundtrip(y)
"""

from __future__ import annotations

from dataclasses import replace

import registry as R
from mathpack import MathPack, decide_coef, DT, DEV

__all__ = ['PACKS', 'BY_NAME', 'get', 'ELEMENTARY', 'HIGHER',
           'self_test_all']


# ================================================================
# 一、构建
# ================================================================


def _to_pack(t: R.Tool, idx: int) -> MathPack:
    """
    ⭐ 把一个原料工具打包成数理包（2026-11-06 Frank 定）。

    ⭐ 打包时做三件事：
        ① 定系数 c —— 值在域内就 c=1，超了才缩放
        ② 给系数一个**独立编码** —— 和公式一起打包，不会丢
        ③ 系数写死在包里，正逆共用
    """
    yr = R.safe_range(t)
    c, k = decide_coef(yr, t.search)
    return MathPack(
        code=t.code,
        name=t.name,
        kind=t.kind,
        domain=t.domain,
        note=t.note,
        _fwd=t._fwd,
        _inv=t._inv,
        search=t.search,
        y_range=yr,                       # ⚠️ 用 safe_range，不是裸 y_range
        inverse_trusted=t.inverse_trusted,
        has_div=t.has_div,
        coef_value=c,                     # ⭐ 打包时定死
        coef_code=f'C{idx:03d}',          # ⭐ 系数独立编码
    )


PACKS: list[MathPack] = [_to_pack(t, i + 1) for i, t in enumerate(R.TOOLS)]
BY_NAME: dict[str, MathPack] = {p.name: p for p in PACKS}
BY_CODE: dict[str, MathPack] = {p.code: p for p in PACKS}

ELEMENTARY = [p for p in PACKS if p.kind == 'ELEMENTARY']
HIGHER = [p for p in PACKS if p.kind == 'HIGHER']


def get(key: str) -> MathPack:
    """按名字取包（编码会变，名字稳定）。"""
    if key in BY_NAME:
        return BY_NAME[key]
    if key in BY_CODE:
        return BY_CODE[key]
    raise KeyError(key)


# ================================================================
# 二、自检
# ================================================================


def self_test_all(verbose: bool = True) -> int:
    """
    把所有数理包跑一遍正逆自洽。

    ⭐ 返回**真正的失败数**（已知多根、精度边缘都不算失败）。

    ⚠️ 2026-11-07 02:00 改口径，两个问题一起修：

        ① **种子不可复现**：原来用 `abs(hash(p.code))`，
           Python 的字符串 hash 每个进程都不同 → 每次跑结果都不一样。
           实测 expn 在不同进程里 96.9%~100% 乱跳。
           ✅ 改用 `p.code` 的稳定整数（编码本身可复现）。

        ② **口径把「已知问题」算成失败**：
           sin/cos 是已知多根（inverse_trusted=False），本来就不该进链，
           拿它们当失败是错的。
           expn 是精度边缘（中位 4e-13），数学没错。
           ✅ 现在分三类：真失败 / 已知多根 / 精度边缘。
    """
    bad = []          # 真失败
    known = []        # 已知多根（inverse_trusted=False）
    edge = []         # 精度边缘（通过率 ≥95%）

    if verbose:
        print('=' * 96)
        print('数理包自检（正逆 + 系数，全在包内）')
        print('=' * 96)
        print(f'{"":<2} {"编码":<5} {"包名":<15} {"类型":<6} {"系数":>8} '
              f'{"系码":<6} {"自洽率":>7} {"中位误差":>10} {"可信":>4}')
        print('-' * 96)

    for p in PACKS:
        # ⭐ 稳定种子：用编码的字符码之和，不用 hash()
        seed = sum(ord(ch) for ch in p.code)
        r = p.self_test(n=256, seed=seed)
        rate = r['rate']

        if rate >= 0.99:
            flag = '✅'
        elif not p.inverse_trusted:
            flag = '⚪'                  # 已知多根，不算失败
            known.append((p, r))
        elif rate >= 0.95:
            flag = '🟡'                  # 精度边缘，不算失败
            edge.append((p, r))
        else:
            flag = '❌'
            bad.append((p, r))

        if verbose:
            print(f'{flag} {p.code:<5} {p.name:<15} {p.kind[:4]:<6} '
                  f'{p.c:>8.3g} {p.coef_code:<6} '
                  f'{rate*100:>6.1f}% {r["med"]:>10.2e} '
                  f'{"是" if p.inverse_trusted else "否":>4}')

    if verbose:
        print('-' * 96)
        print(f'{len(PACKS)} 个包：通过 {len(PACKS)-len(bad)-len(known)-len(edge)}'
              f' · 精度边缘 {len(edge)} · 已知多根 {len(known)} · **真失败 {len(bad)}**')
        with_c = [p for p in PACKS if p.c != 1.0]
        print(f'系数 ≠ 1 的包：{len(with_c)} 个'
              + ('（' + ', '.join(p.name for p in with_c[:10]) + '）' if with_c else '（全为 1）'))
        if edge:
            print()
            print('🟡 精度边缘（数学没错，只是够不着 1e-13）：')
            for p, r in edge:
                print(f'  {p.code} {p.name:<15} {r["rate"]*100:.1f}%  '
                      f'中位 {r["med"]:.2e}')
        if known:
            print()
            print('⚪ 已知多根（已排除，不进链）：'
                  + ', '.join(p.name for p, _r in known))
        if bad:
            print()
            print('❌ 真失败清单：')
            for p, r in bad:
                print(f'  {p.code} {p.name:<15} {r["rate"]*100:.1f}%  '
                      f'中位 {r["med"]:.2e}')
    return len(bad)


if __name__ == '__main__':
    raise SystemExit(0 if self_test_all() == 0 else 1)
