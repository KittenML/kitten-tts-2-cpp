# A pure-Python port of mojimoji 0.0.13 (https://github.com/studio-ousia/mojimoji, Apache License 2.0,
# Copyright 2013 Studio Ousia): the same tables and the same rules as its Cython zen_to_han and han_to_zen, so the
# Japanese front end needs no compiled package.
"""Full-width <-> half-width conversion of ASCII, digits and katakana."""

_ASCII_ZENKAKU = 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿‘｛｜｝～\u3000＼'
_ASCII_HANKAKU = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~ \\'
_KANA_ZENKAKU = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンァィゥェォッャュョ。、・゛゜「」ー'
_KANA_HANKAKU = 'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝｧｨｩｪｫｯｬｭｮ｡､･ﾞﾟ｢｣ｰ'
_DIGIT_ZENKAKU = '０１２３４５６７８９'
_DIGIT_HANKAKU = '0123456789'
_KANA_TEN = 'ガギグゲゴザジズゼゾダヂヅデドバビブベボヴ'         # voiced katakana and their half-width bases (+ U+FF9E)
_KANA_TEN_BASE = 'ｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾊﾋﾌﾍﾎｳ'
_KANA_MARU = 'パピプペポ'         # semi-voiced katakana and their half-width bases (+ U+FF9F)
_KANA_MARU_BASE = 'ﾊﾋﾌﾍﾎ'

_ascii_zh, _ascii_hz = dict(zip(_ASCII_ZENKAKU, _ASCII_HANKAKU)), dict(zip(_ASCII_HANKAKU, _ASCII_ZENKAKU))
_digit_zh, _digit_hz = dict(zip(_DIGIT_ZENKAKU, _DIGIT_HANKAKU)), dict(zip(_DIGIT_HANKAKU, _DIGIT_ZENKAKU))
_kana_zh, _kana_hz = dict(zip(_KANA_ZENKAKU, _KANA_HANKAKU)), dict(zip(_KANA_HANKAKU, _KANA_ZENKAKU))
_ten_zh, _ten_hz = dict(zip(_KANA_TEN, _KANA_TEN_BASE)), dict(zip(_KANA_TEN_BASE, _KANA_TEN))
_maru_zh, _maru_hz = dict(zip(_KANA_MARU, _KANA_MARU_BASE)), dict(zip(_KANA_MARU_BASE, _KANA_MARU))
_DAKUTEN, _HANDAKUTEN = "\uff9e", "\uff9f"          # half-width voiced / semi-voiced marks


def zen_to_han(text, ascii=True, digit=True, kana=True):
    """Full-width -> half-width."""
    out = []
    for c in text:
        if ascii and c in _ascii_zh:
            out.append(_ascii_zh[c])
        elif digit and c in _digit_zh:
            out.append(_digit_zh[c])
        elif kana and c in _kana_zh:
            out.append(_kana_zh[c])
        elif kana and c in _ten_zh:
            out += [_ten_zh[c], _DAKUTEN]
        elif kana and c in _maru_zh:
            out += [_maru_zh[c], _HANDAKUTEN]
        else:
            out.append(c)
    return "".join(out)


def han_to_zen(text, ascii=True, digit=True, kana=True):
    """Half-width -> full-width; a half-width (semi-)voiced mark joins the katakana before it."""
    out = []
    prev = " "
    for c in text:
        if ascii and c in _ascii_hz:
            out.append(_ascii_hz[c])
        elif digit and c in _digit_hz:
            out.append(_digit_hz[c])
        elif kana and c == _DAKUTEN and prev in _ten_hz:
            out[-1] = _ten_hz[prev]
        elif kana and c == _HANDAKUTEN and prev in _maru_hz:
            out[-1] = _maru_hz[prev]
        elif kana and c in _kana_hz:
            out.append(_kana_hz[c])
        else:
            out.append(c)
        prev = c
    return "".join(out)
