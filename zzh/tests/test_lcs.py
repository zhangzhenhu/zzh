from zzh import fastlib


def test_lcs():
    assert fastlib.lcs("中国人", "中国人") == (3, '中国人', 0)
    assert fastlib.lcs("x中国人", "中国人") == (3, '中国人', 1)
    assert fastlib.lcs("中国人x", "中国人") == (3, '中国人', 0)
    assert fastlib.lcs("中国人", "x中国人") == (3, '中国人', 0)
    assert fastlib.lcs("中国人", "中国人x") == (3, '中国人', 0)
    assert fastlib.lcs("中国人x", "中国人b") == (3, '中国人', 0)
    assert fastlib.lcs("x中国人x", "a中国人b") == (3, '中国人', 1)
    assert fastlib.lcs("1中2国3人4", "a中b国c人d") == (1, '中', 1)

    assert fastlib.lcs([1, 2, 3, 4, 5], [2, 3, 4]) == (3, [2, 3, 4], 1)
    assert fastlib.lcs([1, 2, 3, 4, 5], [3, 4, 5]) == (3, [3, 4, 5], 2)
    assert fastlib.lcs([1, 2, 3, 4, 5], [1]) == (1, [1], 0)
    assert fastlib.lcs([1, 2, 3, 4, 5], [1, 2]) == (2, [1, 2], 0)


def test_lcx():
    assert fastlib.lcx("中国人", "中国人") == 3
    assert fastlib.lcx("x中国人", "中国人") == 3
    assert fastlib.lcx("中国人x", "中国人") == 3
    assert fastlib.lcx("中国人", "x中国人") == 3
    assert fastlib.lcx("中国人", "中国人x") == 3
    assert fastlib.lcx("中国人x", "中国人b") == 3
    assert fastlib.lcx("x中国人x", "a中国人b") == 3
    assert fastlib.lcx("1中2国3人4", "a中b国c人d") == 3
    assert fastlib.lcx("1中2国3人4", "中b国c人d") == 3
    assert fastlib.lcx("中2国3人4", "中b国c人d") == 3

    assert fastlib.lcx([1, 2, 3, 4, 5], [2, 3, 4]) == 3
    assert fastlib.lcx([1, 2, 3, 4, 5], [3, 4, 5]) == 3
    assert fastlib.lcx([1, 2, 3, 4, 5], [1]) == 1
    assert fastlib.lcx([1, 2, 3, 4, 5], [1, 2]) == 2


def test_lcx_ex():
    assert fastlib.lcx_ex("中国人", "中国人") == (3, '中国人', [0, 1, 2], [0, 1, 2])
    assert fastlib.lcx_ex("x中国人", "中国人") == (3, '中国人', [1, 2, 3], [0, 1, 2])
    assert fastlib.lcx_ex("中国人x", "中国人") == (3, '中国人', [0, 1, 2], [0, 1, 2])
    assert fastlib.lcx_ex("中国人", "x中国人") == (3, '中国人', [0, 1, 2], [1, 2, 3])
    assert fastlib.lcx_ex("中国人", "中国人x") == (3, '中国人', [0, 1, 2], [0, 1, 2])
    assert fastlib.lcx_ex("中国人x", "中国人b") == (3, '中国人', [0, 1, 2], [0, 1, 2])
    assert fastlib.lcx_ex("x中国人x", "a中国人b") == (3, '中国人', [1, 2, 3], [1, 2, 3])
    assert fastlib.lcx_ex("1中2国3人4", "a中b国c人d") == (3, '中国人', [1, 3, 5], [1, 3, 5])
    assert fastlib.lcx_ex("1中2国3人4", "中b国c人d") == (3, '中国人', [1, 3, 5], [0, 2, 4])
    assert fastlib.lcx_ex("中2国3人4", "中b国c人d") == (3, '中国人', [0, 2, 4], [0, 2, 4])

    assert fastlib.lcx_ex([1, 2, 3, 4, 5], [2, 3, 4]) == (3, [2, 3, 4], [1, 2, 3], [0, 1, 2])
    assert fastlib.lcx_ex([1, 2, 3, 4, 5], [3, 4, 5]) == (3, [3, 4, 5], [2, 3, 4], [0, 1, 2])
    assert fastlib.lcx_ex([1, 2, 3, 4, 5], [1]) == (1, [1, ], [0, ], [0, ])
    assert fastlib.lcx_ex([1, 2, 3, 4, 5], [1, 0, 2, 0, 3, 0, 4, 5]) == (
    5, [1, 2, 3, 4, 5], [0, 1, 2, 3, 4], [0, 2, 4, 6, 7])
