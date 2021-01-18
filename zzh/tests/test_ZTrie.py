from zzh.fastlib import ZTrie


def test_add():
    trie = ZTrie()
    trie.add("中国人")
    trie.add("中国节")
    assert trie.size() == 5
    assert trie.get("中") == (False, 2)
    assert trie.get("中国") == (False, 2)
    assert trie.get("中国人") == (True, 1)
    assert trie.get("中国节") == (True, 1)


def test_insert():
    trie = ZTrie()
    trie.insert("中国人", counter=3, end=True)
    trie.insert("中国节", counter=100, end=False)
    assert trie.size() == 5
    assert trie.get("中") == (False, 1)
    assert trie.get("中国") == (False, 1)
    assert trie.get("中国人") == (True, 3)
    assert trie.get("中国节") == (False, 100)
    assert trie.get("呵呵") is None


def test_equal():
    t1 = ZTrie()
    t1.insert("中国人", counter=3, end=True)
    t1.insert("中国节", counter=100, end=False)
    t2 = ZTrie()
    t2.insert("中国人", counter=3, end=True)
    t2.insert("中国节", counter=100, end=False)
    assert t1.equal(t1)
    assert t1.equal(t2)
    assert t2.equal(t2)
    assert t2.equal(t1)

    t2.add("中国人")
    assert not t1.equal(t2)
    assert t1.equal(t2, counter=False, end=False)

    t2.add("中国菜")
    assert not t1.equal(t2)


def test_copy():
    t1 = ZTrie()
    t1.add("中国人")
    t1.add("中国节")
    t2 = t1.copy()

    assert t1.equal(t2)

    t2.add("中国人")
    assert not t1.equal(t2)
    assert t1.equal(t2, counter=False, end=False)
    t2.add("中国菜")
    assert not t1.equal(t2)


def test_to_dict():
    t1 = ZTrie()
    t1.add("中国人")
    t1.add("中国节")

    assert t1.to_dict(False) == {'children': 1, 'counter': 1, 'end': False}

    assert t1.to_dict(True) == {
        'end': False,
        'counter': 1,
        'children': {
            '中': {
                'end': False,
                'counter': 2,
                'children': {
                    '国': {
                        'end': False,
                        'counter': 2,
                        'children': {
                            '人': {
                                'end': True,
                                'counter': 1,
                                'children': {}
                            },
                            '节': {
                                'end': True,
                                'counter': 1,
                                'children': {}
                            }
                        }
                    }
                }
            }
        }
    }


def test_subtree():
    trie = ZTrie()
    trie.add("中国人")
    trie.add("中国节")
    assert trie.size() == 5
    assert trie.subtree("中")
    assert trie.subtree("中国")
    assert trie.subtree("中国人")
    assert trie.subtree("中国话") is None


def test_save_load():
    import os
    t1 = ZTrie()

    t1.add("中国人")
    t1.add("中国节")
    t1.save("_t_.csv", sep=',')
    t2 = ZTrie().load("_t_.csv", sep=',')
    os.remove("_t_.csv")
    assert t1.equal(t2)


def test_remove():
    t1 = ZTrie()

    t1.add("中国人")
    t1.add("中国节")
    assert len(list(t1)) == 4
    assert t1.size() == 5
    assert not t1.remove("中国菜")
    assert t1.remove("中国节")
    assert t1.size() == 4
    assert len(list(t1)) == 3
    t1.add("中国结")
    assert t1.remove("中国")
    assert t1.size() == 2
    assert len(list(t1)) == 1


def test_pop():
    t1 = ZTrie()

    t1.add("中国人")
    t1.add("中国节日")
    t1.add("中国节选")
    assert len(list(t1)) == 6
    assert t1.size() == 7

    tt = t1.pop("中国菜")
    assert tt is None

    tt = t1.pop("中国节")

    assert t1.size() == 4
    assert len(list(t1)) == 3
    assert tt.size() == 3


def test_longest():
    t1 = ZTrie()
    t1.add("中国人")
    t1.add("中国节").add("我喜欢你")

    assert t1.longest("", 1) is None
    assert t1.longest("中国", 1) is None
    assert t1.longest("我是中国人中国节", 1) is None

    long = t1.longest("中国人中国节", 1)
    assert long["prefix"] == '中国人'
    assert long["start"] == 0
    assert long["len"] == 3

    long = t1.longest("我是中国人中国节", 2)
    assert long["prefix"] == '中国人'
    assert long["start"] == 2
    assert long["len"] == 3

    long = t1.longest("我中国人中国节我喜欢你", 2)
    assert long["prefix"] == '我喜欢你'
    assert long["start"] == 7
    assert long["len"] == 4


def test_search():
    t1 = ZTrie()
    t1.add("中国人")
    t1.add("中国节").add("我喜欢你").add("中国").add("我喜欢")

    se = list(t1.search("他是中国人我喜欢你"))
    assert len(se) == 4
    assert se[0] == {'start': 2, 'len': 4, 'prefix': '中国'}
    assert se[1] == {'start': 2, 'len': 5, 'prefix': '中国人'}
    assert se[2] == {'start': 5, 'len': 8, 'prefix': '我喜欢'}
    assert se[3] == {'start': 5, 'len': 9, 'prefix': '我喜欢你'}
