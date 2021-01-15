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

    l = t1.longest("我是中国人中国节", 1)
    assert l[0] == "我"
    assert l[2] == 0

    l = t1.longest("他是中国人中国节", 1)
    assert l == (None, None, None)

    l = t1.longest("我是中国人中国节", 2)
    assert l[0] == "中国人"
    assert l[2] == 2

    l = t1.longest("我是中国中国人中国节", 2)
    assert l[0] == "中国人"
    assert l[2] == 4
