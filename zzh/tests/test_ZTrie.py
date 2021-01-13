from zzh.fastlib import ZTrie


def test_add():
    trie = ZTrie()
    trie.add("中国人")
    trie.add("中国节")
    assert trie.size() == 5
    assert trie.search("中") == (False, 2)
    assert trie.search("中国") == (False, 2)
    assert trie.search("中国人") == (True, 1)
    assert trie.search("中国节") == (True, 1)


def test_insert():
    trie = ZTrie()
    trie.insert("中国人", counter=3, end=True)
    trie.insert("中国节", counter=100, end=False)
    assert trie.size() == 5
    assert trie.search("中") == (False, 1)
    assert trie.search("中国") == (False, 1)
    assert trie.search("中国人") == (True, 3)
    assert trie.search("中国节") == (False, 100)
    assert trie.search("呵呵") is None


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
