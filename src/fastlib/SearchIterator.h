//
// Created by 张振虎 on 2021/1/15.
//

#ifndef FASTLIB_SEARCHITERATOR_H
#define FASTLIB_SEARCHITERATOR_H

#include <utility>

#include "ZTrie.h"
#include "SearchNode.h"


class SearchIterator {
    friend class ZTrie;

    friend class SearchNode;

protected:

    bool stop;
    const std::wstring text;
    const ZTrie *tree;
    std::stack<SearchNode> stack;

public:

    py::dict operator*();

    SearchNode operator->();

    const SearchIterator &operator++();

    bool operator==(const SearchIterator &other) const noexcept;

    bool operator!=(const SearchIterator &other) const noexcept;


protected:
    SearchIterator(const ZTrie *_tree, const std::wstring &_text, bool _stop = false):text(_text) {
        this->tree = _tree;
        this->stop = _stop;
        if (!this->stop) {
            this->init(_text);
        }

    };

    void init(const std::wstring &_text);
};

#endif //FASTLIB_SEARCHITERATOR_H
