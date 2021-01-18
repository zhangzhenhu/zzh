//
// Created by 张振虎 on 2021/1/15.
//

#ifndef FASTLIB_SEARCHNODE_H
#define FASTLIB_SEARCHNODE_H

#include <string>
#include <utility>
#include "ZTrie.h"

class SearchIterator;

class SearchNode {
    friend class SearchIterator;

public:
//    std::wstring prefix;
    size_t start_pos;
    size_t end_pos;
    const ZTrie *tree;
//    const std::wstring text;

//    std::wstring prefix();

protected:
//    SearchIterator *iter;
    bool stop;


    SearchNode(size_t start, const ZTrie *_tree) {

        this->tree = _tree;
//        this->prefix = prefix;
        this->start_pos = start;
        this->end_pos = start;
        this->stop = false;
//        this->iter = nullptr;
//        this->iter = _iter;
//        cout << "SearchNode int " << &_text << " " << &(this->text) << endl;
//        cout << "start "<<start << " end " <<end_pos << " stop " << stop << endl;
    };

    bool next(const std::wstring &_text);
};


#endif //FASTLIB_SEARCHNODE_H
