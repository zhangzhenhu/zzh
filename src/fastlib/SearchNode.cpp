//
// Created by 张振虎 on 2021/1/15.
//

#include "SearchNode.h"
#include "SearchIterator.h"

//void SearchNode::info()
std::string _ws2s(const std::wstring &w_str) {
    if (w_str.empty()) {
        return "";
    }
    unsigned len = w_str.size() * 4 + 1;
    setlocale(LC_CTYPE, "en_US.UTF-8");
    char *p = new char[len];
    wcstombs(p, w_str.c_str(), len);
    std::string str(p);
    delete[] p;
    return str;
}

bool SearchNode::next(const std::wstring &text) {

    if (this->stop || this->tree == nullptr) {
        return false;
    }

    const ZTrie *cur_ptr = this->tree;
    size_t i = this->end_pos;
    for (; i < text.length(); i++) {

        auto c = text[i];

        auto f_iter = cur_ptr->_children.find(std::wstring() + c);
        if (f_iter != cur_ptr->_children.end()) { // found
            cur_ptr = f_iter->second;
            if (cur_ptr->_end) { // 找到一个尾部标记
                // 记录下当前的匹配位置，下次匹配从这个位置开始
                this->end_pos = i + 1;
                this->tree = cur_ptr;
                this->stop = false;
                return true;
            }
            // 不是尾部，继续寻找
            continue;

        } else { // 没有找到子节点，不用再继续了
            break;
        }
    }
    this->end_pos = i + 1;
    this->stop = true;
    return false;
}


