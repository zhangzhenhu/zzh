//
// Created by 张振虎 on 2021/1/15.
//

#include "SearchIterator.h"
#include "SearchNode.h"


void SearchIterator::init(const std::wstring &_text) {

//    cout << "eere " << _text.length() << endl;
    size_t i = 0;
    for (i = _text.length() - 1;; i--) {
//        cout << " a " << i << endl;
//        return;
        // 依次从每个位置开始扫描
        auto n = SearchNode(i, this->tree);
        if (n.next(_text)) { // 存在匹配项，加到栈里
            this->stack.push(n);
        }
        // 注意 i 是无符号类型，不能成为负值
        if (i == 0) {
            break;

        }
    }
};


py::dict SearchIterator::operator*() {

    if (this->stop || this->tree == nullptr || this->stack.empty()) {
        return py::none();
    }
//    SearchNode n = stack.top();

    auto n = stack.top();
//    n.iter = this;
    py::dict d;
    d["start"] = n.start_pos;
    d["len"] = n.end_pos - n.start_pos;
    d["prefix"] = this->text.substr(n.start_pos, n.end_pos - n.start_pos);
//    cout << "xxxx " << this << " " << &(this->text) << " " << this->text.length() << endl;
    return d;
}

SearchNode SearchIterator::operator->() {
    return stack.top();
}

//SearchNode SearchIterator::value() {
//    return stack.top();
//}

const SearchIterator &SearchIterator::operator++() {
//            std::cout<<"++"<<endl;
    if (stack.empty()) {
        this->stop = true;
        return *this;
    }
    // 弹出来的是复制品，
    auto node = stack.top();
    this->stack.pop();
    if (node.next(this->text)) { // 如果没有匹配了，就去掉
        this->stack.push(node);
    }

    if (this->stack.empty()) {
        this->stop = true;
    }
    return *this;
}


bool SearchIterator::operator==(const SearchIterator &other) const noexcept {
    return this->tree == other.tree && this->stop == other.stop;
}

bool SearchIterator::operator!=(const SearchIterator &other) const noexcept {
    return !(*this == other);
}




