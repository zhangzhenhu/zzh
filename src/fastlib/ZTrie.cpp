//
// Created by 张振虎 on 2021/1/10.
//

#include "ZTrie.h"

#include <utility>


std::wstring bool2pystr(bool v) {
    if (v) { return L"True"; }
    else {
        return L"False";
    }
}


ZTrie::ZTrie(std::wstring name, unsigned long counter, bool end) {
    this->name = std::move(name);
    this->counter = counter;
//    this->value = value;
    this->children = std::map<wchar_t, ZTrie *>();
    this->end = end;

//    this->key;
}

std::wstring ZTrie::toString() {
//    std::wstring::f
    return L"ZTre{'name': '"
           + this->name + L"', 'end': "
           + bool2pystr(this->end)
           + L", 'counter': "
           + std::to_wstring(this->counter)
           + L", 'children': "
           + std::to_wstring(this->children.size())
           + L"}";
}

void ZTrie::insert(const std::wstring &word) {
    if (word.empty()) {
        return;
    }
    ZTrie *cur_ptr = this;
    ZTrie *t = nullptr;

    for (auto c :word) {
        std::wcout << c << endl;
        auto iter = cur_ptr->children.find(c);
        if (iter != cur_ptr->children.end()) {
            // found
            cur_ptr = iter->second;
            cur_ptr->counter += 1;

            continue;

        } else {
            // not found
            t = new ZTrie(std::wstring() += c, 1);
            cur_ptr->children[c] = t;
            cur_ptr = t;
        }
    }
    cur_ptr->end = true;
}

size_t ZTrie::size() {
    size_t c = 1;
    if (this->children.empty()) {
        return c;
    }
    for (auto ch:this->children) {
        c += ch.second->size();
    }
    return c;
}

ZTrie *ZTrie::subtree(const std::wstring &word) {
    if (word.empty()) {
        return nullptr;
    }
    ZTrie *cur_ptr = this;

    if (cur_ptr == nullptr) {
        return nullptr;
    }

    for (auto c :word) {

        auto iter = cur_ptr->children.find(c);
        if (iter != cur_ptr->children.end()) {
            // found
            cur_ptr = iter->second;
            continue;

        } else {
            return nullptr;
        }
    }
    return cur_ptr;
}

py::tuple ZTrie::search(const std::wstring &prefix) {

    ZTrie *sub = this->subtree(prefix);
    if (sub != nullptr) {
        return py::make_tuple(sub->end, sub->counter);
    }
    return py::make_tuple(false, 0);
}

py::dict ZTrie::asDict(bool recursion) {
    auto d = py::dict();
    d["end"] = py::bool_(this->end);
    d["counter"] = py::int_(this->counter);
//    d["children_num"] = py::int_(this->children.size());
    if (recursion) {
        auto cc = py::dict();
        for (auto c:this->children) {
//            auto key = std::wstring()+c.first;
            cc[py::cast(c.first)] = c.second->asDict(recursion);
        }
        d["children"]=cc;
    } else {
        d["children"] = py::int_(this->children.size());
    }


    return d;
}

//unsigned long ZTrie::counter(const std::wstring &prefix) {
//
//    ZTrie *sub = this->subtree(prefix);
//    if (sub != nullptr) {
//        return sub->counter;
//    }
//    return 0;
//}

void ZTrie::clear() {

    for (auto iter :this->children) {
        delete iter.second;
    }
    this->children.clear();
}

py::iterator ZTrie::iter() {

//    return py::make_iterator(this->children.begin(), this->children.end());
//    std::cout<<"dfdf"<<endl;
//    for(auto i =this->iter_begin();i!=this->iter_end();++i){
//        std::cout<<*i<<endl;
//    }
    return py::make_iterator(this->iter_begin(), this->iter_end());
}


ZTrie::~ZTrie() {

    this->clear();
}

//
//py::iterater iter(const std::wstring &prefix, ZTrie *root) {
//    if (root == nullptr) {
//        return nullptr;
//    }
//    if (root->end) {
//        return root;
//    }
//    for (auto i :root->children) {
//        return iter(prefix + i.first, i.second);
//    }
//
//}


//void ZTrie::add_word(const std::wstring &word) {
//    if (word.empty()) {
//        return;
//    }
//    ZTrie *cur_ptr = this->root;
//    ZTrie *t = nullptr;
//
//    for (auto c :word) {
//        std::wcout << c << endl;
//        auto iter = cur_ptr->children.find(c);
//        if (iter != cur_ptr->children.end()) {
//            // found
//            cur_ptr = iter->second;
//            cur_ptr->counter += 1;
//
//            continue;
//
//        } else {
//            // not found
//            t = new ZTrie(std::wstring() += c, 1);
//            cur_ptr->children[c] = t;
//            cur_ptr = t;
//        }
//    }
//    cur_ptr->end = true;
////    cur_ptr->value = value;
//}
//
//ZTrie *ZTrie::get(const std::wstring &word) {
//    if (word.empty()) {
//        return nullptr;
//    }
//    ZTrie *cur_ptr = this->root;
//
//    if (cur_ptr == nullptr) {
//        return nullptr;
//    }
//
//    for (auto c :word) {
//
//        auto iter = cur_ptr->children.find(c);
//        if (iter != cur_ptr->children.end()) {
//            // found
//            cur_ptr = iter->second;
//            continue;
//
//        } else {
//            return nullptr;
//        }
//    }
//    return cur_ptr;
//}
