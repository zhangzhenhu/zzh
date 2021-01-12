//
// Created by 张振虎 on 2021/1/10.
//

#include "ZTrie.h"

#include <utility>
#include <iostream>
#include <fstream>
#include <codecvt>
#include <string>


std::wstring bool2str(bool v) {
    if (v) { return L"True"; }
    else {
        return L"False";
    }
}

void split(const std::string &s, std::vector<std::string> &v, const std::string &c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

std::string ws2s_(const std::wstring &wstr) {
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;

    return converterX.to_bytes(wstr);
}

std::string ws2s(const std::wstring &w_str) {
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

std::wstring strtowstr(const std::string &str) {
    setlocale(LC_CTYPE, "en_US.UTF-8");
    const char *cstr = str.c_str();
    std::size_t cstrlen = std::mbstowcs(nullptr, cstr, 0) + 1;
    wchar_t *wcstr = new wchar_t[cstrlen];
    std::wmemset(wcstr, 0, cstrlen);
    std::mbstowcs(wcstr, cstr, cstrlen);
    std::wstring wstr(wcstr);
    delete[]wcstr;
    return wstr;
}

ZTrie::ZTrie(std::wstring
             name, unsigned long
             counter, bool
             end) {
    this->_name = std::move(name);
    this->_counter = counter;
//    this->value = value;
    this->_children = ChildrenMap();
    this->_end = end;

//    this->key;
}

std::wstring ZTrie::toString() {
//    std::wstring::f
    return L"ZTrie{'name': '"
           + this->_name + L"', 'end': "
           + bool2str(this->_end)
           + L", 'counter': "
           + std::to_wstring(this->_counter)
           + L", 'children': "
           + std::to_wstring(this->_children.size())
           + L"}";
}

ZTrie *ZTrie::insert(const std::wstring &word) {
    if (word.empty()) {
        return this;
    }
    ZTrie *cur_ptr = this;
    ZTrie *t = nullptr;

    for (auto c :word) {
        auto key = std::wstring() + c;
        auto iter = cur_ptr->_children.find(key);
        if (iter != cur_ptr->_children.end()) {
            // found
            cur_ptr = iter->second;
            cur_ptr->_counter += 1;

            continue;

        } else {
            // not found
            t = new ZTrie(key, 1);
            cur_ptr->_children[key] = t;
            cur_ptr = t;
        }
    }
    cur_ptr->_end = true;
    return this;
}

ZTrie *ZTrie::add(const std::wstring &word, size_t count, bool end) {

    ZTrie *cur_ptr = this;
    ZTrie *t = nullptr;

    for (auto c :word) {
        auto key = std::wstring() + c;
        auto iter = cur_ptr->_children.find(key);
        if (iter != cur_ptr->_children.end()) {
            // found
            cur_ptr = iter->second;
//            cur_ptr->counter += 1;
            continue;

        } else {
            // not found
            t = new ZTrie(key, 0, false);
            cur_ptr->_children[key] = t;
            cur_ptr = t;
        }
    }
    cur_ptr->_counter = count;
    cur_ptr->_end = end;
    return this;
}

size_t ZTrie::size() {
    size_t c = 1;
    if (this->_children.empty()) {
        return c;
    }
    for (auto ch:this->_children) {
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

        auto iter = cur_ptr->_children.find(std::wstring() + c);
        if (iter != cur_ptr->_children.end()) {
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
        return py::make_tuple(sub->_end, sub->_counter);
    }
    return py::make_tuple(false, 0);
}

py::dict ZTrie::asDict(bool recursion) {
    auto d = py::dict();
    d["end"] = py::bool_(this->_end);
    d["counter"] = py::int_(this->_counter);
//    d["children_num"] = py::int_(this->children.size());
    if (recursion) {
        auto cc = py::dict();
        for (auto c:this->_children) {
//            auto key = std::wstring()+c.first;
            cc[py::cast(c.first)] = c.second->asDict(recursion);
        }
        d["children"] = cc;
    } else {
        d["children"] = py::int_(this->_children.size());
    }


    return d;
}

ZTrie *ZTrie::copy() {
    auto n = new ZTrie(this->_name, this->_counter, this->_end);
    if (this->_children.empty()) {
        return n;
    }
    for (auto i:this->_children) {

        n->_children[i.first] = i.second->copy();
    }
    return n;
}

ZTrie *ZTrie::merge(ZTrie *other) {
    if (this->_name != other->_name) {
        throw std::invalid_argument("Name not equal");
    }
    this->_counter += other->_counter;

    for (auto oc:other->_children) {
        auto tc = this->_children.find(oc.first);
        if (tc != this->_children.end()) { // 找到了
            // 子树进行合并
            tc->second->merge(oc.second);

        } else {//没有找到
            this->_children[oc.first] = oc.second->copy();
        }
    }
    return this;
}

bool ZTrie::equal(ZTrie *other, bool name, bool counter) {
    if (name && this->_name != other->_name) {
        return false;
    }
    if (counter && this->_counter != other->_counter) {
        return false;
    }
    if (this->_children.size() != other->_children.size()) {
        return false;
    }
    for (const auto &oc:other->_children) {
        auto tc = this->_children.find(oc.first);
        if (tc != this->_children.end()) { // 找到了
            // 子树进行比较
            if (!tc->second->equal(oc.second, name, counter)) {
                return false;
            }

        } else {//没有找到
            return false;
        }
    }


    return true;
}

ZTrie *ZTrie::save(const string &filename, const string &separator) {

    std::ofstream out = std::ofstream();

    out.open(filename);

    if (!out.is_open()) {
        return this;
    }
//    const Node node;
    auto it = this->iter_begin();
    auto end = this->iter_end();

    for (; it != end; ++it) {
        auto node = it.value();
        out << ws2s(node.prefix) << " "
            << std::to_string(node.tree->_counter) << " "
            << std::to_string(node.tree->_end)
            << endl;
    }
    out.close();
    return this;
}

ZTrie *ZTrie::load(const string &filename, const string &separator) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        return this;
    }
    string line;
    std::vector<string> v(3);
    while (getline(fin, line)) {

        split(line, v, separator);
        if (v.size() != 3) {
            cerr << "Invalid data " << line << endl;
            continue;
        };
        this->add(strtowstr(v[0]), std::stoul(v[1]), std::stoi(v[3]));
        v.clear();
    }
    return this;
}


ZTrie *ZTrie::clear() {

    for (const auto &iter :this->_children) {
        delete iter.second;
    }
    this->_children.clear();
    return this;
}

py::iterator ZTrie::iter() {

    return py::make_iterator(this->iter_begin(), this->iter_end());
}


ZTrie::~ZTrie() {

    this->clear();
}

