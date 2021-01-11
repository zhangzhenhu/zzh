//
// Created by 张振虎 on 2021/1/10.
//

#ifndef FASTLIB_ZTRIE_H
#define FASTLIB_ZTRIE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stack>
#include <string>
#include <utility>


using namespace std;

namespace py = pybind11;

std::wstring bool2pystr(bool v);


class ZTrie {
public:
    std::wstring name;
    unsigned long counter;
    std::map<wchar_t, ZTrie *> children;
    bool end;

//    py::object *value;
public:
    explicit ZTrie(std::wstring name = L"", unsigned long counter = 0, bool end = false);

    void insert(const std::wstring &word);

    std::wstring getName() { return this->name; };

    void setName(std::wstring n) { this->name = n; };


//    py::object *getValue() { return this->value; };

//    void setValue(py::object *v = nullptr) { this->value = v; };

    std::map<wchar_t, ZTrie *> *getChildren() { return &(this->children); };

    std::wstring toString();
//    py::iterator keys() {
//
////        for(auto iter :this->children){
////            iter.first
////        }
//        return py::make_iterator(this->children.begin(), this->children.end());
//    };

    ZTrie *subtree(const wstring &word);

    py::tuple search(const wstring &prefix);

//    unsigned long counter(const wstring &prefix);
    ~ZTrie();

    void clear();

    size_t size();

    bool empty() const { return this->children.empty(); };

    py::iterator iter();


private:


    class Node {
    public:
        std::wstring prefix;
        const ZTrie *tree;

        Node(std::wstring p, const ZTrie *t) : prefix(std::move(p)), tree(t) {};
    };

    class ZIterator {
        friend class ZTrie;

    public:
        using value_type          = ZTrie;
        using pointer             = const ZTrie *;
        using reference           = const ZTrie &;
        using iterator_category   = std::input_iterator_tag;

        py::tuple operator*() {
            if (this->end || this->tree == nullptr || this->stack.empty()) {
                return py::make_tuple();
            }
            Node *n = stack.top();
            return py::make_tuple(n->prefix, n->tree->end, n->tree->counter);
        }

//    Node operator->() const {
//        return operator*();
//    }

        const ZIterator &operator++() {
//            std::cout<<"++"<<endl;
            if (stack.empty()) {
                this->end = true;
                return *this;
            }
            auto cur = stack.top();
            stack.pop();

            for (auto c:cur->tree->children) {
                this->stack.push(new Node(cur->prefix + c.first, c.second));
            }
            delete cur;

            if (stack.empty()) {
                this->end = true;
            }

            return *this;
        }

    const ZIterator operator++(int) {
        const auto oldIter = *this;
        ++(*this);
        return oldIter;
    }

        bool operator==(const ZIterator &other) const noexcept {
            return tree == other.tree && end == other.end;
        }

        bool operator!=(const ZIterator &other) const noexcept {
            return !(*this == other);
        }

    protected:
        ZIterator(const ZTrie *t, bool end = false)
                : tree(t) {
            if (end || this->tree->empty()) {
                this->end = true;
            } else {
                for(auto i :this->tree->children){

                    this->stack.push(new Node(std::wstring()+i.first, i.second));
                }
                this->end = false;
            }
        }

        bool end;
        const ZTrie *tree;
        std::stack<Node *> stack;
    };

public:
    ZIterator iter_begin() const {
        return ZIterator(this, false);
    }

    ZIterator iter_end() const {
        return ZIterator(this, true);
    }

    py::dict asDict(bool recursion= false);
};
//class ZTrie {
//
//private:
//    ZTrie *root;
//public:
//    ZTrie() { this->root = new ZTrie(L"root"); };
//
//    void add_word(const std::wstring &word);
//
//    ZTrie *get(const wstring &word);
//
//};
//PYBIND11_MAKE_OPAQUE(std::map<wchar_t, ZTrie *>);


#endif //FASTLIB_ZTRIE_H
