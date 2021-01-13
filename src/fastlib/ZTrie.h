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
#include <cstdlib>
#include <memory>
#include <string>

using namespace std;

namespace py = pybind11;

#define ChildrenMap std::map<std::wstring, ZTrie *>

std::wstring bool2str(bool v);


class ZTrie {
public:
    std::wstring _name;
    unsigned long _counter;
    ChildrenMap _children;
    bool _end;

//    py::object *value;
public:
    explicit ZTrie(std::wstring name = L"", unsigned long counter = 0, bool end = false);

    ~ZTrie();

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
        using node_ptr = std::shared_ptr<Node>;

        py::tuple operator*() {
            if (this->end || this->tree == nullptr || this->stack.empty()) {
                return py::make_tuple();
            }
            Node n = stack.top();
            return py::make_tuple(n.prefix, n.tree->_counter, n.tree->_end);
        }

        Node operator->() {
            return this->value();
        }

        Node value() {
//            if (this->end || this->tree == nullptr || this->stack.empty()) {
////                return ;
//
//            }
            return stack.top();
        }

        const ZIterator &operator++() {
//            std::cout<<"++"<<endl;
            if (stack.empty()) {
                this->end = true;
                return *this;
            }
            auto cur = stack.top();
            stack.pop();

            for (auto c:cur.tree->_children) {
//                this->stack.push(new Node(cur->prefix + c.first, c.second));
                this->stack.push(Node(cur.prefix + c.first, c.second));
            }
//            delete cur;
//            cur.reset();

            if (stack.empty()) {
                this->end = true;
            }

            return *this;
        }

//    const ZIterator operator++(int) {
//        const auto oldIter = *this;
//        ++(*this);
//        return oldIter;
//    }

        bool operator==(const ZIterator &other) const noexcept {
            return tree == other.tree && end == other.end;
        }

        bool operator!=(const ZIterator &other) const noexcept {
            return !(*this == other);
        }

//        ~ZIterator() {
//            while (!this->stack.empty()) {
//                auto x = this->stack.top();
//                this->stack.pop();
//                // 注意这个迭代器有可能被复制，
//                // 如果复制了，stack 也会被复制
//                // 就会有两个 ZIterator 对象的stack中内容相同
//                // 如果在这里直接 delete 会 delete 两次
//                // 最后 stack 中不用指针了，直接用对象
//                std::cout << this << " " << &(this->stack)
//                          << " " << x << " "
//                          << x.use_count() << endl;
//                x.reset();
////                x;
//            }
//        };
    protected:
        ZIterator(const ZTrie *t, bool end = false)
                : tree(t) {
            if (end || this->tree->empty()) {
                this->end = true;
            } else {
                for (const auto &i :this->tree->_children) {
//                    auto xxx=new Node(std::wstring() + i.first, i.second);

                    this->stack.push(Node(std::wstring() + i.first, i.second));
                }
                this->end = false;
            }
        }

        bool end;
        const ZTrie *tree;
        std::stack<Node> stack;
    };

public:
    std::wstring getName() { return this->_name; };

    void setName(const std::wstring &n) { this->_name = n; };

//    ChildrenMap *getChildren() { return &(this->children); };

    std::wstring toString();

    ZIterator iter_begin() const {
        return ZIterator(this, false);
    }

    ZIterator iter_end() const {
        return ZIterator(this, true);
    }

    py::iterator iter();

    py::dict asDict(bool recursion = false);

    ZTrie *merge(ZTrie *other);

    ZTrie *copy();

    ZTrie *add(const std::wstring &word);

    ZTrie *insert(const std::wstring &word, size_t counter = 1, bool end = true);

    ZTrie *subtree(const wstring &word);

    py::tuple search(const wstring &prefix);

    ZTrie *clear();

    size_t size();

    bool empty() const { return this->_children.empty(); };

    ZTrie *save(const string &filename, const string &separator = " ");

    ZTrie *load(const string &filename, const wstring &separator = L" ");

    bool equal(ZTrie *other, bool name = true, bool counter = true,bool end=true);
};

#endif //FASTLIB_ZTRIE_H
