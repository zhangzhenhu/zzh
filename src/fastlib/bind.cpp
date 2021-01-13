//
// Created by 张振虎 on 2021/1/8.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "lcs.hpp"
#include "ZTrie.h"

namespace py = pybind11;


PYBIND11_MAKE_OPAQUE(ChildrenMap);

template<typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;


PYBIND11_MODULE(fastlib, m) {
    m.doc() = R"pbdoc(
        C++实现的高性能算法包，包含一些需要的算法。

        .. currentmodule:: fastlib
        .. autosummary::
           :toctree: _generate
           lcs 最大公共子串
           lcx 最大公共子序列
           ZTrie 前缀树

    )pbdoc";

    m.def("lcx", overload_cast_<const wstring &, const wstring &>()(&lcx), R"pbdoc(
    最大公共子序列。仅返回长度，如果需要公共序列内容或者位置请使用 ``lcx_ex`` 。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。


    :param short: 较短的串
    :param long: 较长的串
    :return: 最大公共子序列的长度

    )pbdoc", py::arg("short").none(false), py::arg("long").none(false));

    m.def("lcx", overload_cast_<std::vector<int> &, std::vector<int> &>()(&lcx), R"pbdoc(
    最大公共子序列。仅返回长度，如果需要公共序列内容或者位置请使用 ``lcx_ex`` 。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。

    :param short: 较短的串
    :param long: 较长的串
    :return: 最大公共子序列的长度

    )pbdoc", py::arg("short").none(false), py::arg("long").none(false));


    m.def("lcx_ex", overload_cast_<const wstring &, const wstring &>()(&lcx_ex), R"pbdoc(
    最大公共子序列，本函数返回最大公共子序列的长度、内容、位置。
    如果仅需要长度，不需要内容和位置，请使用 ``lcx`` ，``lcx`` 性能会更好。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。

    :param short: 较短的串
    :param long: 较长的串
    :return: tuple(最大公共子序列的长度，内容，位置)

    )pbdoc", py::arg("short").none(false), py::arg("long").none(false));


    m.def("lcx_ex", overload_cast_<std::vector<int> &, std::vector<int> &>()(&lcx_ex), R"pbdoc(
    最大公共子序列，本函数返回最大公共子序列的长度、内容、位置。
    如果仅需要长度，不需要内容和位置，请使用 ``lcx`` ，``lcx`` 性能会更好。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。

    :param short: 较短的串
    :param long: 较长的串
    :return: tuple(最大公共子序列的长度，内容，位置)

    )pbdoc", py::arg("short").none(false), py::arg("long").none(false));


    m.def("lcs", overload_cast_<const wstring &, const wstring &>()(&lcs), R"pbdoc(
    最大公共子串。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。

    :param short: 较短的串
    :param long: 较长的串
    :return: tuple("公共子串",在短串的开始位置,长度)

    )pbdoc", py::arg("short").none(false), py::arg("long").none(false));

    m.def("lcs", overload_cast_<std::vector<int> &, std::vector<int> &>()(&lcs), R"pbdoc(
    最大公共子串。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。

    :param short: 较短的串
    :type short: str
    :param long: 较长的串
    :type long: str
    :return: tuple("公共子串",在短串的开始位置,长度)
    :rtype: tuple

    )pbdoc", py::arg("short").none(false), py::arg("long").none(false));

    // 绑定 ChildrenMap
    auto cm = py::bind_map<ChildrenMap >(m, "ChildrenMap",
                                         R"pbdoc(
    ZTrie 用来存储子节点的结构，底层是一个 STL 中的 Map<string,ZTrie *> 结构。
    )pbdoc");
    cm.def("__str__", [](ChildrenMap &m) {
        std::wstringstream s;
        s << L"ChildrenMap" << L'{';
        bool f = false;
        for (auto const &kv : m) {
            if (f)
                s << L", ";
            s << kv.first << L": " << std::to_wstring(kv.second->_children.size());
            f = true;
        }
        s << L'}';
        return s.str();
    });
    cm.def("size", &ChildrenMap::size, R"pbdoc(
    返回子节点的数量。

    :return: 子节点的数量。
    )pbdoc");

    cm.def("keys",
           [](ChildrenMap &m) { return py::make_key_iterator(m.begin(), m.end()); },
           py::keep_alive<0, 1>(), /* Essential: keep list alive while iterator exists */
           R"pbdoc(

    返回所有子节点的 key 的迭代器。

    :return: 迭代器

    )pbdoc");
    cm.def("clear",
           [](ChildrenMap &m) {
               for (auto i:m) { delete i.second; }
               m.clear();
           },

           R"pbdoc(
    清空子节点

    )pbdoc");
    cm.def("get",
           [](ChildrenMap &m, const std::wstring &k) -> ZTrie *& {
               auto it = m.find(k);
               if (it == m.end())
                   throw py::key_error();
               return it->second;
           },
           py::return_value_policy::reference_internal,// ref + keepalive
           R"pbdoc(

    根据 key 获取某个子节点

    :return: 子节点（子树）

    )pbdoc");


    auto z = pybind11::class_<ZTrie>(m,
                                     "ZTrie",
                                     R"pbdoc(
    C++ 实现的前缀树结构。
    )pbdoc");

    z.def(py::init<std::wstring, unsigned long, bool>(),
          py::arg("name") = L"",
          py::arg("counter") = 1,
          py::arg("end") = false,
          R"pbdoc(
    构造函数。

    :param name: 根节点的name，默认空串。
    :param counter: 根节点的计数器，默认为1。
    :param end: 根节点的end标记，默认为False。

    )pbdoc");
//    .def(py::self + py::self);

//    z.def(py::self == py::self);

    z.def("__str__", &ZTrie::toString);
    z.def("__repr__", &ZTrie::toString);

    z.def("add", &ZTrie::add, py::arg("word"),
          R"pbdoc(
    往当前树中增加一个（前缀）词。
    路径中所有节点的 counter 都会加1，最后的尾节点的 end 标记为True。

    :param word: 待插入前缀词。
    :type word: string(, unicode)
    :return: self

    )pbdoc");
    z.def("insert", &ZTrie::insert, R"pbdoc(
    直接插入一个尾部节点。
    如果某个前缀节点已经存在，此前缀节点的counter不会累加，end标记不会变动；
    如果某个前缀结点不存在，新增此前缀节点，并且其counter=1，end=False。
    最后的尾部节点：
    如果尾部节点已经存在，其 counter 累加上入参的counter，end标记赋值为入参的值。
    如果尾部节点不存在，其counter和end分别是入参的值。

    :param word: 待插入前缀词。
    :type word: string(, unicode)
    :param counter: 计数。默认值1.
    :type counter: int
    :param end: 结束标记。默认True。
    :type end: bool
    :return: self
             )pbdoc", py::arg("word"),
          py::arg("counter") = 1,
          py::arg("end") = true);

    z.def("search", &ZTrie::search,
          py::arg("prefix"),
          R"pbdoc(
    查询一个前缀

    :param prefix: 前缀词
    :return: 若存在返回 tuple(end,counter);若不存在，返回 None。
    :rtype: tuple, None

    )pbdoc");

    z.def("subtree", &ZTrie::subtree,
          py::arg("prefix"),
          py::return_value_policy::reference,
          R"pbdoc(
    获取子树

    :param prefix: 前缀

    :return: 子树
    )pbdoc");

    z.def("copy", &ZTrie::copy,
          R"pbdoc(
    深度复制一份。

    :return: 副本。

    )pbdoc");

    z.def("size", &ZTrie::size,
          R"pbdoc(
    返回树中节点的数量，包括根节点。
            )pbdoc");
    z.def("__iter__", &ZTrie::iter);
//                 py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */)

    z.def("to_dict", &ZTrie::asDict,
          py::arg("recursion") = true,
          R"pbdoc(
    把 ``ZTrie`` 转换成 Python 中的 ``Dict`` 结构。
    注意，对于超大规模的树转换的性能是比较差的。

    :param recursion: 是否递归子节点。默认为 True。
    :return:
    :rtype: Dict

    )pbdoc");

    z.def("save", &ZTrie::save,
          py::arg("filename"),
          py::arg("sep") = " ",
          R"pbdoc(
    把树保存到csv文件中。

    :param filename: 文件路径
    :param sep: 分割符，默认空格
    :return: self
    )pbdoc");

    z.def("load", &ZTrie::load,
          py::arg("filename").none(false),
          py::arg("sep") = " ", R"pbdoc(
    从csv文件中加载文件。
    读取的文件必须是 save 函数保存的结果，如果是原始的语料数据请重新创建树。

    :param filename: 文件路径
    :param sep: 分割符，默认空格
    :return: self
    )pbdoc");

    z.def("merge", &ZTrie::merge,
          py::arg("other").none(false),
          R"pbdoc(
    合并另外一棵树结构到当前的树。
    合并的过程中，节点的counter会累加，end 标记 True 会覆盖 False，False 不会覆盖True。

    :param other: 另外的 ZTrie 对象
    :return: self
    )pbdoc");

    z.def("equal", &ZTrie::equal,
          py::arg("other").none(false),
          py::arg("name") = true,
          py::arg("counter") = true,
          py::arg("end") = true,
          R"pbdoc(
    判断两棵树的结构和内容是否相同。
    如果两棵树的结构不同（有不同的子节点）一定会返回 False。

    :param other: 需要比较的 ZTrie 对象
    :param name: 是否需要比较节点的 name 属性，默认为 True。
    :type name: bool
    :param counter: 是否需要比较节点的 counter 属性，默认为 True。
    :type counter: bool
    :param end: 是否需要比较节点的 end 属性，默认为 True。
    :type end: bool
    :return: 是否相同
    :rtype: bool

    )pbdoc");
    z.def_readwrite("end", &ZTrie::_end,
                    R"pbdoc(
    结束标记。
    如果为True，表示从根节点到当前节点是一个是完整的词；
    如果为False，表示当前节点是一个中间节点；
    )pbdoc");
    z.def_readwrite("name", &ZTrie::_name,
                    R"pbdoc(
    节点的名字
    )pbdoc");
    z.def_readonly("children", &ZTrie::_children,
                   py::return_value_policy::reference,
                   R"pbdoc(
    当前节点的子节点。

    )pbdoc");


//    pybind11::class_<ZTrie>(m, "ZTrie")
//            .def(py::init<>())
//            .def("add_word", &ZTrie::add_word)
//            .def("get", &ZTrie::get, py::return_value_policy::copy);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}