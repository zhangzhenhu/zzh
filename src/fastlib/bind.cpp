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
        -----------------------
        .. currentmodule:: fastlib
        .. autosummary::
           :toctree: _generate
           lcs 最大公共子串
           lcx 最大公共子序列
    )pbdoc";

    m.def("lcx", overload_cast_<const wstring &, const wstring &>()(&lcx), R"pbdoc(
    最大公共子序列。仅返回长度，如果需要公共序列内容或者位置请使用 :function:`lcx_ex` 。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: 最大公共子序列的长度

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));

    m.def("lcx", overload_cast_<std::vector<int> &, std::vector<int> &>()(&lcx), R"pbdoc(
    最大公共子序列。仅返回长度，如果需要公共序列内容或者位置请使用 :function:`lcx_ex` 。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: 最大公共子序列的长度

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));


    m.def("lcx_ex", overload_cast_<const wstring &, const wstring &>()(&lcx_ex), R"pbdoc(
    最大公共子序列，本函数返回最大公共子序列的长度、内容、位置。
    如果仅需要长度，不需要内容和位置，请使用 :function:`lcx` ，:function:`lcx` 性能会更好。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: tuple(最大公共子序列的长度，内容，位置)

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));


    m.def("lcx_ex", overload_cast_<std::vector<int> &, std::vector<int> &>()(&lcx_ex), R"pbdoc(
    最大公共子序列，本函数返回最大公共子序列的长度、内容、位置。
    如果仅需要长度，不需要内容和位置，请使用 :function:`lcx` ，:function:`lcx` 性能会更好。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: tuple(最大公共子序列的长度，内容，位置)

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));


    m.def("lcs", overload_cast_<const wstring &, const wstring &>()(&lcs), R"pbdoc(
    最大公共子串。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: tuple("公共子串",在短串的开始位置,长度)

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));

    m.def("lcs", overload_cast_<std::vector<int> &, std::vector<int> &>()(&lcs), R"pbdoc(
    最大公共子串。
    第一个参数是较短的序列，第二个参数是较长的序列。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    如果长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: tuple("公共子串",在短串的开始位置,长度)

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));

//    using ChildrenMap = std::map<wchar_t, ZTrie *>;

    py::bind_map<ChildrenMap >(m, "ChildrenMap")
            .def("__str__", [](ChildrenMap &m) {
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
            })
            .def("size", &ChildrenMap::size)
            .def("keys",
                 [](ChildrenMap &m) { return py::make_key_iterator(m.begin(), m.end()); },
                 py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
            )
            .def("clear",
                 [](ChildrenMap &m) {
                     for (auto i:m) { delete i.second; }
                     m.clear();
                 },
                 py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
            ).def("get",
                  [](ChildrenMap &m, const std::wstring &k) -> ZTrie *& {
                      auto it = m.find(k);
                      if (it == m.end())
                          throw py::key_error();
                      return it->second;
                  },
                  py::return_value_policy::reference_internal // ref + keepalive
            );


    pybind11::class_<ZTrie>(m, "ZTrie")
            .def(py::init<std::wstring, unsigned long, bool>(),
                 py::arg("name") = L"",
                 py::arg("counter") = 1,
                 py::arg("end") = false
            )
            .def("__str__", &ZTrie::toString)
            .def("__repr__", &ZTrie::toString)
            .def("insert", &ZTrie::insert)
            .def("search", &ZTrie::search)
            .def("subtree", &ZTrie::subtree, py::return_value_policy::reference)
            .def("size", &ZTrie::size)
            .def("__iter__", &ZTrie::iter)
//                 py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */)
            .def("asDict", &ZTrie::asDict, py::arg("recursion") = false)
            .def("save", &ZTrie::save,
                 py::arg("filename"),
                 py::arg("sep") = " ")
            .def("load", &ZTrie::load,
                 py::arg("filename").none(false),
                 py::arg("sep") = " ")
            .def("merge", &ZTrie::merge,
                 py::arg("other").none(false))
            .def("equal", &ZTrie::equal,
                 py::arg("other").none(false),
                 py::arg("name") = true,
                 py::arg("counter") = true)
            .def_readwrite("end", &ZTrie::_end)
            .def_readwrite("name", &ZTrie::_name)
            .def_readonly("children", &ZTrie::_children, py::return_value_policy::reference);


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