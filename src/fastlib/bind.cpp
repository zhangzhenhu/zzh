//
// Created by 张振虎 on 2021/1/8.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lcs.hpp"

namespace py = pybind11;


PYBIND11_MODULE(fastlib, m) {
    m.doc() = R"pbdoc(
        C++实现的高性能算法包，包含一些需要的算法。
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           lcs 最大公共子串
           lcx 最大公共子序列
    )pbdoc";

    m.def("lcx", pybind11::overload_cast<const wstring &, const wstring &>(&lcx), R"pbdoc(
    最大公共子序列，本函数近返回最大公共子序列的长度，如果需要公共序列内容或者位置请使用 :function:`lcx_ex` 。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: 最大公共子序列的长度

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));

    m.def("lcx", pybind11::overload_cast<std::vector<int> &, std::vector<int> &>(&lcx), R"pbdoc(
    最大公共子序列，本函数近返回最大公共子序列的长度，如果需要公共序列内容或者位置请使用 :function:`lcx_ex` 。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: 最大公共子序列的长度

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));


    m.def("lcx_ex", pybind11::overload_cast<const wstring &, const wstring &>(&lcx_ex), R"pbdoc(
    最大公共子序列，本函数近返回最大公共子序列的长度、内容、位置。
    如果仅需要长度，不需要内容和位置，请使用 :function:`lcx` ，:function:`lcx` 性能会更好。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: tuple(最大公共子序列的长度，内容，位置)

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));


    m.def("lcx_ex", pybind11::overload_cast<std::vector<int> &, std::vector<int> &>(&lcx_ex), R"pbdoc(
    最大公共子序列，本函数近返回最大公共子序列的长度、内容、位置。
    如果仅需要长度，不需要内容和位置，请使用 :function:`lcx` ，:function:`lcx` 性能会更好。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: tuple(最大公共子序列的长度，内容，位置)

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));


    m.def("lcs", pybind11::overload_cast<const wstring &, const wstring &>(&lcs), R"pbdoc(
    最大公共子串。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: tuple("公共子串",在短串的开始位置,长度)

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));

    m.def("lcs", pybind11::overload_cast<std::vector<int> &, std::vector<int> &>(&lcs), R"pbdoc(
    最大公共子串。
    算法不会自动区分长短串，需要使用者自行区分，并传入对应参数。
    长短传入错误，算法也能正常执行，只是效率会变差并且消耗更多内存。
    -----------------------
    :param short: 较短的串
    :param long: 较长的串
    :return: tuple("公共子串",在短串的开始位置,长度)

    )pbdoc", py::arg("short").none(true), py::arg("long").none(true));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}