//
// Created by 张振虎 on 2021/1/8.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include "simpleArray.h"

using namespace std;
namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

py::tuple lcs(std::vector<int> &_short, std::vector<int> &_long) {

    size_t l1 = _short.size();
    size_t l2 = _long.size();


    size_t *cur = init1D<size_t>(l1 + 1);
    size_t *pre = init1D<size_t>(l1 + 1);
    size_t *tmp = nullptr;
    toZero1D(pre, l1 + 1);
    toZero1D(cur, l1 + 1);

    size_t i, j;
    size_t max = 0;  //#最长匹配的长度
    size_t pos = 0; //#最长匹配对应在s1中的最后一位

    for (i = 0; i < l2; i++) {
        //# 比较结果矩阵只保留两行，交替使用，这样能节省内存空间。
        // 交换一下 两行
        tmp = pre;
        pre = cur;
        cur = tmp;

        for (j = 0; j < l1; j++) {
            if (_short[j] == _long[i]) {
                cur[j + 1] = pre[j] + 1;
                if (cur[j + 1] > max) {

                    max = cur[j + 1];
                    pos = j;
                }
            } else {

                cur[j + 1] = 0;
            }
        }
    };
    free(cur);
    free(pre);
    pos += 1;
//    size_t s = pos - max + 1;
//    size_t e = pos + 1;
//    std::cerr << pos-max << " " << pos << " " << max << " " << *(_short.begin() + pos - max) << " "
//              << *(_short.begin() + pos) << endl;

    return py::make_tuple(max, std::vector<int>(_short.begin() + pos - max, _short.begin() + pos), pos - max);

}


py::tuple lcs(const std::wstring &_short, const std::wstring &_long) {

    size_t l1 = _short.length();
    size_t l2 = _long.length();


    size_t *cur = init1D<size_t>(l1 + 1);
    size_t *pre = init1D<size_t>(l1 + 1);
    size_t *tmp = nullptr;
    toZero1D(pre, l1 + 1);
    toZero1D(cur, l1 + 1);

    size_t i, j;
    size_t max = 0;  //#最长匹配的长度
    size_t pos = 0; //#最长匹配对应在s1中的最后一位

    for (i = 0; i < l2; i++) {
        //# 比较结果矩阵只保留两行，交替使用，这样能节省内存空间。
        // 交换一下 两行
        tmp = pre;
        pre = cur;
        cur = tmp;

        for (j = 0; j < l1; j++) {
            if (_short[j] == _long[i]) {
                cur[j + 1] = pre[j] + 1;
                if (cur[j + 1] > max) {

                    max = cur[j + 1];
                    pos = j;
                }
            } else {

                cur[j + 1] = 0;
            }
        }
    };
    free(cur);
    free(pre);
    pos += 1;
    return py::make_tuple(max, _short.substr(pos - max, max), pos - max);

}


size_t lcx(const std::wstring &_short, const std::wstring &_long) {

    size_t short_len = _short.length();
    size_t long_len = _long.length();

    size_t **cache = init2D<size_t>(2, short_len + 1);
    toZero2D(cache, 2, short_len + 1);

    size_t cur_row = 0;
    size_t pre_row = 0;

    size_t i, j;

    for (i = 0; i < long_len; ++i) {
        cur_row = (i + 1) % 2;
        pre_row = i % 2;

        for (j = 0; j < short_len; ++j) {

            // 比较当前值
            if (_long[i] == _short[j]) {
                cache[cur_row][j + 1] = cache[pre_row][j] + 1;
            } else {
                // 左边的值
//                x1 = cache[cur_row][j];
                // 上边的值
//                x2 = cache[pre_row][j+1];
                cache[cur_row][j + 1] = MAX(cache[cur_row][j], cache[pre_row][j + 1]);
            }


        }
//            print1D(cache[cur_row],b_len+1);


    }
    size_t n = cache[cur_row][short_len];
    free2D(cache, 2);


    return n;


}


size_t lcx(std::vector<int> &_short, std::vector<int> &_long) {

    size_t short_len = _short.size();
    size_t long_len = _long.size();

    size_t **cache = init2D<size_t>(2, short_len + 1);
    toZero2D(cache, 2, short_len + 1);

    size_t cur_row = 0;
    size_t pre_row = 0;

    size_t i, j;

    for (i = 0; i < long_len; ++i) {
        cur_row = (i + 1) % 2;
        pre_row = i % 2;

        for (j = 0; j < short_len; ++j) {

            // 比较当前值
            if (_long[i] == _short[j]) {
                cache[cur_row][j + 1] = cache[pre_row][j] + 1;
            } else {
                // 左边的值
//                x1 = cache[cur_row][j];
                // 上边的值
//                x2 = cache[pre_row][j+1];
                cache[cur_row][j + 1] = MAX(cache[cur_row][j], cache[pre_row][j + 1]);
            }


        }
//            print1D(cache[cur_row],b_len+1);


    }
    size_t n = cache[cur_row][short_len];
    free2D(cache, 2);


    return n;


}

py::tuple lcx_ex(std::vector<int> &_short, std::vector<int> &_long) {

//    if(_short)
    size_t short_len = _short.size();
    size_t long_len = _long.size();


    size_t row_len = long_len + 1;
    size_t col_len = short_len + 1;
    // 保存比较状态
    size_t **cache = init2D<size_t>(2, col_len);
    toZero2D(cache, 2, col_len);
    // 保存比较的位置信息
    char **position = init2D<char>(row_len, col_len);
    toZero2D(position, row_len, col_len);

    size_t cur_row = 0;
    size_t pre_row = 0;
    size_t x1, x2;
    size_t i, j;

    for (i = 0; i < long_len; ++i) {
        cur_row = (i + 1) % 2;
        pre_row = i % 2;
        for (j = 0; j < short_len; ++j) {
            // 比较当前值
            if (_long[i] == _short[j]) {
                cache[cur_row][j + 1] = cache[pre_row][j] + 1;
                position[i + 1][j + 1] = 1; // 1表示来自左上
            } else {
                // 左边的值
                x1 = cache[cur_row][j];
                // 上边的值
                x2 = cache[pre_row][j + 1];
                if (x1 > x2) {
                    cache[cur_row][j + 1] = x1;
                    position[i + 1][j + 1] = 2; //2 代表左边

                } else {
                    cache[cur_row][j + 1] = x2;
                    position[i + 1][j + 1] = 3; // 3 代表上边
                }
            }
        }

    }


    // 寻找匹配序列
    // 匹配的长度
    size_t n_match = cache[cur_row][col_len - 1];

    i = row_len - 1;
    j = col_len - 1;
    size_t k = n_match - 1;

    auto short_v = std::vector<size_t>(n_match);
    auto long_v = std::vector<size_t>(n_match);
    auto match = std::vector<int>(0);


    for (; k >= 0 & i > 0 & j > 0;) {
        // 当前位置来自左上，也即是当前位置是相同的
        if (position[i][j] == 1) {
            long_v[k] = i - 1;
            short_v[k] = j - 1;

            i--;
            j--;
            k--;

        } else if (position[i][j] == 2) {//2 代表左边
            j--;

        } else if (position[i][j] == 3) {  // 3 代表上边
            i--;

        }
    }
    free2D<size_t>(cache, 2);
    free2D<char>(position, row_len);
    for (auto c:short_v) {
        match.push_back(_short[c]);
    }
    return py::make_tuple(n_match, match, short_v, long_v);

};

py::tuple lcx_ex(const std::wstring &_short, const std::wstring &_long) {

//    if(_short)
    size_t short_len = _short.length();
    size_t long_len = _long.length();


    size_t row_len = long_len + 1;
    size_t col_len = short_len + 1;
    // 保存比较状态
    size_t **cache = init2D<size_t>(2, col_len);
    toZero2D(cache, 2, col_len);
    // 保存比较的位置信息
    char **position = init2D<char>(row_len, col_len);
    toZero2D(position, row_len, col_len);

    size_t cur_row = 0;
    size_t pre_row = 0;
    size_t x1, x2;
    size_t i, j;

    for (i = 0; i < long_len; ++i) {
        cur_row = (i + 1) % 2;
        pre_row = i % 2;
        for (j = 0; j < short_len; ++j) {
            // 比较当前值
            if (_long[i] == _short[j]) {
                cache[cur_row][j + 1] = cache[pre_row][j] + 1;
                position[i + 1][j + 1] = 1; // 1表示来自左上
            } else {
                // 左边的值
                x1 = cache[cur_row][j];
                // 上边的值
                x2 = cache[pre_row][j + 1];
                if (x1 > x2) {
                    cache[cur_row][j + 1] = x1;
                    position[i + 1][j + 1] = 2; //2 代表左边

                } else {
                    cache[cur_row][j + 1] = x2;
                    position[i + 1][j + 1] = 3; // 3 代表上边
                }
            }
        }

    }


    // 寻找匹配序列
    // 匹配的长度
    size_t n_match = cache[cur_row][col_len - 1];

    i = row_len - 1;
    j = col_len - 1;
    size_t k = n_match - 1;

    auto short_v = std::vector<size_t>(n_match);
    auto long_v = std::vector<size_t>(n_match);
    std::wstring match;

    for (; k >= 0 & i > 0 & j > 0;) {
        // 当前位置来自左上，也即是当前位置是相同的
        if (position[i][j] == 1) {
            long_v[k] = i - 1;
            short_v[k] = j - 1;

            i--;
            j--;
            k--;

        } else if (position[i][j] == 2) {//2 代表左边
            j--;

        } else if (position[i][j] == 3) {  // 3 代表上边
            i--;

        }
    }
    free2D<size_t>(cache, 2);
    free2D<char>(position, row_len);
    for (auto c:short_v) {
        match += _short[c];
    }
    return py::make_tuple(n_match, match, short_v, long_v);

};
