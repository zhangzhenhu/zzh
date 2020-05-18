//
// Created by 张振虎 on 2019/1/10.
//

#include "lcs.h"
#include <iostream>
#include <stdint.h>
#include <algorithm>


int LCS::fast_score(int *a, size_t a_len, int *b, size_t b_len) {

    int **cache = init2D<int>(2, b_len + 1);
    toZero2D(cache, 2, b_len + 1);

    int cur_index = 0;
    int pre_index = 0;
    int x0, x1, x2, i, j;
    int b_v;
    for (i = 0; i < a_len; ++i) {
        cur_index = i % 2;
        pre_index = (i + 1) % 2;

        for (j = 1; j <= b_len; ++j) {
//            cur_j = j + 1;
            b_v = b[j-1];

            // 左上角的值
            if (a[i] == b_v) {
                cache[cur_index][j] = cache[pre_index][j-1] + 1;
            } else {
//                x0 = cache[pre_index][j];
                // 左边的值
                x1 = cache[cur_index][j-1];
                // 上边的值
                x2 = cache[pre_index][j];
                cache[cur_index][j] = MAX(x1, x2);
            }


        }
            print1D(cache[cur_index],b_len+1);


    }
    free2D(cache,2);
    return cache[cur_index][b_len];


}