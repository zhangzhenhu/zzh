//
// Created by 张振虎 on 2019/1/10.
//

#ifndef LCS_LCS_H
#define LCS_LCS_H


//#include <cstdio>
#include <iostream>

using namespace std;

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))
#define Calloc(type, n) (type *)calloc(n,sizeof(type))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


template<class T>
T max(T a, T b) {

    return a > b ? a : b;
}

template<class T>
T **init2D(size_t size1, size_t size2) {
    T **ar = (T **) Calloc(T *, (size_t) size1);
    for (int i = 0; i < size1; i++)
        ar[i] = (T *) Calloc(T, (size_t) size2);
    return ar;
}


template<typename T>
void free2D(T **ar, size_t size1) {
    for (size_t i = 0; i < size1; i++)
        free(ar[i]);
    free(ar);
    //    &ar = NULL;
}

template<typename T>
void toZero2D(T **ar, size_t size1, size_t size2) {
    for (int i = 0; i < size1; i++)
        for (int j = 0; j < size2; j++)
            ar[i][j] = 0;
}

template<typename T>
void print2D(T **ar, int size1, int size2) {
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++)
            cout << ar[i][j] << " ";
        cout << endl;
    }

}

template<typename T>
void print1D(T *ar, int size1) {
    for (int i = 0; i < size1; i++) {
        cout << ar[i] << " ";
    }
    cout << endl;

}


template<typename T>
int fast_lcs(const T *a, size_t a_len, const T *b, size_t b_len) {

    int **cache = init2D<int>(2, b_len + 1);
    toZero2D(cache, 2, b_len + 1);

    int cur_index = 0;
    int pre_index = 0;
    int x0, x1, x2;
    size_t i, j;
    T b_v;
    for (i = 0; i < a_len; ++i) {
        cur_index = i % 2;
        pre_index = (i + 1) % 2;

        for (j = 1; j <= b_len; ++j) {
//            cur_j = j + 1;
            b_v = b[j - 1];

            // 比较当前值
            if (a[i] == b_v) {
                cache[cur_index][j] = cache[pre_index][j - 1] + 1;
            } else {
//                x0 = cache[pre_index][j];
                // 左边的值
                x1 = cache[cur_index][j - 1];
                // 上边的值
                x2 = cache[pre_index][j];
                cache[cur_index][j] = MAX(x1, x2);
            }


        }
//            print1D(cache[cur_index],b_len+1);


    }

    x0 = cache[cur_index][b_len];
    free2D(cache, 2);
    return x0;


}


template<class T>
class LCS {
private:
    size_t row_len = 0;
    size_t col_len = 0;
    int **cache = 0;
    char **postion = 0;

    void free_cache() {
        if (this->cache != NULL) {
            free2D(this->cache, this->row_len);
            free2D(this->postion, this->row_len);
            this->cache = NULL;
            this->postion = NULL;
            this->row_len = 0;
            this->col_len = 0;
        }

    }


public:
    ~LCS() { this->free_cache(); };

    int length(const T a[], size_t a_len, const T b[], size_t b_len) {

        this->free_cache();

        this->row_len = a_len + 1;
        this->col_len = b_len + 1;
//        cout << "c++ " << this->row_len <<" " << this->col_len << endl;

        this->cache = init2D<int>(this->row_len, this->col_len);
        this->postion = init2D<char>(this->row_len, this->col_len);

        toZero2D(this->cache, this->row_len, this->col_len);
        toZero2D(this->postion, this->row_len, this->col_len);

        int cur_index = 0;
        int pre_index = 0;
        int x1, x2;
        int i, j;
        T b_v, a_v;
        for (i = 1; i < this->row_len; ++i) {
            a_v = a[i - 1];
            for (j = 1; j < this->col_len; ++j) {
                b_v = b[j - 1];
                // 比较当前值
                if (a_v == b_v) {
                    this->cache[i][j] = this->cache[i - 1][j - 1] + 1;
                    postion[i][j] = 1; // 1表示来自左上
                } else {
                    // 左边的值
                    x1 = this->cache[i][j - 1];
                    // 上边的值
                    x2 = this->cache[i - 1][j];
                    if (x1 > x2) {
                        this->cache[i][j] = x1;
                        postion[i][j] = 2; //2 代表左边

                    } else {
//                        cout << " x2 " << x2 << endl;
                        this->cache[i][j] = x2;
                        postion[i][j] = 3; // 3 代表上边
                    }

                }


            }
//            print1D(this->cache[i],this->col_len);


        }
//        cout << "c++ " << "row "<<this->row_len;
//        cout << " col " << this->col_len;
//        cout <<" length " <<this->cache[this->row_len - 1][this->col_len - 1]<<endl;

        return this->cache[this->row_len - 1][this->col_len - 1];


    };

    int sequence_position(int a[], int b[]) {

        int length = this->cache[this->row_len - 1][this->col_len - 1];
        size_t i, j;
        i = this->row_len - 1;
        j = this->col_len - 1;
        length--;

        for (; length >= 0 & i > 0 & j > 0;) {
            // 当前位置来自左上，也即是当前位置是相同的
            if (this->postion[i][j] == 1) {
                a[length] = (int) i - 1;
                b[length] = (int) j - 1;
                i--;
                j--;
                length--;

            } else if (this->postion[i][j] == 2) {//2 代表左边
                j--;

            } else if (this->postion[i][j] == 3) {  // 3 代表上边
                i--;

            }
        }
        return this->cache[this->row_len - 1][this->col_len - 1];

    }
};


#endif //LCS_LCS_H
