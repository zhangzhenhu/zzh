//
// Created by 张振虎 on 2021/1/8.
//

#ifndef CLIB_SIMPLEMATRIX_H
#define CLIB_SIMPLEMATRIX_H


#include <iostream>
//using namespace std;

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))
#define Calloc(type, n) (type *)calloc(n,sizeof(type))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define SAFETY 1e-12 // value to substitute for zero for safe math


#define DEBUG true

template<typename T>
void prsize_t2D(T **ar, size_t size1, size_t size2) {
    for (size_t i = 0; i < size1; i++) {
        for (size_t j = 0; j < size2; j++)
            std::cout << ar[i][j] << " ";
        std::cout << std::endl;
    }

}

template<typename T>
void prsize_t1D(T *ar, size_t size1) {
    for (size_t i = 0; i < size1; i++) {
        std::cout << ar[i] << " ";
    }
    std::cout << std::endl;

}

template<typename T>
void toZero1D(T *ar, size_t size) {
    for (size_t i = 0; i < size; i++)
        ar[i] = 0;
}

template<typename T>
void setConstant1D(T *ar, size_t size, T value) {
    for (size_t i = 0; i < size; i++)
        ar[i] = value;
}

template<typename T>
void toZero2D(T **ar, size_t size1, size_t size2) {
    for (size_t i = 0; i < size1; i++)
        for (size_t j = 0; j < size2; j++)
            ar[i][j] = 0;
}

template<typename T>
void setConstant2D(T **ar, size_t size1, size_t size2, T value) {
    for (size_t i = 0; i < size1; i++)
        for (size_t j = 0; j < size2; j++)
            ar[i][j] = value;
}


template<typename T>
void toZero3D(T ***ar, size_t size1, size_t size2, size_t size3) {
    for (size_t i = 0; i < size1; i++)
        for (size_t j = 0; j < size2; j++)
            for (size_t l = 0; l < size3; l++)
                ar[i][j][l] = 0;
}

template<typename T>
void toZero4D(T ****ar, size_t size1, size_t size2, size_t size3, size_t size4) {
    for (size_t i = 0; i < size1; i++)
        for (size_t j = 0; j < size2; j++)
            for (size_t l = 0; l < size3; l++)
                for (size_t n = 0; n < size4; n++)
                    ar[i][j][l][n] = 0;
}

template<typename T>
T *init1D(size_t size) {
    T *ar = Calloc(T, (size_t) size);
//    toZero1D()
    return ar;
}

template<typename T>
T **init2D(size_t size1, size_t size2) {
    T **ar = (T **) Calloc(T *, (size_t) size1);
    for (size_t i = 0; i < size1; i++)
        ar[i] = (T *) Calloc(T, (size_t) size2);
    return ar;
}

template<typename T>
T ***init3D(size_t size1, size_t size2, size_t size3) {
    size_t i, j;
    T ***ar = Calloc(T **, (size_t) size1);
    for (i = 0; i < size1; i++) {
        ar[i] = Calloc(T*, (size_t) size2);
        for (j = 0; j < size2; j++)
            ar[i][j] = Calloc(T, (size_t) size3);
    }
    return ar;
}

template<typename T>
T ****init4D(size_t size1, size_t size2, size_t size3, size_t size4) {
    size_t i, j, l;
    T ****ar = Calloc(T ***, (size_t) size1);
    for (i = 0; i < size1; i++) {
        ar[i] = Calloc(T**, (size_t) size2);
        for (j = 0; j < size2; j++) {
            ar[i][j] = Calloc(T*, (size_t) size3);
            for (l = 0; l < size3; l++)
                ar[i][j][l] = Calloc(T, (size_t) size4);
        }
    }
    return ar;
}


template<typename T>
void free2D(T **ar, size_t size1) {
    if (ar == NULL) {
        return;
    }
    for (size_t i = 0; i < size1; i++) {
        if (ar[i] != NULL)
            free(ar[i]);
    }
    free(ar);
    //    &ar = NULL;
}

template<typename T>
void free3D(T ***ar, size_t size1, size_t size2) {
    for (size_t i = 0; i < size1; i++) {
        for (size_t j = 0; j < size2; j++)
            free(ar[i][j]);
        free(ar[i]);
    }
    free(ar);
    //    &ar = NULL;
}

template<typename T>
void free4D(T ****ar, size_t size1, size_t size2, size_t size3) {
    for (size_t i = 0; i < size1; i++) {
        for (size_t j = 0; j < size2; j++) {
            for (size_t l = 0; l < size3; l++)
                free(ar[i][j][l]);
            free(ar[i][j]);
        }
        free(ar[i]);
    }
    free(ar);
    //    &ar = NULL;
}


template<typename T>
void cpy1D(T *source, T *target, size_t size) {
    memcpy(target, source, sizeof(T) * (size_t) size);
}

template<typename T>
void cpy2D(T **source, T **target, size_t size1, size_t size2) {
    for (size_t i = 0; i < size1; i++)
        memcpy(target[i], source[i], sizeof(T) * (size_t) size2);
}

template<typename T>
void cpy3D(T ***source, T ***target, size_t size1, size_t size2, size_t size3) {
    for (size_t t = 0; t < size1; t++)
        for (size_t i = 0; i < size2; i++)
            memcpy(target[t][i], source[t][i], sizeof(T) * (size_t) size3);
}

template<typename T>
void cpy4D(T ****source, T ****target, size_t size1, size_t size2, size_t size3, size_t size4) {
    for (size_t t = 0; t < size1; t++)
        for (size_t i = 0; i < size2; i++)
            for (size_t j = 0; j < size3; j++)
                memcpy(target[t][i][j], source[t][i][j], sizeof(T) * (size_t) size4);
}


template<typename T>
void swap1D(T *source, T *target, size_t size) {
    T *buffer = init1D<T>(size); // init1<NUMBER>(size);
    memcpy(buffer, target, sizeof(T) * (size_t) size); // reversed order, destination then source
    memcpy(target, source, sizeof(T) * (size_t) size);
    memcpy(source, buffer, sizeof(T) * (size_t) size);
    free(buffer);
}

template<typename T>
void swap2D(T **source, T **target, size_t size1, size_t size2) {
    T **buffer = init2D<T>(size1, size2);
    cpy2D<T>(buffer, target, size1, size2);
    cpy2D<T>(target, source, size1, size2);
    cpy2D<T>(source, buffer, size1, size2);
    free2D<T>(buffer, size1);
}

template<typename T>
void swap3D(T ***source, T ***target, size_t size1, size_t size2, size_t size3) {
    T ***buffer = init3D<T>(size1, size2, size3);
    cpy3D<T>(buffer, target, size1, size2, size3);
    cpy3D<T>(target, source, size1, size2, size3);
    cpy3D<T>(source, buffer, size1, size2, size3);
    free3D<T>(buffer, size1, size2);
}

template<typename T>
void swap4D(T ****source, T ****target, size_t size1, size_t size2, size_t size3, size_t size4) {
    T ****buffer = init4D<T>(size1, size2, size3, size4);
    cpy4D<T>(buffer, target, size1, size2, size3, size4);
    cpy4D<T>(target, source, size1, size2, size3, size4);
    cpy4D<T>(source, buffer, size1, size2, size3, size4);
    free4D<T>(buffer, size1, size2, size3);
}

template<typename T>
T max1D(T *ar, size_t size1) {
    T value = 0;
    if (size1 < 1) {
        return value;
    }
    value = ar[0];
    for (size_t i = 0; i < size1; ++i) {
        if (ar[i] > value) {
            value = ar[i];
        }
    }
    return value;
}


template<typename T>
T sum1D(T *ar, size_t size1) {
    T sum = 0;

    for (size_t i = 0; i < size1; ++i) {
        sum += ar[i];
    }
    return sum;
}

template<typename T>
T sum2D(T **ar, size_t size1, size_t size2) {
    T sum = 0;

    for (size_t i = 0; i < size1; ++i) {
        for (size_t j = 0; j < size2; ++j) {
            sum += ar[i][j];
        }
    }
    return sum;
}

template<typename T>
double normalize1D(T *ar, size_t size1) {
    T sum = sum1D<T>(ar, size1);
    for (size_t i = 0; i < size1; ++i) {
        ar[i] /= sum;
    }
    return sum;
}

template<typename T>
double normalize2D(T **ar, size_t size1, size_t size2) {
    T sum = sum2D<T>(ar, size1, size2);
    for (size_t i = 0; i < size1; ++i) {
        for (size_t j = 0; j < size2; ++j) {
            ar[i][j] /= sum;
        }
    }
    return sum;

}

template<typename T>
void bounded1D(T *source, T *low, T *upper, size_t size) {
    for (size_t k = 0; k < size; ++k) {
        if (source[k] > upper[k]) {
            source[k] = upper[k];
        }
        if (source[k] < low[k]) {
            source[k] = low[k];
        }
    }
}

template<typename T>
void bounded2D(T **source, T **low, T **upper, size_t size1, size_t size2) {
    for (size_t i = 0; i < size1; ++i) {
        for (size_t k = 0; k < size2; ++k) {
            if (source[i][k] > upper[i][k]) {
                source[i][k] = upper[i][k];
            }
            if (source[i][k] < low[i][k]) {
                source[i][k] = low[i][k];
            }
        }
    }

}

template<class C>
class MatrixView {
public:
    C *data;
    size_t rows;
    size_t cols;

    MatrixView(size_t rows = 0, size_t cols = 0, C *ptr = NULL) {
        this->rows = rows;
        this->cols = cols;
        this->data = ptr;
    };

    C *operator[](size_t k) { return &(this->data[k * this->cols]); }

    void init(size_t rows = 0, size_t cols = 0) {
        this->rows = rows;
        this->cols = cols;
        this->data = init1D<C>(this->rows * this->cols);
    }

    void toZero() {
        toZero1D<C>(this->data, this->rows * this->cols);
    }

    void prsize_t() {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                std::cout << (*this)[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void free_data() {
        free(this->data);
    }

};


// 注意，这个函数不支持传入指针，因为sizeof对于指针和数组名得到不一样的结果，具体可百度
template<typename T>
size_t getArrayLen(T &array) {
    if (array == NULL) {
        return 0;
    }
//    std::cout <<"sizeof(array) " << sizeof(array)<<std::endl;
//    std::cout <<"sizeof(array[0])) " << sizeof(array[0])<<std::endl;
    return (sizeof(array) / sizeof(array[0]));
}


//void prsize_tAlpha(double **alpha, double *cn, size_t n_x, size_t n_stat) {
//    double c = 1;
//    for (size_t t = 0; t < n_x; ++t) {
//        c *= cn[t];
//        std::cout << t;
//        for (size_t i = 0; i < n_stat; ++i) {
//            std::cout << " " << alpha[t][i] * c;
//        }
//        std::cout << std::endl;
//    }
//}
//
//void prsize_tBeta(double **beta, double *cn, size_t n_x, size_t n_stat) {
//    double c = 1;
//    for (size_t t = n_x - 1; t >= 0; --t) {
//        c *= cn[t];
//        std::cout << t;
//        for (size_t i = 0; i < n_stat; ++i) {
//            std::cout << " " << beta[t][i] * c;
//        }
//        std::cout << std::endl;
//    }
//}

template<class T>
size_t *unique_counts(T data[], size_t length, size_t &out_length) {

    if (length < 1 || data == NULL) {
        return NULL;
    }

    size_t total_count = 1;
//    size_t pre = data[0];
    size_t i, j;
    for (i = 1; i < length; ++i) {
        if (data[i] != data[i - 1]) {
            total_count++;
        }
    }
//    std::cout<<"total-"<<total_count<<std::endl;
    out_length = total_count;

    size_t *values = (size_t *) calloc((size_t) total_count, sizeof(size_t));
    size_t cc = 1;
    i = 0;
    for (j = 1; j < length; ++j) {

        if (data[j] != data[j - 1]) {
            values[i] = cc;
            cc = 0;
            i++;
        }
        cc++;

    }

    values[i] = cc;

//    count = total_count;
    return values;
//    return total_count;
}

#endif //CLIB_SIMPLEMATRIX_H
