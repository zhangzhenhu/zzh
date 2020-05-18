#include <iostream>
#include "lcs.h"

using namespace std;

template<class T>
int getArrayLen(T &array) {

    return (sizeof(array) / sizeof(array[0]));

}

int main() {
    std::cout << "Hello, World!" << std::endl;

    LCS<int> *lcs = new LCS<int>();
    int a[] = {3, 9, 6, 10, 7, 8};
    int b[] = {3, 4, 8, 6, 7, 8};

    print1D(a, getArrayLen(a));
    print1D(b, getArrayLen(b));
    int score = lcs->length(a, getArrayLen(a), b, getArrayLen(b));


    cout << " class " << score << endl;
    int aa[] = {3, 1, 2, 0, 4, 5, 8};
    int bb[] = {17, 7, 15, 13, 12, 10, 21, 9, 18, 5, 16, 18, 22, 8, 19, 15, 13, 14, 6, 20, 11, 18, 8};


    int x[] = {50, 1, 3, 5, 10, 43, 46, 11, 4, 3, 5, 8, 9, 37, 29, 42, 13, 51, 6, 1, 30, 7, 17, 38, 14, 48, 0, 56, 7,
               17, 38, 14, 32, 45, 18, 6, 9, 37, 29, 24, 31, 24, 16, 40, 34, 0, 47, 52, 39, 25, 19, 0, 12, 34, 38, 46,
               28, 44, 46, 49, 54, 20, 36, 17, 38, 24, 3, 49, 27, 9, 37, 29, 22, 55, 9, 37, 29, 33, 35, 23, 32, 47, 27,
               2, 53};

    int y[] = {15, 13, 26, 41, 6, 9, 37, 23, 21, 2, 53};

    score = fast_lcs<int>(x, getArrayLen(x), y, getArrayLen(y));

    cout << " function " << score << endl;


    score = lcs->length(x, getArrayLen(x), y, getArrayLen(y));


    cout << " class    " << score << endl;

    cout << "short " << sizeof(short) << endl;
    cout << "int " << sizeof(int) << endl;
    cout << "long " << sizeof(long) << endl;
    cout << "int64_t " << sizeof(int64_t) << endl;

    return 0;
}
