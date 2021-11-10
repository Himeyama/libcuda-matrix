#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>
#include <iostream>
#include <typeinfo>

void cublasCopy(int n, const float *x, int incx, float *y, int incy){
    cublasScopy(n, x, incx, y, incy);
}

void cublasCopy(int n, const double *x, int incx, double *y, int incy){
    cublasDcopy(n, x, incx, y, incy);
}

template <typename T>
T* getRow(T* d_mat, long col, long i){
    T* d_vec;
    cublasAlloc(col, sizeof(T), (void**)&d_vec);
    cublasCopy(col, d_mat + col * i, 1, d_vec, 1);
    return d_vec;
}

template <typename T>
T* getCol(T* d_mat, long row, long col, long i){
    T* d_vec;
    cublasAlloc(row, sizeof(T), (void**)&d_vec);
    cublasCopy(row, d_mat + i, col, d_vec, 1);
    return d_vec;
}

template <typename T>
T* inspect(T* d_mat, long row, long col){
    T *mat = (T*)malloc(sizeof(T) * row * col);
    cublasGetMatrix(row, col, sizeof(T), d_mat, row, mat, row);
    std::string str = "[";
    for(long i = 0; i < row; i++){
        str += "[";
        for(long j = 0; j < col; j++){
            str += std::to_string(mat[i * col + j]) + (j == col - 1 ? "" : ", ");
        }
        str += (i == row - 1 ? "]" : "], ");
    }
    str += "]";
    std::cout << str << std::endl;
    free(mat);
}

int main(){
    cublasInit();

    long m = 3, n = 5, k = 2;
    CuMatrix<float> x, w, h, v;

    x = CuMatrix<float>::rand(n, m) * 20;
    w = CuMatrix<float>::rand(n, k);
    h = CuMatrix<float>::rand(k, m);
    CuMatrix<float> a, b, c, d;

    // x.inspect();
    // float* X = getRow(x.dMat, m, 1);
    // inspect(X, 1, m);

    // float* Y = getCol(x.dMat, n, m, 1);
    // inspect(Y, n, 1);

    // cublasFree(X);
    // cublasFree(Y);
    a = x.tdot(w);
    b = w.tdot(w);
    float *tmp1, *tmp2, *tmp3;
    for(long j = 0; j < k; j++){
        tmp1 = getRow(h.dMat, m, j); // H[j, :]
        tmp2 = getCol(a.dMat, a.rowSize, a.colSize, j); // A[:, j]
        tmp3 = getCol(b.dMat, b.rowSize, b.colSize, j); // B[:, j]
        // tmp4 = 
        // todo: メモリーを使いまわしする
        // H[j, :] + A[:, j] - H.T.dot(B[:, j])

        // cublasFree(tmp1);
        // cublasFree(tmp2);
        // cublasFree(tmp3);
    }

    x.freeMat();
    w.freeMat();
    h.freeMat();


    return 0;
}