#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>
#include <iostream>
#include <typeinfo>

int main(){
    cublasInit();

    long m = 3, n = 5, k = 2;
    CuMatrix<float> x, w, h, tmp;

    tmp = CuMatrix<float>::rand(n, m);
    x = tmp * 20; tmp.freeMat();
    w = CuMatrix<float>::rand(n, k);
    h = CuMatrix<float>::rand(k, m);
    CuMatrix<float> a, b, c, d;


    a = x.tdot(w); // (m x k)
    b = w.tdot(w); // (k x k)


    CuMatrix<float> hj(1, m), aj(m, 1), bj(k, 1);
    // CuMatrix<float> tmp1, tmp2, tmp3, tmp4, tmp5;

    h.inspect();
    for(long j = 0; j < k; j++){
        hj = h.getRow(j, hj.dMat); // H[j, :]
        aj = a.getCol(j, aj.dMat); // A[:, j]
        bj = b.getCol(j, bj.dMat); // B[:, j]
        cublasSaxpy(m, 1, aj.dMat, 1, hj.dMat, 1);
        cublasSgemm('T', 'N', hj.rowSize, hj.colSize, bj.rowSize, -1, h.dMat, h.rowSize, bj.dMat, bj.rowSize, 1, hj.dMat, hj.rowSize);
        hj.inspect();
        cublasScopy(m, hj.dMat, 1, h.dMat + m * j, 1);
        // h.setRow(j, hj);
    }
    h.inspect();

    x.freeMat();
    w.freeMat();
    h.freeMat();

    hj.freeMat();
    aj.freeMat();
    bj.freeMat();

    a.freeMat();
    b.freeMat();

    return 0;
}