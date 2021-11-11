#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>

int main(){
    cublasInit();

    float _a[] = {1, 2, 3, 4};
    float _b[] = {5, 6, 7, 8};

    CuMatrix<float> a(2, 2, _a);
    CuMatrix<float> b(2, 2, _b);

    CuMatrix<float> c(2, 2);

    // cublasSgemm('N', 'N', a.rowSize, b.colSize, b.rowSize, 1, a.dMat, a.rowSize, b.dMat, b.rowSize, 0, c.dMat, c.rowSize);

    c = a * b;


    a.inspect();
    b.inspect();
    c.inspect();
    
    a.freeMat();
    b.freeMat();
    c.freeMat();

    return 0;
}