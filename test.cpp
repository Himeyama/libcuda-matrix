#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>

int main(){
    cublasInit();

    float _a[] = {1, 2, 3, 4, 5, 6};
    float _b[] = {7, 8, 9, 10, 11, 12, 13, 14};

    int m = 3, n = 4, k = 2;

    CuMatrix<float> a(m, k, _a);
    CuMatrix<float> b(n, k, _b);
    CuMatrix<float> c(m, n);
    c = a.dott(b);
    c.inspect();

    a.rowSize = k;
    a.colSize = m; // a(k, m)
    b.rowSize = k;
    b.colSize = n;  // b(k, n)
    c = a.tdot(b);
    c.inspect();

    a.inspect();
    b.inspect();
    
    a.freeMat();
    b.freeMat();
    c.freeMat();

    return 0;
}

