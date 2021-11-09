#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>

int main(){
    cublasInit();

    long m = 3, n = 5, k = 2;
    CuMatrix<float> x, w, h, v;

    x = CuMatrix<float>::rand(n, m) * 20;
    a = CuMatrix<float>::rand(n, k);
    b = CuMatrix<float>::rand(k, m);

    w = x.T * a; 
    v = a.T * a;

    for(long j = 0; j = b.colSize; j++){
        
    }


    return 0;
}