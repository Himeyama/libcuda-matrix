#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>
#include <iostream>

int main(){
    cublasInit();
    long m = 3, n = 5, k = 2;
    CuMatrix<float> x, w, h, v;

    x = CuMatrix<float>::rand(n, m) * 20;
    w = CuMatrix<float>::rand(n, k);
    h = CuMatrix<float>::rand(k, m);


    for(long i = 0; i < 200; i++){
        v = w * h;
        h = h.times((w.tdot(x)).rdivide(w.tdot(w * h)));
        w = w.times(
            (x.dott(h)).rdivide(w * h.dott(h))
        );
        v.inspect();
    }

    x.inspect();
    

    x.freeMat();
    w.freeMat();
    h.freeMat();
    v.freeMat();

    return 0;
}