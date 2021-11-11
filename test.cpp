#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>

int main(){
    cublasInit();

    CuMatrix<float> a = CuMatrix<float>::rand(3, 5);
    CuMatrix<float> b = a * 20; a.freeMat();

    b.inspect();
    b.freeMat();

    return 0;
}