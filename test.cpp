#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>

int main(){
    cublasInit();

    float* a_mat = (float*)malloc(sizeof(float) * 3 * 4);
    float* b_mat = (float*)malloc(sizeof(float) * 3 * 4);

    for(long i = 0; i < 12; i++){
        a_mat[i] = i + 1;
        b_mat[i] = i + 11;
    }
        

    CuMatrix<float> a(3, 4, a_mat);    
    CuMatrix<float> b(3, 4, b_mat);

    CuMatrix<float> c = a.times(b);

    a.inspect();
    b.inspect();
    c.inspect();

    a.freeMat();
    b.freeMat();

    free(a_mat);
    free(b_mat);

    return 0;
}