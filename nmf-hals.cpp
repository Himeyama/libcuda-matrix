#include <cublas.h>
#include <cuda-matrix.hpp>

void NMF(float eps, long m, long n, long k, float *dat){
    CuMatrix<float> x(n, m, dat), w(n, k), h(k, m), y(n, m);

    CuMatrix<float> a(m, k), b(k, k), c(n, k), d(k, k);
    CuMatrix<float> hj(1, m), aj(m, 1), bj(k, 1), cj(n, 1), dj(k, 1), ej(n, 1), fj(k, m), wj(n, 1), djj(1, 1);
    float* h_m_tmp = (float*)malloc(sizeof(float) * m);
    float* h_n_tmp = (float*)malloc(sizeof(float) * n);

    // for(int iter = 0; iter < 10; iter++){
    //     x.tdot(w, a.dMat);
    //     w.tdot(w, b.dMat);

    //     for(long j = 0; j < k; j++){
    //         h.getRow(j, hj.dMat);
    //         a.getCol(j, aj.dMat);
    //         b.getCol(j, bj.dMat);
    //         hj += aj;
    //         h.tdot(bj, fj.dMat);
    //         hj -= fj;

    //         // 最小値設定
    //         cublasGetMatrix(1, m, sizeof(float), hj.dMat, 1, h_m_tmp, 1);
    //         for(int i = 0; i < m; i++)
    //             h_m_tmp[i] = h_m_tmp[i] < eps ? eps : h_m_tmp[i];
    //         cublasSetMatrix(1, m, sizeof(float), h_m_tmp, 1, hj.dMat, 1);
    //         h.setRow(j, hj);
    //     }
    //     x.dott(h, c.dMat);       
    //     h.dott(h, d.dMat);

    //     for(long j = 0; j < k; j++){
    //         w.getCol(j, wj.dMat);
    //         cublasScopy(1, d.dMat + d.colSize * j + j, 1, djj.dMat, 1);
    //         wj.dot(djj, wj.dMat);
    //         c.getCol(j, cj.dMat);
    //         wj += cj;
    //         d.getCol(j, dj.dMat);
    //         w.dot(dj, ej.dMat);
    //         wj -= ej;

    //         cublasGetMatrix(1, n, sizeof(float), wj.dMat, 1, h_n_tmp, 1);
    //         float norm = 0;
    //         for(int i = 0; i < n; i++){
    //             h_n_tmp[i] = h_n_tmp[i] < eps ? eps : h_n_tmp[i];
    //             norm += h_n_tmp[i] * h_n_tmp[i];
    //         }
    //         norm = sqrt(norm);
    //         if(norm > 0)
    //             for(int i = 0; i < n; i++)
    //                 h_n_tmp[i] /= norm;
    //         cublasSetMatrix(1, n, sizeof(float), h_n_tmp, 1, wj.dMat, 1);
    //         w.setCol(j, wj);
    //     }
    // }

    w.dot(h, y.dMat);
    y.inspect();

    hj.freeMat();
    aj.freeMat();
    bj.freeMat();
    cj.freeMat();
    dj.freeMat();
    ej.freeMat();
    fj.freeMat();
    wj.freeMat();
    djj.freeMat();
    a.freeMat();
    b.freeMat();
    c.freeMat();
    d.freeMat();
    x.freeMat();
    w.freeMat();
    h.freeMat();
    y.freeMat();
    free(h_m_tmp);
    free(h_n_tmp);
    h_m_tmp = NULL;
    h_n_tmp = NULL;
}


int main(){
    cublasInit();

    long m = 2, n = 6, k = 2;
    float eps = 1e-4;
    
    float h_x[] = {1,1,2,1,3,1.2,4,1,5,0.8,6,1};
    NMF(eps, m, n, k, h_x);

    return 0;
}