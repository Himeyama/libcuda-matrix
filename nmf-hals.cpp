#include <cstdlib>
#include <cublas.h>
#include <cuda-matrix.hpp>
#include <iostream>
#include <typeinfo>

int main(){
    cublasInit();

    long m = 2, n = 6, k = 2;
    double eps = 1e-4;

    CuMatrix<float> x, w, h, tmp;

    float h_x[] = {1,1,2,1,3,1.2,4,1,5,0.8,6,1};
    x = CuMatrix<float>(n, m, h_x);
    w = CuMatrix<float>(n, k);
    h = CuMatrix<float>(k, m);

    CuMatrix<float> a(m, k), b(k, k), c(n, k), d(k, k);
    CuMatrix<float> hj(1, m), aj(m, 1), bj(k, 1), cj(n, 1), dj(k, 1), wj(n, 1), djj(1, 1);
    float* h_m_tmp = (float*)malloc(sizeof(float) * m);
    float* h_n_tmp = (float*)malloc(sizeof(float) * n);

    for(int iter = 0; iter < 10; iter++){
        a = x.tdot(w);
        b = w.tdot(w);

        for(long j = 0; j < k; j++){
            hj = h.getRow(j, hj.dMat);
            aj = a.getCol(j, aj.dMat);
            bj = b.getCol(j, bj.dMat);
            hj += aj;
            hj -= h.tdot(bj);

            // 最小値設定
            cublasGetMatrix(1, m, sizeof(float), hj.dMat, 1, h_m_tmp, 1);
            for(int i = 0; i < m; i++)
                h_m_tmp[i] = h_m_tmp[i] < eps ? eps : h_m_tmp[i];
            cublasSetMatrix(1, m, sizeof(float), h_m_tmp, 1, hj.dMat, 1);
            h.setRow(j, hj);
        }
        c = x.dott(h);       
        d = h.dott(h);

        for(long j = 0; j < k; j++){
            wj = w.getCol(j, wj.dMat);
            cublasScopy(1, d.dMat + d.colSize * j + j, 1, djj.dMat, 1);
            wj *= djj;
            cj = c.getCol(j, cj.dMat);
            wj += cj;
            dj = d.getCol(j, dj.dMat);
            wj -= w * dj;

            // 最小値設定
            cublasGetMatrix(1, n, sizeof(float), wj.dMat, 1, h_n_tmp, 1);
            float norm = 0;
            for(int i = 0; i < n; i++){
                h_n_tmp[i] = h_n_tmp[i] < eps ? eps : h_n_tmp[i];
                norm += h_n_tmp[i] * h_n_tmp[i];
            }
            norm = sqrt(norm);
            if(norm > 0)
                for(int i = 0; i < n; i++)
                    h_n_tmp[i] /= norm;
            cublasSetMatrix(1, n, sizeof(float), h_n_tmp, 1, wj.dMat, 1);
            w.setCol(j, wj);
        }

        CuMatrix<float> wh = w * h;
        wh.inspect();
        wh.freeMat();
    }

    w.inspect();
    h.inspect();

    x.freeMat();
    w.freeMat();
    h.freeMat();

    hj.freeMat();
    aj.freeMat();
    bj.freeMat();
    wj.freeMat();
    djj.freeMat();

    a.freeMat();
    b.freeMat();

    free(h_m_tmp);
    free(h_n_tmp);

    return 0;
}