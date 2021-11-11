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

    // tmp = CuMatrix<float>::rand(n, m);
    // x = tmp * 20; tmp.freeMat();
    float h_x[] = {1,1,2,1,3,1.2,4,1,5,0.8,6,1};
    float h_w[n * k] = {};
    float h_h[k * m] = {};
    x = CuMatrix<float>(n, m, h_x);
    w = CuMatrix<float>(n, k, h_w);
    h = CuMatrix<float>(k, m, h_h);

    // w = CuMatrix<float>::rand(n, k);
    // h = CuMatrix<float>::rand(k, m);
    CuMatrix<float> a, b, c, d;
    CuMatrix<float> hj(1, m), aj(m, 1), bj(k, 1), cj(n, 1), dj(k, 1), wj(n, 1), djj(1, 1);
    float* h_m_tmp = (float*)malloc(sizeof(float) * m);
    float* h_n_tmp = (float*)malloc(sizeof(float) * n);
    // float djj;

    // h.inspect();
    for(int iter = 0; iter < 1; iter++){
        a = x.tdot(w); // (m x k)
        b = w.tdot(w); // (k x k)

        for(long j = 0; j < k; j++){
            hj = h.getRow(j, hj.dMat); // H[j, :] (k, m)
            aj = a.getCol(j, aj.dMat); // A[:, j] (m, k)
            bj = b.getCol(j, bj.dMat); // B[:, j] (k, k)
            hj += aj;
            cublasSgemm('T', 'N', hj.rowSize, hj.colSize, bj.rowSize, -1, h.dMat, h.rowSize, bj.dMat, bj.rowSize, 1, hj.dMat, hj.rowSize);
            
            // 最小値設定
            cublasGetMatrix(1, m, sizeof(float), hj.dMat, 1, h_m_tmp, 1);
            for(int i = 0; i < m; i++)
                h_m_tmp[i] = h_m_tmp[i] < eps ? eps : h_m_tmp[i];
            cublasSetMatrix(1, m, sizeof(float), h_m_tmp, 1, hj.dMat, 1);
            h.setRow(j, hj);
        }
        // h.inspect();

        // x.inspect();
        c = x * h; // (n, k)
        // cublasSgemm('N', 'N', x.rowSize, h.rowSize, x.colSize, 1, x.dMat, n, h.dMat, k, 0, c.dMat, n);
        x.inspect();
        h.inspect();
        c.inspect();
        // c.inspect();
        d = h.dott(h); // (k, k)
        for(long j = 0; j < k; j++){
            wj = w.getCol(j, wj.dMat); // (n, 1)
            cublasScopy(1, d.dMat + d.colSize * j + j, 1, djj.dMat, 1); // (1)
            wj *= djj;
            cj = c.getCol(j, cj.dMat); // (n, 1)
            // cj.inspect();
            wj += cj;

            // wj.inspect();

            dj = d.getCol(j, dj.dMat); // (k, 1)
            cublasSgemm('N', 'N', wj.rowSize, wj.colSize, dj.rowSize, -1, w.dMat, w.rowSize, dj.dMat, dj.rowSize, 1, wj.dMat, wj.rowSize);
            
            // wj.inspect();

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
            w.setRow(j, wj);
        }

        // CuMatrix<float> wh = w * h;
        // wh.inspect();
        // wh.freeMat();

        // w.inspect();
        // h.inspect();
        // x.inspect();
    }

    

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