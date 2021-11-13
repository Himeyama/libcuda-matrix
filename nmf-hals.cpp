#include <cublas.h>
#include <cuda-matrix.hpp>
#include <iostream>

template <class T>
class NMF{
    public:
    long n_samples, n_components, n_features;
    T *W, *H, *Y;

    void cublasCopy(int n, const float *x, int incx, float *y, int incy){
        cublasScopy(n, x, incx, y, incy);
    }

    void cublasCopy(int n, const double *x, int incx, double *y, int incy){
        cublasDcopy(n, x, incx, y, incy);
    }

    NMF(long m, long n, long k, T *dat, T eps = 1e-4){
        n_samples = n;
        n_features = m;
        n_components = k;

        CuMatrix<T> 
            x(n, m, dat),
            a(m, k), 
            b(k, k), 
            c(n, k), 
            d(k, k),
            hj(1, m),
            aj(m, 1), 
            bj(k, 1), 
            cj(n, 1), 
            dj(k, 1), 
            ej(n, 1), 
            fj(k, m), 
            wj(n, 1), 
            djj(1, 1),
            w(n, k),
            h(k, m),
            y(n, m);
        T* h_m_tmp = (T*)malloc(sizeof(T) * m);
        T* h_n_tmp = (T*)malloc(sizeof(T) * n);

        for(int iter = 0; iter < 10; iter++){
            x.tdot(w, a.dMat);
            w.tdot(w, b.dMat);

            for(long j = 0; j < k; j++){
                h.getRow(j, hj.dMat);
                a.getCol(j, aj.dMat);
                b.getCol(j, bj.dMat);
                hj += aj;
                h.tdot(bj, fj.dMat);
                hj -= fj;

                // 最小値設定
                cublasGetMatrix(1, m, sizeof(T), hj.dMat, 1, h_m_tmp, 1);
                for(int i = 0; i < m; i++)
                    h_m_tmp[i] = h_m_tmp[i] < eps ? eps : h_m_tmp[i];
                cublasSetMatrix(1, m, sizeof(T), h_m_tmp, 1, hj.dMat, 1);
                h.setRow(j, hj);
            }
            x.dott(h, c.dMat);       
            h.dott(h, d.dMat);

            for(long j = 0; j < k; j++){
                w.getCol(j, wj.dMat);
                cublasCopy(1, d.dMat + d.colSize * j + j, 1, djj.dMat, 1);
                wj.dot(djj, wj.dMat);
                c.getCol(j, cj.dMat);
                wj += cj;
                d.getCol(j, dj.dMat);
                w.dot(dj, ej.dMat);
                wj -= ej;

                cublasGetMatrix(1, n, sizeof(T), wj.dMat, 1, h_n_tmp, 1);
                T norm = 0;
                for(int i = 0; i < n; i++){
                    h_n_tmp[i] = h_n_tmp[i] < eps ? eps : h_n_tmp[i];
                    norm += h_n_tmp[i] * h_n_tmp[i];
                }
                norm = sqrt(norm);
                if(norm > 0)
                    for(int i = 0; i < n; i++)
                        h_n_tmp[i] /= norm;
                cublasSetMatrix(1, n, sizeof(T), h_n_tmp, 1, wj.dMat, 1);
                w.setCol(j, wj);
            }
        }

        w.dot(h, y.dMat);
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
        free(h_m_tmp);
        free(h_n_tmp);
        h_m_tmp = NULL;
        h_n_tmp = NULL;

        W = w.toMem();
        H = h.toMem();
        Y = y.toMem();

        w.freeMat();
        h.freeMat();
        y.freeMat();
    }
};

template class NMF<float>;
template class NMF<double>;
