#include <iostream>
#include <cublas.h>
#include <random>
#include <typeinfo>
#include <list>

template <class T>
class CuMatrix{  
    public:
    T *dMat;
    long rowSize;
    long colSize;
    bool alloced;

    CuMatrix(){};

    CuMatrix(long row, long col, T *mat = NULL, bool mode = true){
        // mode == true ? host : device
        rowSize = row;
        colSize = col;
        alloced = false;
        if(mode){
            cublasAlloc(row * col, sizeof(T), (void**)&dMat);
            alloced = true;
        }else{
            dMat = mat;
            alloced = true;
        }
        
        bool zero = false;
        if(mat == NULL){
            mat = (T*)malloc(sizeof(T) * row * col);
            memset(mat, 0, sizeof(T) * row * col);
            zero = true;
        }
        cublasSetMatrix(row, col, sizeof(T), mat, rowSize, dMat, rowSize);
        if(zero) free(mat);
    }

    static CuMatrix rand(long row, long col){
        std::random_device rd;
        std::default_random_engine engine(rd());
        std::uniform_real_distribution<> urd(0, 1);
        T *x = (T*)malloc(sizeof(T) * row * col);
        for(long i = 0; i < row * col; i++)
            x[i] = urd(engine);
        CuMatrix<T> r(row, col, x);
        free(x);
        return r;
    }

    CuMatrix copy(){
        CuMatrix<T> cp(rowSize, colSize);
        cublasCopy(rowSize * colSize, dMat, 1, cp.dMat, 1);
        return cp;
    }

    static CuMatrix I(long n, T k = 1){
        T *x = (T*)malloc(sizeof(T) * n * n);
        memset(x, 0, sizeof(T) * n * n);
        for(long i = 0; i < n; i++)
            x[i * n + i] = k;
        CuMatrix<T> r(n, n, x);
        return r;
    }

    void freeMat(){
        if(alloced)
            cublasFree(dMat);
        alloced = false;
    }

    // 型によって関数を分ける
    void cublasGemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc){
        cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    void cublasGemm(char transa, char transb, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc){
        cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
    }

    CuMatrix<T> tdot(CuMatrix<T> b){
        CuMatrix<T> c(colSize, b.colSize, NULL);
        cublasGemm('T', 'N', c.rowSize, c.colSize, b.rowSize, 1, dMat, rowSize, b.dMat, b.rowSize, 0, c.dMat, c.rowSize);
        return c;
    }

    CuMatrix<T> dott(CuMatrix<T> b){
        CuMatrix<T> c(rowSize, b.rowSize, NULL);
        cublasGemm('N', 'T', c.rowSize, c.colSize, colSize, 1, dMat, rowSize, b.dMat, b.rowSize, 0, c.dMat, c.rowSize);
        return c;
    }

    void cublasAxpy(int n, float alpha, const float *x, int incx, float *y, int incy){
        cublasSaxpy(n, alpha, x, incx, y, incy);
    }

    void cublasAxpy(int n, double alpha, const double *x, int incx, double *y, int incy){
        cublasDaxpy(n, alpha, x, incx, y, incy);
    }

    void operator +=(CuMatrix<T> b){
        if(rowSize == 1 || colSize == 1 || rowSize * colSize == b.rowSize * b.colSize)
            ;// ベクトルの足し算
        else if(rowSize != b.rowSize || colSize != b.colSize)
            std::cerr << "+(CuMatrix): 二つの行列の大きさが異なります" << std::endl;
        cublasAxpy(rowSize * colSize, 1, b.dMat, 1, dMat, 1);
    }

    void operator -=(CuMatrix<T> b){
        if(rowSize == 1 || colSize == 1 || rowSize * colSize == b.rowSize * b.colSize)
            ;// ベクトルの引き算
        else if(rowSize != b.rowSize || colSize != b.colSize)
            std::cerr << "-(CuMatrix): 二つの行列の大きさが異なります" << std::endl;
        cublasAxpy(rowSize * colSize, -1, b.dMat, 1, dMat, 1);
    }

    void operator *=(CuMatrix<T> b){
        cublasGemm('N', 'N', rowSize, colSize, colSize, 1, dMat, rowSize, b.dMat, b.rowSize, 1, dMat, rowSize);
    }

    CuMatrix<T> operator *(CuMatrix<T> b){
        CuMatrix<T> c(rowSize, b.colSize);
        // c.inspect();
        cublasGemm('N', 'N', c.rowSize, c.colSize, colSize, 1, dMat, rowSize, b.dMat, b.rowSize, 0, c.dMat, c.rowSize);
        // c.inspect();
        return c;
    }

    CuMatrix<T> operator *(T b){
        CuMatrix<T> matB = I(colSize, b);
        return *this *(matB);
    }

    CuMatrix<T> times(CuMatrix<T> b){
        CuMatrix<T> r;
        if(!(b.colSize == colSize && b.rowSize == rowSize)){
            std::cerr << "times(): 二つの行列の大きさが異なります" << std::endl;
            return r;
        }
        T *h_a = toMem();
        T *h_b = b.toMem();
        T *h_c = (T*)malloc(sizeof(T) * rowSize * colSize);
        for(long i = 0; i < rowSize * colSize; i++)
            h_c[i] = h_a[i] * h_b[i];
        r = CuMatrix(rowSize, colSize, h_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return r;
    }

    CuMatrix<T> rdivide(CuMatrix<T> b){
        CuMatrix<T> r;
        if(!(b.colSize == colSize && b.rowSize == rowSize)){
            std::cerr << "rdivide(): 二つの行列の大きさが異なります" << std::endl;
            return r;
        }
        T *h_a = toMem();
        T *h_b = b.toMem();
        T *h_c = (T*)malloc(sizeof(T) * rowSize * colSize);
        for(long i = 0; i < rowSize * colSize; i++){
            if(h_b[i] == 0)
                std::cerr << "rdivide(): ゼロ除算" << std::endl;
            h_c[i] = h_a[i] / h_b[i];
        }
        r = CuMatrix(rowSize, colSize, h_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return r;
    }



    void inspect(){
        T *mat = toMem();
        std::string str = "[";
        for(long i = 0; i < rowSize; i++){
            str += "[";
            for(long j = 0; j < colSize; j++){
                str += std::to_string(mat[i * colSize + j]) + (j == colSize - 1 ? "" : ", ");
            }
            str += (i == rowSize - 1 ? "]" : "], ");
        }
        str += "]";
        std::cout << str << std::endl;
        free(mat);
    }

    T* toMem(){
        T *mat = (T*)malloc(sizeof(T) * rowSize * colSize);
        cublasGetMatrix(rowSize, colSize, sizeof(T), dMat, rowSize, mat, rowSize);
        return mat;
    }

    void cublasCopy(int n, const float *x, int incx, float *y, int incy){
        cublasScopy(n, x, incx, y, incy);
    }

    void cublasCopy(int n, const double *x, int incx, double *y, int incy){
        cublasDcopy(n, x, incx, y, incy);
    }

    CuMatrix<T> getRow(long i, T* d_vec = NULL){
        if(d_vec == NULL)
            cublasAlloc(colSize, sizeof(T), (void**)&d_vec);
        cublasCopy(colSize, dMat + colSize * i, 1, d_vec, 1);
        return CuMatrix(1, colSize, d_vec, false);
    }

    CuMatrix<T> getCol(long i, T* d_vec = NULL){
        if(d_vec == NULL)
            cublasAlloc(rowSize, sizeof(T), (void**)&d_vec);
        cublasCopy(rowSize, dMat + i, colSize, d_vec, 1);
        return CuMatrix(rowSize, 1, d_vec, false);
    }

    void setRow(long i, CuMatrix<T> b){
        cublasCopy(colSize, b.dMat, 1, dMat + colSize * i, 1);
    }
};

template class CuMatrix<float>;
template class CuMatrix<double>;
