#include <iostream>
#include <cublas.h>
#include <random>

template <class T>
class CuMatrix{
    private:
    T *dMat;

    public:
    long rowSize;
    long colSize;

    CuMatrix(){};

    CuMatrix(long row, long col, T *mat){
        rowSize = row;
        colSize = col;
        cublasAlloc(row * col, sizeof(T), (void**)&dMat);
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

    static CuMatrix I(long n, T k = 1){
        T *x = (T*)malloc(sizeof(T) * n * n);
        memset(x, 0, sizeof(T) * n * n);
        for(long i = 0; i < n; i++)
            x[i * n + i] = k;
        CuMatrix<T> r(n, n, x);
        return r;
    }

    void freeMat(){
        cublasFree(dMat);
    }

    CuMatrix<T> tdot(CuMatrix<T> b){
        CuMatrix<T> c(colSize, b.colSize, NULL);
        if(typeid(T) == typeid(float))
            cublasSgemm('T', 'N', c.rowSize, c.colSize, b.rowSize, 1, (const float*)dMat, rowSize, (const float*)b.dMat, b.rowSize, 0, (float*)c.dMat, c.rowSize);
        else if(typeid(T) == typeid(double))
            cublasDgemm('T', 'N', c.rowSize, c.colSize, b.rowSize, 1, (const double*)dMat, rowSize, (const double*)b.dMat, b.rowSize, 0, (double*)c.dMat, c.rowSize);
        return c;
    }

    CuMatrix<T> dott(CuMatrix<T> b){
        CuMatrix<T> c(rowSize, b.rowSize, NULL);
        if(typeid(T) == typeid(float))
            cublasSgemm('N', 'T', c.rowSize, c.colSize, rowSize, 1, (const float*)dMat, rowSize, (const float*)b.dMat, b.rowSize, 0, (float*)c.dMat, c.rowSize);
        else if(typeid(T) == typeid(double))
            cublasDgemm('N', 'T', c.rowSize, c.colSize, rowSize, 1, (const double*)dMat, rowSize, (const double*)b.dMat, b.rowSize, 0, (double*)c.dMat, c.rowSize);
        return c;
    }

    CuMatrix<T> operator*(CuMatrix<T> b){
        CuMatrix<T> c(rowSize, b.colSize, NULL);
        if(typeid(T) == typeid(float))
            cublasSgemm('N', 'N', c.rowSize, c.colSize, colSize, 1, (const float*)dMat, rowSize, (const float*)b.dMat, b.rowSize, 0, (float*)c.dMat, c.rowSize);
        else if(typeid(T) == typeid(double))
            cublasDgemm('N', 'N', c.rowSize, c.colSize, colSize, 1, (const double*)dMat, rowSize, (const double*)b.dMat, b.rowSize, 0, (double*)c.dMat, c.rowSize);
        return c;
    }

    CuMatrix<T> operator*(T b){
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
};

template class CuMatrix<float>;
template class CuMatrix<double>;
