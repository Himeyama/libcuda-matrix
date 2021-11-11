#ifndef CUDA_MATRIX_HPP
#define CUDA_MATRIX_HPP

template <class T>
class CuMatrix{
    private:
    

    public:
        T *dMat;
        long rowSize;
        long colSize;

    public:
        CuMatrix();
        CuMatrix(long row, long col, T *mat = NULL, bool mode = true);
        
        static CuMatrix rand(long row, long col);
        static CuMatrix I(long, T = 1);

        CuMatrix<T> tdot(CuMatrix<T> b);
        CuMatrix<T> dott(CuMatrix<T> b);
        CuMatrix<T> operator-=(CuMatrix<T> b);
        CuMatrix<T> operator+=(CuMatrix<T> b);
        CuMatrix<T> operator*(T b);
        CuMatrix<T> operator*(CuMatrix<T> b);
        CuMatrix<T> times(CuMatrix<T> b);
        CuMatrix<T> rdivide(CuMatrix<T> b);
        CuMatrix<T> getRow(long i, T* d_vec = NULL);
        CuMatrix<T> getCol(long i, T* d_vec = NULL);
        CuMatrix<T> setRow(long i, CuMatrix<T> b);
        CuMatrix copy();

        void freeMat();
        void inspect();
        T* toMem();


};

#endif