#ifndef CUDA_MATRIX_HPP
#define CUDA_MATRIX_HPP

template <class T>
class CuMatrix{
    private:
    T *dMat;

    public:
        long rowSize;
        long colSize;

    public:
        CuMatrix();
        CuMatrix(long row, long col, T *mat);
        
        static CuMatrix rand(long row, long col);
        static CuMatrix I(long, T = 1);

        CuMatrix<T> tdot(CuMatrix<T> b);
        CuMatrix<T> dott(CuMatrix<T> b);
        CuMatrix<T> operator*(T b);
        CuMatrix<T> operator*(CuMatrix<T> b);
        CuMatrix<T> times(CuMatrix<T> b);
        CuMatrix<T> rdivide(CuMatrix<T> b);


        void freeMat();
        void inspect();
        T* toMem();
};

#endif