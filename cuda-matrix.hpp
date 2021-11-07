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
    CuMatrix(long row, long col, T *mat);
    CuMatrix<T> operator*(CuMatrix<T> b);
    void freeMat();
    void inspect();
};

#endif