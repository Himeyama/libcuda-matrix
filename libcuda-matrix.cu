#include <iostream>
#include <cublas.h>


template <class T>
class CuMatrix{
    private:
    T *dMat;

    public:
    long rowSize;
    long colSize;

    public:
    CuMatrix(long row, long col, T *mat){
        rowSize = row;
        colSize = col;
        cublasAlloc(row * col, sizeof(T), (void**)&dMat);
        if(mat != NULL)
            cublasSetMatrix(row, col, sizeof(T), mat, rowSize, dMat, rowSize);
    }

    void freeMat(){
        cublasFree(dMat);
    }


    CuMatrix<T> operator*(CuMatrix<T> b){
        CuMatrix<T> c(rowSize, b.colSize, NULL);
        if(typeid(T) == typeid(float))
            cublasSgemm('N', 'N', rowSize, b.colSize, colSize, 1, (const float*)dMat, rowSize, (const float*)b.dMat, colSize, 0, (float*)c.dMat, rowSize);
        else if(typeid(T) == typeid(double))
            cublasDgemm('N', 'N', rowSize, b.colSize, colSize, 1, (const double*)dMat, rowSize, (const double*)b.dMat, colSize, 0, (double*)c.dMat, rowSize);
        return c;
    }

    void inspect(){
        T *mat = (T*)malloc(sizeof(T) * rowSize * colSize);
        cublasGetMatrix(rowSize, colSize, sizeof(T), dMat, rowSize, mat, rowSize);

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
};

template class CuMatrix<float>;
template class CuMatrix<double>;
