#include <nmf-hals.hpp>

template <typename T>
void inspect(T* mat, std::int64_t row, std::int64_t col){
    std::string str = "[";
    for(std::int64_t i = 0; i < row; i++){
        str += "[";
        for(std::int64_t j = 0; j < col; j++){
            str += std::to_string(mat[i * col + j]) + (j == col - 1 ? "" : ", ");
        }
        str += (i == row - 1 ? "]" : "], ");
    }
    str += "] (" +  std::to_string(row) + ", " + std::to_string(col) + ")";
    std::cout << str << std::endl;
}

int main(){
    cublasInit();

    long m = 2, n = 6, k = 2;
    double data[] = {1,1,2,1,3,1.2,4,1,5,0.8,6,1};
    NMF<double> nmf(m, n, k, data);

    inspect<double>(nmf.W, n, k);
    inspect<double>(nmf.H, k, m);
    inspect<double>(nmf.Y, n, m);
    inspect<double>(data, n, m);

    return 0;
}
