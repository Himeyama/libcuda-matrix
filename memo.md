## cublasSetMatrix()
```cpp
cublasStatus_t
cublasSetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda, void *B, int ldb)
```

この関数は、`行 × 列` の要素のタイルをホストメモリ空間の行列 `A` から GPU メモリ空間の行列 `B` にコピーします。各要素は `elemSize` バイトのストレージを必要とし、両方の行列が列マジョール形式で保存されていることを想定しており、コピー元の行列 `A` とコピー先の行列 `B` の先頭次元はそれぞれ `lda` と `ldb` で指定されています。先頭の次元は、割り当てられた行列の行数を示しますが、行列の部分行列のみが使用される場合もあります。

## cublas<T>gemm()

```cpp
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
```

