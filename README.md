# Libcuda-matrix
NMF (HALS) の Cuda 実装。

使用例: [nmf-test.cpp](nmf-test.cpp)

> 関数とメンバー
```cpp
NMF<T>::NMF(long m, long n, long k, T *dat, T eps = 1e-4);
// long m: 列数
// long n: 行数
// long k: コンポーネント数
// T *data: 配列の先頭ポインタ (n x m)
// T eps: W, H の最小値

T *NMF<T>::H
// H 配列の先頭ポインタ

T *NMF<T>::W
// W 配列の先頭ポインタ

T *NMF<T>::Y
// WH 配列の先頭ポインタ
```

