NVCC = nvcc
 CXX = /usr/bin/g++
  CC = /usr/bin/gcc
 OPT = -ccbin $(CC) -std=c++11 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75

libcuda-matrix.so: libcuda-matrix.cu
	$(NVCC) $(OPT) -lcublas --shared -Xcompiler -fPIC $^ -o $@

test: test.cpp
	$(NVCC) $(OPT) -lcublas -lcuda-matrix -L. -I. $^ -o $@

test1: test.cu
	$(NVCC) $(OPT) $^ -o $@

install: libcuda-matrix.so
	install -s $^ $(libdir)
	cp cuda-nmf.hpp $(incdir)