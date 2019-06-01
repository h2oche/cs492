CC = nvcc
TARGET = dmm lssolver sparse

all: $(TARGET)
	rm *.o

lssolver: lssolver.cu
	$(CC) -o $@ $^ -std=c++11

dmm: dmm.cu
	$(CC) -o $@ $^ -std=c++11

sparse: sparse.cu mmreader.cpp
	$(CC) -o $@ $^ -std=c++11

max: max.cu
	$(CC) -o $@ $^ -std=c++11
 
clean:
	rm $(TARGET)
	rm *.o
	rm *.o.tmp

adder: adder.cu
	nvcc $^ -o $@

gpu_spec: gpu_spec.cu
	nvcc $^ -o $@