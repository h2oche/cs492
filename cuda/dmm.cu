#include <iostream>
#include <cmath>
#include <unistd.h>
#include <sys/time.h>
#include <cstdint>
#include <stdlib.h>
#define BLOCK_WIDTH 16

uint64_t GetTimeStamp();
void random_init(void);
void fillRandom(float*, int);


using namespace std;

void random_init() {
  srand48((long)time(NULL));
}

void print_mtrx(float* _mtrx, int _size) {
  printf("--------------------------------------------------\n");
  for(int i = 0 ; i < _size ; i ++) {
    for(int j = 0 ; j < _size ; j++) {
      printf("%5.2f", _mtrx[i * _size + j]);
    }
    printf("\n");
  }
  printf("--------------------------------------------------\n");
}

void fillRandom(float* _mtrx, int _size) {
  for(int i = 0 ; i < _size ; i++)
    _mtrx[i] = drand48();
}

uint64_t GetTimeStamp() {
  struct timeval tv;
  gettimeofday(&tv,NULL); 
  return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

void multiply_single(float* _A, float* _B, float* _C, int _size) {
  for(int i = 0 ; i < _size ; i++) {
    for(int j = 0 ; j < _size ; j++) { 
      double sum = 0;
      int iBuf = i * _size;
      for(int k = 0 ; k < _size ; k++)
        sum += _A[iBuf + k] * _B[k * _size + j];
      _C[iBuf + j] = sum;
    }
  }
}

__global__
void multiply_cuda(float* _A, float* _B, float* _C, int _size) {
  __shared__ float bufA[BLOCK_WIDTH][BLOCK_WIDTH];
  __shared__ float bufB[BLOCK_WIDTH][BLOCK_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * BLOCK_WIDTH + ty;
  int col = bx * BLOCK_WIDTH + tx;
  int mtrxSize = _size * _size;

  double sum = 0;

  bufA[ty][tx] = 0;
  bufB[ty][tx] = 0;
  __syncthreads();

  for(int i = 0 ; i < (_size + BLOCK_WIDTH - 1) / BLOCK_WIDTH ; i++ ) {
    if(row * _size + i * BLOCK_WIDTH + tx < mtrxSize)
      bufA[ty][tx] = _A[row * _size + i * BLOCK_WIDTH + tx];
    if((i * BLOCK_WIDTH + ty) * _size + col < mtrxSize)
      bufB[ty][tx] = _B[(i * BLOCK_WIDTH + ty) * _size + col];
    __syncthreads();

    // int remain = row - i * BLOCK_WIDTH;
    // remain = remain > BLOCK_WIDTH ? BLOCK_WIDTH : remain;
    
    for(int k = 0; k < BLOCK_WIDTH ; k++)
      sum += bufA[ty][k] * bufB[k][tx];
    __syncthreads();

    bufA[ty][tx] = 0;
    bufB[ty][tx] = 0;
    __syncthreads();
  }

  if(row < _size && col < _size) {
    // printf("%d, %d = %10.2f\n", row, col, sum);
    _C[row * _size + col] = sum;
  }
}

/* A, B, C1, C2 */
float* A;
float* B;
float* C1;
float* C2;

int main(int argc, char** argv) {
  /* initialization */
  random_init();

  int size = atoi(argv[1]);
  int mtrxSize = size * size;
  
  A = (float *)calloc(mtrxSize, sizeof(float));
  B = (float *)calloc(mtrxSize, sizeof(float));
  C1 = (float *)calloc(mtrxSize, sizeof(float));
  C2 = (float *)calloc(mtrxSize, sizeof(float));

  fillRandom(A, mtrxSize);
  fillRandom(B, mtrxSize);

  uint64_t start, end;
  float error = 0;

  /* dense matrix multiplication
  1. serial mm
  2. parallel mm */

  /* serial mm */
  std::cout << "Single Thread Computation Start" << std::endl;
  start = GetTimeStamp();
  multiply_single(A, B, C1, size);
  end = GetTimeStamp();
  std::cout << "Single Thread Computation End: " << end - start  << " us." << std::endl;

  /* parallel mm */
  std::cout << "Multi Thread Computation Start" << std::endl;
  start = GetTimeStamp();

  /* init device memory */
  float* dA;
  float* dB;
  float* dC;
  cudaMalloc(&dA, mtrxSize * sizeof(float));
  cudaMemcpy(dA, A, mtrxSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&dB, mtrxSize * sizeof(float));
  cudaMemcpy(dB, B, mtrxSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&dC, mtrxSize * sizeof(float));
  cudaMemset(dC, 0, mtrxSize * sizeof(float));

  cudaDeviceSynchronize();

  //launch kernel
  dim3 grid((size + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (size + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 1);
  dim3 block(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  multiply_cuda<<<grid,block>>>(dA, dB, dC, size);

  cudaDeviceSynchronize();
  cudaMemcpy(C2, dC, mtrxSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  end = GetTimeStamp();
  std::cout << "Multi Thread Computation End: " << end - start << " us." << std::endl;

  /* verification */
  error = 0;
  for(int i = 0 ; i < mtrxSize ; i ++)
    error += abs(C1[i] - C2[i]);
  cout << "error : " << error << endl;

  // print_mtrx(A, size);
  // print_mtrx(B, size);
  // print_mtrx(C1, size);
  // print_mtrx(C2, size);

  free(A);
  free(B);
  free(C1);
  free(C2);
}