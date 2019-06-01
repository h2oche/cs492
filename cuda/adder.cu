#include <iostream>
#include <math.h>

// __global__
// void add(int n, float* x, float* y) {
//   for(int i = 0 ; i < n ; i ++) 
//     y[i] = x[i] + y[i];
// }

// __global__
// void add(int n, float* x, float* y) {
//   int index = threadIdx.x;
//   int stride = blockDim.x;
//   for(int i = index ; i < n ; i += stride)
//     y[i] = y[i] + x[i];
// }

__global__
void add(int n, float* x, float* y) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index ; i < n ; i += stride)
    y[i] = y[i] + x[i];
}

int main(void) {
  int N = 1 << 20;
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for(int i = 0 ; i < N ; i ++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);
  
  cudaDeviceSynchronize();

  float error = 0.0f;
  for(int i = 0 ; i < N ; i ++)
    error = fmax(error, fabs(y[i] - 3.0f));
  std::cout << "Max error : " << error << std::endl;
  
  cudaFree(x);
  cudaFree(y);

  return 0;
}