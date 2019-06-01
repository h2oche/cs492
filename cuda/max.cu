#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

__global__
void findMax(int* _A, int _j, int _size, int* _maxBuffer) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int bd = blockDim.x;
  int gd = gridDim.x;

  int base = bx * bd * 2;
  int mid = bx * bd * 2 + bd;

  _maxBuffer[base + tx] = base + tx + _j;
  _maxBuffer[mid + tx] = mid + tx + _j;
  __syncthreads();
  int stride = gd == bx + 1 ? (_size - _j - base + 1)/2 : bd;

  for(; stride > 0 ;stride = (stride + 1) >> 1) {
    if(tx < stride) {
      int firstIdx = _maxBuffer[base + tx];
      int secondIdx = _maxBuffer[base + tx];
      if(base + tx + stride < _size) secondIdx = _maxBuffer[base + tx + stride];
      _maxBuffer[base + tx] = abs(_A[firstIdx]) > abs(_A[secondIdx]) ? firstIdx : secondIdx;
    }
    if(stride == 1) break;
    __syncthreads();
  }
}

void print_buffer(int* _buf, int _size) {
  for(int i = 0 ; i < _size ; i ++)
    printf("%5d(%04d)", _buf[i], i);
  printf("\n");
}

int buffer[100000];

int main(int argc, char** argv) {
  int TEST_SIZE = atoi(argv[1]);
  int REDUCE_WIDTH = atoi(argv[2]);
  int j = atoi(argv[3]);

  struct timeval tv;
  gettimeofday(&tv, NULL);
  int usec = tv.tv_usec;
  srand48(usec);

  for(int i = 0 ; i < TEST_SIZE ; i ++) buffer[i] = -i;

  // print_buffer(buffer, TEST_SIZE);

  /* shuffle */
  for(int i = TEST_SIZE - 1; i >= 0 ; i --) {
    int j = (int) (drand48() * (i+1));
    int t = buffer[i];
    buffer[i] = buffer[j];
    buffer[j] = t;
  }

  print_buffer(buffer, TEST_SIZE);
  int* dbuf;
  int* maxBuffer;
  cudaMalloc(&dbuf, sizeof(int) * TEST_SIZE);
  cudaMemcpy(dbuf, buffer, sizeof(int) * TEST_SIZE, cudaMemcpyHostToDevice);
  cudaMalloc(&maxBuffer, sizeof(int) * TEST_SIZE);

  dim3 grid( ((TEST_SIZE - j + 1)/2 + REDUCE_WIDTH -1)/REDUCE_WIDTH, 1, 1);
  dim3 block(REDUCE_WIDTH, 1, 1);
  findMax<<<grid, block>>>(dbuf, j, TEST_SIZE, maxBuffer);

  for(int i = 0 ; i < ((TEST_SIZE - j + 1)/2 + REDUCE_WIDTH -1)/REDUCE_WIDTH ; i ++) {
    int answer = -1;
    cudaMemcpy(&answer, maxBuffer + i * REDUCE_WIDTH * 2, sizeof(int), cudaMemcpyDeviceToHost);
    printf("answer : %dth = %d\n", answer, buffer[answer]);
  }
  

  cudaFree(dbuf);
  cudaFree(maxBuffer);
  return 0;
}