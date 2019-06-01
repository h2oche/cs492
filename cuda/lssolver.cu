#include <vector>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <algorithm>
#include <climits>
#include <float.h>
#include <cstdint>
#include <cstdlib>
#define REDUCE_WIDTH 512

uint64_t GetTimeStamp();
void random_init(void);
void fillRandom(float*, int);

void random_init() {
  srand48((long)time(NULL));
}

void print_mtrx(float* _mtrx, int _size) {
  printf("--------------------------------------------------\n");
  for(int i = 0 ; i < _size ; i ++) {
    for(int j = 0 ; j < _size ; j++) {
      printf("%10.2f", _mtrx[i * _size + j]);
    }
    printf("\n");
  }
  printf("--------------------------------------------------\n");
}

void print_vector(float* _vec, int _size) {
  printf("--------------------------------------------------\n");
  for(int i = 0 ; i < _size ; i ++)
    printf("%5.2f", _vec[i]);
  printf("\n");
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

using namespace std;

/* single linear system solver 
1. gaussian elimination
2. backward propagation */
void solve_single(float* _A, float* _b, float* _x, int _size) {
  /* gaussian elimination */
  for(int j = 0 ; j < _size ; j ++) {
    // print_mtrx(_A, _size);
    // print_vector(_b, _size);
    //find max index x >= j s.t _A[x, j] \ne 0
    int nonZeroJ = j;
    float maxVal = abs(_A[j * _size + j]);
    for(int k = j ; k < _size ; k++) {
      float testVal = abs(_A[k * _size + j]);
      if(testVal > maxVal) {
        nonZeroJ = k;
        maxVal = testVal;
      }
    }
    // printf("[FINAL - %d]max val : %f, nonZeroJ : %d\n", j, maxVal, nonZeroJ);

    /*swap
    1. A[j:] and A[nonZeroJ:]
    2. b[j] and b[nonZeroJ]
    */
    if(j != nonZeroJ) {
      float temp;
      /* swap A[j:] and A[nonZeroJ:] */
      int jBuf = j * _size;
      int nonZeroJBuf = nonZeroJ * _size;
      for(int k = 0 ; k < _size; k ++) {
        temp = _A[jBuf + k];
        _A[jBuf + k] = _A[nonZeroJBuf + k];
        _A[nonZeroJBuf + k] = temp;
      }
      /* b[j] and b[nonZeroJ] */
      temp = _b[j];
      _b[j]  = _b[nonZeroJ];
      _b[nonZeroJ] = temp;
    }
    /*for all k s.t _A.n > k > j
    1. A[k:] = A[k:] - m * A[j:]
    2. b[k] = b[k] - m * b[j]
    where m = A[k,j] / A[j,j]*/
    for(int k = j + 1 ; k < _size ; k++) {
      float m = _A[k * _size + j] / _A[j * _size + j];
      for(int i = j ; i < _size ; i++)
        _A[k * _size + i] -= m * _A[j*_size + i];
      _b[k] -= m * _b[j];
    }
  }

  /* backward propagation */
  for(int j = _size - 1 ; j >= 0 ; j--) {
    _x[j] = _b[j] / _A[j * _size + j];
    for(int k = j - 1 ; k >= 0 ; k--) {
      _b[k] -= _x[j] * _A[k * _size + j];
    }
  }
}

__global__
void print_A(float* _A, int _size) {
  printf("--------------------------------------------------\n");
  for(int i = 0 ; i < _size ; i ++) {
    for(int j = 0 ; j < _size ; j++) {
      printf("%10.2f", _A[i * _size + j]);
    }
    printf("\n");
  }
  printf("--------------------------------------------------\n");
}

__global__
void print_b(float* _b, int _size) {
  printf("--------------------------------------------------\n");
  for(int i = 0 ; i < _size ; i ++)
    printf("%5.2f", _b[i]);
  printf("\n");
  printf("--------------------------------------------------\n");
}

__global__
void find_max_single(float* _A, int j, int _size, int* _maxBuffer) {
  float maxVal = abs(_A[j * _size + j]);
  int nonZeroJ = j;

  for(int k = j ; k < _size ; k++) {
    float testVal = abs(_A[k * _size + j]);
    if(testVal > maxVal) {
      nonZeroJ = k;
      maxVal = testVal;
    }
  }
  _maxBuffer[0] = nonZeroJ;
}

/* find max value */
__global__
void find_max(float* _A, int _j, int _size, int* _maxBuffer) {
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
      _maxBuffer[base + tx] = abs(_A[firstIdx * _size + _j]) > abs(_A[secondIdx * _size + _j]) ? firstIdx : secondIdx;
    }
    __syncthreads();
    if(stride == 1) break;
  }
}

/* swap row */
__global__
void swap_row(float* _A, int _j1, int _j2, int _size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int bd = blockDim.x;
  int idx = tx + bx * bd;

  if(idx < _size) {
    float temp = _A[_j1 * _size + idx];
    _A[_j1 * _size + idx] = _A[_j2 * _size + idx];
    _A[_j2 * _size + idx] = temp;
  }
}

__global__
void swap_one(float* _b, int _j1, int _j2) {
  float temp = _b[_j1];
  _b[_j1] = _b[_j2];
  _b[_j2] = temp;
}

__global__
void get_m(float* _A, float* _mBuffer, int _j, int _size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int bd = blockDim.x;

  int row = tx + bx * bd;
  if(row > _j && row < _size)
    _mBuffer[row] = _A[row * _size + _j] / _A[_j * _size + _j];
}

__global__
void subtract_row(float* _A, float* _b, float* _mBuffer, int _j, int _size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x; int by = blockIdx.y;
  int bdx = blockDim.x;
  
  int row = by + 1 + _j;
  int col = bx * bdx + tx;
  float m = _mBuffer[row];
  if(col >= _j && col < _size) _A[row * _size + col] -= m * _A[_j*_size + col];
  if(col == 0) _b[row] -= m * _b[_j];
}

__global__
void subtract_row_single(float* _A, float* _b, int j, int _size) {
  for(int k = j + 1 ; k < _size ; k++) {
    float m = _A[k * _size + j] / _A[j * _size + j];
    for(int i = j ; i < _size ; i++)
      _A[k * _size + i] -= m * _A[j*_size + i];
    _b[k] -= m * _b[j];
  }
}

__global__
void back_propa(float* _A, float* _x, float* _b, int _j, int _size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int bd = blockDim.x;

  int idx = tx + bx * bd;
  float new_x = _b[_j] / _A[_j * _size + _j];
  if(idx < _j )
    _b[idx] -= new_x * _A[idx * _size + _j];
  else if(idx == _j)
    _x[_j] = new_x;
} 

void solve_cuda(float* _A, float* _b, float* _x, int _size) {
  /* device memory initialization */
  float* dA;
  float* db; 
  float* dx;
  int* maxBuffer;
  float* mBuffer;

  int size = _size;
  int mtrxSize = _size * size;

  cudaMalloc(&dA, mtrxSize * sizeof(float));
  cudaMemcpy(dA, _A, mtrxSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&db, size * sizeof(float));
  cudaMemcpy(db, _b, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&dx, size * sizeof(float));
  cudaMemset(dx, 0, size * sizeof(float));

  cudaMalloc(&maxBuffer, size * sizeof(int));
  cudaMalloc(&mBuffer, size * sizeof(float));

  /* gaussian elimination */
  for(int j = 0 ; j < _size - 1 ; j ++) {
    // print_A<<<1,1>>>(dA, size);
    // print_b<<<1,1>>>(db, size);
    /* max pivoting */
    int nonZeroJ = j; float maxVal = 0;
    int gridSize = ( (_size - j + 1)/2 + REDUCE_WIDTH - 1) / REDUCE_WIDTH;
    dim3 reduceGrid( gridSize,1,1);
    dim3 reduceBlock(REDUCE_WIDTH,1,1);
    find_max<<<reduceGrid, reduceBlock>>>(dA, j, size, maxBuffer);
    
    for(int i = 0 ; i < gridSize ; i ++) {
      int idx = -1;
      float val = 0;
      cudaMemcpy(&idx, maxBuffer + i * REDUCE_WIDTH * 2, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&val, dA + idx * size + j, sizeof(float), cudaMemcpyDeviceToHost);
      if(abs(val) > maxVal) {
        maxVal = abs(val);
        nonZeroJ = idx;

        // printf("max val : %f, nonZeroJ : %d\n", maxVal, nonZeroJ);
      }
    }
    // find_max_single<<<1,1>>>(dA, j, size, maxBuffer);
    // cudaMemcpy(&nonZeroJ, maxBuffer, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&maxVal, dA + nonZeroJ * size + j, sizeof(float), cudaMemcpyDeviceToHost);
    // printf("[FINAL - %d]max val : %f, nonZeroJ : %d\n", j, maxVal, nonZeroJ);

    /*swap row */
    if(j != nonZeroJ) {
      /* swap b[j] and b[nonZeroJ] */ 
      swap_one<<<1,1>>>(db, j, nonZeroJ);

      /* swap A[j:] and A[nonZeroJ:] */
      dim3 swapGrid( (size + REDUCE_WIDTH - 1) / REDUCE_WIDTH,1,1);
      dim3 swapBlock(REDUCE_WIDTH, 1, 1);
      swap_row<<<swapGrid, swapBlock>>>(dA, j, nonZeroJ, size);
    }
    /*row subtraction*/
    dim3 mGrid( (size + REDUCE_WIDTH - 1) / REDUCE_WIDTH, 1, 1);
    dim3 mBlock(REDUCE_WIDTH, 1, 1);
    get_m<<<mGrid, mBlock>>>(dA, mBuffer, j, size);
    
    dim3 subtractGrid( (size + REDUCE_WIDTH - 1) / REDUCE_WIDTH,size-(j+1),1);
    dim3 subtractBlock(REDUCE_WIDTH,1,1);
    subtract_row<<<subtractGrid, subtractBlock>>>(dA, db, mBuffer, j, size);
  }

  /* backward propagation */
  for(int j = _size - 1 ; j >= 0 ; j--) {
    dim3 backGrid((j + REDUCE_WIDTH) / REDUCE_WIDTH,1,1);
    dim3 backBlock(REDUCE_WIDTH, 1, 1);
    back_propa<<<backGrid, backBlock>>>(dA, dx, db, j, _size);
  }

  cudaMemcpy(_x, dx, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(db);
  cudaFree(dx);
  cudaFree(maxBuffer);
  cudaFree(mBuffer);
}

/* error detection */
float L2Norm(float* _A, float* _x, float* _b, int _size) {
  float error = 0;
  for(int i = 0; i < _size; i++) {
    float sum = 0;
    for(int j = 0 ; j < _size ; j++)
      sum += _A[i * _size + j] * _x[j];
    error += pow(abs(sum - _b[i]),2);
  }
  return error;
}

/* A, b, x_s, ACopy1, bCopy1, x_p, ACopy2, bCopy2 */
float* A;
float* b;
float* x1;
float* A_copy1;
float* b_copy1;
float* x2;

int main(int argc, char** argv) {
  /* initialization */
  random_init();

  int size = atoi(argv[1]);
  int mtrxSize = size * size;

  A = (float *)calloc(mtrxSize, sizeof(float)); fillRandom(A, mtrxSize);
  b = (float *)calloc(size, sizeof(float)); fillRandom(b, size);
  x1 = (float *)calloc(size, sizeof(float));

  A_copy1 = (float *)malloc(mtrxSize * sizeof(float)); memcpy(A_copy1, A, mtrxSize * sizeof(float));
  b_copy1 = (float *)malloc(size * sizeof(float)); memcpy(b_copy1, b, size * sizeof(float));
  x2 = (float *)calloc(size, sizeof(float));
  
  uint64_t start, end;

  /* solve linear system
  1. single solver
  2. parallel solver*/

  /* single solver */
  std::cout << "Single Thread Computation Start" << std::endl;
  start = GetTimeStamp();
  solve_single(A_copy1, b_copy1, x1, size);
  end = GetTimeStamp();
  cout << "error : " << L2Norm(A, x1, b, size) << endl;
  std::cout << "Single Thread Computation End: " << end - start  << " us." << std::endl;
  // print_vector(x1, size);

  /* parallel solver */
  std::cout << "Multi Thread Computation Start" << std::endl;
  start = GetTimeStamp();
  solve_cuda(A, b, x2, size);
  end = GetTimeStamp();
  cout << "error : " << L2Norm(A, x2, b, size) << endl;
  std::cout << "Multi Thread Computation End: " << end - start << " us." << std::endl;
  // print_vector(x2, size);

  free(A);
  free(b);
  free(x1);
  free(A_copy1);
  free(b_copy1);
  free(x2);

  return 0;
}