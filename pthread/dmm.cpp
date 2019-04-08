#include <vector>
#include <iostream>
#include <cmath>
#include <pthread.h>
#include "helper.hpp"
#define BLOCK 32

using namespace std;

typedef std::vector<double> vd;

struct pthread_args {
  int i, nThread, size;
  vd* A;
  vd* B;
  vd* C;
};

typedef std::vector<pthread_t> vth;
typedef std::vector<pthread_args> vthargs;

void multiply_single(vd* _A, vd* _B, vd* _C, int _size) {
  for(int i = 0 ; i < _size ; i++) {
    for(int j = 0 ; j < _size ; j++) {
      double sum = 0;
      int iBuf = i * _size;
      for(int k = 0 ; k < _size ; k++)
        sum += (*_A)[iBuf + k] * (*_B)[k * _size + j];
      (*_C)[iBuf + j] = sum;
    }
  }
}

static void* multiply_pthread_func(void* _args) {
  pthread_args* args = (pthread_args*)_args;

  int startRow = args->size * (args->i) / args->nThread;
  int endRow = args->size * (args->i + 1) / args->nThread;
  int size = args->size;

  
  for(int i = startRow ; i < endRow ; i += BLOCK) {
    for(int j = 0 ; j < size ; j += BLOCK) {
      for(int k = 0 ; k < size ; k += BLOCK) {
        int iMax = std::min(endRow, i + BLOCK);
        int jMax = std::min(size, j + BLOCK);
        int kMax = std::min(size, k + BLOCK);
        for(int i1 = i ; i1 < iMax ; i1 ++) {
          for(int j1 = j ; j1 < jMax ; j1 ++) {
            double sum = 0;
            for(int k1 = k ; k1 < kMax ; k1 ++)
              sum += (*args->A)[i1 * size + k1] * (*args->B)[k1 * size + j1];
            (*args->C)[i1 * size + j1] += sum;
          }
        }
      }
    }
  }
}

void multiply_pthread(vd* _A, vd* _B, vd* _C, int _size, int _nThread) {
  vth threads = vth(_nThread);
  vthargs threadArgs(_nThread);
  int status;
  
  for(int i = 0 ; i < _nThread ; i ++) {
    threadArgs[i].i = i;
    threadArgs[i].nThread = _nThread;
    threadArgs[i].size = _size;
    threadArgs[i].A = _A;
    threadArgs[i].B = _B;
    threadArgs[i].C = _C;
  }

  for(int i = 1 ; i < _nThread ; i ++)
    pthread_create(&threads[i], NULL, multiply_pthread_func, &threadArgs[i]);
  multiply_pthread_func(&threadArgs[0]);
  for(int i = 1 ; i < _nThread ; i ++)
    pthread_join(threads[i], (void **)&status);
}

/* A, B, C1, C2 */
vd mtrx[4];

int main(int argc, char** argv) {
  /* initialization */
  random_init();

  int size = atoi(argv[1]);
  int nThread = atoi(argv[2]);
  int mtrxSize = size * size;
  for(int i = 0 ; i < 4 ; i ++)
    mtrx[i] = vd(mtrxSize);
  for(int i = 0 ; i < 2 ; i ++)
    fillRandom(&mtrx[i]);
  uint64_t start, end;
  double error = 0;

  // cout << size << endl << nThread << endl;
  
  /* dense matrix multiplication
  1. serial mm
  2. parallel mm */
  /* serial mm */
  std::cout << "Single Thread Computation Start" << std::endl;
  start = GetTimeStamp();
  multiply_single(&mtrx[0], &mtrx[1], &mtrx[2], size);
  end = GetTimeStamp();
  std::cout << "Single Thread Computation End: " << end - start  << " us." << std::endl;

  /* parallel mm */
  std::cout << "Multi Thread Computation Start" << std::endl;
  start = GetTimeStamp();
  multiply_pthread(&mtrx[0], &mtrx[1], &mtrx[3], size, nThread);
  end = GetTimeStamp();
  std::cout << "Multi Thread Computation End: " << end - start << " us." << std::endl;

  /* verification */
  error = 0;
  for(int i = 0 ; i < mtrxSize ; i ++)
    error += abs(mtrx[2][i] - mtrx[3][i]);
  cout << "error : " << error << endl;
}