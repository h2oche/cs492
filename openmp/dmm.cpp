#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>
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

void multiply_omp(vd* _A, vd* _B, vd* _C, int _size, int _nThread) {
  #pragma omp parallel num_threads(_nThread)
  #pragma omp for
  {
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
  multiply_omp(&mtrx[0], &mtrx[1], &mtrx[3], size, nThread);
  end = GetTimeStamp();
  std::cout << "Multi Thread Computation End: " << end - start << " us." << std::endl;

  /* verification */
  error = 0;
  for(int i = 0 ; i < mtrxSize ; i ++)
    error += abs(mtrx[2][i] - mtrx[3][i]);
  cout << "error : " << error << endl;
}