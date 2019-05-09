#include <vector>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <algorithm>
#include <climits>
#include <float.h>
#include <omp.h>
#include <cstdlib>
#include "helper.hpp"
#define MAX_THREAD 6

using namespace std;

typedef std::vector<double> vd;

void print_vd(vd* _vd, int _size) {
  cout << "----------------------------------------" << endl;
  for(int i = 0 ; i < _vd->size() / _size ; i ++) {
    for(int j = 0 ; j < _size ; j ++) 
      cout << (*_vd)[i * _size + j] << '\t';
    cout << endl;
  }
  cout << "----------------------------------------" << endl;
}

/* single linear system solver 
1. gaussian elimination
2. backward propagation */
void solve_single(vd* _A, vd* _b, vd* _x, int _size) {
  /* gaussian elimination */
  for(int j = 0 ; j < _size ; j ++) {

    //find max index x >= j s.t _A[x, j] \ne 0
    int nonZeroJ = j;
    double maxVal = abs((*_A)[j * _size + j]);
    for(int k = j ; k < _size ; k++) {
      double testVal = abs((*_A)[k * _size + j]);
      if(testVal > maxVal) {
        nonZeroJ = k;
        maxVal = testVal;
      }
    }

    /*swap
    1. A[j:] and A[nonZeroJ:]
    2. b[j] and b[nonZeroJ]
    */
    if(j != nonZeroJ) {
      double temp;
      /* swap A[j:] and A[nonZeroJ:] */
      int jBuf = j * _size;
      int nonZeroJBuf = nonZeroJ * _size;
      for(int k = 0 ; k < _size; k ++) {
        temp = (*_A)[jBuf + k];
        (*_A)[jBuf + k] = (*_A)[nonZeroJBuf + k];
        (*_A)[nonZeroJBuf + k] = temp;
      }
      /* b[j] and b[nonZeroJ] */
      temp = (*_b)[j];
      (*_b)[j]  = (*_b)[nonZeroJ];
      (*_b)[nonZeroJ] = temp;
    }
    /*for all k s.t _A.n > k > j
    1. A[k:] = A[k:] - m * A[j:]
    2. b[k] = b[k] - m * b[j]
    where m = A[k,j] / A[j,j]*/
    for(int k = j + 1 ; k < _size ; k++) {
      double m = (*_A)[k * _size + j] / (*_A)[j * _size + j];
      for(int i = j ; i < _size ; i++)
        (*_A)[k * _size + i] -= m * (*_A)[j*_size + i];
      (*_b)[k] -= m * (*_b)[j];
    }
  }

  /* backward propagation */
  for(int j = _size - 1 ; j >= 0 ; j--) {
    (*_x)[j] = (*_b)[j] / (*_A)[j * _size + j];
    for(int k = j - 1 ; k >= 0 ; k--) {
      (*_b)[k] -= (*_x)[j] * (*_A)[k * _size + j];
    }
  }
}

void solve_omp(vd* _A, vd* _b, vd* _x, int _size, int _nThread) {
  #pragma omp parallel num_threads(_nThread)
  {
    /* gaussian elimination */
    for(int j = 0 ; j < _size ; j ++) {
      #pragma omp single
      {
        //find max index x >= j s.t _A[x, j] \ne 0
        int nonZeroJ = j;
        double maxVal = abs((*_A)[j * _size + j]);
        for(int k = j ; k < _size ; k++) {
          double testVal = abs((*_A)[k * _size + j]);
          if(testVal > maxVal) {
            nonZeroJ = k;
            maxVal = testVal;
          }
        }

        /*swap
        1. A[j:] and A[nonZeroJ:]
        2. b[j] and b[nonZeroJ]
        */
        if(j != nonZeroJ) {
          double temp;
          /* swap A[j:] and A[nonZeroJ:] */
          int jBuf = j * _size;
          int nonZeroJBuf = nonZeroJ * _size;
          for(int k = 0 ; k < _size; k ++) {
            temp = (*_A)[jBuf + k];
            (*_A)[jBuf + k] = (*_A)[nonZeroJBuf + k];
            (*_A)[nonZeroJBuf + k] = temp;
          }
          /* b[j] and b[nonZeroJ] */
          temp = (*_b)[j];
          (*_b)[j]  = (*_b)[nonZeroJ];
          (*_b)[nonZeroJ] = temp;
        }
      }
      //#pragma omp barrier
      /*for all k s.t _A.n > k > j
      1. A[k:] = A[k:] - m * A[j:]
      2. b[k] = b[k] - m * b[j]
      where m = A[k,j] / A[j,j]*/
      #pragma omp for
      for(int k = j + 1 ; k < _size ; k++) {
        double m = (*_A)[k * _size + j] / (*_A)[j * _size + j];
        for(int i = j ; i < _size ; i++)
          (*_A)[k * _size + i] -= m * (*_A)[j*_size + i];
        (*_b)[k] -= m * (*_b)[j];
      }
    }

    #pragma omp single
    {
      /* backward propagation */
      for(int j = _size - 1 ; j >= 0 ; j--) {
        (*_x)[j] = (*_b)[j] / (*_A)[j * _size + j];
        for(int k = j - 1 ; k >= 0 ; k--) {
          (*_b)[k] -= (*_x)[j] * (*_A)[k * _size + j];
        }
      }
    }
  }
}

/* error detection */
double L2Norm(vd* _A, vd* _x, vd* _b, int _size) {
  double error = 0;
  for(int i = 0; i < _size; i++) {
    double sum = 0;
    for(int j = 0 ; j < _size ; j++)
      sum += (*_A)[i * _size + j] * (*_x)[j];
    error += pow(abs(sum - (*_b)[i]),2);
  }
  return error;
}

/* A, b, x_s, ACopy1, bCopy1, x_p, ACopy2, bCopy2 */
vd mtrx[8];
int main(int argc, char** argv) {
  /* initialization */
  random_init();

  int size = atoi(argv[1]);
  int nThread = atoi(argv[2]);
  mtrx[0] = vd(size * size); fillRandom(&mtrx[0]);
  mtrx[1] = vd(size); fillRandom(&mtrx[1]);
  mtrx[2] = vd(size);
  mtrx[3].assign(mtrx[0].begin(), mtrx[0].end());
  mtrx[4].assign(mtrx[1].begin(), mtrx[1].end());
  mtrx[5] = vd(size);
  mtrx[6].assign(mtrx[0].begin(), mtrx[0].end());
  mtrx[7].assign(mtrx[1].begin(), mtrx[1].end());

  uint64_t start, end;
  double error = 0;

  /* solve linear system
  1. single solver
  2. parallel solver*/

  /* single solver */
  std::cout << "Single Thread Computation Start" << std::endl;
  start = GetTimeStamp();
  solve_single(&mtrx[0], &mtrx[1], &mtrx[2], size);
  end = GetTimeStamp();
  cout << "error : " << L2Norm(&mtrx[6], &mtrx[2], &mtrx[7], size) << endl;
  std::cout << "Single Thread Computation End: " << end - start  << " us." << std::endl;

  /* parallel solver */
  std::cout << "Multi Thread Computation Start" << std::endl;
  start = GetTimeStamp();
  solve_omp(&mtrx[3], &mtrx[4], &mtrx[5], size, nThread);
  end = GetTimeStamp();
  cout << "error : " << L2Norm(&mtrx[6], &mtrx[5], &mtrx[7], size) << endl;
  std::cout << "Multi Thread Computation End: " << end - start << " us." << std::endl;

  return 0;
}