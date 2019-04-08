#include <vector>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <algorithm>
#include <climits>
#include <float.h>
#include <pthread.h>
#include <cstdlib>
#include "helper.hpp"
#define MAX_THREAD 6

using namespace std;

struct pthread_args {
  int i, nThread, size;
  vd* A;
  vd* b;
  vd* x;
  bool is_main;
};

typedef std::vector<double> vd;
typedef std::vector<pthread_t> vth;
typedef std::vector<pthread_args> vthargs;

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

pthread_mutex_t forward_mutex[MAX_THREAD];
pthread_cond_t pivot_cond[MAX_THREAD];
pthread_cond_t subtract_cond[MAX_THREAD];
bool pivot_end[MAX_THREAD];
bool subtract_end[MAX_THREAD];

int start_j = 0;
bool forward_end = false;

static void* row_subtract_pthread_func(void* _args) {
  pthread_args* args = (pthread_args*)_args;
  int thread_num = args->i;

  while(!forward_end) {
    /* wait for pivot */
    if(!args->is_main) {
      pthread_mutex_lock(&forward_mutex[thread_num]);
      while(!pivot_end[thread_num])
        pthread_cond_wait(&pivot_cond[thread_num], &forward_mutex[thread_num]);
    }
    
    /* row subtraction */
    int rowCnt = args->size - (start_j + 1);
    int startRow = rowCnt * thread_num / args->nThread + start_j + 1;
    int endRow = rowCnt * (thread_num + 1) / args->nThread + start_j + 1;
    int size = args->size;

    // printf("thread num(%d) compute from %d to %d\n", thread_num, startRow, endRow);

    for(int i = startRow ; i < endRow ; i++) {
      double m = (*args->A)[i * size + start_j] / (*args->A)[start_j*size + start_j];
      for(int k = start_j ; k < size ; k++)
        (*args->A)[i * size + k] -= m * (*args->A)[start_j*size + k];
      (*args->b)[i] -= m * (*args->b)[start_j];
    }

    if(!args->is_main) {
      pivot_end[thread_num] = false;
      subtract_end[thread_num] = true;
      pthread_cond_signal(&subtract_cond[thread_num]);
      pthread_mutex_unlock(&forward_mutex[thread_num]);
    }
    else break;
  }
  
}

void solve_pthread_init(int _nThread) {
  for(int i = 0 ; i < _nThread ; i ++) {
    // pivot_mutex[i] = PTHREAD_MUTEX_INITIALIZER;
    // subtract_mutex[i] = PTHREAD_MUTEX_INITIALIZER;
    forward_mutex[i] = PTHREAD_MUTEX_INITIALIZER;
    
    pivot_cond[i] = PTHREAD_COND_INITIALIZER;
    subtract_cond[i] = PTHREAD_COND_INITIALIZER;

    pivot_end[i] = false;
    subtract_end[i] = true;
  }
}

void solve_pthread(vd* _A, vd* _b, vd* _x, int _size, int _nThread) {
  bool first = true;

  /* initialize */
  solve_pthread_init(_nThread);

  vth threads = vth(_nThread);
  vthargs threadArgs(_nThread);
  int status;

  for(int i = 0 ; i < _nThread ; i ++) {
    threadArgs[i].i = i;
    threadArgs[i].nThread = _nThread;
    threadArgs[i].size = _size;
    threadArgs[i].A = _A;
    threadArgs[i].b = _b;
    threadArgs[i].x = _x;
    threadArgs[i].is_main = false;
  }
  threadArgs[0].is_main = true;

  for(int i = 1 ; i < _nThread ; i ++)
    pthread_create(&threads[i], NULL, row_subtract_pthread_func, &threadArgs[i]);

  /* gaussian elimination */
  for(int j = 0 ; j < _size ; j ++) {
    if(j == _size - 1) {
      forward_end = true;
      
      for(int k = 1 ; k < _nThread ; k++) {
        pivot_end[k] = true;
        pthread_cond_signal(&pivot_cond[k]);
      }
      break;
    }

    /* wait for subtraction */
    for(int k = 1 ; k < _nThread ; k++) {
      pthread_mutex_lock(&forward_mutex[k]);
      while(!subtract_end[k])
        pthread_cond_wait(&subtract_cond[k], &forward_mutex[k]);
    }

    // printf("partial pivoting(%d)\n", j);

    /* partial pivot */
    int nonZeroJ = j;
    double maxVal = abs((*_A)[j * _size + j]);
    for(int k = j ; k < _size ; k++) {
      double testVal = abs((*_A)[k * _size + j]);
      if(testVal > maxVal) {
        nonZeroJ = k;
        maxVal = testVal;
      }
    }

    /*swap*/
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
    
    /* set argument */
    start_j = j;

    /* signal to other thread */
    for(int k = 1 ; k < _nThread ; k++) {
      pivot_end[k] = true;
      subtract_end[k] = false;
      pthread_cond_signal(&pivot_cond[k]);
      pthread_mutex_unlock(&forward_mutex[k]);
    }

    row_subtract_pthread_func(&threadArgs[0]);
  }

  for(int i = 1 ; i < _nThread ; i ++)
    pthread_join(threads[i], (void **)&status);

  /* backward propagation */
  for(int j = _size - 1 ; j >= 0 ; j--) {
    (*_x)[j] = (*_b)[j] / (*_A)[j * _size + j];
    for(int k = j - 1 ; k >= 0 ; k--) {
      (*_b)[k] -= (*_x)[j] * (*_A)[k * _size + j];
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

vd mtrx[8];
int main(int argc, char** argv) {
  /* A, b, x_s, ACopy1, bCopy1, x_p, ACopy2, bCopy2 */
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
  solve_pthread(&mtrx[3], &mtrx[4], &mtrx[5], size, nThread);
  end = GetTimeStamp();
  cout << "error : " << L2Norm(&mtrx[6], &mtrx[5], &mtrx[7], size) << endl;
  std::cout << "Multi Thread Computation End: " << end - start << " us." << std::endl;

  return 0;
}