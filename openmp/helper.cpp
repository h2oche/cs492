#include "helper.hpp"

#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>

void random_init() {
  srand48((long)time(NULL));
}

void fillRandom(vd* _mtrx) {
  
  for(int i = 0 ; i < _mtrx->size() ; i++)
    (*_mtrx)[i] = drand48();
}

uint64_t GetTimeStamp() {
  struct timeval tv;
  gettimeofday(&tv,NULL); 
  return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}