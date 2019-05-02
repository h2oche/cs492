#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include <vector>
#include <unistd.h>
#include <cstdint>
typedef std::vector<double> vd;
uint64_t GetTimeStamp();
void random_init(void);
void fillRandom(vd*);
#endif // !__HELPER_HPP__ 