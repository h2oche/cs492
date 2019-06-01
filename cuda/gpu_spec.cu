#include <stdio.h>

int main() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for(int i = 0 ; i < nDevices ; i ++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  device overlap: %d\n", prop.deviceOverlap);
    printf("  max grid dim: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  max thread dim: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  # of multi-processors: %d\n", prop.multiProcessorCount); 
    printf("  shared mem per block: %lu\n", prop.sharedMemPerBlock);
    printf("  total const memory: %lu\n", prop.totalConstMem);
    printf("  total global memory: %lu\n", prop.totalGlobalMem);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  can map host memory: %d\n\n", prop.canMapHostMemory);
  }

  return 0;
}