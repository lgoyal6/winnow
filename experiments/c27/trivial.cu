#include <cstdio>
__global__ void k(float* o, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=o[i]*2.0f+1.0f;}
int main(){float* d; cudaMalloc(&d, 1024*1024*4); k<<<1024,1024>>>(d,1024*1024); cudaDeviceSynchronize();
 printf("err=%s\n", cudaGetErrorString(cudaGetLastError())); cudaFree(d); return 0;}
