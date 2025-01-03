#include <stdio.h>
#include <stdlib.h>

// each thread calculates a column of results of tile C
__global__ void gemm_1d_blocktiling_k(const float *A, const float *B, float *C, size_t m, size_t n, size_t k) {
    // for now
    const int BM = 32;
    const int BK = 16;
    const int BN = 32;
    // C thread global mapping
    const int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    // C tile mapping
    const int trow = blockIdx.y; // ?
    const int tcol = blockIdx.x; // ? 
    // allocate shared memory
    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // pointers
    float *A_ptr = A + trow * k * BK;
    float *B_ptr = B + tcol * BK;
}