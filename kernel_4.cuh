#include <stdio.h>
#include <stdlib.h>

// adds new inner loop for calculating multiple C entries per thread
// each thread calculates a column of results of tile C (multiple results)
__global__ void gemm_1d_blocktiling_k(const float *A, const float *B, float *C, size_t m, size_t n, size_t k) {
    // tiles dimmensions 
    const int BM = 64, BN = 64;
    const int BK = 8;
    const int TM = BM/BK; // BM/blockDim.y;
    // C thread global mapping
    const int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    // C tile mapping
    int trow = blockIdx.y;
    int tcol = blockIdx.x; 
    // allocate shared memory
    __shared__ float A_shared[BM * BK]; // 64x8
    __shared__ float B_shared[BK * BN]; // 8x64

    // pointers
    float *A = A + trow * k * BM; // points to 1st tile element of each trow
    float *B = B + tcol * BK; // points to 1st tile element of each tcol
    float *C = C + trow * (n * BK) + tcol * BK; // ?

    float sum = 0.0f;
    int nchunks = k/BK;
    for (int ch = 0; ch < nchunks; ch++) {
        float threadResults[TM]; // store all result of that thread (column of results per thread)
        // load into smem
        for (int i = 0; i < BK; i++) {
            A_shared[threadIdx.y * BK + threadIdx.x + i] = A[c_row * BK + c_col + i];
            B_shared[threadIdx.y * BK + threadIdx.x + i] = B[c_row * BK + c_col + (i * n + c_col)]; // n or BN
        }
    
    
        // move pointers
        //...
    }
}

/*
OBS:
- each thread block is responsible for calculating one tile of C
- each thread is also responsible for loading multiple values into memory

Reminders:
- global index calculation: tells which element each thread should handle in the matrix. The calculation is:
    crow = blockIdx.y * blockDim.y + threadIdx.y;
    ccol = blockIdx.x * blockDim.x + threadIdx.x;

- shared memory index: tells the index within the shared memory that the thread(threadIdx.x, threadIdx.y) will load 
  the element from global memory:
        smem[threadIdx.y * blockDim.x + threadIdx.x]; // is the flattened index of smem, 
  instead of smem[threadIdx.x][threadIdx.y]
*/