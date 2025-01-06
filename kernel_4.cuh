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
    for (int ch = 0; ch < nchunks; ch++) { // loop through tiles
        float threadResults[TM]; // store all result of that thread (column of results per thread)
        // load into smem
        for (int i = 0; i < BK; i++) {
            // + i: to load the next element of the column or row that the thread is loading 
            // lets change it to each thread loads and entire BK column of the tile B, and all (TM,BK) tile A.
            // A_shared[threadIdx.y * BK + threadIdx.x + i] = A[c_row * k + c_col + i];
            // B_shared[threadIdx.y * BK + threadIdx.x + i] = B[c_row * n + c_col + (i * n + c_col)]; // n or BN
            A_shared[threadIdx.y * BK + threadIdx.x] = A[c_row * k + c_col];
            B_shared[threadIdx.y * BK + threadIdx.x] = B[c_row * n + c_col + (i * n + c_col)]; // loads all elements from this col?
        }

        // --- just experimenting!
        int offset = 0;
        for (int i = 0; i < BK; i++) {
            for (int j = 0; j < TM; j++) {
                A_shared[threadIdx.y * BK + threadIdx.x + offset] = A[c_row * k + c_col + i]; // load all elements from tile A (TM,BK)
                offset++;
            }
            B_shared[threadIdx.y * BK + threadIdx.x + i] = B[c_row * n + c_col + (i * n + c_col)]; // load all elements from the tile column of B
        }
        // --- end experimenting!
        __syncthreads();

    
    
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
    crow = blockIdx.y * blockDim.y + threadIdx.y; // represent the 'y' coordinate on linear global memory
    ccol = blockIdx.x * blockDim.x + threadIdx.x; // represent the 'x' coordinate on linear global memory

    crow * width + ccol; // represent the [x,y] coordinates of that specific element on the linear global memory

- shared memory index: tells the index within the shared memory that the thread(threadIdx.x, threadIdx.y) will load 
  the element from global memory:
        smem[threadIdx.y * blockDim.x + threadIdx.x]; // is the flattened index of smem, 
  instead of smem[threadIdx.x][threadIdx.y]
*/