#include <stdio.h>
#include <iostream>

#define TILE_SIZE 32 // tile = 32x32 = 1024

// A=(M,k), B=(k,N)
__global__ void gemm_smem_cache_blocking_k(const float *A, const float *B, float *C, size_t m, size_t n, size_t k,) {
    // allocate shared memory
    __shared__ float A_s[TILE_SIZE*TILE_SIZE]; //[TILE_SIZE];
    __shared__ float B_s[TILE_SIZE*TILE_SIZE]; //[TILE_SIZE];
    // each thread is responsible for a tile of C
    const size_t c_col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t c_row = blockIdx.y * blockDim.y + threadIdx.y;
    // tile row and column
    const int trow = blockIdx.y; // coalesced
    const int tcol = blockIdx.x;

    // pointers that point to the 1st element of the current tile in GMEM (tileA, tileB, tileC)
    float *ptr_A = ptr_A + trow * TILE_SIZE * k;
    float *ptr_B = ptr_B + tcol * TILE_SIZE;
    float *ptr_C = ptr_C + (trow * TILE_SIZE * N) + (tcol * TILE_SIZE); // point to the 1st element of current tile C

    //for (int tileix = 0; tileix < k/TILE_SIZE; tileix++) { // loops for the number of tiles in the row or the col
    for (int i = 0; i < k; i++) {
        // fill shared (same for storing in C naive matmul)
        A_s[c_row * TILE_SIZE + c_col] = A[ptr_A * tileix ]
        B_s[c_row * TILE_SIZE + c_col] = 
    }
}

// mapping thread to gmem = threadIdx.y * blockDim.x + threadIdx.x;
// mapping block  to gmem = blockIdx.y * gridDim.x + blockIdx.x;

for (int i = 0; i < k; i++) {
    sum +=
}
C[row * k + col] = sum

/*
- assing a thread block to a C_tile (resulting tile)
- so the thread block will load one tile of A and one tile of B into smem
- each thread of that block will load one element of tile A and one of tile B
- shared memory: shared across all threads of the same block
- shared memory will load an entire row of tiles and col of tiles
*/