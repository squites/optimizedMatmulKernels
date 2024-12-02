#include <stdio.h>
#include <iostream>

#define TILE_SIZE 32 // tile = 32x32 = 1024

__device__ void load_smem();


// A=(M,k), B=(k,N)
__global__ void gemm_smem_cache_blocking_k(const float *A, const float *B, float *C, size_t m, size_t n, size_t k,) {
    // allocate shared memory
    __shared__ float A_s[TILE_SIZE*TILE_SIZE]; //[TILE_SIZE];
    __shared__ float B_s[TILE_SIZE*TILE_SIZE]; //[TILE_SIZE];
    // each thread is responsible for a tile of C
    const int c_col = blockIdx.x * blockDim.x + threadIdx.x; // blockdim == TILE_SIZE
    const int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    // tile row and column
    const int trow = blockIdx.y; // coalesced
    const int tcol = blockIdx.x;

    // pointers to the 1st element of the current tile in GMEM (tileA, tileB, tileC)
    float *ptr_A = ptr_A +  trow * TILE_SIZE * k;
    float *ptr_B = ptr_B +  tcol * TILE_SIZE;
    float *ptr_C = ptr_C + (trow * TILE_SIZE * N) + (tcol * TILE_SIZE); // point to the 1st element of current tile C

    // do if barrier here?

    //for (int i = 0; i < k; i+=TILE_SIZE {
    for (int tileix = 0; tileix < k/TILE_SIZE; tileix++) { // loops for the number of tiles in the row or the col
        for (int ki = 0; ki < k; ki++) {
        // fill shared (same for storing in C naive matmul)
        // all threads load one element of A and one B into shared
        A_s[threadIdx.y * TILE_SIZE + threadIdx.x] = ptr_A[c_row * k + ki]; // se tile(2x2), para pular para os elementos da proxima linha do mesmo tile, basta multiplicar pelo num de elementos em cada linha (k)
        B_s[threadIdx.y * TILE_SIZE + threadIdx.x] = ptr_B[trow * n + tcol]; // aqui é *n porque tem n elementos em cada row na matrix B
        // x0, x1, y0, y1, z0, z1,
        // x2, x3, y2, y3, z2, z3
        // tile 2x2
        // memory: x0, x1 | y0, y1 | z0, z1 | x2, x3,| y2, y3,| z2, z3
        // wait for all threads to finish loading
        }
        __syncthreads();

        // dot product
        float sum = 0.0f;
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_s[tileix * TILE_SIZE + i] * B_s[tileix * TILE_SIZE + i];
        }
        C[]

        // move pointers to the next tile
        ptr_A += tileix * TILE_SIZE;
        ptr_B += 

    }
}

// mapping thread to gmem = threadIdx.y * blockDim.x + threadIdx.x;
// mapping block  to gmem = blockIdx.y * gridDim.x + blockIdx.x;


/*
- assing a thread block to a C_tile (resulting tile)
- so the thread block will load one tile of A and one tile of B into smem
- each thread of that block will load one element of tile A and one of tile B
- shared memory: shared across all threads of the same block
- shared memory will load an entire row of tiles and col of tiles
*/