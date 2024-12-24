#include <iostream>
#include <stdio.h>

// from blog simoeh 
/*
__global__ void gmem_coalesce_2_k(const float *A, const float *B, float *C, int width) {
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    if (x < width && y < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) { 
            sum += A[x * width + i] * B[i * width + y];
        }
        C[x * width + y] = sum;
    }
}*/

// from leimao blog
__global__ void gemm_coalesced_v2_k(const float *A, const float *B, float *C, size_t m, size_t n, size_t k) {
    size_t const c_col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const c_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (c_col < n && c_row < m) {
        float psum = 0.0f;
        for (int i = 0; i < k; i++) {
            psum += A[c_row * k + i] * B[i * n + c_col];
        }
        C[c_row * k + c_col] = psum;
    }
}


// from PMPP book
__global__ void gemm_coalesced_k(const float *A, const float *B, float *C, size_t m, size_t n, size_t k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[k * n + col];
        }
    }
}

/*
# Threads of the same warp execute the same instruction
# great: when all threads in a warp access consecutive global memory locations
# when warp threads perform LOAD instruction, the hardware check if the accesses are consecutive in global memory. If so,
  the hardware combine (coalesce) all these accesses into a consolidate access to consecutive DRAM locations
# this coalesces allows DRAM to deliver data as burst
# on matmul, each thread access a row of M and a col of N.
# coalesce happens among threads, not amongst different iterations of the loop within each thread execution
# in matmul, each thread access a col of N. So each thread will access the 0th element of its corresponded col,
  which means that all this elements combined forms a row 0 of matrix N. Which can coalesce

  t0  t1  t2
   |   |   |
   x0  x1  x2
   x3  x4  x5
   x6  x7  x8

1st iteration, t0 access x0, t1 access x1 and t2 access x2, which this row can coalesce

# to clarify, each thread is assigned to a position in the resulting matrix C, and calculates the row and column dot product
corresponded to that position element. the K is the number of elements in the row (the same for column).
# so, the loop of a thread that reads one element of this coalesced method is the same for all the other threads, 
meaning they all execute the same iteration. In the example above, all threads read the 1st element o

   t0  t1  t2 <     x0  x1  x2      x0  x1  x2
   |   |   |        t0  t1  t2 <    x3  x4  x5
   x0  x1  x2       |   |   |       t0  t1  t2 <
   x3  x4  x5       x3  x4  x5      |   |   |
   x6  x7  x8       x6  x7  x8      x6  x7  x8

# this way coalesces because the matrices are stored in row-major order in memory, so the columns are stored in consecutive
memory locations
*/