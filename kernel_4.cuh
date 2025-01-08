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
    int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
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
        // load into SMEM
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < BK; j++) {
                // load TMxBK block into smem
                A_shared[threadIdx.y * BK + threadIdx.x + i*BK + j] = A[(c_row + i) * k + (c_col + j)];
                // "threadIdx.y * BK + threadIdx.x": base unique index for each thread
                // "i * BK": row offset. i iterates over TM rows that the thread is responsible for
                // "+ j": col offset. iterates over the columns of BK 
            }
            // if TM == BK, we can load to B outside the inner loop
            // load the respective column with BK elements
            B_shared[threadIdx.x * BK + i] = B[(c_row + i) * n + c_col];  
        }
        __syncthreads();

        // dot product
        for (int i = 0; i < BK; i++) {
            float B_tmp = B_shared[i * BN + threadIdx.x]; // the author uses threadCol instead of threadIdx.x; Each iteration here, B_tmp will be one element from the column that's in B_shared
            for (int res = 0; res < TM; res++) {    
                // dotprod between each row from (TMxBK) with respective col (BK)
                threadResults[res] += A_shared[res * BK + ] * B_tmp; // fix A_shareD indexing
            }
        }

    
    
        // move pointers
        //...
    }
}

/*
OBS:
- each thread block is responsible for calculating one tile of C
- each thread is also responsible for loading multiple values into memory
- each thread is responsible for load TM rows of BK,(TM,BK), elements from tile A(BM, BK). And load a single column
of BK elements from tile B(BK, BN). So, load a block (TM,BK) from tileA, and load a column (BK,1) from tileB.
- After the loading, each thread will be responsible for loading the necessary elements to calculate a column of
tile C of TM elements. 
Remember: each thread is supposed to compute a column (subcolumn more specific), of size (TM,1) elements. Is a
subcolumn because the entire tile C will have size (BM,BN).

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