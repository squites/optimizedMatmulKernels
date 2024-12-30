#include <stdio.h>
#include <iostream>


//__device__ void load_smem();

/*
#define TILE_SIZE 32 // tile = 32x32 = 1024
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
// each thread assigned to one element of C
// 1 - load chunk of A and chunk of B from GMEM to SMEM
// 2 - work of these chunks
// move the chunks along the row and column
*/


#define CHUNK_SIZE 32
__global__ void gemm_smem_cache_blocking_v2_k(const float *A, const float *B, float *C, 
                                              size_t m, size_t n, size_t k) {
    // allocate shared memory
    __shared__ float A_shared[CHUNK_SIZE * CHUNK_SIZE]; // 32x32
    __shared__ float B_shared[CHUNK_SIZE * CHUNK_SIZE];
    // C thread mapping
    int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    // C tile mapping
    int trow = blockIdx.y;
    int tcol = blockIdx.x;

    // initializing pointers to the start of the first chunk
    float *A_ptr = trow * k * CHUNK_SIZE; // points to 1st element of 1st chunk on each tile row
    float *B_ptr = tcol * CHUNK_SIZE;     // points to 1st element of 1st chunk on each tile column
    float *C_ptr = trow * (n * CHUNK_SIZE) + tcol * CHUNK_SIZE;
    // row_idx * width: pula para o começo da linha correspondente com o indice.
    // + CHUNKSIZE: chega no primeiro elemento to próximo chunk na mesma linha.

    // loop over tiles
    float sum = 0.0f;
    const int nchunks = k/CHUNK_SIZE;
    // while (chunk < nchunks) {
    for (int chunki = 0; chunki < nchunks; chunki++) { // (each thread will eventually load one element of each chunk)
        // thread loads element of A and B chunks into SMEM. All threads are doing the same thing on different data at "same" time
        A_shared[threadIdx.y * CHUNK_SIZE + threadIdx.x] = A_ptr[c_row * k + c_col]; 
        B_shared[threadIdx.y * CHUNK_SIZE + threadIdx.x] = B_ptr[c_row * n + c_col];
        __syncthreads();

        // barrier?
        // calculate dot-product (at this point, all threads of that block loaded the corresponded element of that chunk into shared, so we can already do matmul on this tile)
        for (int i = 0; i < CHUNK_SIZE*CHUNK_SIZE; i++) { // should be CHUNK_SIZE*CHUNK_SIZE ?
            sum += A_shared[c_row * CHUNK_SIZE + i] * B_shared[i * n + c_col]
        }
        __syncthreads();
        C_ptr[c_row * n + c_col] = sum;
    }

}

// OBS (IMPORTANTE): cada thread block, é responsável por um TILE da matrix C. Para calcular esse C tile, dentro de  
// um loop, o thread block vai carregar a shared memory com um tile de A e um tile de B, e computar o tile C 
// parcialmente. Depois, na próxima iteração, o MESMO thread block, vai carregar a shared memory com o próximo tile
// de A o próximo tile de B, e adicionar o novo resultado ao resultado parcial anterior. Repete isso para todos os 
// tiles daquela tilerow e daquela tilecol.
// Para computar cada resultado parcial do tile C, em cada iteração, o thread block computa uma matmul entre 
// o tile A e o tile B. Com isso ele tem um resultado parcial do tile C. faz isso até acabar os tiles daquela tilerow
// e daquela tilecol.
//
// OBS 2 (IMPORTANTE): o loop itera por cada chunk. O porquê disso é que um thread_block é responsável por computar
// um tile de C. Para isso, ele precisa calcular uma tileRow inteira de A e uma tileCol inteira de B, e para isso,
// é necessário carregar chunks de A e de B para a shared memory. Só que pensa comigo, o ideal seria colocar toda a
// tileRow e tileCol na shared_memory, só que não cabe, pois ela comporta poucos dados. Então, em cada iteração 
// carregamos um chunk de A e um chunk de B, onde cada thread desse thread_block carrega apenas 1 elemento do chunk
// A e um elemento do chunk B. Porém, devemos lembrar que para completar a tileRow e tileCol inteira, é necessário 
// carregar vários chunks para a shared memory. Com isso, a mesma thread que carregou um elemento específico do chunk
// A e do chunk B para a shared_memory, será também responsável de carregar os "mesmos" elementos correspondentes
// de cada próximo chunk. Por isso o loop itera por chunks e não por elementos. 
// Devemos pensar que o kernel roda em uma thread, então precisamos pensar como uma thread. Dessa forma, no loop só 
// é feito um load de A e um load de B em cada iteração, porém, cada thread do block estará fazendo a mesma coisa
// porém de coordenadas diferentes!

// mapping thread to gmem = threadIdx.y * blockDim.x + threadIdx.x;
// mapping block  to gmem = blockIdx.y * gridDim.x + blockIdx.x;


/*
- assing a thread block to a C_tile (resulting tile)
- so the thread block will load one tile of A and one tile of B into smem
- each thread of that block will load one element of tile A and one of tile B
- shared memory: shared across all threads of the same block
- shared memory will load an entire row of tiles and col of tiles
- (reminder) blockIdx.x * blockDim.x + threadIdx.x : global thread ID in x dimension
- number of blocks in a grid: gridDim.x * gridDim.y
- number of threads in a block: blockDim.x * blockDim.y * blockDim.z
- full global thread ID in 2D: 
    x = blockIdx.x * blockDim.x + threadIdx.x
    y = blockIdx.y * blockDim.y + threadIdx.y
*/