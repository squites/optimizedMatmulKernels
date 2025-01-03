#include <stdio.h>
#include <iostream>

#define CHUNK_SIZE 32
__global__ void gemm_smem_cache_blocking_v2_k(float *A, float *B, float *C, 
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
    float *A_ptr = A + trow * k * CHUNK_SIZE; // points to 1st element of 1st chunk on each tile row
    float *B_ptr = B + tcol * CHUNK_SIZE;     // points to 1st element of 1st chunk on each tile column
    float *C_ptr = C + trow * (n * CHUNK_SIZE) + tcol * CHUNK_SIZE;
    // row_idx * width: pula para o come�o da linha correspondente com o indice.
    // + CHUNKSIZE: chega no primeiro elemento to pr�ximo chunk na mesma linha.

    // loop over tiles
    float sum = 0.0f;
    const int nchunks = k/CHUNK_SIZE;
    // while (chunk < nchunks) {
    for (int chunki = 0; chunki < nchunks; chunki++) { // (each thread will eventually load one element of each chunk)
        // thread loads element of A and B chunks into SMEM. All threads are doing the same thing on different data at "same" time. So when they synchronize, the entire chunk of A and B will be on SMEM, which makes possible to perform the matmul on these tiles 
        A_shared[threadIdx.y * CHUNK_SIZE + threadIdx.x] = A_ptr[c_row * k + c_col]; 
        B_shared[threadIdx.y * CHUNK_SIZE + threadIdx.x] = B_ptr[c_row * n + c_col];
        __syncthreads();

        // barrier?
        // calculate dot-product (at this point, all threads of that block loaded the corresponded element of that chunk into shared, so we can already do matmul on this tile)
        for (int i = 0; i < CHUNK_SIZE; i++) { // this loops to each element of the row and col designated to that thread
            sum += A_shared[c_row * CHUNK_SIZE + i] * B_shared[i * n + c_col];
        }
        __syncthreads();

        // move pointers
        A_ptr += CHUNK_SIZE; // position + 32. jumps to the start of the next tile of A of that thread block
        B_ptr += CHUNK_SIZE * n; // 
    }
    C_ptr[c_row * n + c_col] = sum; // this needs to be outside because in order to calculate the specific element of the C tile, the thread needs to calculate the partial matmul of each A-B pair of tiles. So, sum on first iteration is matmul between 1st tile of A and B. Then, on next iteration, sum adds up the 2nd tile of A and B to the previous result
}

// OBS (IMPORTANTE): cada thread block, � respons�vel por um TILE da matrix C. Para calcular esse C tile, dentro de  
// um loop, o thread block vai carregar a shared memory com um tile de A e um tile de B, e computar o tile C 
// parcialmente. Depois, na pr�xima itera��o, o MESMO thread block, vai carregar a shared memory com o pr�ximo tile
// de A o pr�ximo tile de B, e adicionar o novo resultado ao resultado parcial anterior. Repete isso para todos os 
// tiles daquela tilerow e daquela tilecol.
// Para computar cada resultado parcial do tile C, em cada itera��o, o thread block computa uma matmul entre 
// o tile A e o tile B. Com isso ele tem um resultado parcial do tile C. faz isso at� acabar os tiles daquela tilerow
// e daquela tilecol.
//
// OBS 2 (IMPORTANTE): o loop itera por cada chunk. O porqu� disso � que um thread_block � respons�vel por computar
// um tile de C. Para isso, ele precisa calcular uma tileRow inteira de A e uma tileCol inteira de B, e para isso,
// � necess�rio carregar chunks de A e de B para a shared memory. S� que pensa comigo, o ideal seria colocar toda a
// tileRow e tileCol na shared_memory, s� que n�o cabe, pois ela comporta poucos dados. Ent�o, em cada itera��o 
// carregamos um chunk de A e um chunk de B, onde cada thread desse thread_block carrega apenas 1 elemento do chunk
// A e um elemento do chunk B. Por�m, devemos lembrar que para completar a tileRow e tileCol inteira, � necess�rio 
// carregar v�rios chunks para a shared memory. Com isso, a mesma thread que carregou um elemento espec�fico do chunk
// A e do chunk B para a shared_memory, ser� tamb�m respons�vel de carregar os "mesmos" elementos correspondentes
// de cada pr�ximo chunk. Por isso o loop itera por chunks e n�o por elementos. 
// Devemos pensar que o kernel roda em uma thread, ent�o precisamos pensar como uma thread. Dessa forma, no loop s� 
// � feito um load de A e um load de B em cada itera��o, por�m, cada thread do block estar� fazendo a mesma coisa
// por�m de coordenadas diferentes!
//
// OBS 3 (IMPORTANTE): o segundo loop calcula o elemento parcial de cada respectiva thread. Para isso, esse loop
// calcula um dotproduct entre a respectiva linha do tile_i de A e a respectiva coluna do tile_i de B, logo, i < CHUNK_SIZE.
// Fazendo isso, a vari�vel "sum" ter� o resultado de apenas 1 elemento do tile C. Lembrando que na multiplica��o
// de matrizes, para calcular um elemento i de C, � necess�rio fazer um dotproduct entre a linha correspondente com
// a coluna correspondente. � isso que est� sendo feito aqui, "sum" representa o resultado parcial 
//
//
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