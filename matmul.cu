#include <stdio.h>
#include <math.h>

// kernel 1: Naive implementation
// to get all elements of a col. i*cols gets to the beginning of each row. +iy gets to the same column every time
__global__ void matmul_naive(const float *A, const float *B, float *out, int rows, int cols, int K) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread x index (pos. in *out responsible for this thread)
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread y index (pos. in *out responsible for this thread)
    unsigned int idx = ix * cols + iy; // mapping to output
    // barrier
    if (ix < rows && iy < cols) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[ix * K + i] * B[i * cols + iy] 
        }
        out[idx] = sum;
    }
    __syncthreads();
}

// initializes the matrix with random values. (add to ./utils.cpp)
void init_data(float *data, const int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() & 0xFF)/10.0f;
    }
}


int main(int argc, char** argv) {
    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // matrix size
    const int rows = 4092;
    const int cols = 4092;
    const int size = rows*cols;
    printf("Matrix size (%d, %d)\n", sqrt(size), sqrt(size));

    // allocate host memory
    size_t nbytes = size * sizeof(float);
    float *h_A   = (float*)malloc(nbytes);
    float *h_B   = (float*)malloc(nbytes);
    // output for host and device
    float *h_ref = (float*)malloc(nbytes);
    float *d_ref = (float*)malloc(nbytes);

    // initialize data with random values
    init_data(h_A, size);
    init_data(h_B, size);
    memset(h_out, 0, nbytes);
    memset(d_out, 0, nbytes);

    // allocate device memory
    float *d_A, *d_B, *d_out;
    cudaMalloc((float**)&d_A, nbytes);
    cudaMalloc((float**)&d_B, nbytes);
    cudaMalloc((float**)&d_out, nbytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);

    // invoke kernel
    //dim3 block (size);
    //dim3 grid (size/block.x);
    //matmul_naive<<<grid, block>>>(d_A, d_B, d_out, rows, cols, size);
    //printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    // copy result back from device to host
    cudaMemcpy(d_ref, d_out, size);

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    // free host memory
    free(h_A);
    free(h_B);
    free(h_ref);
    free(d_ref);

    return 0;
}