// can change for c++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
//#include "kernel_1.cuh"
#include "caller.cuh"

// working only with square matrices first
#define ROW 1024//8//2048 
#define COL 1024//8//2048
// CUDA dims
#define BLOCK_SIZE 32//2048

// generate random values and fill the matrix
void randfloat(float *data, const int size, float min, float max) {
    static int seed_init = 0;
    if (!seed_init) {
        srand((unsigned int)time(NULL));
        seed_init = 1;
    }
    for (int i = 0; i < size; i++) {
        data[i] = min + (float)rand() / (float)(RAND_MAX / (max - min));
    }
} 

void print_data(float *data, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.2f, ", data[i]);
    }
    printf("\n");
}

void print_matrix(float *data, int n) {
    printf("[");
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (j == COL) {
            printf("\n");
            j = 0;
        }
        printf("%.2f, ", data[i]);
        j++;
    }
    printf("]\n");
}

// cpu matmul
void matmul_h(float *A, float *B, float *C, int m, int n, int K) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * n + k] * B[k * m + j];
            }
            C[i * m + j] = sum;
        }
    }
}
/*
#define ABS_TOL 1e-5
#define REL_TOL 1e-3
void verify(float *h_data, float *d_data, int N) {
    for (int i = 0; i < N; i++) {
        float diff = fabs(h_data[i] - d_data[i]);
        float max = fmax(fabs(h_data[i]), fabs(d_data[i])); 
        if (fabs(h_data[i] - d_data[i]) > eps) {
            printf("These numbers don't match: %.4f, %.4f\n", h_data[i], d_data[i]);
            printf("Error! The results don't match.\n");
            exit(EXIT_FAILURE);
        }
    }
    printf("Results match!\n");
}
*/

int main(int argc, char **argv) {
    // set device (needed?)
    int dev = 0;
    cudaSetDevice(dev);
    // matrix size
    const int n = ROW*COL; // 64
    const size_t size = n * sizeof(float);
    // allocate host memory for the matrices
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size); 
    float *gpuResult = (float*)malloc(size);

    // initialize matrices 
    randfloat(h_A, n, -50, 50);
    randfloat(h_B, n, -50, 50);

    matmul_h(h_A, h_B, h_C, ROW, COL, COL);

    // debugging
    printf("h_A: "); //print_data(h_A, n);
    printf("h_B: "); //print_data(h_B, n);
    printf("h_C: "); //print_data(h_C, n);

    // allocate device memory for the matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, size);
    cudaMalloc((float**)&d_B, size);
    cudaMalloc((float**)&d_C, size);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // invoke kernel (write a kernel runner after)
    dim3 const blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 const gridDim(BLOCK_SIZE/blockDim.x, BLOCK_SIZE/blockDim.y);
    
    // kernel 1
    gemm_matmul_naive_k<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROW, COL, COL); // the last "COL" is to represent the width
    cudaMemcpy(gpuResult, d_C, size, cudaMemcpyDeviceToHost); // transfer result back to host
    printf("kernel 1: "); print_data(gpuResult, n);

    // kernel 2
    //gemm_coalesced_k<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROW, COL, COL);
    //cudaMemcpy(gpuResult, d_C, size, cudaMemcpyDeviceToHost);
    //printf("kernel 2: "); print_data(gpuResult, n);

    dim3 const blockDim2(CHUNK_SIZE, CHUNK_SIZE);
    dim3 const gridDim2(CHUNK_SIZE/blockDim2.x, CHUNK_SIZE/blockDim2.y);
    // kernel 3
    gemm_smem_cache_blocking_v2_k<<<gridDim2, blockDim2>>>(d_A, d_B, d_C, ROW, COL, COL);
    cudaMemcpy(gpuResult, d_C, size, cudaMemcpyDeviceToHost);
    printf("kernel 3: "); print_data(gpuResult, n);

    // verify result
    //verify(h_C, gpuResult, n);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // frees host allocated memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}