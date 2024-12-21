// can change for c++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "kernel_1.cuh"

// working only with square matrices first
#define ROW 5 
#define COL 5

#define BLOCK_SIZE 16

void init_randf(float *data, const int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() & 0xFF)/10.0f;
    }
}

void print_data(float *data, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%.4f, ", data[i]);
    }
    printf("]\n");
}

void print_matrix(float *data, int n) {
    printf("[");
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (j == COL) {
            printf("\n");
            j = 0;
        }
        printf("%.4f, ", data[i]);
        j++;
    }
    printf("]\n\n");
}

// cpu matmul
void matmul_h(const float *A, const float *B, float *C,
              const size_t m, const size_t n, const size_t k) {
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int ki = 0; ki < k; ki++) {
                sum += A[i * n + ki] * B[ki * m + j];
            }
            // compare the result with the kernels
            //assert(sum == C[i*m+j]);
            C[i * m + j] = sum;
        }
    }
}

void verify(float *h_data, float *d_data, int N) {
    for (int i = 0; i < N; i++) {
        if (h_data[i] != d_data[i]) {
            printf("Error! The results don't match.\n");
            return 1;
        }
    }
    printf("Results match!");
}

int main(int argc, char **argv) {
    // set device
    int dev = 0;
    cudaSetDevice(dev);
    // matrix size
    const int n = ROW*COL;
    const size_t size = n * sizeof(float);
    // allocate host memory for the matrices
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size); // need to change this size
    float *gpuRef = (float*)malloc(size);

    // initialize matrices 
    init_randf(h_A, n);
    init_randf(h_B, n);

    matmul_h(h_A, h_B, h_C, ROW, COL, n);

    //print_data(h_A, n);
    // print matrix
    //print_matrix(h_A, n);
    //print_matrix(B_h, n);
    //print_matrix(C_h, n);

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
    gemm_matmul_naive_k<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS, COLS, n);

    // transfer result back to host
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);

    // verify result
    verify(h_C, gpuRef, n);


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
