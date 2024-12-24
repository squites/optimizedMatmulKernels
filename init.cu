// can change for c++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "kernel_1.cuh"

// working only with square matrices first
#define ROW 3 
#define COL 3
// CUDA dims
#define BLOCK_SIZE 3

/*
__global__ void gemm_matmul_naive_k(const float *A, const float *B, float *C, int rows, int cols, int K) {
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < rows && iy < cols) {
		float sum = 0.0f;
	        for (int i = 0; i < K; i++) {
	        	sum += A[ix * K + i] * B[i * cols + iy];
		}
		C[ix * cols + iy] = sum;
	}
}*/

void init_randf(float *data, const int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() & 0xFF)/10.0f;
    }
}

void init_float(float *data, int m, int n, float min, float max) {
    static int is_seeded = 0;
    if (!is_seeded) {
        srand((unsigned int)time(NULL));
        is_seeded = 1;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            data[i * n + j] = min + ((float)rand() / (float)RAND_MAX) * (max - min);
        }
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
void matmul_h(float *A, float *B, float *C,
              int m, int n, int K) {
    //float sum = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * n + k] * B[k * m + j];
            }
            C[i * m + j] = sum;
            //printf("sum: %.2f\n", sum);
        }
    }
}

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

int main(int argc, char **argv) {
    // set device
    int dev = 0;
    cudaSetDevice(dev);
    // matrix size
    const int n = ROW*COL; // 64
    const size_t size = n * sizeof(float);
    // allocate host memory for the matrices
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size); 
    float *gpuRef = (float*)malloc(size);

    // initialize matrices 
//    init_float(h_A, ROW, COL, -150, 150);
 //   init_float(h_B, ROW, COL, -150, 150);
    init_randf(h_A, n);
    init_randf(h_B, n);

    matmul_h(h_A, h_B, h_C, ROW, COL, COL);
    //mat(h_A, h_B, h_C, ROW, COL, n);

    printf("h_A: ");
    print_data(h_A, n);
    printf("h_B: ");
    print_data(h_B, n);
    printf("h_C: ");
    print_data(h_C, n);
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
    gemm_matmul_naive_k<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROW, COL, COL); // the last "COL" is to represent the width

    // transfer result back to host
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);

    printf("gpuRef: ");
    print_data(gpuRef, n);

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