// can change for c++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>

#define ROW 5 
#define COL 5

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

int main(int argc, char **argv) {
    // set device
    int dev = 0;
    cudaSetDevice(dev);
    // matrix size
    const int n = ROW*COL;
    const size_t bytes = n * sizeof(float);
    // allocate host memory for the matrices
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes); // need to change this size
    float *gpuRef = (float*)malloc(bytes);

    // populate the allocated space
    init_randf(h_A, n);
    init_randf(h_B, n);

    matmul_h(h_A, h_B, h_C, ROW, COL, n);

    print_data(h_A, n);
    // print matrix
    print_matrix(h_A, n);
    //print_matrix(B_h, n);
    //print_matrix(C_h, n);

    // allocate device memory for the matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, bytes);
    cudaMalloc((float**)&d_B, bytes);
    cudaMalloc((float**)&d_C, bytes);

    // transfer data to gpu
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // invoke kernel

    // transfer kernel result data back to host
//    cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost);

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
