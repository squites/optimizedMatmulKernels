// can change for c++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define ROW 3
#define COL 2

void init_randf(float *data, const int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() & 0xFF)/10.0f;
    }
}

void init_2(float *data, const int size) {
    srand((unsigned int) time(NULL));
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand()/RAND_MAX);//(float)rand());
    }
}

void print_data(float *data, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%.4f, ", data[i]);
    }
    printf("]");
}

void print_matrix(float *data, int n) {
    printf("[");
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (j == COL) {
            //printf("\n");
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
    // matrix size
    const int n = ROW*COL;
    const size_t bytes = n * sizeof(float);
    // allocating memory in host for the matrices
    float *A_h = (float*)malloc(bytes);
    float *B_h = (float*)malloc(bytes);
    float *C_h = (float*)malloc(bytes); // need to change this size

    // populate the allocated space
    init_2(A_h, n);
    init_2(B_h, n);

    matmul_h(A_h, B_h, C_h, ROW, COL, n);

    print_data(A_h, n);
    // print matrix
    //print_matrix(A_h, n);
    //print_matrix(B_h, n);
    //print_matrix(C_h, n);



    // frees host allocated memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}