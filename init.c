// can change for c++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_randf(float *data, const int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() & 0xFF)/10.0f;
    }
}

void init_2(float *data, const int size) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand()/(float)RAND_MAX) * 10;
    }
}

void print_matrix(float *data, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%f, ", data[i]);
    }
    printf("]\n");
}

x, x,       y, y, y
x, x,   X   y, y, y
x, x, 

void matmul_h(const float *A, const float *B, float *C,
              const size_t m, const size_t n, const size_t k) {
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int ki = 0; ki < k; ki++) {
                sum += A[i * n + ki] * B[ki * m + i]; // something like this
            }
            C[]
        }
    }

}

#define ROW 3
#define COL 3
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

    print_matrix(A_h, n);
    print_matrix(B_h, n);



    // frees host allocated memory
    free(A_h);
    free(B_h);
    free(C_h);
}