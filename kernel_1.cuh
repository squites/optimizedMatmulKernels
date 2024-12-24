// naive kernel 
__global__ void gemm_matmul_naive_k(const float *A, const float *B, float *C, int rows, int cols, int K) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; // thread x index (pos. in *out responsible for this thread)
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // thread y index (pos. in *out responsible for this thread)
    unsigned int idx = ix * cols + iy; // mapping to output
    // barrier
    if (ix < rows && iy < cols) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[ix * K + i] * B[i * cols + iy]; 
        }
        C[idx] = sum;
    }
}

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