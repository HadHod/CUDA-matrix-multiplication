// C = A * B

const int TILE_SIZE = 16;

__global__
void matrixMultiplication(float* C, float* A, float* B, int rowsC, int colsC, int rowsA, int colsA, int rowsB, int colsB) {

    __device__ __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __device__ __shared__ float ds_B[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    float cValue = 0.0;

    for (int t = 0; t < (colsA-1) / TILE_SIZE+1; t++) {

        if (t * TILE_SIZE + tx < colsA && row < rowsA) {
            ds_A[ty][tx] = A[row * colsA + t * TILE_SIZE + tx];
        } else {
            ds_A[ty][tx] = 0.0;
        }

        if (t * TILE_SIZE + ty < rowsB && col < colsB) {
            ds_B[ty][tx] = B[(t * TILE_SIZE + ty) * colsB + col];
        } else {
            ds_B[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            cValue += ds_A[ty][i] * ds_B[i][tx];
        }

        __syncthreads();

    }

    if (row < rowsC && col < colsC) {
        C[row * colsC + col] = cValue;
    }

}
