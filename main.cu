#include <iostream>

#include "kernel/matrix.cu"

using namespace std;

const int TILE_SIZE = 16;

int main(int argc, char* argv[]) {

    int rowsA = 12;
    int colsA = 12;
    int rowsB = 12;
    int colsB = 12;
    int rowsC = rowsA;
    int colsC = colsB;

    float* matrixA;
    float* matrixB;
    float* matrixC = (float*) malloc(rowsC * colsC * sizeof(float));

    float* dev_matrixA;
    float* dev_matrixB;
    float* dev_matrixC;

    cudaMalloc((void**) &dev_matrixA, rowsA * colsA * sizeof(float));
    cudaMalloc((void**) &dev_matrixB, rowsB * colsB * sizeof(float));
    cudaMalloc((void**) &dev_matrixC, rowsC * colsC * sizeof(float));

    cudaMemcpy(dev_matrixA, matrixA, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixB, matrixB, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize((colsC-1) / TILE_SIZE + 1, (rowsC-1) / TILE_SIZE + 1, 1);
    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);

    matrixMultiplication<<<gridSize, blockSize>>>(
        dev_matrixC, dev_matrixA, dev_matrixB,
        rowsC, colsC, rowsA, colsA, rowsB, colsB,
        TILE_SIZE
    )

    cudaDeviceSynchronize();

    cudaMemcpy(matrixC, dev_matrixC, rowsC * colsC * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixC);

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return EXIT_SUCCESS;
}