#include <iostream>

#include "kernel/matrix.cuh"
#include "utils/matrix.hpp"

using namespace std;

int main(int argc, char* argv[]) {

    if (argc != 1 && argc != 5) {
        cout << "You have to pass none or 4 arguments. Passed: " << argc << " argument(s)\n";
        return EXIT_FAILURE;
    }

    const int TILE_SIZE = 16;

    const int rowsA = (argc == 5) ? atoi(argv[1]) : 4;
    const int colsA = (argc == 5) ? atoi(argv[2]) : 4;
    const int rowsB = (argc == 5) ? atoi(argv[3]) : 4;
    const int colsB = (argc == 5) ? atoi(argv[4]) : 4;
    const int rowsC = rowsA;
    const int colsC = colsB;

    float* matrixA = initRandomMatrix(rowsA, colsA);
    float* matrixB = initRandomMatrix(rowsB, colsB);
    float* matrixC = (float*) malloc(rowsC * colsC * sizeof(float));

    printMatrix(matrixA, rowsA, colsC);
    printMatrix(matrixB, rowsB, colsB);

    float* dev_matrixA;
    float* dev_matrixB;
    float* dev_matrixC;

    cudaMalloc((void**) &dev_matrixA, rowsA * colsA * sizeof(float));
    cudaMalloc((void**) &dev_matrixB, rowsB * colsB * sizeof(float));
    cudaMalloc((void**) &dev_matrixC, rowsC * colsC * sizeof(float));

    cudaMemcpy(dev_matrixA, matrixA, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixB, matrixB, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    dim3 gridSize((colsC-1) / TILE_SIZE + 1, (rowsC-1) / TILE_SIZE + 1, 1);
    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);

    matrixMultiplication<<<gridSize, blockSize>>>(
        dev_matrixC, dev_matrixA, dev_matrixB,
        rowsC, colsC, rowsA, colsA, rowsB, colsB
    );

    cudaDeviceSynchronize();

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(matrixC, dev_matrixC, rowsC * colsC * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "\nResult:";
    printMatrix(matrixC, rowsC, colsC);

    cout << "Elapsed time: " << elapsedTime << " ms\n";

    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixC);

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return EXIT_SUCCESS;
}
