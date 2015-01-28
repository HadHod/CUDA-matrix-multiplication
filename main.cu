#include <iostream>
#include <ctime>
#include <chrono>

#include "kernel/matrix.cu"

using namespace std;

void printMatrix(const float* matrix, const int rows, const int columns) {

    for (int i=0; i<rows * columns; i++) {
        if (i % columns == 0) {
            cout << "\n";
        }
        cout << matrix[i] << " ";
    }

    cout << "\n";
}

float* initRandomMatrix(const int rows, const int columns) {
    srand(chrono::high_resolution_clock::now().time_since_epoch().count());

    const int numberOfElements = rows * columns;
    float* matrix = (float*) malloc(numberOfElements * sizeof(float));

    for (int i=0; i<numberOfElements; i++) {
        matrix[i] = static_cast<int>( rand() ) % 10; // TODO back to floats in future
    }

    return matrix;
}

int main(int argc, char* argv[]) {

    const int TILE_SIZE = 16;
    const int size = 4;

    int rowsA = size;
    int colsA = size;
    int rowsB = size;
    int colsB = size;
    int rowsC = rowsA;
    int colsC = colsB;

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