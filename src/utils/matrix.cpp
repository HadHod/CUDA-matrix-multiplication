#include <iostream>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

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
    srand(high_resolution_clock::now().time_since_epoch().count());

    const int numberOfElements = rows * columns;
    float* matrix = (float*) malloc(numberOfElements * sizeof(float));

    for (int i=0; i<numberOfElements; i++) {
        matrix[i] = static_cast<int>( rand() ) % 10; // TODO back to floats in future
    }

    return matrix;
}
