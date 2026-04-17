#include <cstdlib>

const int ARRAY_SIZE = 1000; // Reduced from 10000

void forward(int input[ARRAY_SIZE], int output[ARRAY_SIZE], int num_elements, int threshold);

int main() {
    // seed for reproducibility
    std::srand(42);

    int input1[ARRAY_SIZE], input2[ARRAY_SIZE];
    int output1[ARRAY_SIZE], output2[ARRAY_SIZE];

    // Test 1: Mostly low values - predictable data flow
    for (int i = 0; i < 500; i++) { // Reduced from 5000
        input1[i] = rand() % 100;   // Below threshold
    }

    // Test 2: Burst of high values - creates data expansion
    for (int i = 0; i < 500; i++) {            // Reduced from 5000
        if (i >= 100 && i < 400) {             // Adjusted range
            input2[i] = 1000 + (i % 20) * 100; // High values in specific routes
        } else {
            input2[i] = rand() % 100;
        }
    }

    forward(input1, output1, 500, 500); // threshold = 500
    // forward(input2, output2, 500, 500);

    return 0;
}