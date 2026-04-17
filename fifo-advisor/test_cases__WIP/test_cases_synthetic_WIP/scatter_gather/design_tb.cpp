#include <cstdlib>

const int ARRAY_SIZE = 1000;

void forward(int input[ARRAY_SIZE], int output[ARRAY_SIZE], int num_elements, int aggregate_threshold);

int main() {
    srand(7); // For reproducibility

    // int input1[ARRAY_SIZE], input2[ARRAY_SIZE];
    // int output1[ARRAY_SIZE], output2[ARRAY_SIZE];

    int input1[ARRAY_SIZE], output1[ARRAY_SIZE];
    int input2[ARRAY_SIZE], output2[ARRAY_SIZE];

    // // Test 1: Below threshold - no aggregation
    // for (int i = 0; i < ARRAY_SIZE; i++) {
    //     input1[i] = rand() % 500;
    // }

    // Test 2: Burst of high values - creates 4x data expansion
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (i >= 200 && i < 300) {
            input2[i] = 2000 + (i % 16) * 256; // High values per scatter channel
        } else {
            input2[i] = rand() % 500;
        }
    }

    // forward(input1, output1, ARRAY_SIZE, 1000);
    forward(input2, output2, ARRAY_SIZE, 1000);

    return 0;
}
