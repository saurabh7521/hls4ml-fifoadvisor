
#include <cstdlib>
#include <iostream>

const int ARRAY_SIZE = 2000;
const int FILTER_THRESHOLD = 500;
const int ENERGY_MULTIPLIER = 10;

void forward(int input_signals[ARRAY_SIZE], int characteristics[ARRAY_SIZE], int output_signals[ARRAY_SIZE],
             int num_signals);

// Testbench
int main() {
    int input_signals1[ARRAY_SIZE] = {0};
    int characteristics1[ARRAY_SIZE] = {0};
    int output_signals1[ARRAY_SIZE] = {0};

    int input_signals2[ARRAY_SIZE] = {0};
    int characteristics2[ARRAY_SIZE] = {0};
    int output_signals2[ARRAY_SIZE] = {0};

    std::srand(7); // Seed for reproducibility

    // Test case 1: Mostly bypass (low energy signals)
    for (int i = 0; i < 500; ++i) {
        input_signals1[i] = std::rand() % 50;   // Low values, likely to bypass
        characteristics1[i] = std::rand() % 32; // Random bit pattern
    }

    std::cout << "Test 1: Mostly bypass, low energy\n";
    forward(input_signals1, characteristics1, output_signals1, 500);
    std::cout << "Output count (Test 1): ";
    int count1 = 0;
    for (int i = 0; i < 500; ++i) {
        if (output_signals1[i] != 0)
            count1++;
    }
    std::cout << count1 << " nonzero outputs\n";
    std::cout << "Sample outputs: ";
    for (int i = 0; i < 10 && i < 500; ++i) {
        std::cout << output_signals1[i] << " ";
    }
    std::cout << "\n\n";

    // Test case 2: Heavy filter use (high energy signals)
    // for (int i = 0; i < 500; ++i) {
    //     if (i >= 100 && i < 400) {
    //         input_signals2[i] = 600 + (std::rand() % 400); // High value, high energy
    //         characteristics2[i] = 0x1F;                    // All filter bits on
    //     } else {
    //         input_signals2[i] = std::rand() % 200;
    //         characteristics2[i] = std::rand() % 32;
    //     }
    // }

    // std::cout << "Test 2: Heavy filter use, high energy\n";
    // forward(input_signals2, characteristics2, output_signals2, 500);
    // std::cout << "Output count (Test 2): ";
    // int count2 = 0;
    // for (int i = 0; i < 500; ++i) {
    //     if (output_signals2[i] != 0)
    //         count2++;
    // }
    // std::cout << count2 << " nonzero outputs\n";
    // std::cout << "Sample outputs: ";
    // for (int i = 0; i < 10 && i < 500; ++i) {
    //     std::cout << output_signals2[i] << " ";
    // }
    // std::cout << "\n";

    std::cout << "Testbench completed.\n";
    return 0;
}