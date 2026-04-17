
// Updated testbench
#include <cstdlib>
#include <iostream>

const int INPUT_SIZE = 1000;
const int ARRAY_SIZE = 1000;

void forward(int input[ARRAY_SIZE], int output[ARRAY_SIZE], int threshold, int num_elements, int &actual_output_count);

void generate_uniform_data(int data[ARRAY_SIZE], int size, int min_val, int max_val) {
    for (int i = 0; i < size; i++) {
        data[i] = min_val + (rand() % (max_val - min_val + 1));
    }
}

void generate_burst_data(int data[ARRAY_SIZE], int size, int burst_value, int burst_lane) {
    for (int i = 0; i < size; i++) {
        if (i >= size / 3 && i < 2 * size / 3) {
            data[i] = burst_value + (burst_lane * 10);
        } else {
            data[i] = rand() % 100;
        }
    }
}

int main() {
    int input1[ARRAY_SIZE], input2[ARRAY_SIZE];
    int output1[ARRAY_SIZE], output2[ARRAY_SIZE];
    int output_count1, output_count2;

    // seed for reproducibility
    std::srand(42);

    generate_uniform_data(input1, 500, 0, 1000);
    generate_burst_data(input2, 500, 800, 5);

    std::cout << "Test 1: Uniform input with threshold=100\n";
    forward(input1, output1, 100, 500, output_count1);
    std::cout << "Output count: " << output_count1 << "\n";

    // std::cout << "Test 2: Burst input with threshold=50\n";
    // forward(input2, output2, 50, 500, output_count2);
    // std::cout << "Output count: " << output_count2 << "\n";

    return 0;
}