#include "ap_int.h"
#include <cassert>
#include <iostream>

constexpr int N = 100;         // Number of input/output FIFOs
constexpr int IN_WORDS = 4096; // Input size per FIFO
constexpr int OUT_WORDS = 4096;

typedef uint32_t data_t;
typedef uint32_t addr_t; // log2(N) = 7

void forward(data_t in[N][IN_WORDS], bool fifo_enable[N], data_t out[N][OUT_WORDS]);

int main() {
    data_t in1[N][IN_WORDS], in2[N][IN_WORDS];
    data_t out1[N][OUT_WORDS] = {0}, out2[N][OUT_WORDS] = {0};
    bool enable1[N] = {false}, enable2[N] = {false};

    // Initialize all inputs to zero
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < IN_WORDS; ++j) {
            in1[i][j] = 0;
            in2[i][j] = 0;
        }
    }

    // Test Case 1: Sparse Clustered Pattern
    // Only enable FIFOs 10-29 (20 FIFOs = 20% of 100)
    std::cout << "=== Test Case 1: Sparse Clustered Pattern ===\n";
    std::cout << "Active FIFOs: 10-29 (20 FIFOs)\n";

    // Set up enable array for clustered pattern
    int active_fifos_1 = 0;
    for (int i = 10; i < 30; ++i) {
        enable1[i] = true;
        active_fifos_1++;

        // Only fill data for enabled FIFOs
        for (int j = 0; j < IN_WORDS; ++j) {
            data_t base_value = (data_t)(i * 100 + j);
            in1[i][j] = base_value;
        }
    }

    std::cout << "Enabled " << active_fifos_1 << " FIFOs for clustered test\n";

    forward(in1, enable1, out1);

    std::cout << "\n";

    // // Test Case 2: Sparse Scattered Pattern
    // // Only enable every 5th FIFO (0, 5, 10, 15, ..., 95)
    // std::cout << "=== Test Case 2: Sparse Scattered Pattern ===\n";
    // std::cout << "Active FIFOs: Every 5th FIFO (0,5,10,15...95)\n";

    // // Set up enable array for scattered pattern
    // int active_fifos_2 = 0;
    // for (int i = 0; i < 100; i += 5) {
    //     enable2[i] = true;
    //     active_fifos_2++;

    //     // Only fill data for enabled FIFOs
    //     for (int j = 0; j < IN_WORDS; ++j) {
    //         data_t base_value = (data_t)(i * 73 + j * 41 + 12345);
    //         in2[i][j] = base_value;
    //     }
    // }

    // std::cout << "Enabled " << active_fifos_2 << " FIFOs for scattered test\n";

    // forward(in2, enable2, out2);

    // // Test Case 3: Minimal Pattern (only 3 FIFOs)
    // std::cout << "=== Test Case 3: Minimal Pattern ===\n";
    // std::cout << "Active FIFOs: Only 0, 50, 99 (3 FIFOs)\n";

    // data_t in3[N][IN_WORDS] = {0};
    // data_t out3[N][OUT_WORDS] = {0};
    // bool enable3[N] = {false};

    // // Enable only 3 FIFOs
    // enable3[0] = true;
    // enable3[50] = true;
    // enable3[99] = true;

    // for (int i = 0; i < N; ++i) {
    //     if (enable3[i]) {
    //         for (int j = 0; j < IN_WORDS; ++j) {
    //             in3[i][j] = (data_t)(i * 1000 + j);
    //         }
    //     }
    // }

    // forward(in3, enable3, out3);

    std::cout << "\n";
    std::cout << "=== Performance Benefits ===\n";
    std::cout << "• No unnecessary processing of disabled FIFOs\n";
    std::cout << "• Reduced memory bandwidth usage\n";
    std::cout << "• Faster execution with sparse workloads\n";
    std::cout << "• Runtime configurability for different use cases\n";

    return 0;
}