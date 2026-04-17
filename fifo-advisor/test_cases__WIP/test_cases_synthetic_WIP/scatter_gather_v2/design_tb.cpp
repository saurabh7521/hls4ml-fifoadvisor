#include "ap_int.h"
#include <cassert>
#include <iostream>

constexpr int N = 100;         // Number of input/output FIFOs
constexpr int IN_WORDS = 4096; // Input size per FIFO
constexpr int OUT_WORDS = 4096;

typedef ap_uint<32> data_t;
typedef ap_uint<7> addr_t; // log2(N) = 7

void forward(data_t in[N][IN_WORDS], data_t out[N][OUT_WORDS]);

int main() {
    data_t in1[N][IN_WORDS], in2[N][IN_WORDS];
    data_t out1[N][OUT_WORDS] = {0}, out2[N][OUT_WORDS] = {0};

    // Input Set 1: "spread" traffic (likely needs large FIFOs)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < IN_WORDS; ++j)
            in1[i][j] = (data_t)(i * 31 + j * 13);

    // Input Set 2: "localized" traffic (likely fits in small FIFOs)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < IN_WORDS; ++j)
            in2[i][j] = (data_t)i; // will mostly target own FIFO

    // ---- Run with "tight" FIFO depths (simulate tool restricting depth to 1024) ----
    // (Assume you run HLS with pragma depth=1024 above.)

    // Should NOT deadlock for in2, but may deadlock for in1
    std::cout << "Test with input set 1 (scattered):\n";
    try {
        forward(in1, out1);
        std::cout << "Finished without deadlock!\n";
    } catch (...) {
        std::cout << "Deadlocked!\n";
    }

    // std::cout << "Test with input set 2 (localized):\n";
    // try {
    //     forward(in2, out2);
    //     std::cout << "Finished without deadlock!\n";
    // } catch (...) {
    //     std::cout << "Deadlocked!\n";
    // }

    // // Basic correctness: check outputs are "something" (more elaborate checking is possible)
    // std::cout << "out1[0][0]=" << out1[0][0] << " out2[0][0]=" << out2[0][0] << "\n";
    return 0;
}
