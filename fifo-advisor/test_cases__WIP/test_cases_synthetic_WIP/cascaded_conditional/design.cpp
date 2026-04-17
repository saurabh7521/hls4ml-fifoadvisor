#include <hls_stream.h>

const int INPUT_SIZE = 1000;
const int ARRAY_SIZE = 1000;

void forward(int input[ARRAY_SIZE], int output[ARRAY_SIZE], int threshold, int num_elements, int &actual_output_count) {
#pragma HLS DATAFLOW

    // 100 FIFOs arranged in a 10x10 grid with cross-connections
    hls::stream<int> fifo[10][10];
#pragma HLS STREAM variable = fifo depth = 2048

    // Track data count per stage to ensure balanced processing
    hls::stream<int> stage_counts[10];
#pragma HLS STREAM variable = stage_counts depth = 10

    // Input stage - write array to streams
    int input_counts[10] = {0};
    for (int iter = 0; iter < num_elements; iter++) {
#pragma HLS PIPELINE II = 1
        int val = input[iter];
        int route = (val % 10);
        fifo[0][route].write(val);
        input_counts[route]++;
    }

    // Write initial counts
    for (int i = 0; i < 10; i++) {
        stage_counts[0].write(input_counts[i]);
    }

    // Processing stages with proper data tracking
    for (int stage = 1; stage < 9; stage++) {
        int next_counts[10] = {0};

        // Read expected counts for this stage
        int expected_counts[10];
        for (int i = 0; i < 10; i++) {
            expected_counts[i] = stage_counts[stage - 1].read();
        }

        // Process all data in this stage
        for (int lane = 0; lane < 10; lane++) {
            for (int cnt = 0; cnt < expected_counts[lane]; cnt++) {
                // Remove the .empty() check - we know exactly how much data to read
                int data = fifo[stage - 1][lane].read();

                if (data > threshold * stage) {
                    // Only forward to next lane (remove cross-connection to avoid
                    // amplification)
                    int next_lane = (lane + 1) % 10;
                    fifo[stage][next_lane].write(data);
                    next_counts[next_lane]++;
                } else {
                    // Forward to same lane without multiplication to avoid data growth
                    fifo[stage][lane].write(data);
                    next_counts[lane]++;
                }
            }
        }

        // Write counts for next stage
        for (int i = 0; i < 10; i++) {
            stage_counts[stage].write(next_counts[i]);
        }
    }

    // Output stage: Collect all results
    int out_idx = 0;

    // Read final stage counts
    int final_counts[10];
    for (int i = 0; i < 10; i++) {
        final_counts[i] = stage_counts[8].read();
    }

    // Collect exactly the amount of data present
    for (int lane = 0; lane < 10; lane++) {
        for (int cnt = 0; cnt < final_counts[lane]; cnt++) {
            // Remove the .empty() check and array bounds check
            // We rely on the count to ensure we don't overflow
            if (out_idx < ARRAY_SIZE) {
                output[out_idx++] = fifo[8][lane].read();
            } else {
                // If we would overflow, still consume the data to avoid deadlock
                fifo[8][lane].read();
            }
        }
    }

    actual_output_count = out_idx;

    // Fill remaining output with zeros
    for (int i = out_idx; i < ARRAY_SIZE; i++) {
        output[i] = 0;
    }
}
