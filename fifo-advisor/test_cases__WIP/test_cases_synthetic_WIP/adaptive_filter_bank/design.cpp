#include <cstdlib>
#include <hls_stream.h>

const int ARRAY_SIZE = 2000;
const int FILTER_THRESHOLD = 500;
const int ENERGY_MULTIPLIER = 10;

struct Signal {
    int value;
    int energy;
    int characteristics;
};

// Process 1: Input distribution with token counting
void input_processor(int input_signals[ARRAY_SIZE], int characteristics[ARRAY_SIZE], int num_signals,
                     hls::stream<Signal> bypass[10], hls::stream<Signal> to_filter[10],
                     hls::stream<int> bypass_counts[10], hls::stream<int> filter_counts[10]) {

    int bypass_tokens[10] = {0};
    int filter_tokens[10] = {0};

    // Process all inputs
    for (int i = 0; i < num_signals; i++) {
#pragma HLS PIPELINE II = 1
        Signal sig;
        sig.value = input_signals[i];
        sig.characteristics = characteristics[i];
        sig.energy = (sig.value * sig.characteristics) % 1000;
        int bank = sig.characteristics % 10;

        if (sig.energy > FILTER_THRESHOLD) {
            to_filter[bank].write(sig);
            filter_tokens[bank]++;
        } else {
            bypass[bank].write(sig);
            bypass_tokens[bank]++;
        }
    }

    // Send token counts to downstream processes
    for (int b = 0; b < 10; b++) {
#pragma HLS UNROLL
        bypass_counts[b].write(bypass_tokens[b]);
        filter_counts[b].write(filter_tokens[b]);
    }
}

// Process 2: Filter processor with exact token counting
void filter_processor(hls::stream<Signal> to_filter[10], hls::stream<int> filter_counts[10],
                      hls::stream<Signal> filtered_out[10], hls::stream<int> output_counts[10]) {

    int tokens_to_read[10];
    int output_tokens[10] = {0};

    // Read expected token counts
    for (int b = 0; b < 10; b++) {
#pragma HLS UNROLL
        tokens_to_read[b] = filter_counts[b].read();
    }

    // Process exact number of tokens
    int total_to_process = 0;
    for (int b = 0; b < 10; b++) {
#pragma HLS UNROLL
        total_to_process += tokens_to_read[b];
    }

    for (int i = 0; i < total_to_process; i++) {
#pragma HLS PIPELINE II = 1
        // Round-robin through banks
        for (int b = 0; b < 10; b++) {
#pragma HLS UNROLL
            if (tokens_to_read[b] > 0) {
                Signal sig = to_filter[b].read();
                tokens_to_read[b]--;

                // Apply all filter stages in sequence for signals that qualify
                for (int stage = 0; stage < 5; stage++) {
#pragma HLS UNROLL
                    if ((sig.characteristics & (1 << stage)) != 0) {
                        // Apply filter transformation
                        sig.value = (sig.value * (stage + 1)) / 2;
                        sig.energy = sig.energy * 8 / 10;

                        // Check if should continue to next stage
                        if (sig.energy <= FILTER_THRESHOLD / 2) {
                            break; // Exit filter chain early
                        }
                    }
                }

                // Send to output
                filtered_out[b].write(sig);
                output_tokens[b]++;
                break; // Move to next iteration
            }
        }
    }

    // Send output token counts
    for (int b = 0; b < 10; b++) {
#pragma HLS UNROLL
        output_counts[b].write(output_tokens[b]);
    }
}

// Process 3: Output collector with exact token counting
void output_collector(hls::stream<Signal> bypass[10], hls::stream<int> bypass_counts[10],
                      hls::stream<Signal> filtered_out[10], hls::stream<int> output_counts[10],
                      int output_signals[ARRAY_SIZE]) {

    int bypass_tokens[10];
    int filter_tokens[10];
    int output_count = 0;

    // Read expected token counts
    for (int b = 0; b < 10; b++) {
#pragma HLS UNROLL
        bypass_tokens[b] = bypass_counts[b].read();
        filter_tokens[b] = output_counts[b].read();
    }

    // Collect all bypass tokens
    for (int b = 0; b < 10; b++) {
        for (int i = 0; i < bypass_tokens[b] && output_count < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE II = 1
            Signal sig = bypass[b].read();
            output_signals[output_count++] = sig.value;
        }
    }

    // Collect all filter output tokens
    for (int b = 0; b < 10; b++) {
        for (int i = 0; i < filter_tokens[b] && output_count < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE II = 1
            Signal sig = filtered_out[b].read();
            output_signals[output_count++] = sig.value;
        }
    }

    // Fill remaining with zeros
    for (int i = output_count; i < ARRAY_SIZE; i++) {
        output_signals[i] = 0;
    }
}

void forward(int input_signals[ARRAY_SIZE], int characteristics[ARRAY_SIZE], int output_signals[ARRAY_SIZE],
             int num_signals) {
#pragma HLS DATAFLOW

    // Signal streams
    hls::stream<Signal> bypass[10];
    hls::stream<Signal> to_filter[10];
    hls::stream<Signal> filtered_out[10];

    // Token count streams
    hls::stream<int> bypass_counts[10];
    hls::stream<int> filter_counts[10];
    hls::stream<int> output_counts[10];

#pragma HLS STREAM variable = bypass depth = 512
#pragma HLS STREAM variable = to_filter depth = 512
#pragma HLS STREAM variable = filtered_out depth = 512
#pragma HLS STREAM variable = bypass_counts depth = 512
#pragma HLS STREAM variable = filter_counts depth = 512
#pragma HLS STREAM variable = output_counts depth = 512

    // Three processes with token counting
    input_processor(input_signals, characteristics, num_signals, bypass, to_filter, bypass_counts, filter_counts);

    filter_processor(to_filter, filter_counts, filtered_out, output_counts);

    output_collector(bypass, bypass_counts, filtered_out, output_counts, output_signals);
}
