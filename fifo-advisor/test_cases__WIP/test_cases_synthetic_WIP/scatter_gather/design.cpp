#include <cstdlib>
#include <hls_stream.h>

const int ARRAY_SIZE = 1000;

// Scatter process - tracks how many items go to each scatter FIFO
void scatter_proc(int input[ARRAY_SIZE], int num_elements, hls::stream<int> scatter_fifos[16],
                  hls::stream<int> scatter_count_stream[16]) {
    int scatter_write_count[16] = {0};

    for (int i = 0; i < num_elements; i++) {
#pragma HLS PIPELINE II = 1
        int val = input[i];
        int target = (val / 256) % 16;
        scatter_fifos[target].write(val);
        scatter_write_count[target]++;
    }

    // Send count information to process_proc
    for (int s = 0; s < 16; s++) {
        scatter_count_stream[s].write(scatter_write_count[s]);
    }
}

// Aggregation/conditional scatter - uses actual scatter counts
void process_proc(int num_elements, int aggregate_threshold, hls::stream<int> scatter_fifos[16],
                  hls::stream<int> process_fifos[64], hls::stream<int> scatter_count_stream[16],
                  hls::stream<int> process_count_stream[64]) {
    int scatter_read_count[16];
    int process_write_count[64] = {0};

    // Read how many items each scatter FIFO contains
    for (int s = 0; s < 16; s++) {
        scatter_read_count[s] = scatter_count_stream[s].read();
    }

    // Process each scatter FIFO based on actual counts
    for (int s = 0; s < 16; s++) {
        for (int i = 0; i < scatter_read_count[s]; i++) {
#pragma HLS PIPELINE II = 1
            int val = scatter_fifos[s].read();

            if (val > aggregate_threshold) {
                // Split into 4 parts and send to 4 process FIFOs
                for (int p = 0; p < 4; p++) {
                    process_fifos[s * 4 + p].write(val / 4);
                    process_write_count[s * 4 + p]++;
                }
            } else {
                // Send to first process FIFO for this scatter group
                process_fifos[s * 4].write(val);
                process_write_count[s * 4]++;
            }
        }
    }

    // Send count information to aggregate_proc
    for (int p = 0; p < 64; p++) {
        process_count_stream[p].write(process_write_count[p]);
    }
}

// Processing phase - uses actual process counts
void aggregate_proc(int num_elements, hls::stream<int> process_fifos[64], hls::stream<int> gather_fifos[16],
                    hls::stream<int> gather_count_stream[16], hls::stream<int> process_count_stream[64]) {
    int process_read_count[64];
    int gather_write_count[16] = {0};

    // Read how many items each process FIFO contains
    for (int p = 0; p < 64; p++) {
        process_read_count[p] = process_count_stream[p].read();
    }

    // Process each process FIFO based on actual counts
    for (int p = 0; p < 64; p++) {
        for (int i = 0; i < process_read_count[p]; i++) {
#pragma HLS PIPELINE II = 1
            int val = process_fifos[p].read();
            val = val * 2; // Simple processing
            int g = p / 4;
            gather_fifos[g].write(val);
            gather_write_count[g]++;
        }
    }

    // Output the count for each gather FIFO via the stream
    for (int g = 0; g < 16; g++) {
        gather_count_stream[g].write(gather_write_count[g]);
    }
}

// Gather process - uses actual gather counts
void gather_proc(int num_elements, int output[ARRAY_SIZE], hls::stream<int> gather_fifos[16],
                 hls::stream<int> gather_count_stream[16]) {
    int gather_read_count[16];

    // Read the count for each gather FIFO from the stream
    for (int g = 0; g < 16; g++) {
        gather_read_count[g] = gather_count_stream[g].read();
    }

    int out_idx = 0;

    // Gather outputs based on actual counts
    for (int g = 0; g < 16; g++) {
        for (int i = 0; i < gather_read_count[g] && out_idx < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE II = 1
            output[out_idx++] = gather_fifos[g].read();
        }
    }
}

// Top-level HLS function
void forward(int input[ARRAY_SIZE], int output[ARRAY_SIZE], int num_elements, int aggregate_threshold) {
#pragma HLS DATAFLOW

    hls::stream<int> scatter_fifos[16];
    hls::stream<int> process_fifos[64];
    hls::stream<int> gather_fifos[16];
    hls::stream<int> gather_count_stream[16];
    hls::stream<int> scatter_count_stream[16];
    hls::stream<int> process_count_stream[64];

#pragma HLS STREAM variable = scatter_fifos depth = 2048
#pragma HLS STREAM variable = process_fifos depth = 2048
#pragma HLS STREAM variable = gather_fifos depth = 2048
#pragma HLS STREAM variable = gather_count_stream depth = 2048
#pragma HLS STREAM variable = scatter_count_stream depth = 16
#pragma HLS STREAM variable = process_count_stream depth = 64

    scatter_proc(input, num_elements, scatter_fifos, scatter_count_stream);
    process_proc(num_elements, aggregate_threshold, scatter_fifos, process_fifos, scatter_count_stream,
                 process_count_stream);
    aggregate_proc(num_elements, process_fifos, gather_fifos, gather_count_stream, process_count_stream);
    gather_proc(num_elements, output, gather_fifos, gather_count_stream);
}