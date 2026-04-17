#include <hls_stream.h>
const int ARRAY_SIZE = 1000; // Reduced from 10000

void forward(int input[ARRAY_SIZE], int output[ARRAY_SIZE], int num_elements, int threshold) {
#pragma HLS DATAFLOW

    // 100 FIFOs: 20 splitters, 40 workers, 40 collectors
    hls::stream<int> split_fifos[20];
    hls::stream<int> work_fifos[40];
    hls::stream<int> collect_fifos[40];

    // Separate streams for count communication between stages
    hls::stream<int> worker_count_to_work[40];    // From process stage to work stage
    hls::stream<int> worker_count_to_collect[40]; // From work stage to collect stage

#pragma HLS STREAM variable = split_fifos depth = 2048
#pragma HLS STREAM variable = work_fifos depth = 2048
#pragma HLS STREAM variable = collect_fifos depth = 2048
#pragma HLS STREAM variable = worker_count_to_work depth = 2
#pragma HLS STREAM variable = worker_count_to_collect depth = 2

    // Fork stage - split input based on value
    for (int i = 0; i < num_elements; i++) {
#pragma HLS PIPELINE II = 1
        int val = input[i];
        int route = val % 20;
        split_fifos[route].write(val);
    }

    // Process stage with data-dependent fanout
    // Pre-calculate routing counts to avoid using .empty()
    int splitter_counts[20] = {0};
    for (int i = 0; i < num_elements; i++) {
#pragma HLS PIPELINE II = 1
        int route = input[i] % 20;
        splitter_counts[route]++;
    }

    // This process writes to work_fifos and worker_count_to_work
    for (int s = 0; s < 20; s++) {
        int local_counts[2] = {0}; // Local count for this splitter's two workers

        // Process exact number of items in this splitter FIFO
        for (int i = 0; i < splitter_counts[s]; i++) {
#pragma HLS PIPELINE II = 1
            int val = split_fifos[s].read();
            if (val > threshold) {
                // High values fan out to multiple workers
                work_fifos[s * 2].write(val);
                work_fifos[s * 2 + 1].write(val / 2);
                local_counts[0]++;
                local_counts[1]++;
            } else {
                // Low values go to single worker
                work_fifos[s * 2].write(val);
                local_counts[0]++;
            }
        }

        // Write counts to streams going to work stage
        worker_count_to_work[s * 2].write(local_counts[0]);
        worker_count_to_work[s * 2 + 1].write(local_counts[1]);
    }

    // Work stage - reads from worker_count_to_work, writes to worker_count_to_collect
    for (int w = 0; w < 40; w++) {
        // Read the count for this worker
        int count = worker_count_to_work[w].read();

        // Process exact number of items in this worker FIFO
        for (int i = 0; i < count; i++) {
#pragma HLS PIPELINE II = 1
            int val = work_fifos[w].read();
            // Simple computation
            val = (val * 3) / 2;
            collect_fifos[w].write(val);
        }

        // Pass the count to the collection stage
        worker_count_to_collect[w].write(count);
    }

    // Join stage - collect results
    int out_idx = 0;
    for (int c = 0; c < 40; c++) {
        // Read the count for this collector
        int count = worker_count_to_collect[c].read();

        // Collect exact number of results from this collector FIFO
        for (int i = 0; i < count && out_idx < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE II = 1
            output[out_idx++] = collect_fifos[c].read();
        }
    }
}
