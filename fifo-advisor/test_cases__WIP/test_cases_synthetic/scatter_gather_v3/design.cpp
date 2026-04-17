#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

constexpr int N = 100;         // Number of input/output FIFOs
constexpr int IN_WORDS = 4096; // Input size per FIFO
constexpr int OUT_WORDS = 4096;

typedef uint32_t data_t;
typedef uint32_t addr_t; // log2(N) = 7

struct Packet {
    data_t value;
    addr_t dest;
    bool is_end; // End-of-stream marker

    // Default constructor
    Packet() : value(0), dest(0), is_end(false) {
    }

    // Constructor for full initialization
    Packet(data_t v, addr_t d, bool end) : value(v), dest(d), is_end(end) {
    }

    // Constructor for data packets
    Packet(data_t v, addr_t d) : value(v), dest(d), is_end(false) {
    }
};

const Packet END_MARKER = {0, 0, true};

// Data-dependent scatter: sends values to different output FIFOs based on their value
// Only processes FIFOs marked as enabled in fifo_enable array
void scatter(data_t in[N][IN_WORDS], bool fifo_enable[N], hls::stream<Packet> scatter_out[N]) {
    // #pragma HLS inline

    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
#pragma HLS PIPELINE off

        // Send all data packets for this enabled FIFO
        for (int j = 0; j < IN_WORDS; ++j) {
#pragma HLS PIPELINE off
            addr_t dest = (in[i][j] ^ (i * 17 + j)) % N; // data-dependent dest
            Packet pkt(in[i][j], dest);
            if (fifo_enable[i]) {
                scatter_out[i].write(pkt);
            }
            if (fifo_enable[i] && (j == IN_WORDS - 1)) {
                Packet pkt(in[i][j], dest, true);
                scatter_out[i].write(pkt);
            }
        }
    }
}

// Round-robin gather with end-of-stream detection
// Now knows which FIFOs to expect data from via fifo_enable array
// void gather(hls::stream<Packet> gather_in[N], bool fifo_enable[N], data_t out[N][OUT_WORDS]) {
//     // #pragma HLS inline

//     int out_counts[N] = {0};
//     bool fifo_done[N] = {false};

//     // Count how many FIFOs we expect to process
//     int fifos_remaining = 0;
//     for (int i = 0; i < N; ++i) {
// #pragma HLS PIPELINE off
//         fifos_remaining++; // All FIFOs will send at least an end marker
//     }

//     int current_fifo = 0;

//     while (fifos_remaining > 0) {
// #pragma HLS PIPELINE off

//         // Find next FIFO that isn't done
//         bool found_active = false;
//         for (int attempt = 0; attempt < N && !found_active; attempt++) {
// #pragma HLS PIPELINE off
//             if (!fifo_done[current_fifo]) {
//                 found_active = true;
//             } else {
//                 current_fifo = (current_fifo + 1) % N;
//             }
//         }

//         if (found_active) {
//             // Read packet from current FIFO
//             Packet pkt = gather_in[current_fifo].read();

//             if (pkt.is_end) {
//                 // Mark this FIFO as done
//                 fifo_done[current_fifo] = true;
//                 fifos_remaining--;
//             } else {
//                 // Process data packet - route to destination
//                 // Only enabled FIFOs should send data packets
//                 if (out_counts[pkt.dest] < OUT_WORDS) {
//                     out[pkt.dest][out_counts[pkt.dest]++] = pkt.value;
//                 }
//                 // Note: packets that would overflow are silently dropped
//             }

//             // Move to next FIFO in round-robin
//             current_fifo = (current_fifo + 1) % N;
//         }
//     }
// }

void gather(hls::stream<Packet> gather_in[N], bool fifo_enable[N], data_t out[N][OUT_WORDS]) {
#pragma HLS ARRAY_PARTITION variable = out complete dim = 1
#pragma HLS ARRAY_PARTITION variable = fifo_enable complete

    // Use local arrays for better performance
    ap_uint<12> out_counts[N];
#pragma HLS ARRAY_PARTITION variable = out_counts complete

    bool fifo_done[N];
#pragma HLS ARRAY_PARTITION variable = fifo_done complete

    // Initialize arrays
INIT_LOOP:
    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
        out_counts[i] = 0;
        fifo_done[i] = false;
    }

    // Count enabled FIFOs for termination condition
    ap_uint<8> fifos_remaining = 0;
COUNT_LOOP:
    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
        if (fifo_enable[i]) {
            fifos_remaining++;
        } else {
            fifo_done[i] = true; // Mark disabled FIFOs as done
        }
    }

    ap_uint<8> current_fifo = 0;

    // Main processing loop - blocking reads only
MAIN_LOOP:
    while (fifos_remaining > 0) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = N * IN_WORDS

    // Skip disabled/done FIFOs
    SKIP_LOOP:
        for (int skip_count = 0; skip_count < N; skip_count++) {
#pragma HLS PIPELINE off
            if (!fifo_done[current_fifo]) {
                break;
            }
            current_fifo = (current_fifo + 1) % N;
        }

        // Process active FIFO with blocking read
        if (!fifo_done[current_fifo]) {
            // Blocking read - will wait until data is available
            Packet pkt = gather_in[current_fifo].read();

            if (pkt.is_end) {
                // Mark this FIFO as done
                fifo_done[current_fifo] = true;
                fifos_remaining--;
            } else {
                // Process data packet - route to destination
                if (out_counts[pkt.dest] < OUT_WORDS) {
                    out[pkt.dest][out_counts[pkt.dest]] = pkt.value;
                    out_counts[pkt.dest]++;
                }
                // Note: packets that would overflow are silently dropped
            }
        }

        // Move to next FIFO in round-robin
        current_fifo = (current_fifo + 1) % N;
    }
}

void forward(data_t in[N][IN_WORDS], bool fifo_enable[N], data_t out[N][OUT_WORDS]) {
#pragma HLS DATAFLOW

    hls::stream<Packet> fifos[N];
    hls::stream<Packet> fifos_2[N];
#pragma HLS STREAM variable = fifos depth = 5000
#pragma HLS ARRAY_PARTITION variable = fifos complete

    scatter(in, fifo_enable, fifos);
    gather(fifos, fifo_enable, out);
}