#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

constexpr int N = 100;         // Number of input/output FIFOs
constexpr int IN_WORDS = 4096; // Input size per FIFO
constexpr int OUT_WORDS = 4096;

typedef ap_uint<32> data_t;
typedef ap_uint<7> addr_t; // log2(N) = 7

struct Packet {
    data_t value;
    addr_t dest;
};

// Data-dependent scatter: sends values to different output FIFOs based on their value
void scatter(data_t in[N][IN_WORDS], hls::stream<Packet> scatter_out[N]) {
#pragma HLS DATAFLOW
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < IN_WORDS; ++j) {
#pragma HLS PIPELINE II = 1
            addr_t dest = (in[i][j] ^ (i * 17 + j)) % N; // data-dependent dest
            Packet pkt;
            pkt.value = in[i][j];
            pkt.dest = dest;
            scatter_out[i].write(pkt); // write to FIFO for downstream processing
        }
    }
}

void gather(hls::stream<Packet> gather_in[N], data_t out[N][OUT_WORDS]) {
#pragma HLS DATAFLOW
    int out_counts[N] = {0};

    // Process all packets from all FIFOs
    for (int total = 0; total < N * IN_WORDS; total++) {
#pragma HLS PIPELINE II = 1

        // Read from FIFOs in round-robin, but route by packet destination
        int fifo_idx = total % N;
        Packet pkt = gather_in[fifo_idx].read();

        // Route to destination based on packet's dest field
        if (out_counts[pkt.dest] < OUT_WORDS) {
            out[pkt.dest][out_counts[pkt.dest]++] = pkt.value;
        }
    }
}

void forward(data_t in[N][IN_WORDS], data_t out[N][OUT_WORDS]) {
#pragma HLS DATAFLOW
    hls::stream<Packet> fifos[N];
#pragma HLS STREAM variable = fifos depth = 5000 type = fifo // Start with 5000, but tool will explore higher
#pragma HLS ARRAY_PARTITION variable = fifos complete

    scatter(in, fifos);
    gather(fifos, out);
}
