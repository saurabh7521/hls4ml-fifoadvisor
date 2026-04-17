open_project -reset project
set_top GCN_compute_graphs

add_files src/GCN_compute.cc
add_files src/conv_layer.cc
add_files src/finalize.cc
add_files src/globals.cc
add_files src/linear.cc
add_files src/load_inputs.cc
add_files src/message_passing.cc
add_files src/node_embedding.cc
add_files -tb testbench/main.cc -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb testbench/load.cc -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb g1_node_feature.bin -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb g1_info.txt -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb g1_edge_list.bin -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb g1_edge_attr.bin -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb gcn_ep1_dim100.weights.all.bin -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"

open_solution "solution1" -flow_target vivado
# set_part xcu200-fsgd2104-2-e
set_part {xcu280-fsvh2892-2L-e}
create_clock -name ap_clk -period 3.33
config_compile -unsafe_math_optimizations
# config_interface -m_axi_addr64 -m_axi_offset off -register_io off
# if [lindex $argv 1] is csim, then csim_design, else if [lindex $argv 1] is csynth, then csynth_design
if {[lindex $argv 1] == "csim"} {
  csim_design
} elseif {[lindex $argv 1] == "cosim"} {
  cosim_design
} elseif {[lindex $argv 1] == "syn"} {
  # config_op fmacc -impl auto -precision high
  csynth_design
} elseif {[lindex $argv 1] == "pnr"} {
  export_design -flow impl
  get_clock_period -name ap_clk -ns
} elseif {[lindex $argv 1] == "toCosim"} {
  csim_design
  csynth_design
  cosim_design
} elseif {[lindex $argv 1] == "all"} {
  csim_design
  csynth_design
  cosim_design
  export_design -flow impl
  get_clock_period -name ap_clk -ns
} else {
  puts "Error: [lindex $argv 1] is not a valid argument"
}
exit