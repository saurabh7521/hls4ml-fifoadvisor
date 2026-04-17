#pragma once

#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_vector.h"

#include <iostream>

#ifndef __SYNTHESIS__
#define CSIM_STATIC static
#else
#define CSIM_STATIC
#endif

template <typename T> constexpr T t_multiply(T arg) { return arg; }

template <typename T, typename... Args>
constexpr T t_multiply(T arg, Args... args) {
  return arg * t_multiply(args...);
}

template <const int... _dim_sizes> struct array_shape {
  static constexpr int n_dims = sizeof...(_dim_sizes);
  static constexpr int dim_sizes[sizeof...(_dim_sizes)] = {_dim_sizes...};
  static constexpr int N = t_multiply(_dim_sizes...);
  array_shape() = default;
};

template <int N>
constexpr bool compare_array_shape(const int a[N], const int b[N]) {
  return (N == 0) ? true
                  : (a[N - 1] == b[N - 1]) && compare_array_shape<N - 1>(a, b);
}

// implement a constexpr function that takes an array and a start_index
// it shoudl return the mutiplied vlaues of the array from start_index to the
// end
template <int N, int start_index> constexpr int get_product(const int a[N]) {
  int result = 1;
  for (int i = start_index; i < N; i++) {
    result *= a[i];
  }
  return result;
}

template <typename T, typename array_shape, const int stream_block_size = 1>
struct array_stream {
  using T_data = T;

  static constexpr array_shape shape{};
  // using shape = array_shape;
  static constexpr int block_size = stream_block_size;
  static constexpr int n_blocks =
      (array_shape::N + stream_block_size - 1) / stream_block_size;

  using T_block = hls::vector<T, stream_block_size>;

  hls::stream<T_block> data;

  array_stream() {}
};

template <typename T, int M, int stream_block_size = 1>
void array_1d_to_array_stream(
    T array_in[M],
    array_stream<T, array_shape<M>, stream_block_size> &array_stream_out) {
#pragma HLS INLINE off

  const int n_total = M;
  const int block_size = array_stream_out.block_size;
  const int n_blocks = array_stream_out.n_blocks;

  for (int n_block = 0; n_block < n_blocks; n_block++) {
    typename array_stream<T, array_shape<M>, stream_block_size>::T_block block =
        T(0.0);

    for (int n_elem = 0; n_elem < block_size; n_elem++) {
      int n = n_block * block_size + n_elem;
      if (n < n_total) {
        block[n_elem] = array_in[n];
      }
    }

    array_stream_out.data.write(block);
  }
}

template <typename T, int M, int N, int stream_block_size = 1>
void array_2d_to_array_stream(
    T array_in[M][N],
    array_stream<T, array_shape<M, N>, stream_block_size> &array_stream_out) {
#pragma HLS INLINE off

  const int n_total = M * N;
  const int block_size = array_stream_out.block_size;
  const int n_blocks = array_stream_out.n_blocks;

  for (int n_block = 0; n_block < n_blocks; n_block++) {
    typename array_stream<T, array_shape<M, N>, stream_block_size>::T_block
        block = T(0.0);

    for (int n_elem = 0; n_elem < block_size; n_elem++) {
      int n = n_block * block_size + n_elem;
      if (n < n_total) {
        block[n_elem] = array_in[n / N][n % N];
      }
    }

    array_stream_out.data.write(block);
  }
}

template <typename T, int M, int N, int O, int stream_block_size = 1>
void array_3d_to_array_stream(
    T array_in[M][N][O],
    array_stream<T, array_shape<M, N, O>, stream_block_size>
        &array_stream_out) {
#pragma HLS INLINE off

  const int n_total = M * N * O;
  const int block_size = array_stream_out.block_size;
  const int n_blocks = array_stream_out.n_blocks;

  for (int n_block = 0; n_block < n_blocks; n_block++) {
    typename array_stream<T, array_shape<M, N, O>, stream_block_size>::T_block
        block = T(0.0);

    for (int n_elem = 0; n_elem < block_size; n_elem++) {
      int n = n_block * block_size + n_elem;
      if (n < n_total) {
        block[n_elem] = array_in[n / (N * O)][(n / O) % N][n % O];
      }
    }

    array_stream_out.data.write(block);
  }
}

template <typename T, int M, int stream_block_size = 1>
void array_stream_to_array_1d(
    array_stream<T, array_shape<M>, stream_block_size> &array_stream_in,
    T array_out[M]) {
#pragma HLS INLINE off

  const int n_total = M;
  const int block_size = array_stream_in.block_size;
  const int n_blocks = array_stream_in.n_blocks;

  for (int n_block = 0; n_block < n_blocks; n_block++) {
    typename array_stream<T, array_shape<M>, stream_block_size>::T_block block =
        array_stream_in.data.read();

    for (int n_elem = 0; n_elem < block_size; n_elem++) {
      int n = n_block * block_size + n_elem;
      if (n < n_total) {
        array_out[n] = block[n_elem];
      }
    }
  }
}

template <typename T, int M, int N, int stream_block_size = 1>
void array_stream_to_array_2d(
    array_stream<T, array_shape<M, N>, stream_block_size> &array_stream_in,
    T array_out[M][N]) {
#pragma HLS INLINE off

  const int n_total = M * N;
  const int block_size = array_stream_in.block_size;
  const int n_blocks = array_stream_in.n_blocks;

  for (int n_block = 0; n_block < n_blocks; n_block++) {
    typename array_stream<T, array_shape<M, N>, stream_block_size>::T_block
        block = array_stream_in.data.read();

    for (int n_elem = 0; n_elem < block_size; n_elem++) {
      int n = n_block * block_size + n_elem;
      if (n < n_total) {
        array_out[n / N][n % N] = block[n_elem];
      }
    }
  }
}

template <typename T, int M, int N, int O, int stream_block_size = 1>
void array_stream_to_array_3d(
    array_stream<T, array_shape<M, N, O>, stream_block_size> &array_stream_in,
    T array_out[M][N][O]) {
#pragma HLS INLINE off

  const int n_total = M * N * O;
  const int block_size = array_stream_in.block_size;
  const int n_blocks = array_stream_in.n_blocks;

  for (int n_block = 0; n_block < n_blocks; n_block++) {
    typename array_stream<T, array_shape<M, N, O>, stream_block_size>::T_block
        block = array_stream_in.data.read();

    for (int n_elem = 0; n_elem < block_size; n_elem++) {
      int n = n_block * block_size + n_elem;
      if (n < n_total) {
        array_out[n / (N * O)][(n / O) % N][n % O] = block[n_elem];
      }
    }
  }
}

template <typename T, typename array_shape, int stream_block_size = 1>
void array_stream_to_array(
    array_stream<T, array_shape, stream_block_size> &array_stream_in,
    T *array_out) {
#pragma HLS INLINE off

  const int n_dims = array_shape::n_dims;

  const bool is_valid_dims = (n_dims == 1) || (n_dims == 2) || (n_dims == 3);
  static_assert(is_valid_dims,
                "array_stream_to_array: invalid number of dimensions");

  if (n_dims == 1) {
    array_stream_to_array_1d<T, array_shape::M, stream_block_size>(
        array_stream_in, array_out);
  }
  if (n_dims == 2) {
    array_stream_to_array_2d<T, array_shape::M, array_shape::N,
                             stream_block_size>(array_stream_in, array_out);
  }
  if (n_dims == 3) {
    array_stream_to_array_3d<T, array_shape::M, array_shape::N, array_shape::O,
                             stream_block_size>(array_stream_in, array_out);
  }
}

template <typename T_array_in, typename T_array_out>
void cast_stream(T_array_in &a, T_array_out &b) {
#pragma HLS INLINE off

  static_assert(T_array_in::block_size == T_array_out::block_size,
                "Input and output arrays must have the same block size");

  typedef typename T_array_out::T output_data_type;

  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array_in::T_block a_in = a.data.read();
    typename T_array_out::T_block b_out = typename T_array_in::T_data(0.0);
    for (int j = 0; j < T_array_in::block_size; j++) {
#pragma HLS unroll
      b_out[j] = output_data_type(a_in[j]);
    }

    b.data.write(b_out);
  }
}

template <typename T = float, const int M>
bool compare_array_1d(T arr1[M], T arr2[M], float eps) {
  for (int i = 0; i < M; i++) {
    if (std::abs(arr1[i] - arr2[i]) > eps) {
      return false;
    }
  }
  return true;
}

template <typename T = float, const int M, const int N>
bool compare_array_2d(T arr1[M][N], T arr2[M][N], float eps) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (std::abs(arr1[i][j] - arr2[i][j]) > eps) {
        return false;
      }
    }
  }
  return true;
}

template <typename T = float, const int M, const int N, const int O>
bool compare_array_3d(T arr1[M][N][O], T arr2[M][N][O], float eps) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < O; k++) {
        if (std::abs(arr1[i][j][k] - arr2[i][j][k]) > eps) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename T_array_stream>
void empty_the_stream(T_array_stream &array_stream_in) {
  for (int i = 0; i < array_stream_in.n_blocks; i++) {
    array_stream_in.data.read();
  }
}

template <typename T_array, typename... T_arrays>
void copy_stream_write(typename T_array::T_block element, T_array &dst,
                       T_arrays &...dsts) {
#pragma HLS inline
  dst.data.write(element);
  copy_stream_write(element, dsts...);
}
template <typename T_array>
void copy_stream_write(typename T_array::T_block element, T_array &dst) {
#pragma HLS inline
  dst.data.write(element);
}
template <typename T_array, typename... T_arrays>
void copy_stream(T_array &src, T_arrays &...dsts) {
#pragma HLS inline off
  for (int i = 0; i < T_array::n_blocks; i++) {
    typename T_array::T_block element = src.data.read();
    copy_stream_write(element, dsts...);
  }
}

// element wise addition of two arrays
template <typename T_array>
void elementwise_add(T_array &a, T_array &b, T_array &c) {
#pragma HLS INLINE off
  for (int i = 0; i < c.n_blocks; i++) {
    typename T_array::T_block a_in = a.data.read();
    typename T_array::T_block b_in = b.data.read();
    typename T_array::T_block c_out = a_in + b_in;
    c.data.write(c_out);
  }
}

// // element wise multiplication of two arrays
template <typename T_array>
void elementwise_mul(T_array &a, T_array &b, T_array &c) {
#pragma HLS INLINE off
  for (int i = 0; i < c.n_blocks; i++) {
    typename T_array::T_block a_in = a.data.read();
    typename T_array::T_block b_in = b.data.read();
    typename T_array::T_block c_out = a_in * b_in;
    c.data.write(c_out);
  }
}

// element wise negate of an array
template <typename T_array> void elementwise_negate(T_array &a, T_array &b) {
#pragma HLS INLINE off
  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array::T_block a_in = a.data.read();
    typename T_array::T_block b_out = typename T_array::T_data(0.0);
    for (int j = 0; j < T_array::block_size; j++) {
#pragma HLS unroll
      b_out[j] = -a_in[j];
    }
    b.data.write(b_out);
  }
}

// element wise square of an array
template <typename T_array> void elementwise_square(T_array &a, T_array &b) {
#pragma HLS INLINE off
  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array::T_block a_in = a.data.read();
    typename T_array::T_block b_out = a_in * a_in;
    b.data.write(b_out);
  }
}

template <typename T_array> void elementwise_sin(T_array &a, T_array &b) {
#pragma HLS INLINE off
  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array::T_block a_in = a.data.read();

    typename T_array::T_block b_out = typename T_array::T_data(0.0);
    for (int j = 0; j < T_array::block_size; j++) {
#pragma HLS unroll
      b_out[j] = m_sin(a_in[j]);
    }

    b.data.write(b_out);
  }
}

template <typename T_array> void elementwise_cos(T_array &a, T_array &b) {
#pragma HLS INLINE off
  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array::T_block a_in = a.data.read();

    typename T_array::T_block b_out = typename T_array::T_data(0.0);
    for (int j = 0; j < T_array::block_size; j++) {
#pragma HLS unroll
      b_out[j] = m_cos(a_in[j]);
    }

    b.data.write(b_out);
  }
}

template <typename T_array>
void elementwise_add_const(T_array &a, typename T_array::T_data b, T_array &c) {
#pragma HLS INLINE off
  for (int i = 0; i < c.n_blocks; i++) {
    typename T_array::T_block a_in = a.data.read();
    typename T_array::T_block c_out = a_in + b;
    c.data.write(c_out);
  }
}

template <typename T_array>
void elementwise_mul_const(T_array &a, typename T_array::T_data b, T_array &c) {
  for (int i = 0; i < c.n_blocks; i++) {
    typename T_array::T_block a_in = a.data.read();
    typename T_array::T_block c_out = a_in * b;
    c.data.write(c_out);
  }
}

template <typename T_array_in, typename T_array_out>
void transpose_2d(T_array_in &input, T_array_out &output) {
#pragma HLS INLINE off

  static_assert(T_array_in::block_size == T_array_out::block_size,
                "Input and output arrays must have the same block size");

  const int B_in = T_array_in::block_size;
  const int B_out = T_array_out::block_size;

  // transpose buffer
  const int M_in = T_array_in::shape.dim_sizes[0];
  const int N_in = T_array_in::shape.dim_sizes[1];

  const int M_out = T_array_out::shape.dim_sizes[0];
  const int N_out = T_array_out::shape.dim_sizes[1];

  typename T_array_in::T_data transpose_buffer[M_in][N_in];

// cyclic partitioning of the transpose buffer based on the block size
#pragma HLS ARRAY_PARTITION variable = transpose_buffer cyclic factor =        \
    B_in dim = 1
#pragma HLS ARRAY_PARTITION variable = transpose_buffer cyclic factor =        \
    B_out dim = 2

  // helper variables
  const int n_blocks = input.n_blocks;
  const int block_size = input.block_size;
  const int n_leftover = input.shape.N % block_size;

  // read in the input data into the transpose buffer
  int i_in = 0;
  int j_in = 0;
  int n_in = 0;

  for (int n_block = 0; n_block < n_blocks; n_block++) {
    typename T_array_in::T_block block = input.data.read();

    for (int n_elem = 0; n_elem < block_size; n_elem++) {
      if (n_in < T_array_in::shape.N) {
        transpose_buffer[i_in][j_in] = block[n_elem];
        j_in++;
        if (j_in == N_in) {
          j_in = 0;
          i_in++;
        }
        n_in++;
      }
    }
  }

  // write the data to the output but in transpose order
  int i_out = 0;
  int j_out = 0;
  int n_out = 0;

  for (int n_block = 0; n_block < n_blocks; n_block++) {
    typename T_array_out::T_block block = typename T_array_out::T_data(0.0);

    for (int n_elem = 0; n_elem < block_size; n_elem++) {
      if (n_out < T_array_out::shape.N) {
        block[n_elem] = transpose_buffer[j_out][i_out];
        j_out++;
        if (j_out == N_out) {
          j_out = 0;
          i_out++;
        }
        n_out++;
      }
    }

    output.data.write(block);
  }
}

template <typename T_array_a, typename T_array_b, typename T_array_c>
void mm(T_array_a &a, T_array_b &b, T_array_c &c) {
  // static_assert(false, "Not implemented");
  static_assert(T_array_a::block_size == T_array_b::block_size,
                "Input arrays must have the same block size");
  static_assert(
      T_array_a::block_size == T_array_c::block_size,
      "Input \"a\" and output array \"c\" must have the same block size");
  static_assert(
      T_array_a::block_size == T_array_c::block_size,
      "Input \"b\" and output array \"c\" must have the same block size");

  static_assert(T_array_a::shape.dim_sizes[1] == T_array_b::shape.dim_sizes[0],
                "Input arrays must have compatible dimensions");
  static_assert(
      T_array_a::shape.dim_sizes[0] == T_array_c::shape.dim_sizes[0],
      "Input \"a\" and output array \"c\" must have compatible dimensions");
  static_assert(
      T_array_b::shape.dim_sizes[1] == T_array_c::shape.dim_sizes[1],
      "Input \"b\" and output array \"c\" must have compatible dimensions");

  const int M = T_array_a::shape.dim_sizes[0];
  const int N = T_array_b::shape.dim_sizes[0];
  const int O = T_array_c::shape.dim_sizes[1];

  const int B_a = T_array_a::block_size;
  const int B_b = T_array_b::block_size;
  const int B_c = T_array_c::block_size;

  typedef typename T_array_a::T_data T_a;
  typedef typename T_array_b::T_data T_b;
  typedef typename T_array_c::T_data T_c;

  // #FIXME: this is not a good way to prevent stack overflow since there can be
  // many instances of this function with different sizes T_a a_buffer[M][N];
  // T_b b_buffer[N][O];
  // T_c c_buffer[M][O];
  CSIM_STATIC T_a a_buffer[M][N];
  CSIM_STATIC T_b b_buffer[N][O];
  CSIM_STATIC T_c c_buffer[M][O];

  // array_stream_to_array_2d(a, a_buffer);
  array_stream_to_array_2d<T_a, M, N, B_a>(a, a_buffer);
  array_stream_to_array_2d<T_b, N, O, B_b>(b, b_buffer);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < O; j++) {
      T_c sum = 0.0;
      for (int k = 0; k < N; k++) {
        sum += a_buffer[i][k] * b_buffer[k][j];
      }
      c_buffer[i][j] = sum;
    }
  }

  array_2d_to_array_stream<T_c, M, O, B_c>(c_buffer, c);
}

// create a heler object called an auto_stream_writer that is given an output
// stream the auto_stream_writer will buffer incoming data and write it to the
// output stream when the buffer is full the auto_stream_writer will allso have
// a flush method that will write a partially full buffer to the output stream
template <typename T_array_out> class auto_stream_writer {
public:
  const int block_size = T_array_out::block_size;
  typename T_array_out::T_data buffer[T_array_out::block_size];
  int n_buffered = 0;
  T_array_out &output;

  auto_stream_writer(T_array_out &output) : output(output) {}

  void write(typename T_array_out::T_data data) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = buffer complete dim = 1
    buffer[n_buffered] = data;
    n_buffered++;
    if (n_buffered == block_size) {
      flush();
    }
  }

  void flush() {
    // make a vector of the buffered data
    typename T_array_out::T_block block = typename T_array_out::T_data(0.0);
    for (int i = 0; i < n_buffered; i++) {
#pragma HLS UNROLL
      block[i] = buffer[i];
    }

    // write the vector to the output stream
    output.data.write(block);

    // reset the buffer
    for (int i = 0; i < n_buffered; i++) {
#pragma HLS UNROLL
      buffer[i] = typename T_array_out::T_data(0.0);
    }
    n_buffered = 0;
  }

  void flush_if_not_empty() {
    if (n_buffered > 0) {
      flush();
    }
  }
};

template <const int dim, const int index, typename T_array_in,
          typename T_array_out>
void select(T_array_in &a, T_array_out &b) {
#pragma HLS INLINE off

  static_assert(T_array_in::block_size == T_array_out::block_size,
                "Input and output arrays must have the same block size");
  static_assert(
      T_array_out::shape.n_dims == T_array_in::shape.n_dims - 1,
      "Output array must have one less dimension than the input array");
  static_assert(T_array_in::shape.dim_sizes[dim] > index,
                "Index must be less than the size of the specified dimension");
  static_assert(T_array_out::shape.N ==
                    T_array_in::shape.N / T_array_in::shape.dim_sizes[dim],
                "Output array must have the correct number of elements");

  const int B_in = T_array_in::block_size;
  const int B_out = T_array_out::block_size;

  // start_index = index * product of dimensions after dim
  // chunk_size = product of dimensions after dim
  // chunk_stride = product of dimensions including and after dim
  const int start_index = get_product<T_array_in::shape.n_dims, dim + 1>(
                              T_array_in::shape.dim_sizes) *
                          index;
  const int chunk_size = get_product<T_array_in::shape.n_dims, dim + 1>(
      T_array_in::shape.dim_sizes);
  const int chunk_stride =
      get_product<T_array_in::shape.n_dims, dim>(T_array_in::shape.dim_sizes);

  int current_linear_index = 0;
  int total_output_count = 0;
  auto_stream_writer<T_array_out> asw(b);
  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array_in::T_block block = a.data.read();
    for (int j = 0; j < B_in; j++) {
      if (total_output_count < T_array_out::shape.N) {
        // #pragma HLS PIPELINE

        bool past_start_index = current_linear_index >= start_index;

        int i = (current_linear_index - start_index) / chunk_stride;
        int chunk_start = start_index + i * chunk_stride;
        int chunk_end = chunk_start + chunk_size;
        bool inside_chunk = (chunk_start <= current_linear_index) &&
                            (current_linear_index < chunk_end);

        if (past_start_index && inside_chunk) {
          asw.write(block[j]);
          total_output_count++;
        }
        current_linear_index++;
      }
    }
  }
  asw.flush_if_not_empty();
}

template <const int dim, const int num, typename T_array_in,
          typename T_array_out>
void repeat_singleton_dim_2d(T_array_in &a, T_array_out &b) {
#pragma HLS INLINE off

  static_assert(T_array_in::block_size == T_array_out::block_size,
                "Input and output arrays must have the same block size");
  static_assert(
      T_array_out::shape.n_dims == T_array_in::shape.n_dims,
      "Output array must have one more dimension than the input array");
  static_assert(
      T_array_in::shape.dim_sizes[dim] == 1,
      "Input array must have a dimension of size 1 at the specified dimension");
  static_assert(T_array_out::shape.dim_sizes[dim] == num,
                "Output array must have a dimension of size num at the "
                "specified dimension");

  static_assert(T_array_in::shape.n_dims == 2,
                "Input array must have two dimensions");
  static_assert(T_array_out::shape.n_dims == 2,
                "Output array must have two dimensions");

  static_assert(0 <= dim && dim < T_array_in::shape.n_dims,
                "Dimension must be between 0 and the number of dimensions in "
                "the input array");
  static_assert(num > 1, "num must be greater than 1");

  const int B_in = T_array_in::block_size;
  const int B_out = T_array_out::block_size;

  typedef typename T_array_in::T_data T_data_in;
  typedef typename T_array_out::T_data T_data_out;
  static_assert(std::is_same<T_data_in, T_data_out>::value,
                "Input and output arrays must have the same data type");

  const int non_singleton_dim = (dim == 0) ? 1 : 0;
  const int non_singleton_dim_size =
      T_array_in::shape.dim_sizes[non_singleton_dim];

  T_data_in buffer[non_singleton_dim_size];

  int n = 0;
  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array_in::T_block block = a.data.read();
    for (int j = 0; j < B_in; j++) {
      if (n < non_singleton_dim_size) {
        buffer[n] = block[j];
        n++;
      } else {
        break;
      }
    }
  }

  auto_stream_writer<T_array_out> asw(b);
  if (dim == 0) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < non_singleton_dim_size; j++) {
#pragma HLS PIPELINE
        asw.write(buffer[j]);
      }
    }
    asw.flush_if_not_empty();
  } else {
    for (int i = 0; i < non_singleton_dim_size; i++) {
      for (int j = 0; j < num; j++) {
#pragma HLS PIPELINE
        asw.write(buffer[i]);
      }
    }
    asw.flush_if_not_empty();
  }
}

// template <typename T_array_a, typename T_array_b, typename T_array_c>
// void elementwise_add_2d_brodcast(T_array_a &a, T_array_b &b, T_array_c &c){
//     #pragma HLS INLINE off

//     static_assert(T_array_a::shape.n_dims == 2, "Input array a must have two
//     dimensions"); static_assert(T_array_b::shape.n_dims == 2, "Input array b
//     must have two dimensions"); static_assert(T_array_c::shape.n_dims == 2,
//     "Output array c must have two dimensions");

//     static_assert(T_array_a::block_size == T_array_b::block_size, "Input
//     arrays a and b must have the same block size");
//     static_assert(T_array_a::block_size == T_array_c::block_size, "Input
//     arrays a and c must have the same block size");
//     static_assert(T_array_b::block_size == T_array_c::block_size, "Input
//     arrays b and c must have the same block size");

//     const int M_a = T_array_a::shape.dim_sizes[0];
//     const int N_a = T_array_a::shape.dim_sizes[1];

//     const int M_b = T_array_b::shape.dim_sizes[0];
//     const int N_b = T_array_b::shape.dim_sizes[1];

//     const int M_c = T_array_c::shape.dim_sizes[0];
//     const int N_c = T_array_c::shape.dim_sizes[1];

//     const bool same_dim = (M_a == M_b) && (N_a == N_b);
//     const bool singleton_N = (M_a == M_b) && ((N_a == 1) || (N_b == 1));
//     const bool singleton_M = (N_a == N_b) && ((M_a == 1) || (M_b == 1));

//     static_assert(same_dim || singleton_N || singleton_M, "Input arrays must
//     have the same dimensions or one of the dimensions must be of size 1");

//     const int B_a = T_array_a::block_size;
//     const int B_b = T_array_b::block_size;
//     const int B_c = T_array_c::block_size;

//     typedef typename T_array_a::T_data T_a;
//     typedef typename T_array_b::T_data T_b;
//     typedef typename T_array_c::T_data T_c;

//     if(same_dim){
//         elementwise_add(a, b, c);
//     }
//     else
//     {
//         array_stream<T_c, array_shape<M_c, N_c>, B_c> s_brodcast;
//         if (singleton_N)
//         {
//             if(N_a == 1){
//                 repeat_singleton_dim_2d<0, N_b>(a, s_brodcast);
//                 elementwise_add(s_brodcast, b, c);
//             }
//             else{
//                 repeat_singleton_dim_2d<0, N_a>(b, s_brodcast);
//                 elementwise_add(a, s_brodcast, c);
//             }
//         }
//         else
//         {
//             if(M_a == 1){
//                 repeat_singleton_dim_2d<1, M_b>(a, s_brodcast);
//                 elementwise_add(s_brodcast, b, c);
//             }
//             else{
//                 repeat_singleton_dim_2d<1, M_a>(b, s_brodcast);
//                 elementwise_add(a, s_brodcast, c);
//             }
//         }
//     }
// }

template <typename T_array_in, typename T_array_out>
void unsqueeze(T_array_in &a, T_array_out &b, const int dim) {
#pragma HLS INLINE off
  static_assert(T_array_in::block_size == T_array_out::block_size,
                "Input and output arrays must have the same block size");
  static_assert(
      T_array_in::shape.N == T_array_out::shape.N,
      "Input and output arrays must have the same number of elements");
  static_assert(
      T_array_out::shape.n_dims == T_array_in::shape.n_dims + 1,
      "Output array must have one more dimension than the input array");
  // static_assert(T_array_out::shape.dim_sizes[dim] == 1, "Output array must
  // have a dimension of size 1 at the specified dimension");

  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array_in::T_block block = a.data.read();
    b.data.write(block);
  }
}

template <typename T_array_in, typename T_array_out>
void stream_block_adapter(T_array_in &a, T_array_out &b) {
#pragma HLS INLINE off

  static_assert(
      T_array_in::shape.n_dims == T_array_out::shape.n_dims,
      "Input and output arrays must have the same number of dimensions");
  static_assert(
      T_array_in::shape.N == T_array_out::shape.N,
      "Input and output arrays must have the same number of elements");
  // static_assert(compare_array_shape<T_array_in::shape.n_dims>(T_array_in::shape.dim_sizes,
  // T_array_out::shape.dim_sizes), "Input and output arrays must have the same
  // dimensions");

  const int block_size_in = T_array_in::block_size;
  const int block_size_out = T_array_out::block_size;
  static_assert(block_size_in >= 0, "Input block size must be greater than 0");
  static_assert(block_size_out >= 0,
                "Output block size must be greater than 0");

  typedef typename T_array_in::T_data T_data_in;
  typedef typename T_array_out::T_data T_data_out;
  static_assert(std::is_same<T_data_in, T_data_out>::value,
                "Input and output arrays must have the same data type");

  auto_stream_writer<T_array_out> asw(b);

  int input_counter = 0;
  for (int i = 0; i < a.n_blocks; i++) {
    typename T_array_in::T_block block = a.data.read();

    for (int j = 0; j < block_size_in; j++) {
#pragma HLS PIPELINE
      asw.write(block[j]);
      input_counter++;
    }
  }
  asw.flush_if_not_empty();
}

template <typename T_array_in, typename T_array_out, int M>
void cast_array_1d(T_array_in a[M], T_array_out b[M]) {
#pragma HLS INLINE off

  for (int i = 0; i < M; i++) {
    b[i] = T_array_out(a[i]);
  }
}

template <typename T_array_in, typename T_array_out, int M, int N>
void cast_array_2d(T_array_in a[M][N], T_array_out b[M][N]) {
#pragma HLS INLINE off

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      b[i][j] = T_array_out(a[i][j]);
    }
  }
}

template <typename T_array_in, typename T_array_out, int M, int N, int O>
void cast_array_3d(T_array_in a[M][N][O], T_array_out b[M][N][O]) {
#pragma HLS INLINE off

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < O; k++) {
        b[i][j][k] = T_array_out(a[i][j][k]);
      }
    }
  }
}

template <typename T, const int M> void copy_1d(T from[M], T to[M]) {
  for (int i = 0; i < M; i++) {
    to[i] = from[i];
  }
}

template <typename T, const int M, const int N>
void copy_2d(T from[M][N], T to[M][N]) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      to[i][j] = from[i][j];
    }
  }
}

template <typename T, const int M, const int N, const int O>
void copy_3d(T from[M][N][O], T to[M][N][O]) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < O; k++) {
        to[i][j][k] = from[i][j][k];
      }
    }
  }
}

template <const int M, typename T_in, typename T_out>
void cast_1d(T_in in[M], T_out out[M]) {
  for (int i = 0; i < M; i++) {
    out[i] = T_out(in[i]);
  }
}

template <const int M, const int N, typename T_in, typename T_out>
void cast_2d(T_in in[M][N], T_out out[M][N]) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      out[i][j] = T_out(in[i][j]);
    }
  }
}

template <const int M, const int N, const int O, typename T_in, typename T_out>
void cast_3d(T_in in[M][N][O], T_out out[M][N][O]) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < O; k++) {
        out[i][j][k] = T_out(in[i][j][k]);
      }
    }
  }
}

template <const int M, typename T = float>
void load_data_1d(const char *fp, T arr[M]) {
  FILE *f;
  f = fopen(fp, "r");
  fread(arr, sizeof(T), M, f);
  fclose(f);
}

template <const int M, const int N, typename T = float>
void load_data_2d(const char *fp, T arr[M][N]) {
  FILE *f;
  f = fopen(fp, "r");
  fread(arr, sizeof(T), M * N, f);
  fclose(f);
}

template <const int M, const int N, const int O, typename T = float>
void load_data_3d(const char *fp, T arr[M][N][O]) {
  FILE *f;
  f = fopen(fp, "r");
  fread(arr, sizeof(T), M * N * O, f);
  fclose(f);
}

template <const int M, typename T = float>
void load_data_var_1d(const char *fp, T arr[M], int i) {
  FILE *f;
  f = fopen(fp, "r");
  fread(arr, sizeof(T), i, f);
  fclose(f);
}

template <const int M, const int N, typename T = float>
void load_data_var_2d(const char *fp, T arr[M][N], int i, int j) {
  FILE *f;
  f = fopen(fp, "r");
  fread(arr, sizeof(T), i * j, f);
  fclose(f);
}

template <const int M, const int N, const int O, typename T = float>
void load_data_var_3d(const char *fp, T arr[M][N][O], int i, int j, int k) {
  FILE *f;
  f = fopen(fp, "r");
  fread(arr, sizeof(T), i * j * k, f);
  fclose(f);
}

template <const int M, typename T = float>
bool compare_data_1d(T arr1[M], T arr2[M], float eps) {
  for (int i = 0; i < M; i++) {
    if (std::abs(arr1[i] - arr2[i]) > eps) {
      return false;
    }
  }
  return true;
}

template <const int M, const int N, typename T = float>
bool compare_data_2d(T arr1[M][N], T arr2[M][N], float eps) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (std::abs(arr1[i][j] - arr2[i][j]) > eps) {
        return false;
      }
    }
  }
  return true;
}

template <const int M, const int N, const int O, typename T = float>
bool compare_data_3d(T arr1[M][N][O], T arr2[M][N][O], float eps) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < O; k++) {
        if (std::abs(arr1[i][j][k] - arr2[i][j][k]) > eps) {
          return false;
        }
      }
    }
  }
  return true;
}

// compute_mae_1d
template <const int M, typename T = float>
float compute_mae_1d(T arr1[M], T arr2[M]) {
  float mae = 0;
  for (int i = 0; i < M; i++) {
    mae += std::abs(arr1[i] - arr2[i]);
  }
  return mae / M;
}

// compute_mae_2d
template <const int M, const int N, typename T = float>
float compute_mae_2d(T arr1[M][N], T arr2[M][N]) {
  float mae = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      mae += std::abs(arr1[i][j] - arr2[i][j]);
    }
  }
  return mae / (M * N);
}

// compute_mae_3d
template <const int M, const int N, const int O, typename T = float>
float compute_mae_3d(T arr1[M][N][O], T arr2[M][N][O]) {
  float mae = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < O; k++) {
        mae += std::abs(arr1[i][j][k] - arr2[i][j][k]);
      }
    }
  }
  return mae / (M * N * O);
}

template <int M, typename T = float> void print_array_1d(T x[M]) {

  float x_cast[M];
  cast_array_1d<T, float, M>(x, x_cast);

  if (M < 6) {
    std::cout << "[ ";
    for (int i = 0; i < M; i++) {
      printf(" %010.6f ", x_cast[i]);
    }
    std::cout << " ]";
  } else {
    std::cout << "[ ";
    for (int i = 0; i < 3; i++) {
      printf(" %010.6f ", x_cast[i]);
    }
    std::cout << " ... ";
    for (int i = M - 3; i < M; i++) {
      printf(" %010.6f ", x_cast[i]);
    }
    std::cout << " ]";
  }

  std::cout << std::endl;
}

template <int M, int N, typename T = float> void print_array_2d(T x[M][N]) {

  float x_cast[M][N];
  cast_array_2d<T, float, M, N>(x, x_cast);

  if ((M < 6 && N < 6) || (M < 6 && N >= 6)) {
    std::cout << "[ ";
    for (int i = 0; i < M; i++) {
      if (i > 0) {
        std::cout << "  ";
      }
      print_array_1d<float, N>(x_cast[i]);
      if (i < M - 1) {
        std::cout << std::endl;
      }
    }
    std::cout << " ]" << std::endl;
  } else {
    std::cout << "[ ";
    for (int i = 0; i < 3; i++) {
      if (i > 0) {
        std::cout << "  ";
      }
      print_array_1d<float, N>(x_cast[i]);
      if (i < M - 1) {
        std::cout << std::endl;
      }
    }
    std::cout << "  ." << std::endl;
    std::cout << "  ." << std::endl;
    std::cout << "  ." << std::endl;
    for (int i = M - 3; i < M; i++) {
      if (i > 0) {
        std::cout << "  ";
      }
      print_array_1d<float, N>(x_cast[i]);
      if (i < M - 1) {
        std::cout << std::endl;
      }
    }
    std::cout << " ]" << std::endl;
  }

  std::cout << std::endl;
}