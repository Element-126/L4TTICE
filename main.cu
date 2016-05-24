#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <utility>
#include <cassert>
#include <cstdio>
#include "H5Cpp.h"

/******************************************************************************/

// Geometry & parameters
// =====================

// Block size
constexpr size_t B0 = 8; // ideally = warp size, for coalesced read & write
constexpr size_t B1 = 8;
constexpr size_t B2 = 8;
constexpr size_t B3 = 8;
// Number of threads 8³ = 512
// Loop over the last dimension
// Shared memory usage: 40000o including halos.
// Then grid-stride loop to reuse the RNG state

// Grid size
constexpr size_t G0 = 2;
constexpr size_t G1 = 2;
constexpr size_t G2 = 2;
constexpr size_t G3 = 2;
  
// Lattice size
constexpr size_t N0 = B0*G0;
constexpr size_t N1 = B1*G1;
constexpr size_t N2 = B2*G2;
constexpr size_t N3 = B3*G3;

// Data array size (including ghost cells)
constexpr size_t M0 = N0+2;
constexpr size_t M1 = N1+2;
constexpr size_t M2 = N2+2;
constexpr size_t M3 = N3+2;
constexpr size_t M_count = M0*M1*M2*M3;
constexpr size_t M_bytes = M_count*sizeof(float);

// Lattice spacing
constexpr float a = 0.5f;

// Physical parameters
constexpr float m2 = 1.0f;
constexpr float lambda = 1.0f;

// Monte-Carlo parameters
constexpr unsigned int N_cor = 20;
constexpr unsigned int N_cf  = 100;
constexpr unsigned int N_th  = 10*N_cor;
constexpr float epsilon = 0.7f;

// Output
const H5std_string file_name("correlations.h5");
const H5std_string dataset_name("corr");

/******************************************************************************/

// Variation of the action
// =======================

// Change in the action when φ(i) → φ(i) + ζ
// Idx: array index, including ghost cells
__device__ float delta_S_kin(float * f, const size_t Idx, const float zeta) {

  return a*a*zeta*( 4.0f*zeta + 8.0f*f[Idx]
                    - f[Idx+1]        - f[Idx-1]        // ± (1,0,0,0)
                    - f[Idx+M0]       - f[Idx-M0]       // ± (0,1,0,0)
                    - f[Idx+M0*M1]    - f[Idx-M0*M1]    // ± (0,0,1,0)
                    - f[Idx+M0*M1*M2] - f[Idx-M0*M1*M2] // ± (0,0,0,1)
                    );
}

// Free field: V(φ) = ½m²φ²
__device__ float delta_S_free(float * f, const size_t Idx, const float zeta) {

  const float fi = f[Idx];
  const float delta_V = 0.5f*m2*zeta*(2.0f*fi+zeta);
  return delta_S_kin(f, Idx, zeta) + a*a*a*a*delta_V;
}

// Interacting field: V(φ) = ½m²φ² + ¼λφ⁴
__device__ float delta_S_phi4(float * f, const size_t Idx, const float zeta) {

  const float fi = f[Idx];     // φi
  const float fiP = fi + zeta; // φi + ζ
  const float delta_V = 0.5f*m2*( fiP*fiP - fi*fi ) + 0.25f*lambda*( fiP*fiP*fiP*fiP - fi*fi*fi*fi );
  return delta_S_kin(f, Idx, zeta) + a*a*a*a*delta_V;
}

// Choice of the action used in the simulation
constexpr auto dS = delta_S_free;

/******************************************************************************/

// Kernels
// =======

// // Main kernel, performing one Monte-Carlo iteration
// template <float (*delta_S)(float*, const size_t, const float)>
// __global__ void mc_kernel(float * lat, float * lo, curandState * states) {

//   // Global thread index = lattice site
//   const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//   // Array index
//   const size_t Idx = array_idx(tid);

//   curandState state = states[tid];
//   float zeta = (2.0f*curand_uniform(&state) - 1.0f) * epsilon; // ζ ∈ [-ε,+ε]

//   // Compute change in the action due to variation ζ at site Idx
//   const float delta_S_i = delta_S(lo, Idx, zeta);
  
//   // Update the lattice depending on the variation ΔSi
//   const float update = (float) (delta_S_i < 0.0f || (exp(-delta_S_i) > curand_uniform(&state)));
//   // Is the above really branchless ?
//   lat[Idx] += update * zeta;

//   states[tid] = state;
// }

// MC iteration over "black" indices
template<float (*delta_S)(float*, const size_t, const float)>
__global__ void mc_update_black(float * lat, curandState * states) {

  // Global thread index
  const size_t t0 = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t t1 = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t t2 = blockIdx.z * blockDim.z + threadIdx.z;

  size_t tid = t0 + (N0/2)*t1 + (N0/2)*N1*t2;
  auto state = states[tid];

  // Physical index: 2·(t0   + N0/2·t1     + N0/2·N1·t2     + N0/2·N1·N2·t3)
  // Array index:    2·(t0+1 + M0/2·(t1+1) + M0/2·M1·(t2+1) + M0/2·M1·M2·(t3+1))

  // Grid stride loop in direction 3
  for (size_t g3 = 0 ; g3 < G3 ; ++g3) {

    // Start (array) index
    size_t Idx = 2*(t0+1 + (M0/2)*(t1+1) + (M0/2)*M1*(t2+1) + (M0/2)*M1*M2*(g3*B3+1));

    // Small loop in direction 3
    for (size_t t3 = 0 ; t3 < B3 ; ++t3) {

      const float zeta = (2.0f*curand_uniform(&state) - 1.0f) * epsilon; // ζ ∈ [-ε,+ε]
      // Compute change in the action due to variation ζ at site Idx
      const float delta_S_i = delta_S(lat, Idx, zeta);

      // Update the lattice depending on the variation ΔSi
      const float update = (float) (delta_S_i < 0.0f || (exp(-delta_S_i) > curand_uniform(&state)));
      // Is the above really branchless ?
      lat[Idx] += update * zeta;

      // Virtually increment t3
      Idx += (M0/2)*M1*M2;
    }
  }

  // Write RNG state back to global memory
  states[tid] = state;
}

// MC iteration over "white" indices
template<float (*delta_S)(float*, const size_t, const float)>
__global__ void mc_update_white(float * lat, curandstate * states) {

  // TODO
}

// initialize rng state
__global__ void rng_init(unsigned long long seed, curandstate * states) {

  const size_t idx = blockidx.x * blockdim.x + threadidx.x;
  curand_init(seed, idx, 0, &states[idx]);
}

/******************************************************************************/

// exchange of the 3d "faces" of the 4d lattice
// ============================================

// face 0 (stride = 1)
__global__ void exchange_faces_0(float * lat) {

  const size_t i1 = blockidx.x * blockdim.x + threadidx.x + 1;
  const size_t i2 = blockidx.y * blockdim.y + threadidx.y + 1;
  const size_t i3 = blockidx.z * blockdim.z + threadidx.z + 1;
  const size_t idx = m0*i1 + m0*m1*i2 + m0*m1*m2*i3;

  lat[idx         ] = lat[idx + n0];
  lat[idx + (n0+1)] = lat[idx +  1];
}

// face 1 (stride = m0)
__global__ void exchange_faces_1(float * lat) {

  const size_t i0 = blockidx.x * blockdim.x + threadidx.x + 1;
  const size_t i2 = blockidx.y * blockdim.y + threadidx.y + 1;
  const size_t i3 = blockidx.z * blockdim.z + threadidx.z + 1;
  const size_t idx = i0 + m0*m1*i2 + m0*m1*m2*i3;

  lat[idx            ] = lat[idx + m0*n1];
  lat[idx + m0*(n1+1)] = lat[idx + m0   ];
}

// face 2 (stride = m0·m1)
__global__ void exchange_faces_2(float * lat) {

  const size_t i0 = blockidx.x * blockdim.x + threadidx.x + 1;
  const size_t i1 = blockidx.y * blockdim.y + threadidx.y + 1;
  const size_t i3 = blockidx.z * blockdim.z + threadidx.z + 1;
  const size_t idx = i0 + m0*i1 + m0*m1*m2*i3;

  lat[idx               ] = lat[idx + m0*m1*n2];
  lat[idx + m0*m1*(n2+1)] = lat[idx + m0*m1   ];
}

// face 3 (stride = m0·m1·m2)
__global__ void exchange_faces_3(float * lat) {

  const size_t i0 = blockidx.x * blockdim.x + threadidx.x + 1;
  const size_t i1 = blockidx.y * blockdim.y + threadidx.y + 1;
  const size_t i2 = blockidx.z * blockdim.z + threadidx.z + 1;
  const size_t idx = i0 + m0*i1 + m0*m1*i2;

  lat[idx                  ] = lat[idx + m0*m1*m2*n3];
  lat[idx + m0*m1*m2*(n3+1)] = lat[idx + m0*m1*m2   ];
}

// exchange all faces
__host__ void exchange_faces(float * lat) {

  exchange_faces_0<<<dim3(g1,g2,g3),dim3(b1,b2,b3)>>>(lat);
  exchange_faces_1<<<dim3(g0,g2,g3),dim3(b0,b2,b3)>>>(lat);
  exchange_faces_2<<<dim3(g0,g1,g3),dim3(b0,b1,b3)>>>(lat);
  exchange_faces_3<<<dim3(g0,g1,g2),dim3(b0,b1,b2)>>>(lat);
  cudadevicesynchronize();
}

/******************************************************************************/

// host-side logic
// ===============

// perform one monte-carlo iteration
template <float (*delta_s)(float*, const size_t, const float)>
void mc_update(float* lat, float * lat_old, curandstate * states) {

  mc_kernel<delta_s><<<gridsize,blocksize>>>(lat, lat_old, states);
  cudadevicesynchronize();
  exchange_faces(lat);
  std::swap(lat, lat_old);
}


// compute the space-average of the time-slice correlator value over many configurations.
__host__ void mc_average() {

  fprintf(stderr, "lattice: (%d,%d,%d,%d)\n", n0, n1, n2, n3);
  fprintf(stderr, "array:   (%d,%d,%d,%d)\n", m0, m1, m2, m3);
  fprintf(stderr, "m_count = %d\n", m_count);
  
  fprintf(stderr, "allocating lattice arrays...\n");
  // allocate lattice on device (double buffered)
  float * lat     = nullptr;
  float * lat_old = nullptr;
  fprintf(stderr, "requesting 2×%d bytes...", m_bytes);
  assert(cudamalloc(&lat    , m_bytes) == cudasuccess);
  assert(cudamalloc(&lat_old, m_bytes) == cudasuccess);
  fprintf(stderr, " done.\n");
  fprintf(stderr, "memset'ting to 0...");
  assert(cudamemset(lat    , 0., m_count) == cudasuccess);
  assert(cudamemset(lat_old, 0., m_count) == cudasuccess);
  fprintf(stderr, " done.\n");

  // seed rng on each thread
  fprintf(stderr, "allocating rng...\n");
  fprintf(stderr, "requesting %d bytes...", m_count*sizeof(curandstate));
  curandstate * states;
  assert(cudamalloc(&states, m_count*sizeof(curandstate)) == cudasuccess);
  fprintf(stderr, " done.\n");
  fprintf(stderr, "initializing rng...");
  rng_init<<<gridsize,blocksize>>>(clock(), states);
  assert(cudadevicesynchronize() == cudasuccess);
  fprintf(stderr, " done.\n");

  // allocate memory used to store correlation data
  // host-side buffer
  float * corr_buf_h = (float*) calloc(n1*n2*n3, sizeof(float));
  assert(corr_buf_h);
  // device-side buffer
  float * corr_buf_d = nullptr;
  assert(cudamalloc(&corr_buf_d, n1*n2*n3*sizeof(float)) == cudasuccess);
  // array storing the final results
  float * corr = (float*) calloc(n0*n_cf, sizeof(float));
  assert(corr);

  // thermalize lattice
  cudaevent_t start, stop;
  cudaeventcreate(&start);
  cudaeventcreate(&stop);
  fprintf(stderr, "thermalizing lattice...");
  cudaeventrecord(start);
  for (size_t i = 0 ; i < n_th ; ++i) {
    mc_update<ds>(lat, lat_old, states);
  }
  cudaeventrecord(stop);
  cudaeventsynchronize(stop);
  float ms;
  cudaeventelapsedtime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);

  // run metropolis algorithm
  fprintf(stderr, "running mc...");
  cudaeventrecord(start);
  for (size_t i = 0 ; i < n_cf ; ++i) {
    // drop n_cor iterations to damp correlations between successive configurations.
    for (size_t j = 0 ; j < n_cor ; ++j) {
      mc_update<ds>(lat, lat_old, states);
    }
    fprintf(stderr, " %d", i);
    // compute the euclidean time correlations within one configuration.
    // compute_correlations(lat_old, corr, i, corr_buf_h, corr_buf_d);
  }
  cudaeventrecord(stop);
  cudaeventsynchronize(stop);
  cudaeventelapsedtime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);

  // write output to file
  fprintf(stderr, "writing to file...");
  write_correlations(corr);
  fprintf(stderr, " done.\n");
  
  // finalization
  // ============

  fprintf(stderr, "finalization...");
  // free device memory
  cudafree(lat);
  cudafree(lat_old);
  cudafree(states);
  cudafree(corr_buf_d);
  lat        = nullptr;
  lat_old    = nullptr;
  states     = nullptr;
  corr_buf_d = nullptr;

  // free host memory
  free(corr_buf_h);
  free(corr);
  corr_buf_h = nullptr;
  corr       = nullptr;
  fprintf(stderr, " done.\n");
}

void generate_single_conf() {

  fprintf(stderr, "lattice: (%d,%d,%d,%d)\n", n0, n1, n2, n3);
  fprintf(stderr, "array:   (%d,%d,%d,%d)\n", m0, m1, m2, m3);
  fprintf(stderr, "m_count = %d\n", m_count);
  
  fprintf(stderr, "allocating lattice arrays...\n");
  // allocate lattice on device (double buffered)
  float * lat     = nullptr;
  float * lat_old = nullptr;
  fprintf(stderr, "requesting 2×%d bytes...", m_bytes);
  assert(cudamalloc(&lat    , m_bytes) == cudasuccess);
  assert(cudamalloc(&lat_old, m_bytes) == cudasuccess);
  fprintf(stderr, " done.\n");
  fprintf(stderr, "memset'ting to 0...");
  assert(cudamemset(lat    , 0., m_count) == cudasuccess);
  assert(cudamemset(lat_old, 0., m_count) == cudasuccess);
  fprintf(stderr, " done.\n");

  // seed rng on each thread
  fprintf(stderr, "allocating rng...\n");
  fprintf(stderr, "requesting %d bytes...", m_count*sizeof(curandstate));
  curandstate * states;
  assert(cudamalloc(&states, m_count*sizeof(curandstate)) == cudasuccess);
  fprintf(stderr, " done.\n");
  fprintf(stderr, "initializing rng...");
  rng_init<<<gridsize,blocksize>>>(clock(), states);
  assert(cudadevicesynchronize() == cudasuccess);
  fprintf(stderr, " done.\n");

  // thermalize lattice
  cudaevent_t start, stop;
  cudaeventcreate(&start);
  cudaeventcreate(&stop);
  fprintf(stderr, "thermalizing lattice...");
  cudaeventrecord(start);
  for (size_t i = 0 ; i < n_th ; ++i) {
    mc_update<ds>(lat, lat_old, states);
  }
  cudaeventrecord(stop);
  cudaeventsynchronize(stop);
  float ms;
  cudaeventelapsedtime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);

  // write result to file
  write_configuration(lat_old);

  fprintf(stderr, "finalization...");
  // free device memory
  cudafree(lat);
  cudafree(lat_old);
  cudafree(states);
  lat     = nullptr;
  lat_old = nullptr;
  states  = nullptr;
  fprintf(stderr, " done.\n");
}

__host__ int main() {

  //generate_single_conf();
  mc_average();

  return 0;
}
