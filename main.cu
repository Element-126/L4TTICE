// CUDA
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
// Standard library
#include <utility>
#include <cassert>
#include <cstdio>
#include <vector>
#include <numeric>
#include <string>
// HDF5
#include "H5Cpp.h"
// CUB
#include <cub/cub.cuh>

/******************************************************************************/

// Geometry & parameters
// =====================

// Block size
constexpr size_t B0 = 8;
constexpr size_t Bi = 8;
// Number of threads 8³ = 512
// Loop over the last dimension
// Shared memory usage: 44000o including halos.
// Then grid-stride loop to reuse the RNG state

// Grid size
constexpr size_t G0 = 1;
constexpr size_t Gi = 2;
  
// Lattice size
constexpr size_t N0 = B0*G0;
constexpr size_t Ni = Bi*Gi;

// Data array size (including ghost cells)
constexpr size_t M0 = N0+2;
constexpr size_t Mi = Ni+2;
constexpr size_t M_count = M0*Mi*Mi*Mi;
constexpr size_t M_bytes = M_count*sizeof(float);
// Strides
constexpr size_t S1 = M0;
constexpr size_t S2 = M0*Mi;
constexpr size_t S3 = M0*Mi*Mi;

// Physical parameters
constexpr float m2 = -1./8.;
constexpr float lambda = 1./32.;

// Assumptions which must be satisfied for the simulation to work as expected
static_assert(B0 % 2 == 0, "B0 must be even");
static_assert(Bi % 2 == 0, "Bi must be even");

/******************************************************************************/

// Variation of the action
// =======================

// Change in the action when φ(i) → φ(i) + ζ
// Idx: array index, including ghost cells
__device__ float delta_S_kin(float * f, const size_t Idx, const float zeta, const float a) {

  return a*a*zeta*( 4.0f*zeta + 8.0f*f[Idx]
                    - f[Idx+1 ] - f[Idx-1 ] // ± (1,0,0,0)
                    - f[Idx+S1] - f[Idx-S1] // ± (0,1,0,0)
                    - f[Idx+S2] - f[Idx-S2] // ± (0,0,1,0)
                    - f[Idx+S3] - f[Idx-S3] // ± (0,0,0,1)
                    );
}

// Free field: V(φ) = ½m²φ²
__device__ float delta_S_free(float * f, const size_t Idx, const float zeta, const float a) {

  const float fi = f[Idx];
  const float delta_V = 0.5f*m2*zeta*(2.0f*fi+zeta);
  return delta_S_kin(f, Idx, zeta, a) + a*a*a*a*delta_V;
}

// Interacting field: V(φ) = ½m²φ² + ¼λφ⁴
__device__ float delta_S_phi4(float * f, const size_t Idx, const float zeta, const float a) {

  const float fi = f[Idx];     // φi
  const float fiP = fi + zeta; // φi + ζ
  const float delta_V = 0.5f*m2*( fiP*fiP - fi*fi ) + 0.25f*lambda*( fiP*fiP*fiP*fiP - fi*fi*fi*fi );
  return delta_S_kin(f, Idx, zeta, a) + a*a*a*a*delta_V;
}

// Choice of the action used in the simulation
constexpr auto dS = delta_S_phi4;

/******************************************************************************/

// Kernels
// =======

// Main kernels, performing one Monte-Carlo iteration on either black or white indices.

/*  MC iteration over "black" indices
 *
 *  Blocksize should be (B0/2,Bi,Bi) and stride Bi
 *  Gridsize should be (G0,Gi,Gi) and grid stride Gi
 */ 
template<float (*delta_S)(float*, const size_t, const float, const float)>
__global__ void mc_update_black(float * lat, curandState * states,
                                const float a, const float epsilon) {

  // Global thread index
  const size_t t0 = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t t1 = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t t2 = blockIdx.z * blockDim.z + threadIdx.z;

  // Linear thread index
  const size_t tid = t0 + (N0>>1)*t1 + (N0*Ni>>1)*t2;
  
  auto state = states[tid];

  /*  Indices, assuming dimension 0 is even
   *
   *  Physical index: 2·t0 + N0·t1 + N0·N1·t2 + N0·N1·N2·t3 + parity of (t1+t2+t3)
   *  Ex: 4×4 lattice
   *      +–––––––––+
   *      | 0 · 4 · |
   *      | · 2 · 6 |
   *      | 1 · 5 · |
   *      | · 3 · 7 |
   *      +–––––––––+
   *
   *  Array index:    2·t0+1 + M0*(t1+1) + M0·M1·(t2+1) + M0·M1·M2·(t3+1) + parity of (t1+t2+t3)
   *  Ex: 4×4 lattice
   *
   *      |  halos  |
   *      v         v 
   *    +–––––––––––––+
   *    | × · × · × · | <– halo
   *    | · 0 · 4 · × |
   *    | × · 2 · 6 · |
   *    | · 1 · 5 · × |
   *    | × · 3 · 7 · |
   *    | · × · × · × | <– halo
   *    +–––––––––––––+
   */

  // Grid stride loop in direction 3
  for (size_t g3 = 0 ; g3 < Gi ; ++g3) {

    // Small loop in direction 3
    for (size_t b3 = 0 ; b3 < Bi ; ++b3) {

      const size_t t3 = g3*Bi+b3;
      
      // Array index
      const size_t parity = (t1 + t2 + t3) & 1; // 0 if t1+t2+t3 even, 1 otherwise
      const size_t Idx = 2*t0+1 + S1*(t1+1) + S2*(t2+1) + S3*(t3+1) + parity;

      const float zeta = (2.0f*curand_uniform(&state) - 1.0f) * epsilon; // ζ ∈ [-ε,+ε]
      // Compute change in the action due to variation ζ at site Idx
      const float delta_S_i = delta_S(lat, Idx, zeta, a);

      // Update the lattice depending on the variation ΔSi
      const float update = (float) (delta_S_i < 0.0f || (exp(-delta_S_i) > curand_uniform(&state)));
      // Slightly faster than the divergent version
      lat[Idx] += update * zeta;
    }
  }

  // Write RNG state back to global memory
  states[tid] = state;
}

/*  MC iteration over "white" indices
 *
 *  Same grid and block sizes as for black indices.
 */
template<float (*delta_S)(float*, const size_t, const float, const float)>
__global__ void mc_update_white(float * lat, curandState * states,
                                const float a, const float epsilon) {

  // Global thread index
  const size_t t0 = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t t1 = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t t2 = blockIdx.z * blockDim.z + threadIdx.z;

  // Linear thread index
  const size_t tid = t0 + (N0>>1)*t1 + (N0*Ni>>1)*t2;
  
  auto state = states[tid];

  // Grid stride loop in direction 3
  for (size_t g3 = 0 ; g3 < Gi ; ++g3) {

    // Small loop in direction 3
    for (size_t b3 = 0 ; b3 < Bi ; ++b3) {

      const size_t t3 = g3*Bi+b3;
      const size_t parity = (t1 + t2 + t3) & 1; // 0 if t1+t2+t3 even, 1 otherwise
      // Main difference with "black" indices: opposite parity
      const size_t Idx = 2*t0+1 + S1*(t1+1) + S2*(t2+1) + S3*(t3+1) + !parity;

      const float zeta = (2.0f*curand_uniform(&state) - 1.0f) * epsilon; // ζ ∈ [-ε,+ε]
      // Compute change in the action due to variation ζ at site Idx
      const float delta_S_i = delta_S(lat, Idx, zeta, a);

      // Update the lattice depending on the variation ΔSi
      const float update = (float) (delta_S_i < 0.0f || (exp(-delta_S_i) > curand_uniform(&state)));
      // Is the above really branchless ?
      lat[Idx] += update * zeta;
    }
  }

  // Write RNG state back to global memory
  states[tid] = state;
}

// Initialization of device side resources
// =======================================

/*
 * Initialize RNG state
 *
 * Grid size: (G0,Gi,Gi)
 * Block size: (N0/2,Ni,Ni)
 */
__global__ void rng_init(unsigned long long seed, curandState * states) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t I2 = blockIdx.z * blockDim.z + threadIdx.z;
  const size_t Idx = I0 + (N0>>1)*I1 + (N0*Ni>>1)*I2;
  curand_init(seed, Idx, 0, &states[Idx]);
}

/*
 * Sets the whole lattice (except halos) to a constant value
 *
 * Must be called with grid size (G0,Gi,Gi) and block size (B0,Bi,Bi)
 */
__global__ void fill(float * lat, const float val) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I2 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  for (size_t I3 = 1 ; I3 <= Ni ; ++I3) {
    const size_t Idx = I0 + I1*S1 + I2*S2 + I3*S3;
    lat[Idx] = val;
  }
}

/******************************************************************************/

// Halo related kernels
// ====================

/*
 * Set the 3d halos to zero so they do not affect the reduction
 *
 * Can be undone by calling exchange_faces
 */
__global__ void erase_halos_0(float * lat) {

  const size_t I1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = S1*I1 + S2*I2 + S3*I3;

  lat[Idx         ] = 0.0f;
  lat[Idx + (N0+1)] = 0.0f;
}

__global__ void erase_halos_1(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S2*I2 + S3*I3;

  lat[Idx            ] = 0.0f;
  lat[Idx + S1*(Ni+1)] = 0.0f;
}

__global__ void erase_halos_2(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S1*I1 + S3*I3;

  lat[Idx            ] = 0.0f;
  lat[Idx + S2*(Ni+1)] = 0.0f;
}

__global__ void erase_halos_3(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I2 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S1*I1 + S2*I2;

  lat[Idx            ] = 0.0f;
  lat[Idx + S3*(Ni+1)] = 0.0f;
}

__host__ void erase_halos(float * lat) {

  erase_halos_0<<<dim3(Gi,Gi,Gi),dim3(Bi,Bi,Bi)>>>(lat);
  erase_halos_1<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
  erase_halos_2<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
  erase_halos_3<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
}

// Exchange of the 3d "faces" of the 4d lattice
// ============================================

// Face 0 (stride = 1)
__global__ void update_halos_0(float * lat) {

  const size_t I1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = S1*I1 + S2*I2 + S3*I3;

  lat[Idx         ] = lat[Idx + N0];
  lat[Idx + (N0+1)] = lat[Idx +  1];
}

// Face 1 (stride S1 = M0)
__global__ void update_halos_1(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S2*I2 + S3*I3;

  lat[Idx            ] = lat[Idx + S1*Ni];
  lat[Idx + S1*(Ni+1)] = lat[Idx + S1   ];
}

// Face 2 (stride S2 = M0·M1)
__global__ void update_halos_2(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S1*I1 + S3*I3;

  lat[Idx            ] = lat[Idx + S2*Ni];
  lat[Idx + S2*(Ni+1)] = lat[Idx + S2   ];
}

// Face 3 (stride S3 = M0·M1·M2)
__global__ void update_halos_3(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I2 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S1*I1 + S2*I2;

  lat[Idx            ] = lat[Idx + S3*Ni];
  lat[Idx + S3*(Ni+1)] = lat[Idx + S3   ];
}

// Exchange all faces
__host__ void update_halos(float * lat) {

  update_halos_0<<<dim3(Gi,Gi,Gi),dim3(Bi,Bi,Bi)>>>(lat);
  update_halos_1<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
  update_halos_2<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
  update_halos_3<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
}

/******************************************************************************/

// Host-side logic
// ===============

// Perform one Monte-Carlo iteration
template <float (*delta_S)(float*, const size_t, const float, const float a)>
void mc_update(float* lat, curandState * states, const float a, const float epsilon) {

  mc_update_black<delta_S><<<dim3(G0,Gi,Gi),dim3(B0/2,Bi,Bi)>>>(lat, states, a, epsilon);
  update_halos(lat);
  mc_update_white<delta_S><<<dim3(G0,Gi,Gi),dim3(B0/2,Bi,Bi)>>>(lat, states, a, epsilon);
  update_halos(lat);
}

// Resource management
// -------------------

__host__ float* new_lattice(const bool verbose = false) {

  float * lat = nullptr;

  if (verbose) {
    fprintf(stderr, "Lattice: (%d,%d,%d,%d)\n", N0, Ni, Ni, Ni);
    fprintf(stderr, "Array:   (%d,%d,%d,%d)\n", M0, Mi, Mi, Mi);
    fprintf(stderr, "M_count = %d\n", M_count);
  
    fprintf(stderr, "Allocating lattice array...\n");
    fprintf(stderr, "Requesting %d bytes...", M_bytes);
  }
  // Allocate lattice on device
  assert(cudaMalloc(&lat, M_bytes) == cudaSuccess);
  if (verbose) {
    fprintf(stderr, " done.\n");
    fprintf(stderr, "Memset'ting to 0...");
  }
  fill<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat, 0.0f);
  if (verbose) {
    fprintf(stderr, " done.\n");
  }

  return lat;
}

__host__ void delete_lattice(float ** lat) {

  assert(cudaFree(*lat) == cudaSuccess);
  *lat = nullptr;
}

__host__ curandState* new_rng(const bool verbose = false, unsigned long long seed = 0) {

  curandState * states;

  // Seed RNG on each thread
  if (verbose) {
    fprintf(stderr, "Allocating RNG...\n");
    fprintf(stderr, "Requesting %d bytes...", N0/2*Ni*Ni*sizeof(curandState));
  }
  assert(cudaMalloc(&states, N0/2*Ni*Ni*sizeof(curandState)) == cudaSuccess);
  if (verbose) {
    fprintf(stderr, " done.\n");
    fprintf(stderr, "Initializing RNG...");
  }
  if (seed == 0) {
    seed = clock();
  }
  rng_init<<<dim3(G0,Gi,Gi),dim3(B0/2,Bi,Bi)>>>(seed, states);
  if (verbose) {
    fprintf(stderr, " done.\n");
  }

  return states;
}

__host__ void delete_rng(curandState ** states) {

  assert(cudaFree(*states) == cudaSuccess);
  *states = nullptr;
}

/*
 * Thermalize a configuration
 *
 * a    = lattice spacing
 * N_th = number of MC iterations to run
 * ε    = uniform distribution parameter: [-ε,+ε]
 */
float thermalize_conf(float * lat, curandState * states,
                      const float a, const size_t N_th, const float epsilon,
                      const bool verbose = false, const size_t poll = 1) {

  // Thermalize lattice
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  if (verbose) {
    fprintf(stderr, "Thermalizing lattice...");
  }
  cudaEventRecord(start);
  for (size_t i = 1 ; i <= N_th ; ++i) {
    mc_update<dS>(lat, states, a, epsilon);
    if (verbose && i % poll == 0) {
      fprintf(stderr, " %llu", i);
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  if (verbose) {
    fprintf(stderr, " done in %fs.\n", 1e-3*ms);
  }
  return ms*1e-3;
}

/*
 * Generate N_cf configurations, thermalize them and compute their means.
 *
 * Mandatory parameters:
 *   N_cf = number of configurations to generate
 *   N_th = number of MC iterations to thermalize the lattice
 *   a    = lattice spacing
 *
 * Optional parameters:
 *   verbose  = verbosity level (0: nothing | 1: debug | 2: display progress | 3: print result)
 *   poll_th  = number of MC updates between status updates
 *   filename = name of the file to write the output to
 *       If empty, nothing is written
 */
void mc_mean(const size_t N_cf, const size_t N_th, const float a, const float epsilon,
             float * out, float * lat, curandState * states,
             float * sum_d, float * sum_h, void * cub_tmp_storage,
             size_t cub_tmp_bytes, const int verbose = 0, const size_t poll_th = 500) {

  for (size_t k = 0 ; k < N_cf ; ++k) {

    // Thermalize the configuration
    if (verbose >= 2) {
      fprintf(stderr, "%.2llu: ", k+1);
    }
    float time = thermalize_conf(lat, states, a, N_th, epsilon, (verbose >= 2), poll_th);
    // Erase the halos in order not to interfere with the summation
    erase_halos(lat);
    // Actually run the summation
    *sum_h = 0.0f;
    assert(cudaMemset(sum_d, 0.0f, 1) == cudaSuccess);
    CubDebugExit(cub::DeviceReduce::Sum(cub_tmp_storage, cub_tmp_bytes, lat, sum_d, M_count));
    // Retreive the result
    assert(cudaMemcpy(sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    float mean = *sum_h / (N0*Ni*Ni*Ni);
    // Reset the lattice to zero for the next run
    fill<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat, 0.0f);
    if (verbose >= 3) {
      // Print the result
      fprintf(stderr, "Mean = %f\n", mean);
    }
    if (out) {
      out[3*k    ] = N0*a;
      out[3*k + 1] = mean;
      out[3*k + 2] = time;
    }
  }
}

/*
 * For each inverse temperature β, generate N_cf configurations and compute their means.
 *
 * All vectors must have the same size.
 */
void mc_mean_temp(const std::vector<float> beta, const std::vector<size_t> N_cf,
                  const std::vector<size_t> N_th, const std::vector<float> epsilon,
                  const int verbose = 0, const size_t poll_th = 1000,
                  const std::string filename = "",
                  const unsigned long long seed = 0) {

  assert(beta.size() == N_cf.size());
  assert(beta.size() == N_th.size());
  assert(beta.size() == epsilon.size());

  /*
   * Allocate resources for the simulation
   */
  auto lat = new_lattice(verbose > 0);
  auto states = new_rng(verbose > 0, seed);

  //Prepare resources for CUB
  float * sum_d = nullptr;
  assert(cudaMalloc(&sum_d, sizeof(float)) == cudaSuccess);
  void * cub_tmp_storage = nullptr;
  size_t cub_tmp_bytes = 0;
  cub::CachingDeviceAllocator g_allocator(true);
  float * sum_h = (float*) malloc(sizeof(float));
  assert(sum_h);
  // Call once to initialize resources
  CubDebugExit(cub::DeviceReduce::Sum(cub_tmp_storage, cub_tmp_bytes, lat, sum_d, M_count));
  CubDebugExit(g_allocator.DeviceAllocate(&cub_tmp_storage, cub_tmp_bytes));
  
  /*
   * Allocate output array
   *
   * Columns: β | mean | time
   */
  const size_t N_cf_tot = std::accumulate(N_cf.begin(), N_cf.end(), 0);
  auto out = (float *) calloc(N_cf_tot*3, sizeof(float));
  auto cursor = out;

  // Do the computations for each temperature
  (verbose > 0) && fprintf(stderr, "Running simulation...\n");
  for (size_t i = 0 ; i < beta.size() ; ++i) {

    (verbose > 0) && fprintf(stderr, ">>> β = %f\n", beta[i]);
    mc_mean(N_cf[i], N_th[i], beta[i]/N0, epsilon[i],
            cursor, lat, states, sum_d, sum_h, cub_tmp_storage,
            cub_tmp_bytes, verbose, poll_th);
    cursor += N_cf[i]*3;
  }
  (verbose > 0) && fprintf(stderr, "...done\n");

  if(!filename.empty()) {
    // Write results to file
    (verbose > 0) && fprintf(stderr, "Writing to file...");
    H5::H5File file(filename, H5F_ACC_TRUNC);
    hsize_t dims[2] = {N_cf_tot, 3};
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset = file.createDataSet("means", H5::PredType::NATIVE_FLOAT, dataspace);
    dataset.write(out, H5::PredType::NATIVE_FLOAT);
    (verbose > 0) && fprintf(stderr, " done\n");
  }

  // Free resources
  (verbose > 0) && fprintf(stderr, "Freeing resources...");
  delete_lattice(&lat);
  delete_rng(&states);
  cudaFree(sum_d);
  free(sum_h);
  CubDebugExit(g_allocator.DeviceFree(cub_tmp_storage));
  free(out);
  (verbose > 0) && fprintf(stderr, " done\n");
}

// Presets
// =======

// Values used to obtain figure 2
__host__ void full_run() {

  mc_mean_temp(std::vector<float>({8., 6., 5., 4., 3., 2.5, 2., 1.5, 1.25, 1.}),
               std::vector<size_t>({50, 50, 50, 50, 50, 50, 36, 20, 16, 12}),
               std::vector<size_t>({2000, 6000, 10000, 17000, 30000, 38000, 50000, 85000, 110000, 150000}),
               std::vector<float>({1., 3., 6., 10., 18., 24., 32., 80., 125., 200.}),
               2, 10000,
               "means.h5");
}

// The benchmark used to study scaling (where Bi = 2 B0 is varied)
__host__ void benchmark() {

  mc_mean_temp(std::vector<float>({8.}),
               std::vector<size_t>({12}),
               std::vector<size_t>({2000}),
               std::vector<float>({1.}),
               0, 0,
               "out.h5");
}

// Test the overall behaviour at high temperature to fine-tune the parameters
// Can also be used as a reproducible test case to compare with the shared memory implementation
__host__ void test(const unsigned long long seed, const std::string name = "test.h5") {

  mc_mean_temp(std::vector<float>({8., 4., 2., 1.}),
               std::vector<size_t>({32, 16, 8, 4}),
               std::vector<size_t>({2000, 10000, 50000, 250000}),
               std::vector<float>({1., 4., 16., 64.}),
               2, 10000,
               name,
               seed);
}

__host__ void short_test(const unsigned long long seed, const std::string name = "test.h5") {

  mc_mean_temp(std::vector<float>({8., 4., 2., 1.}),
               std::vector<size_t>({12, 8, 4, 2}),
               std::vector<size_t>({2000, 17000, 50000, 150000}),
               std::vector<float>({1., 4., 16., 64.}),
               2, 10000,
               name,
               seed);
}

// Exactly one iteration
__host__ void debug(const unsigned long long seed) {

  mc_mean_temp(std::vector<float>({8.}),
               std::vector<size_t>({1}),
               std::vector<size_t>({1}),
               std::vector<float>({1.}),
               3, 1,
               "",
               seed);
}

/*
 * Entry point of the program
 *
 * Call the desired function from here
 */
__host__ int main() {

  // full_run();
  // benchmark();
  short_test(42);

  return 0;
}
