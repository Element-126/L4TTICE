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

/*
 * In general, caps correspond to the full lattice while small letters are
 * related to the sub-lattice.
 */

// Sub-lattice size (without halos)
constexpr size_t n0 = 8;
constexpr size_t ni = 8;

// Block size (with halos)
constexpr size_t m0 = n0+2;
constexpr size_t mi = ni+2;
constexpr size_t m_count = m0*mi*mi*mi;
constexpr size_t m_bytes = m_count*sizeof(float);
// Number of threads per block: 10³ = 1000
// Loop over the last dimension
// Shared memory usage for the sub-lattice: 40000o including halos.
// Then grid-stride loop to reuse the RNG state

// Grid size
constexpr size_t G0 = 1;
constexpr size_t Gi = 2;
  
// Lattice size
constexpr size_t N0 = G0*n0;
constexpr size_t Ni = Gi*ni;

// Data array size (including ghost cells)
constexpr size_t M0 = N0+2;
constexpr size_t Mi = Ni+2;
constexpr size_t M_count = M0*Mi*Mi*Mi;
constexpr size_t M_bytes = M_count*sizeof(float);
// Strides
constexpr size_t S1 = M0;
constexpr size_t S2 = M0*Mi;
constexpr size_t S3 = M0*Mi*Mi;
constexpr size_t s1 = m0;
constexpr size_t s2 = m0*mi;
constexpr size_t s3 = m0*mi*mi;

// Physical parameters
constexpr float m2 = -1./8.;
constexpr float lambda = 1./32.;

// Assumptions which must be satisfied for the simulation to work as expected
static_assert(n0 % 2 == 0, "n0 must be even");
static_assert(ni % 2 == 0, "ni must be even");

// Initial lattice value (to debug convergence issues)
constexpr float INIT_VAL = 0.0f;

/******************************************************************************/

// Variation of the action
// =======================

/*  Change in the action when φ(i) → φ(i) + ζ
 *
 *  idx: local array index, including ghost cells
 *  f:   reference to the sub-lattice
 */
__device__ float delta_S_kin(float (&f)[m_count], const size_t idx, const float zeta, const float a) {

  return a*a*zeta*( 4.0f*zeta + 8.0f*f[idx]
                    - f[idx+1 ] - f[idx-1 ] // ± (1,0,0,0)
                    - f[idx+s1] - f[idx-s1] // ± (0,1,0,0)
                    - f[idx+s2] - f[idx-s2] // ± (0,0,1,0)
                    - f[idx+s3] - f[idx-s3] // ± (0,0,0,1)
                    );
}

// Free field: V(φ) = ½m²φ²
__device__ float delta_S_free(float (&f)[m_count], const size_t idx, const float zeta, const float a) {

  const float fi = f[idx];
  const float delta_V = 0.5f*m2*zeta*(2.0f*fi+zeta);
  return delta_S_kin(f, idx, zeta, a) + a*a*a*a*delta_V;
}

// Interacting field: V(φ) = ½m²φ² + ¼λφ⁴
__device__ float delta_S_phi4(float (&f)[m_count], const size_t idx, const float zeta, const float a) {

  const float fi = f[idx];     // φi
  const float fiP = fi + zeta; // φi + ζ
  const float delta_V = 0.5f*m2*( fiP*fiP - fi*fi ) + 0.25f*lambda*( fiP*fiP*fiP*fiP - fi*fi*fi*fi );
  return delta_S_kin(f, idx, zeta, a) + a*a*a*a*delta_V;
}

// Choice of the action used in the simulation
constexpr auto dS = delta_S_phi4;

/******************************************************************************/

// Kernels
// =======

// Main kernels, performing one Monte-Carlo iteration on either black or white indices.

/*  MC iteration over "black" indices
 *
 *  Blocksize should be (m0,mi,mi) and stride mi
 *  Gridsize should be (G0,Gi,Gi) and grid stride Gi
 *
 *  parity: 0 for black sites, 1 for white ones
 */ 
template<float (*delta_S)(float(&)[m_count], const size_t, const float, const float)>
__global__ void mc_update(float * lat, curandState * states, const int parity,
                          const float a, const float epsilon) {

  // 4D sub-lattice
  __shared__ float sub[m_count];

  // Global thread index
  const size_t T0 = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t T1 = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t T2 = blockIdx.z * blockDim.z + threadIdx.z;

  // Linear thread index
  const size_t tid = T0 + S1*T1 + S2*T2;

  // Whether the thread corresponds to a halo or not.
  const bool halo = (threadIdx.x == 0 || threadIdx.x == m0-1 ||
                     threadIdx.y == 0 || threadIdx.y == mi-1 ||
                     threadIdx.z == 0 || threadIdx.z == mi-1);

  // Load RNG state
  curandState state;
  if (!halo) {
    state = states[tid];
  }

  // First 3 indices of the global multidimensional array
  const size_t I0 = blockIdx.x * n0 + threadIdx.x;
  const size_t I1 = blockIdx.y * ni + threadIdx.y;
  const size_t I2 = blockIdx.z * ni + threadIdx.z;

  // Global array index (starting value)
  const size_t Idx0 = I0 + S1*I1 + S2*I2;
  // Local array index (starting value)
  const size_t idx0 = threadIdx.x + s1*threadIdx.y + s2*threadIdx.z;

  // if (!halo) {
  //   printf("[%lu/%lu] State <- %lu\n", idx0, Idx0, tid);
  // }

  // Grid stride loop in direction 3
  for (size_t b3 = 0 ; b3 < Gi ; ++b3) {

    // if (tid == 0) {
    //   printf(">> b3 = %lu\n", b3);
    // }

    // Load values (including halos) in shared array
    for (size_t t3 = 0 ; t3 < mi ; ++t3) {

      const size_t I3 = b3 * ni + t3;
      // Global array index
      const size_t Idx = Idx0 + S3*I3;
      // Local array index
      const size_t idx = idx0 + s3*t3;
      
      sub[idx] = lat[Idx];

      // printf("%lu@(%u,%u,%u,%lu)[%.3f] <- %lu@(%lu,%lu,%lu,%lu)[%.3f]\n",
      //        idx, threadIdx.x, threadIdx.y, threadIdx.z, t3, sub[idx],
      //        Idx, I0, I1, I2, I3, lat[Idx]);
    }

    __syncthreads();

    // Small loop in direction 3 (ignoring all halos)
    if (!halo) {
      for (size_t t3 = 1 ; t3 < mi-1 ; ++t3) {

        // Parity of t0+t1+t2+t3 (4D parity = color on the checkerboard)
        const size_t par = (threadIdx.x + threadIdx.y + threadIdx.z + t3) & 1; 

        // Update only sites matching the parity passed to the kernel
        if (par == parity) {

          // Local array index
          const size_t idx = idx0 + s3*t3;
 
          const float zeta = (2.0f*curand_uniform(&state) - 1.0f) * epsilon; // ζ ∈ [-ε,+ε]
          // Compute change in the action due to variation ζ at site Idx
          const float delta_S_i = delta_S(sub, idx, zeta, a);
 
          // Update the lattice depending on the variation ΔSi
          const float update = (float) (delta_S_i < 0.0f || (exp(-delta_S_i) > curand_uniform(&state)));

          // // DEBUG
          // const size_t I3 = b3 * ni + t3;
          // const size_t Idx = Idx0 + S3*I3;
          // printf("[%d:%lu:%lu] %.3f -> (ζ = %.3f) -> (ΔS = %.3f) -> %.3f\n",
          //        parity, idx, Idx, sub[idx], zeta, delta_S_i, sub[idx]+update*zeta);

          sub[idx] += update * zeta;
        }
      }
    }

    // End divergence before writing back to global memory
    __syncthreads();
 
    // Store values (except halos) in global array
    if (!halo) {
      for (size_t t3 = 1 ; t3 < mi-1 ; ++t3) {

        const size_t I3 = b3 * ni + t3;
        // Global array index
        const size_t Idx = Idx0 + S3*I3;
        // Local array index
        const size_t idx = idx0 + s3*t3;

        lat[Idx] = sub[idx];

        // printf("%lu@(%u,%u,%u,%lu)[%.3f] -> %lu@(%lu,%lu,%lu,%lu)[%.3f]\n",
        //        idx, threadIdx.x, threadIdx.y, threadIdx.z, t3, sub[idx],
        //        Idx, I0, I1, I2, I3, lat[Idx]);
      }
    }
  }

  // Write RNG state back to global memory
  if (!halo) {
    states[tid] = state;
  }
}

/******************************************************************************/

// Initialization of device side resources
// =======================================

/*
 * Initialize RNG state
 *
 * Grid size: (G0,Gi,Gi)
 * Block size: (m0,mi,mi)
 */
__global__ void rng_init(unsigned long long seed, curandState * states) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t I2 = blockIdx.z * blockDim.z + threadIdx.z;
  const size_t Idx = I0 + S1*I1 + S2*I2;
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
 * Set the 3D halos to zero so they do not affect the reduction
 * The lower-dimensional halos are assumed to be zero at all times
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

  erase_halos_0<<<dim3(Gi,Gi,Gi),dim3(ni,ni,ni)>>>(lat);
  erase_halos_1<<<dim3(G0,Gi,Gi),dim3(n0,ni,ni)>>>(lat);
  erase_halos_2<<<dim3(G0,Gi,Gi),dim3(n0,ni,ni)>>>(lat);
  erase_halos_3<<<dim3(G0,Gi,Gi),dim3(n0,ni,ni)>>>(lat);
}

/*
 * Update the 3D "faces" of the 4d lattice
 *
 * Lower dimensional halos are unused and don't need updating
 */

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

  update_halos_0<<<dim3(Gi,Gi,Gi),dim3(ni,ni,ni)>>>(lat);
  update_halos_1<<<dim3(G0,Gi,Gi),dim3(n0,ni,ni)>>>(lat);
  update_halos_2<<<dim3(G0,Gi,Gi),dim3(n0,ni,ni)>>>(lat);
  update_halos_3<<<dim3(G0,Gi,Gi),dim3(n0,ni,ni)>>>(lat);
}

/******************************************************************************/

// Host-side logic
// ===============

// Perform one Monte-Carlo iteration
template <float (*delta_S)(float(&)[m_count], const size_t, const float, const float a)>
void mc_update(float* lat, curandState * states, const float a, const float epsilon) {

  mc_update<delta_S><<<dim3(G0,Gi,Gi),dim3(m0,mi,mi)>>>(lat, states, 0, a, epsilon);
  update_halos(lat);
  mc_update<delta_S><<<dim3(G0,Gi,Gi),dim3(m0,mi,mi)>>>(lat, states, 1, a, epsilon);
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
    fprintf(stderr, "Shared memory per block: %d bytes\n", m_bytes);
  
    fprintf(stderr, "Allocating lattice array...\n");
    fprintf(stderr, "Requesting %d bytes...", M_bytes);
  }
  assert(cudaMalloc(&lat, M_bytes) == cudaSuccess);
  if (verbose) {
    fprintf(stderr, " done.\n");
    fprintf(stderr, "Memset'ting to 0...");
  }
  assert(cudaMemset(lat, 0, M_bytes) == cudaSuccess);
  if (INIT_VAL != 0.0f) {
    fill<<<dim3(G0,Gi,Gi),dim3(n0,ni,ni)>>>(lat, INIT_VAL);
    update_halos(lat);
  }
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
  const size_t bytes = M0*Mi*Mi*sizeof(curandState);

  if (verbose) {
    fprintf(stderr, "Allocating RNG...\n");
    fprintf(stderr, "Requesting %d bytes...", bytes);
  }
  assert(cudaMalloc(&states, bytes) == cudaSuccess);
  if (verbose) {
    fprintf(stderr, " done.\n");
    fprintf(stderr, "Initializing RNG...");
  }
  if (seed == 0) {
    seed = clock();
  }
  // Seed RNG on each thread
  rng_init<<<dim3(G0,Gi,Gi),dim3(m0,mi,mi)>>>(seed, states);
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
    fill<<<dim3(G0,Gi,Gi),dim3(n0,ni,ni)>>>(lat, INIT_VAL);
    update_halos(lat);

    if (verbose >= 3) {
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

__host__ void short_run() {

  mc_mean_temp(std::vector<float>({8., 6., 5., 4., 3., 2.5, 2., 1.5, 1.25, 1.}),
               std::vector<size_t>({16, 12, 10, 8, 6, 6, 4, 4, 2, 2}),
               std::vector<size_t>({2000, 6000, 10000, 17000, 30000, 38000, 50000, 85000, 110000, 150000}),
               std::vector<float>({1., 3., 6., 10., 18., 24., 32., 80., 125., 200.}),
               2, 10000,
               "short_run.h5");
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

__host__ void debug_2(const unsigned long long seed) {

  mc_mean_temp(std::vector<float>({8.}),
               std::vector<size_t>({1}),
               std::vector<size_t>({1}),
               std::vector<float>({0.1}),
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
  short_test(42, "test_shared_mem.h5");
  // short_run();
  // debug(2);

  return 0;
}
