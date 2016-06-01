#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <utility>
#include <cassert>
#include <cstdio>
// #include "H5Cpp.h"

/******************************************************************************/

// Geometry & parameters
// =====================

// Block size
constexpr size_t B0 = 4;
constexpr size_t Bi = 4;
// Number of threads 8³ = 512
// Loop over the last dimension
// Shared memory usage: 44000o including halos.
// Then grid-stride loop to reuse the RNG state

// Grid size
constexpr size_t G0 = 1;
constexpr size_t Gi = 1;
  
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

// Lattice spacing
constexpr float a = 0.5f;

// Physical parameters
constexpr float m2 = 1.0f;
constexpr float lambda = 1.0f;

// Monte-Carlo parameters
constexpr unsigned int N_cor = 1;
constexpr unsigned int N_cf  = 100;
constexpr unsigned int N_th  = 1;
constexpr float epsilon = 0.7f;

// Output
// const H5std_string file_name("correlations.h5");
// const H5std_string dataset_name("corr");

/******************************************************************************/

// Variation of the action
// =======================

// Change in the action when φ(i) → φ(i) + ζ
// Idx: array index, including ghost cells
__device__ float delta_S_kin(float * f, const size_t Idx, const float zeta) {

  return a*a*zeta*( 4.0f*zeta + 8.0f*f[Idx]
                    - f[Idx+1 ] - f[Idx-1 ] // ± (1,0,0,0)
                    - f[Idx+S1] - f[Idx-S1] // ± (0,1,0,0)
                    - f[Idx+S2] - f[Idx-S2] // ± (0,0,1,0)
                    - f[Idx+S3] - f[Idx-S3] // ± (0,0,0,1)
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

// Main kernels, performing one Monte-Carlo iteration on either black or white indices.

/*  MC iteration over "black" indices
 *
 *  Blocksize should be (B0/2,Bi,Bi) and stride Bi
 *  Gridsize should be (G0,Gi,Gi) and grid stride Gi
 */ 
template<float (*delta_S)(float*, const size_t, const float)>
__global__ void mc_update_black(float * lat, curandState * states) {

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
      
      // Array index (TODO: move this outside of the loop)
      const size_t parity = (t1 + t2 + t3) & 1; // 0 if t1+t2+t3 even, 1 otherwise
      const size_t Idx = 2*t0+1 + S1*(t1+1) + S2*(t2+1) + S3*(t3+1) + parity;

      const float zeta = (2.0f*curand_uniform(&state) - 1.0f) * epsilon; // ζ ∈ [-ε,+ε]
      // Compute change in the action due to variation ζ at site Idx
      const float delta_S_i = delta_S(lat, Idx, zeta);

      // Update the lattice depending on the variation ΔSi
      const float update = (float) (delta_S_i < 0.0f || (exp(-delta_S_i) > curand_uniform(&state)));
      // Is the above really branchless ?
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
template<float (*delta_S)(float*, const size_t, const float)>
__global__ void mc_update_white(float * lat, curandState * states) {

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
      const float delta_S_i = delta_S(lat, Idx, zeta);

      // Update the lattice depending on the variation ΔSi
      const float update = (float) (delta_S_i < 0.0f || (exp(-delta_S_i) > curand_uniform(&state)));
      // Is the above really branchless ?
      lat[Idx] += update * zeta;
    }
  }

  // Write RNG state back to global memory
  states[tid] = state;
}

// Initialize rng state
__global__ void rng_init(unsigned long long seed, curandState * states) {

  const size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, Idx, 0, &states[Idx]);
}

/******************************************************************************/

// Exchange of the 3d "faces" of the 4d lattice
// ============================================

// Face 0 (stride = 1)
__global__ void exchange_faces_0(float * lat) {

  const size_t I1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = S1*I1 + S2*I2 + S3*I3;

  lat[Idx         ] = lat[Idx + N0];
  lat[Idx + (N0+1)] = lat[Idx +  1];
}

// Face 1 (stride S1 = M0)
__global__ void exchange_faces_1(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S2*I2 + S3*I3;

  lat[Idx            ] = lat[Idx + S1*Ni];
  lat[Idx + S1*(Ni+1)] = lat[Idx + S1   ];
}

// Face 2 (stride S2 = M0·M1)
__global__ void exchange_faces_2(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S1*I1 + S3*I3;

  lat[Idx            ] = lat[Idx + S2*Ni];
  lat[Idx + S2*(Ni+1)] = lat[Idx + S2   ];
}

// Face 3 (stride S3 = M0·M1·M2)
__global__ void exchange_faces_3(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I2 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + S1*I1 + S2*I2;

  lat[Idx            ] = lat[Idx + S3*Ni];
  lat[Idx + S3*(Ni+1)] = lat[Idx + S3   ];
}

// Exchange all faces
__host__ void exchange_faces(float * lat) {

  exchange_faces_0<<<dim3(Gi,Gi,Gi),dim3(Bi,Bi,Bi)>>>(lat);
  exchange_faces_1<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
  exchange_faces_2<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
  exchange_faces_3<<<dim3(G0,Gi,Gi),dim3(B0,Bi,Bi)>>>(lat);
  cudaDeviceSynchronize();
}

/******************************************************************************/

// Host-side logic
// ===============

// Perform one Monte-Carlo iteration
template <float (*delta_S)(float*, const size_t, const float)>
void mc_update(float* lat, curandState * states) {

  mc_update_black<delta_S><<<dim3(G0,Gi,Gi),dim3(B0/2,Bi,Bi)>>>(lat, states);
  cudaDeviceSynchronize();
  mc_update_white<delta_S><<<dim3(G0,Gi,Gi),dim3(B0/2,Bi,Bi)>>>(lat, states);
  cudaDeviceSynchronize();
  exchange_faces(lat);
}

__host__ void init_lattice(float ** lat, curandState ** states) {

  fprintf(stderr, "Lattice: (%d,%d,%d,%d)\n", N0, Ni, Ni, Ni);
  fprintf(stderr, "Array:   (%d,%d,%d,%d)\n", M0, Mi, Mi, Mi);
  fprintf(stderr, "M_count = %d\n", M_count);
  
  fprintf(stderr, "Allocating lattice array...\n");
  // Allocate lattice on device
  fprintf(stderr, "Requesting %d bytes...", M_bytes);
  assert(cudaMalloc(lat, M_bytes) == cudaSuccess);
  fprintf(stderr, " done.\n");
  fprintf(stderr, "Memset'ting to 0...");
  assert(cudaMemset(*lat, 0., M_count) == cudaSuccess);
  fprintf(stderr, " done.\n");

  // Seed RNG on each thread
  fprintf(stderr, "Allocating RNG...\n");
  fprintf(stderr, "Requesting %d bytes...", N0/2*Ni*Ni*sizeof(curandState));
  assert(cudaMalloc(states, N0/2*Ni*Ni*sizeof(curandState)) == cudaSuccess);
  fprintf(stderr, " done.\n");
  fprintf(stderr, "Initializing RNG...");
  rng_init<<<dim3(G0,Gi,Gi),dim3(B0/2,Bi,Bi)>>>(clock(), *states);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  fprintf(stderr, " done.\n");
}

__host__ void free_lattice(float ** lat, curandState ** states) {

  cudaFree(*lat);
  cudaFree(*states);
  *lat    = nullptr;
  *states = nullptr;
}

// Compute the space-average of the time-slice correlator value over many configurations.
__host__ void mc_average() {

  float * lat = nullptr;
  curandState * states = nullptr;
  init_lattice(&lat, &states);

  // Allocate memory used to store correlation data
  // Host-side buffer
  float * corr_buf_h = (float*) calloc(Ni*Ni*Ni, sizeof(float));
  assert(corr_buf_h);
  // Device-side buffer
  float * corr_buf_d = nullptr;
  assert(cudaMalloc(&corr_buf_d, Ni*Ni*Ni*sizeof(float)) == cudaSuccess);
  // Array storing the final results
  float * corr = (float*) calloc(N0*N_cf, sizeof(float));
  assert(corr);

  // Thermalize lattice
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  fprintf(stderr, "Thermalizing lattice...");
  cudaEventRecord(start);
  for (size_t i = 0 ; i < N_th ; ++i) {
    mc_update<dS>(lat, states);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);

  // Run Metropolis algorithm
  fprintf(stderr, "Running mc...");
  cudaEventRecord(start);
  for (size_t i = 0 ; i < N_cf ; ++i) {
    // Drop N_cor iterations to damp correlations between successive configurations.
    for (size_t j = 0 ; j < N_cor ; ++j) {
      mc_update<dS>(lat, states);
    }
    fprintf(stderr, " %d", i);
    // Compute the euclidean time correlations within one configuration.
    // Compute_correlations(lat_old, corr, i, corr_buf_h, corr_buf_d);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);

  // Write output to file
  // fprintf(stderr, "Writing to file...");
  // write_correlations(corr);
  // fprintf(stderr, " done.\n");
  
  // Finalization
  // ============

  fprintf(stderr, "Finalization...");

  // Free device memory
  free_lattice(&lat, &states);
  cudaFree(corr_buf_d);
  corr_buf_d = nullptr;

  // Free host memory
  free(corr_buf_h);
  free(corr);
  corr_buf_h = nullptr;
  corr       = nullptr;
  fprintf(stderr, " done.\n");
}

void generate_single_conf() {

  float * lat = nullptr;
  curandState * states = nullptr;
  init_lattice(&lat, &states);

  // Thermalize lattice
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  fprintf(stderr, "Thermalizing lattice...");
  cudaEventRecord(start);
  for (size_t i = 0 ; i < N_th ; ++i) {
    mc_update<dS>(lat, states);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);

  // Write result to file
  // write_configuration(lat_old);

  fprintf(stderr, "Finalization...");
  // Free device memory
  free_lattice(&lat, &states);
  fprintf(stderr, " done.\n");
}

__host__ int main() {

  generate_single_conf();
  // mc_average();

  return 0;
}
