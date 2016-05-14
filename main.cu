#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <utility>
#include <cassert>
#include <cstdio>
// #include "H5Cpp.h"

/* Geometry & parameters ******************************************************/

// Block size
constexpr size_t B0 = 32; // ideally = warp size, for coalesced read & write
constexpr size_t B1 = 4;
constexpr size_t B2 = 4;
constexpr size_t B3 = 2;
constexpr size_t blockSize = B0*B1*B2*B3;
// Maximal number of threads 32*4*4*2 = 1024
// Shared memory usage: 28kio including RNG state.

// Grid size
constexpr size_t G0 = 1;
constexpr size_t G1 = 4;
constexpr size_t G2 = 4;
constexpr size_t G3 = 8;
constexpr size_t gridSize = G0*G1*G2*G3;
  
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
constexpr unsigned int N_cf  = 50;
constexpr unsigned int N_th  = 10*N_cor;
constexpr float epsilon = 0.5f;

/******************************************************************************/

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

__device__ float delta_S_free(float * f, const size_t Idx, const float zeta) {

  const float fi = f[Idx];
  const float delta_V = 0.5f*m2*zeta*(2.0f*fi+zeta);
  return delta_S_kin(f, Idx, zeta) + a*a*a*a*delta_V;
}

__device__ float delta_S_phi4(float * f, const size_t Idx, const float zeta) {

  const float fi = f[Idx];     // φi
  const float fiP = fi + zeta; // φi + ζ
  const float delta_V = 0.5f*m2*( fiP*fiP - fi*fi ) + 0.25f*lambda*( fiP*fiP*fiP*fiP - fi*fi*fi*fi );
  return delta_S_kin(f, Idx, zeta) + a*a*a*a*delta_V;
}


// Compute array index (includes ghost cells)
__device__ size_t array_idx(size_t Idx) {

  const size_t l = Idx / (N0*N1*N2);
  Idx -= l * N0*N1*N2;
  const size_t k = Idx / (N0*N1);
  Idx -= k * N0*N1;
  const size_t j = Idx / N0;
  Idx -= j * N0;

  return Idx+1 + M0*(j+1) + M0*M1*(k+1) + M0*M1*M2*(l+1);
}

template <float (*delta_S)(float*, const size_t, const float)>
__global__ void mc_kernel(float * lat, float * lo, curandState * states) {

  // Global thread index = lattice site
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Array index
  const size_t Idx = array_idx(tid);

  curandState state = states[tid];
  float zeta = (2.0f*curand_uniform(&state) - 1.0f) * epsilon; // ζ ∈ [-ε,+ε]

  // Compute change in the action due to variation ζ at size Idx
  const float delta_S_i = delta_S(lo, Idx, zeta);
  
  // Update the lattice depending on the variation ΔSi
  const float update = (float) (delta_S_i < 0.0f || (exp(-delta_S_i) > curand_uniform(&state)));
  // Is the above really branchless ?
  lat[Idx] += update * zeta;

  states[tid] = state;
}

// Initialize RNG state
__global__ void rng_init(curandState * states) {

  const size_t Idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init((unsigned long long)clock() + Idx, 0, 0, &states[Idx]);
}

// Exchange 3D "faces" of the 4D lattice
// Face 0 (stride = 1)
__global__ void exchange_faces_0(float * lat) {

  const size_t I1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = M0*I1 + M0*M1*I2 + M0*M1*M2*I3;

  lat[Idx         ] = lat[Idx + N0];
  lat[Idx + (N0+1)] = lat[Idx +  1];
}

// Face 1 (stride = M0)
__global__ void exchange_faces_1(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + M0*M1*I2 + M0*M1*M2*I3;

  lat[Idx            ] = lat[Idx + M0*N1];
  lat[Idx + M0*(N1+1)] = lat[Idx + M0   ];
}

// Face 2 (stride = M0·M1)
__global__ void exchange_faces_2(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + M0*I1 + M0*M1*M2*I3;

  lat[Idx               ] = lat[Idx + M0*M1*N2];
  lat[Idx + M0*M1*(N2+1)] = lat[Idx + M0*M1   ];
}

// Face 3 (stride = M0·M1·M2)
__global__ void exchange_faces_3(float * lat) {

  const size_t I0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const size_t I1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const size_t I2 = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const size_t Idx = I0 + M0*I1 + M0*M1*I2;

  lat[Idx                  ] = lat[Idx + M0*M1*M2*N3];
  lat[Idx + M0*M1*M2*(N3+1)] = lat[Idx + M0*M1*M2   ];
}

// Exchange all faces
__host__ void exchange_faces(float * lat) {

  exchange_faces_0<<<dim3(G1,G2,G3),dim3(B1,B2,B3)>>>(lat);
  exchange_faces_1<<<dim3(G0,G2,G3),dim3(B0,B2,B3)>>>(lat);
  exchange_faces_2<<<dim3(G0,G1,G3),dim3(B0,B1,B3)>>>(lat);
  exchange_faces_3<<<dim3(G0,G1,G2),dim3(B0,B1,B2)>>>(lat);
  cudaDeviceSynchronize();
}

template <float (*delta_S)(float*, const size_t, const float)>
void mc_update(float* lat, float * lat_old, curandState * states) {

  mc_kernel<delta_S><<<gridSize,blockSize>>>(lat, lat_old, states);
  cudaDeviceSynchronize();
  exchange_faces(lat);
  std::swap(lat, lat_old);
}

constexpr auto dS = delta_S_free;

__host__ void mc_average() {

  fprintf(stderr, "Lattice: (%d,%d,%d,%d)\n", N0, N1, N2, N3);
  fprintf(stderr, "Array:   (%d,%d,%d,%d)\n", M0, M1, M2, M3);
  fprintf(stderr, "M_count = %d\n", M_count);
  
  fprintf(stderr, "Allocating lattice arrays...\n");
  // Allocate lattice on device (double buffered)
  float * lat     = nullptr;
  float * lat_old = nullptr;
  fprintf(stderr, "Requesting 2×%d bytes...", M_bytes);
  cudaMalloc(&lat    , M_bytes);
  cudaMalloc(&lat_old, M_bytes);
  fprintf(stderr, " done.\n");
  fprintf(stderr, "Memset'ting to 0...");
  cudaMemset(lat    , 0., M_count);
  cudaMemset(lat_old, 0., M_count);
  fprintf(stderr, " done.\n");

  // Seed rng on each thread
  fprintf(stderr, "Allocating RNG...\n");
  fprintf(stderr, "Requesting %d bytes...", M_count*sizeof(curandState));
  curandState * states;
  cudaMalloc(&states, M_count*sizeof(curandState));
  fprintf(stderr, " done.\n");
  fprintf(stderr, "Initializing RNG...");
  rng_init<<<gridSize,blockSize>>>(states);
  cudaDeviceSynchronize();
  fprintf(stderr, " done.\n");

  // Thermalize lattice
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  fprintf(stderr, "Thermalizing lattice...");
  cudaEventRecord(start);
  for (size_t i = 0 ; i < N_th ; ++i) {
    mc_update<dS>(lat, lat_old, states);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);

  // Run Metropolis algorithm
  fprintf(stderr, "Running MC...");
  cudaEventRecord(start);
  for (size_t i = 0 ; i < N_cf ; ++i) {
    for (size_t j = 0 ; j < N_cor ; ++j) {
      mc_update<dS>(lat, lat_old, states);
    }
    fprintf(stderr, " %d", i);
    // TODO - Write result back here... (using lat_old)
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);


  // Finalization
  // ============

  fprintf(stderr, "Finalization...");
  // Free device memory
  cudaFree(lat);
  cudaFree(lat_old);
  cudaFree(states);
  lat     = nullptr;
  lat_old = nullptr;
  states  = nullptr;
  fprintf(stderr, " done.\n");
}

__host__ int main() {

  mc_average();

  return 0;
}