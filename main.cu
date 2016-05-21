#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <utility>
#include <cassert>
#include <cstdio>
#include "H5Cpp.h"

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
constexpr size_t G0 = 2;
constexpr size_t G1 = 8;
constexpr size_t G2 = 8;
constexpr size_t G3 = 16;
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
constexpr float a = 1.0f;

// Physical parameters
constexpr float m2 = 1.0f;
constexpr float lambda = 1.0f;

// Monte-Carlo parameters
constexpr unsigned int N_cor = 20;
constexpr unsigned int N_cf  = 100;
constexpr unsigned int N_th  = 10*N_cor;
constexpr float epsilon = 0.5f;

// Output
const H5std_string file_name("correlations.h5");
const H5std_string dataset_name("corr");

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

__global__ void time_slice_corr(float * lat, const size_t delta, float * corr) {

  const size_t I1 = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t I2 = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t I3 = blockIdx.z * blockDim.z + threadIdx.z;
  const size_t Idx = 1 + (I1+1)*M0 + (I2+1)*M0*M1 + (I3+1)*M0*M1*M2;

  float acc = 0.0f;
  for (size_t I0 = 0 ; I0 < N0 ; ++I0) {

    acc += lat[Idx+I0]*lat[Idx+(I0+delta)%N0];
  }

  corr[I1 + N1*I2 + N1*N2*I3] = acc / N0;
}

__host__ float reduce_3d(float * corr) {

  float acc = 0.0f;
  for (size_t I3 = 0 ; I3 < N3 ; ++I3)
    for (size_t I2 = 0 ; I2 < N2 ; ++I2)
      for (size_t I1 = 0 ; I1 < N1 ; ++I1) {

        acc += corr[I1 + N1*I2 + N1*N2*I3];
      }

  return acc / (N1*N2*N3);
}

__host__ void compute_correlations(float * lat, float * res, size_t run,
                                   float * corr_buf_h, float * corr_buf_d) {

  for (size_t delta = 0 ; delta < N0 ; ++delta) {

    time_slice_corr<<<dim3(B1,B2,B3),dim3(G1,G2,G3)>>>(lat, delta, corr_buf_d);
    cudaDeviceSynchronize();
    cudaMemcpy(corr_buf_h, corr_buf_d, N1*N2*N3*sizeof(float), cudaMemcpyDeviceToHost);
    res[run + N_cf*delta] = reduce_3d(corr_buf_h);
  }
}

int write_correlations(float * corr) {

  try {

    H5::H5File file(file_name, H5F_ACC_TRUNC);
    hsize_t dims[2] = {N_cf, N0};
    H5::DataSpace dataspace(2, dims);
    auto dataset = file.createDataSet(dataset_name, H5::PredType::NATIVE_FLOAT, dataspace);
    dataset.write(corr, H5::PredType::NATIVE_FLOAT);
    dataset.close();
    file.close();
  }

  // catch failure caused by the H5File operations
  catch(H5::FileIException error)
    {
      error.printError();
      return -1;
    }
  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error)
    {
      error.printError();
      return -1;
    }
  // catch failure caused by the DataSpace operations
  catch(H5::DataSpaceIException error)
    {
      error.printError();
      return -1;
    }
  // catch failure caused by the DataSpace operations
  catch(H5::DataTypeIException error)
    {
      error.printError();
      return -1;
    }
  return 0;  // successfully terminated  
}

int write_configuration(float * lat) {

  H5::H5File file("configuration.h5", H5F_ACC_TRUNC);
  hsize_t dims[4]   = {N0,N1,N2,N3};
  hsize_t offset[4] = { 1, 1, 1, 1};
  H5::DataSpace dataspace(4, dims);
  // dataspace.selectHyperslab();   <----------------------- TO-DO
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

  // Allocate memory used to store correlation data
  // Host-side buffer
  float * corr_buf_h = (float*) calloc(N1*N2*N3, sizeof(float));
  assert(corr_buf_h);
  // Device-side buffer
  float * corr_buf_d = nullptr;
  cudaMalloc(&corr_buf_d, N1*N2*N3*sizeof(float));
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
    // Drop N_cor iterations to damp correlations between successive configurations.
    for (size_t j = 0 ; j < N_cor ; ++j) {
      mc_update<dS>(lat, lat_old, states);
    }
    fprintf(stderr, " %d", i);
    // Compute the Euclidean time correlations within one configuration.
    // compute_correlations(lat_old, corr, i, corr_buf_h, corr_buf_d);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  fprintf(stderr, " done in %fs.\n", 1e-3*ms);

  // Write output to file
  fprintf(stderr, "Writing to file...");
  write_correlations(corr);
  fprintf(stderr, " done.\n");
  
  // Finalization
  // ============

  fprintf(stderr, "Finalization...");
  // Free device memory
  cudaFree(lat);
  cudaFree(lat_old);
  cudaFree(states);
  cudaFree(corr_buf_d);
  lat        = nullptr;
  lat_old    = nullptr;
  states     = nullptr;
  corr_buf_d = nullptr;

  // Free host memory
  free(corr_buf_h);
  free(corr);
  corr_buf_h = nullptr;
  corr       = nullptr;
  fprintf(stderr, " done.\n");
}

void generate_single_conf() {

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

  // Write result to file
  write_configuration(lat_old);

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

  //generate_single_conf();
  mc_average();

  return 0;
}
