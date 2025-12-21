#ifndef _EC_DUMP_CUH
#define _EC_DUMP_CUH

#include <cuda.h>
#include <cuda_runtime.h>

namespace ecdump {

// Debug structure for forensic analysis
struct DebugStep {
  int word_idx;
  int bit_idx;
  int bit_val;
  int did_add;
  unsigned int acc_x[8];
  unsigned int acc_y[8]; // Full Y for complete verification
};

// CUDA kernel for computing P = k*G for a batch of private keys
// Input:  privateKeys - array of scalars in device memory (8 x uint32 per key,
// little-endian) Output: xCoords - array of x coordinates (8 x uint32 per
// point, big-endian for output)
//         yParity - array of y parity bits (1 byte per point, 0 or 1)
// numKeys: number of keys to process
// gxPtr, gyPtr: precomputed base point table (256 points: G, 2G, 4G, ...,
// 2^255*G)
#ifdef __CUDACC__
__global__ void computePointsKernel(const unsigned int *privateKeys,
                                    unsigned int *xCoords,
                                    unsigned char *yParity, int numKeys,
                                    const unsigned int *gxPtr,
                                    const unsigned int *gyPtr,
                                    DebugStep *debugTrace);
#endif

// Host function to allocate and initialize base point table
cudaError_t initBasePointTable(unsigned int **gxPtr, unsigned int **gyPtr);

// Host function to free base point table
void freeBasePointTable(unsigned int *gxPtr, unsigned int *gyPtr);

// Host wrapper to launch kernel with timing
cudaError_t
launchComputePoints(const unsigned int *d_privateKeys, unsigned int *d_xCoords,
                    unsigned char *d_yParity, int numKeys,
                    const unsigned int *d_gxPtr, const unsigned int *d_gyPtr,
                    int blocks, int threads, cudaEvent_t start,
                    cudaEvent_t stop, DebugStep *d_debugTrace = nullptr);

// ============================================================================
// DIFFERENTIAL EXPERIMENT: Kernel Declaration
// ============================================================================

// Compute differential: ΔP = P′ − P where P′ = (k + δ)·G and P = k·G
#ifdef __CUDACC__
__global__ void computeDifferentialKernel(
    const unsigned int *privateKeys, // Input scalars k
    uint32_t delta_value,            // The δ to add
    unsigned int *delta_P_x,         // Output: ΔP.x coordinates
    uint32_t *delta_P_x_mod_m1,      // Output: ΔP.x mod 65535
    uint32_t *delta_P_x_mod_m2,      // Output: ΔP.x mod 4294967291
    uint32_t batch_id,               // Batch ID for metadata
    uint8_t family_id,               // Scalar family ID
    uint8_t param,                   // mask_bits or stride parameter
    int numKeys, const unsigned int *gxPtr, const unsigned int *gyPtr);
#endif

// Host wrapper to launch differential kernel with timing
cudaError_t launchDifferentialKernel(
    const unsigned int *d_keys, uint32_t delta, unsigned int *d_delta_P_x,
    uint32_t *d_delta_P_x_mod_m1, uint32_t *d_delta_P_x_mod_m2,
    uint32_t batch_id, uint8_t family_id, uint8_t param, int numKeys,
    const unsigned int *d_gxPtr, const unsigned int *d_gyPtr, int blocks,
    int threads, cudaEvent_t start, cudaEvent_t stop);

// Launch endomorphism kernel wrapper
cudaError_t launchEndomorphismKernel(
    const unsigned int *d_keys, const unsigned int *d_lambda_keys,
    unsigned int *d_x, unsigned int *d_beta_x, unsigned int *d_lambda_x,
    unsigned int *d_flags, const unsigned int *d_gx, const unsigned int *d_gy,
    int numKeys, int blocks, int threads, cudaEvent_t start, cudaEvent_t stop);

// Endomorphism Differential Kernel Declaration
#ifdef __CUDACC__
__global__ void computeEndoDiffKernel(const unsigned int *keys,  // k
                                      uint32_t delta_value,      // delta
                                      unsigned int *delta_phi_x, // Output Δφ.x
                                      uint32_t *flags_out,       // Output flags
                                      const unsigned int *gxPtr,
                                      const unsigned int *gyPtr, int num_keys);
#endif

cudaError_t launchEndoDiffKernel(
    const unsigned int *d_keys, uint32_t delta, unsigned int *d_delta_phi_x,
    unsigned int *d_flags, const unsigned int *d_gx, const unsigned int *d_gy,
    int numKeys, int blocks, int threads, cudaEvent_t start, cudaEvent_t stop);

} // namespace ecdump

#endif // _EC_DUMP_CUH
