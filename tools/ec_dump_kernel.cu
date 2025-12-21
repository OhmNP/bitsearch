#include "ec_dump.cuh"
#include "secp256k1.cuh"
#include "secp256k1.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

namespace ecdump {

// CUDA kernel to compute P = k*G for each private key
__global__ void computePointsKernel(const unsigned int *privateKeys,
                                    unsigned int *xCoords,
                                    unsigned char *yParity, int numKeys,
                                    const unsigned int *gxPtr,
                                    const unsigned int *gyPtr,
                                    DebugStep *debugTrace) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  int totalThreads = gridDim.x * blockDim.x;

  if (threadId >= numKeys) {
    return;
  }

  // Read private key for this thread (contiguous format: 8 words per key)
  unsigned int k[8];
  int base = threadId * 8; // Each key occupies 8 consecutive words
  for (int i = 0; i < 8; i++) {
    k[i] = privateKeys[base + i];
  }

  // Initialize result point to point at infinity
  unsigned int x[8];
  unsigned int y[8];
  for (int i = 0; i < 8; i++) {
    x[i] = 0xFFFFFFFF;
    y[i] = 0xFFFFFFFF;
  }

  // Perform point multiplication: P = k*G using double-and-add
  // Iterate through each bit of k from MSB to LSB
  for (int i = 0; i < 256; i++) {
    int wordIdx = i / 32;
    int bitIdx = i % 32;

    // Check if bit is set (reading from MSB, which is word 7, bit 31 down to
    // word 0, bit 0)
    int actualWordIdx = 7 - wordIdx;
    int actualBitIdx = 31 - bitIdx;
    bool bitSet = (k[actualWordIdx] & (1U << actualBitIdx)) != 0;

    // Get the base point for this bit position: 2^(255-i) * G
    // Table stores: index 0 = G, index 1 = 2G, index 2 = 4G, ..., index 255 =
    // 2^255*G For bit i (counting from MSB), we need 2^(255-i) * G, which is at
    // index (255-i)
    unsigned int gx[8];
    unsigned int gy[8];
    int baseIdx = (255 - i) * 8; // Correct index for 2^(255-i) * G
    for (int j = 0; j < 8; j++) {
      gx[j] = gxPtr[baseIdx + j];
      gy[j] = gyPtr[baseIdx + j];
    }

    bool did_add = false;

    if (bitSet) {
      did_add = true;
      // Add this base point to the result
      if (isInfinity(x)) {
        // Result is infinity, so just copy the base point
        copyBigInt(gx, x);
        copyBigInt(gy, y);
      } else {
        // Perform point addition: result = result + base_point
        unsigned int newX[8];
        unsigned int newY[8];

        // Check if points are equal (need point doubling)
        if (equal(x, gx)) {
          // Point doubling: 2*P
          // s = (3*x^2) / (2*y)
          unsigned int x2[8];
          unsigned int tx2[8];
          mulModP(x, x, x2);
          addModP(x2, x2, tx2);
          addModP(x2, tx2, tx2); // tx2 = 3*x^2

          unsigned int y2[8];
          addModP(y, y, y2); // y2 = 2*y

          unsigned int y2inv[8];
          invModP(y2, y2inv);

          unsigned int s[8];
          mulModP(tx2, y2inv, s);

          // newX = s^2 - 2*x
          unsigned int s2[8];
          mulModP(s, s, s2);
          subModP(s2, x, newX);
          subModP(newX, x, newX);

          // newY = s*(x - newX) - y
          unsigned int diff[8];
          subModP(x, newX, diff);
          mulModP(s, diff, newY);
          subModP(newY, y, newY);
        } else {
          // Point addition: P + Q
          // s = (gy - y) / (gx - x)
          unsigned int rise[8];
          subModP(gy, y, rise);

          unsigned int run[8];
          subModP(gx, x, run);

          unsigned int runInv[8];
          invModP(run, runInv);

          unsigned int s[8];
          mulModP(rise, runInv, s);

          // newX = s^2 - gx - x
          unsigned int s2[8];
          mulModP(s, s, s2);
          subModP(s2, gx, newX);
          subModP(newX, x, newX);

          // newY = s*(x - newX) - y  (using accumulated point's x, not base
          // point's gx)
          unsigned int diff[8];
          subModP(x, newX, diff);
          mulModP(s, diff, newY);
          subModP(newY, y, newY);
        }

        copyBigInt(newX, x);
        copyBigInt(newY, y);
      }
    }

    // Write debug trace for thread 0 (since we test with single thread
    // usually/first thread matters)
    if (debugTrace != nullptr && threadId == 0) {
      debugTrace[i].word_idx = actualWordIdx;
      debugTrace[i].bit_idx = actualBitIdx;
      debugTrace[i].bit_val = bitSet ? 1 : 0;
      debugTrace[i].did_add = did_add ? 1 : 0;

      // Copy accumulator state (handling infinity marker)
      // If infinity, we store 0xFFFFFFFF... which is our markers
      for (int w = 0; w < 8; w++) {
        debugTrace[i].acc_x[w] = x[w];
        debugTrace[i].acc_y[w] = y[w];
      }
    }
  }

  // Write x coordinate to output (contiguous format, already BigEndian)
  base = threadId * 8; // Each point occupies 8 consecutive words
  for (int i = 0; i < 8; i++) {
    xCoords[base + i] = x[i]; // No reversal needed - already BigEndian
  }

  // Write y parity (LSB of y - in BigEndian format, LSB is in y[7])
  yParity[threadId] = (y[7] & 1) ? 1 : 0;
}

// Initialize base point table: G, 2G, 4G, ..., 2^255*G
cudaError_t initBasePointTable(unsigned int **gxPtr, unsigned int **gyPtr) {
  // Allocate device memory for 256 points (x and y coordinates)
  cudaError_t err = cudaMalloc(gxPtr, 256 * 8 * sizeof(unsigned int));
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaMalloc(gyPtr, 256 * 8 * sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(*gxPtr);
    return err;
  }

  // Compute base point table on CPU
  std::vector<unsigned int> gxTable(256 * 8);
  std::vector<unsigned int> gyTable(256 * 8);

  secp256k1::ecpoint p = secp256k1::G();

  for (int i = 0; i < 256; i++) {
    // Store point in BigEndian format (to match how kernel reads it)
    p.x.exportWords(&gxTable[i * 8], 8, secp256k1::uint256::BigEndian);
    p.y.exportWords(&gyTable[i * 8], 8, secp256k1::uint256::BigEndian);

    // Double for next iteration
    p = secp256k1::doublePoint(p);
  }

  // Copy to device
  err = cudaMemcpy(*gxPtr, gxTable.data(), 256 * 8 * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(*gxPtr);
    cudaFree(*gyPtr);
    return err;
  }

  err = cudaMemcpy(*gyPtr, gyTable.data(), 256 * 8 * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(*gxPtr);
    cudaFree(*gyPtr);
    return err;
  }

  return cudaSuccess;
}

// Launch differential kernel wrapper
cudaError_t launchDifferentialKernel(
    const unsigned int *d_keys, uint32_t delta, unsigned int *d_delta_P_x,
    uint32_t *d_delta_P_x_mod_m1, uint32_t *d_delta_P_x_mod_m2,
    uint32_t batch_id, uint8_t family_id, uint8_t param, int numKeys,
    const unsigned int *d_gxPtr, const unsigned int *d_gyPtr, int blocks,
    int threads, cudaEvent_t start, cudaEvent_t stop) {

  cudaEventRecord(start);

  computeDifferentialKernel<<<blocks, threads>>>(
      d_keys, delta, d_delta_P_x, d_delta_P_x_mod_m1, d_delta_P_x_mod_m2,
      batch_id, family_id, param, numKeys, d_gxPtr, d_gyPtr);

  cudaEventRecord(stop);

  return cudaGetLastError();
}

void freeBasePointTable(unsigned int *gxPtr, unsigned int *gyPtr) {
  if (gxPtr)
    cudaFree(gxPtr);
  if (gyPtr)
    cudaFree(gyPtr);
}

cudaError_t launchComputePoints(const unsigned int *d_privateKeys,
                                unsigned int *d_xCoords,
                                unsigned char *d_yParity, int numKeys,
                                const unsigned int *d_gxPtr,
                                const unsigned int *d_gyPtr, int blocks,
                                int threads, cudaEvent_t start,
                                cudaEvent_t stop, DebugStep *d_debugTrace) {
  cudaEventRecord(start);

  computePointsKernel<<<blocks, threads>>>(d_privateKeys, d_xCoords, d_yParity,
                                           numKeys, d_gxPtr, d_gyPtr,
                                           d_debugTrace);

  cudaEventRecord(stop);

  return cudaGetLastError();
}

// ============================================================================
// DIFFERENTIAL EXPERIMENT: Modular Reduction Functions
// ============================================================================

// Reduce 256-bit value modulo (2^16 - 1) = 65535
__device__ uint16_t mod_65535(const unsigned int x[8]) {
  // Sum all 32-bit words, then reduce mod 65535
  // Since 2^32 ≡ 2^16 * 2^16 ≡ 1 * 1 ≡ 1 (mod 65535)
  // We can sum all words and reduce

  uint64_t sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += x[i];
  }

  // Reduce sum mod 65535
  // 2^32 mod 65535 = 65537 mod 65535 = 2
  uint32_t high = (uint32_t)(sum >> 32);
  uint32_t low = (uint32_t)(sum & 0xFFFFFFFF);

  uint32_t result = low + (high * 2);

  // Final reduction
  while (result >= 65535) {
    result = (result & 0xFFFF) + (result >> 16);
  }

  return (uint16_t)result;
}

// Reduce 256-bit value modulo (2^32 - 5) = 4294967291
__device__ uint32_t mod_4294967291(const unsigned int x[8]) {
  // For p = 2^32 - 5, we have 2^32 ≡ 5 (mod p)
  // So x = x7*2^224 + x6*2^192 + ... + x1*2^32 + x0
  // Reduce each term modulo p

  uint64_t result = 0;
  uint64_t power = 1; // 2^(32*i) mod p
  const uint64_t p = 4294967291ULL;

  for (int i = 0; i < 8; i++) {
    result = (result + ((uint64_t)x[i] * power) % p) % p;
    power = (power * 5) % p; // 2^32 ≡ 5 (mod p)
  }

  return (uint32_t)result;
}

// ============================================================================
// DIFFERENTIAL EXPERIMENT: Kernel
// ============================================================================

// Compute differential: ΔP = P′ − P where P′ = (k + δ)·G and P = k·G
__global__ void computeDifferentialKernel(
    const unsigned int *privateKeys, // Input scalars k
    uint32_t delta_value,            // The δ to add
    unsigned int *delta_P_x,         // Output: ΔP.x coordinates
    uint32_t *delta_P_x_mod_m1,      // Output: ΔP.x mod 65535
    uint32_t *delta_P_x_mod_m2,      // Output: ΔP.x mod 4294967291
    uint32_t batch_id,               // Batch ID for metadata
    uint8_t family_id,               // Scalar family ID
    uint8_t param,                   // mask_bits or stride parameter
    int numKeys, const unsigned int *gxPtr, const unsigned int *gyPtr) {

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  int totalThreads = gridDim.x * blockDim.x;

  if (threadId >= numKeys) {
    return;
  }

  // Read private key k for this thread (contiguous format: 8 words per key)
  unsigned int k[8];
  int base = threadId * 8; // Each key occupies 8 consecutive words
  for (int i = 0; i < 8; i++) {
    k[i] = privateKeys[base + i];
  }

  // Compute k' = k + δ (with overflow handling)
  unsigned int k_prime[8];
  uint64_t carry = delta_value;
  for (int i = 0; i < 8; i++) {
    uint64_t sum = (uint64_t)k[i] + carry;
    k_prime[i] = (unsigned int)(sum & 0xFFFFFFFF);
    carry = sum >> 32;
  }

  // Helper lambda to compute P = scalar * G
  auto computePoint = [&](const unsigned int *scalar, unsigned int *px,
                          unsigned int *py) {
    // Initialize to point at infinity
    for (int i = 0; i < 8; i++) {
      px[i] = 0xFFFFFFFF;
      py[i] = 0xFFFFFFFF;
    }

    // Double-and-add algorithm (same as computePointsKernel)
    for (int i = 0; i < 256; i++) {
      int wordIdx = i / 32;
      int bitIdx = i % 32;
      int actualWordIdx = 7 - wordIdx;
      int actualBitIdx = 31 - bitIdx;
      bool bitSet = (scalar[actualWordIdx] & (1U << actualBitIdx)) != 0;

      if (bitSet) {
        unsigned int gx[8], gy[8];
        int baseIdx = i * 8;
        for (int j = 0; j < 8; j++) {
          gx[j] = gxPtr[baseIdx + j];
          gy[j] = gyPtr[baseIdx + j];
        }

        if (isInfinity(px)) {
          copyBigInt(gx, px);
          copyBigInt(gy, py);
        } else {
          unsigned int newX[8], newY[8];

          if (equal(px, gx)) {
            // Point doubling
            unsigned int x2[8], tx2[8];
            mulModP(px, px, x2);
            addModP(x2, x2, tx2);
            addModP(x2, tx2, tx2);

            unsigned int y2[8];
            addModP(py, py, y2);

            unsigned int y2inv[8];
            invModP(y2, y2inv);

            unsigned int s[8];
            mulModP(tx2, y2inv, s);

            unsigned int s2[8];
            mulModP(s, s, s2);
            subModP(s2, px, newX);
            subModP(newX, px, newX);

            unsigned int diff[8];
            subModP(px, newX, diff);
            mulModP(s, diff, newY);
            subModP(newY, py, newY);
          } else {
            // Point addition
            unsigned int rise[8];
            subModP(gy, py, rise);

            unsigned int run[8];
            subModP(gx, px, run);

            unsigned int runInv[8];
            invModP(run, runInv);

            unsigned int s[8];
            mulModP(rise, runInv, s);

            unsigned int s2[8];
            mulModP(s, s, s2);
            subModP(s2, gx, newX);
            subModP(newX, px, newX);

            unsigned int diff[8];
            subModP(gx, newX, diff);
            mulModP(s, diff, newY);
            subModP(newY, gy, newY);
          }

          copyBigInt(newX, px);
          copyBigInt(newY, py);
        }
      }
    }
  };

  // Compute P = k·G
  unsigned int P_x[8], P_y[8];
  computePoint(k, P_x, P_y);

  // Compute P′ = (k + δ)·G
  unsigned int P_prime_x[8], P_prime_y[8];
  computePoint(k_prime, P_prime_x, P_prime_y);

  // Compute ΔP = P′ − P (point subtraction: P′ + (-P))
  // -P = (P.x, -P.y mod p)
  unsigned int neg_P_y[8];
  // Negate P.y: -y mod p = p - y
  // Use device constant _P directly (defined in secp256k1.cuh)
  subModP(_P, P_y, neg_P_y);

  unsigned int delta_P_x_local[8], delta_P_y_local[8];

  // P′ + (-P)
  if (equal(P_prime_x, P_x)) {
    // Special case: P′ == P, so ΔP = O (point at infinity)
    for (int i = 0; i < 8; i++) {
      delta_P_x_local[i] = 0;
      delta_P_y_local[i] = 0;
    }
  } else {
    // Standard point addition
    unsigned int rise[8];
    subModP(neg_P_y, P_prime_y, rise);

    unsigned int run[8];
    subModP(P_x, P_prime_x, run);

    unsigned int runInv[8];
    invModP(run, runInv);

    unsigned int s[8];
    mulModP(rise, runInv, s);

    unsigned int s2[8];
    mulModP(s, s, s2);
    subModP(s2, P_x, delta_P_x_local);
    subModP(delta_P_x_local, P_prime_x, delta_P_x_local);

    unsigned int diff[8];
    subModP(P_x, delta_P_x_local, diff);
    mulModP(s, diff, delta_P_y_local);
    subModP(delta_P_y_local, neg_P_y, delta_P_y_local);
  }

  // Write ΔP.x to output (big-endian)
  base = threadId;
  for (int i = 0; i < 8; i++) {
    delta_P_x[base] = delta_P_x_local[7 - i]; // Reverse for big-endian
    base += totalThreads;
  }

  // Compute and write modular reductions
  delta_P_x_mod_m1[threadId] = mod_65535(delta_P_x_local);
  delta_P_x_mod_m2[threadId] = mod_4294967291(delta_P_x_local);
}

// Endomorphism Kernel
__global__ void computeEndomorphismKernel(
    const unsigned int *keys, // k
    const unsigned int
        *lambda_keys,           // k_lambda = lambda * k (precomputed on host)
    unsigned int *x_out,        // P.x
    unsigned int *beta_x_out,   // P_phi.x
    unsigned int *lambda_x_out, // P_lambda.x
    unsigned int *flags_out,    // Flags: 0:Py, 1:Pphi.y, 2:Plambda.y, 3:Match
    const unsigned int *gxPtr, const unsigned int *gyPtr, int num_keys) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId >= num_keys)
    return;

  unsigned int k[8];
  unsigned int k_lam[8];

  // Load keys (contiguous 8 words per key)
  int base_in = threadId * 8;
  for (int i = 0; i < 8; i++) {
    k[i] = keys[base_in + i];
    k_lam[i] = lambda_keys[base_in + i];
  }

  // Lambda to compute point from scalar
  auto computePoint = [&](const unsigned int *scalar, unsigned int *px,
                          unsigned int *py) {
    // Init infinity
    for (int i = 0; i < 8; i++) {
      px[i] = 0xFFFFFFFF;
      py[i] = 0xFFFFFFFF;
    }

    for (int i = 0; i < 256; i++) {
      int wordIdx = i / 32;
      int bitIdx = i % 32;
      int actualWordIdx = 7 - wordIdx;
      int actualBitIdx = 31 - bitIdx;
      bool bitSet = (scalar[actualWordIdx] & (1U << actualBitIdx)) != 0;

      if (bitSet) {
        unsigned int gx[8], gy[8];
        int baseIdx = (255 - i) * 8; // Correct index for 2^(255-i) * G
        for (int j = 0; j < 8; j++) {
          gx[j] = gxPtr[baseIdx + j];
          gy[j] = gyPtr[baseIdx + j];
        }

        if (isInfinity(px)) {
          copyBigInt(gx, px);
          copyBigInt(gy, py);
        } else {
          unsigned int newX[8], newY[8];
          if (equal(px, gx)) {
            // Double
            unsigned int x2[8], tx2[8], y2[8], y2inv[8], s[8], s2[8], diff[8];
            mulModP(px, px, x2);
            addModP(x2, x2, tx2);
            addModP(x2, tx2, tx2); // 3x^2
            addModP(py, py, y2);
            invModP(y2, y2inv);
            mulModP(tx2, y2inv, s);
            mulModP(s, s, s2);
            subModP(s2, px, newX);
            subModP(newX, px, newX);
            subModP(px, newX, diff);
            mulModP(s, diff, newY);
            subModP(newY, py, newY);
          } else {
            // Add
            unsigned int rise[8], run[8], runInv[8], s[8], s2[8], diff[8];
            subModP(gy, py, rise);
            subModP(gx, px, run);
            invModP(run, runInv);
            mulModP(rise, runInv, s);
            mulModP(s, s, s2);
            subModP(s2, gx, newX);
            subModP(newX, px, newX);
            subModP(gx, newX, diff);
            mulModP(s, diff, newY);
            subModP(newY, gy, newY);
          }
          copyBigInt(newX, px);
          copyBigInt(newY, py);
        }
      }
    }
  };

  // 1. Compute P = kG
  unsigned int P_x[8], P_y[8];
  computePoint(k, P_x, P_y);

  // 2. Compute P_phi = (beta * P.x, P.y)
  unsigned int P_phi_x[8]; // y is same as P_y
  mulModP(_BETA, P_x, P_phi_x);

  // 3. Compute P_lambda = k_lam * G
  unsigned int P_lam_x[8], P_lam_y[8];
  computePoint(k_lam, P_lam_x, P_lam_y);

  // 4. Verify P_phi == P_lambda
  bool match = equal(P_phi_x, P_lam_x) && equal(P_y, P_lam_y);

  // 5. Output
  // Assume BigEndian output desired (word 0 is MSB)
  // Structure of Arrays output
  int base_out = threadId * 8; // Contiguous for each array
  for (int i = 0; i < 8; i++) {
    x_out[base_out + i] = P_x[i];
    beta_x_out[base_out + i] = P_phi_x[i];
    lambda_x_out[base_out + i] = P_lam_x[i];
  }

  // Flags
  int p_y_parity = (P_y[7] & 1);
  int p_phi_y_parity = p_y_parity; // Same Y
  int p_lam_y_parity = (P_lam_y[7] & 1);
  int match_flag = match ? 1 : 0;

  unsigned int flags = (match_flag << 3) | (p_lam_y_parity << 2) |
                       (p_phi_y_parity << 1) | p_y_parity;
  flags_out[threadId] = flags;
}

cudaError_t launchEndomorphismKernel(
    const unsigned int *d_keys, const unsigned int *d_lambda_keys,
    unsigned int *d_x, unsigned int *d_beta_x, unsigned int *d_lambda_x,
    unsigned int *d_flags, const unsigned int *d_gx, const unsigned int *d_gy,
    int numKeys, int blocks, int threads, cudaEvent_t start, cudaEvent_t stop) {
  cudaEventRecord(start);
  computeEndomorphismKernel<<<blocks, threads>>>(d_keys, d_lambda_keys, d_x,
                                                 d_beta_x, d_lambda_x, d_flags,
                                                 d_gx, d_gy, numKeys);
  cudaEventRecord(stop);
  return cudaGetLastError();
}

// Endomorphism Differential Kernel
__global__ void computeEndoDiffKernel(const unsigned int *keys,  // k
                                      uint32_t delta_value,      // delta
                                      unsigned int *delta_phi_x, // Output Δφ.x
                                      uint32_t *flags_out,       // Output flags
                                      const unsigned int *gxPtr,
                                      const unsigned int *gyPtr, int num_keys) {

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId >= num_keys)
    return;

  // Load private key k
  unsigned int k[8];
  int base_in = threadId * 8;
  for (int i = 0; i < 8; i++) {
    k[i] = keys[base_in + i];
  }

  // Compute k' = k + delta
  unsigned int k_prime[8];
  uint64_t carry = delta_value;
  for (int i = 0; i < 8; i++) {
    uint64_t sum = (uint64_t)k[i] + carry;
    k_prime[i] = (unsigned int)(sum & 0xFFFFFFFF);
    carry = sum >> 32;
  }

  // Lambda to compute point from scalar
  auto computePoint = [&](const unsigned int *scalar, unsigned int *px,
                          unsigned int *py) {
    for (int i = 0; i < 8; i++)
      px[i] = py[i] = 0xFFFFFFFF; // Init infinity

    for (int i = 0; i < 256; i++) {
      int wordIdx = i / 32;
      int bitIdx = i % 32;
      int actualWordIdx = 7 - wordIdx;
      int actualBitIdx = 31 - bitIdx;
      bool bitSet = (scalar[actualWordIdx] & (1U << actualBitIdx)) != 0;

      if (bitSet) {
        unsigned int gx[8], gy[8];
        int baseIdx = (255 - i) * 8; // Correct index order
        for (int j = 0; j < 8; j++) {
          gx[j] = gxPtr[baseIdx + j];
          gy[j] = gyPtr[baseIdx + j];
        }

        if (isInfinity(px)) {
          copyBigInt(gx, px);
          copyBigInt(gy, py);
        } else {
          unsigned int newX[8], newY[8];
          if (equal(px, gx)) {
            // Double
            unsigned int x2[8], tx2[8], y2[8], y2inv[8], s[8], s2[8], diff[8];
            mulModP(px, px, x2);
            addModP(x2, x2, tx2);
            addModP(x2, tx2, tx2); // 3x^2
            addModP(py, py, y2);
            invModP(y2, y2inv);
            mulModP(tx2, y2inv, s);
            mulModP(s, s, s2);
            subModP(s2, px, newX);
            subModP(newX, px, newX);
            subModP(px, newX, diff);
            mulModP(s, diff, newY);
            subModP(newY, py, newY);
          } else {
            // Add
            unsigned int rise[8], run[8], runInv[8], s[8], s2[8], diff[8];
            subModP(gy, py, rise);
            subModP(gx, px, run);
            invModP(run, runInv);
            mulModP(rise, runInv, s);
            mulModP(s, s, s2);
            subModP(s2, gx, newX);
            subModP(newX, px, newX);
            subModP(gx, newX, diff);
            mulModP(s, diff, newY);
            subModP(newY, gy, newY);
          }
          copyBigInt(newX, px);
          copyBigInt(newY, py);
        }
      }
    }
  };

  // 1. Compute P1 = kG
  unsigned int P1_x[8], P1_y[8];
  computePoint(k, P1_x, P1_y);

  // 2. Compute P2 = k'G
  unsigned int P2_x[8], P2_y[8];
  computePoint(k_prime, P2_x, P2_y);

  // 3. Apply Endomorphism phi(x,y) = (beta*x, y)
  // Store directly back into P variables to save registers
  // P1_x = beta * P1_x
  unsigned int temp[8];
  mulModP(_BETA, P1_x, temp);
  copyBigInt(temp, P1_x);

  // P2_x = beta * P2_x
  mulModP(_BETA, P2_x, temp);
  copyBigInt(temp, P2_x);

  // 4. Compute Delta = P2 - P1 = P2 + (-P1)
  // -P1 = (x, -y)
  unsigned int neg_P1_y[8];
  subModP(_P, P1_y, neg_P1_y);

  unsigned int D_x[8], D_y[8];

  if (equal(P2_x, P1_x)) {
    // Check if P2 == P1 (Delta = 0) or P2 == -P1 (Delta = 2*P2)
    // For endomorphism, x coordinates match only if points are equal or
    // negations. If equal, difference is infinity.
    for (int i = 0; i < 8; i++) {
      D_x[i] = 0;
      D_y[i] = 0;
    }
  } else {
    // Standard addition P2 + (-P1)
    unsigned int rise[8], run[8], runInv[8], s[8], s2[8], diff[8];
    subModP(neg_P1_y, P2_y,
            rise); // (-y1) - y2 ?? No, P2.y - (-P1.y) = y2 + y1?
    // Wait. Point subtraction P2 - P1 is P2 + (-P1).
    // -P1 = (P1.x, -P1.y)
    // Slope s = (P2.y - (-P1.y)) / (P2.x - P1.x)
    //         = (P2.y + P1.y) / (P2.x - P1.x) ... IF we use mod arithmetic
    //         carefully.
    // But let's stick to standard subModP:
    // rise = (-P1.y) - P2.y  <-- Wait, standard slope is (y2 - y1) / (x2 - x1)
    // Target: (P2 + (-P1))
    // Point A = P2, Point B = -P1
    // slope = (B.y - A.y) / (B.x - A.x)
    //       = (neg_P1_y - P2_y) / (P1_x - P2_x)

    subModP(neg_P1_y, P2_y, rise);
    subModP(P1_x, P2_x, run); // using P1_x because -P1.x = P1.x

    invModP(run, runInv);
    mulModP(rise, runInv, s);
    mulModP(s, s, s2);

    // x3 = s^2 - x1 - x2
    subModP(s2, P2_x, D_x);
    subModP(D_x, P1_x, D_x);

    // y3 = s(x1 - x3) - y1
    // Use P2 as anchor (A)
    subModP(P2_x, D_x, diff);
    mulModP(s, diff, D_y);
    subModP(D_y, P2_y, D_y);
  }

  // Output in AoS format (compatible with host expectations)
  int base_out = threadId * 8;
  for (int i = 0; i < 8; i++) {
    delta_phi_x[base_out + i] = D_x[i]; // Internal is likely Big Endian? Or we
                                        // want to reverse reversal.
  }

  // Flags: P1.y parity (bit 0), P2.y parity (bit 1)
  flags_out[threadId] = ((P2_y[7] & 1) << 1) | (P1_y[7] & 1);
}

cudaError_t launchEndoDiffKernel(
    const unsigned int *d_keys, uint32_t delta, unsigned int *d_delta_phi_x,
    unsigned int *d_flags, const unsigned int *d_gx, const unsigned int *d_gy,
    int numKeys, int blocks, int threads, cudaEvent_t start, cudaEvent_t stop) {
  cudaEventRecord(start);
  computeEndoDiffKernel<<<blocks, threads>>>(d_keys, delta, d_delta_phi_x,
                                             d_flags, d_gx, d_gy, numKeys);
  cudaEventRecord(stop);
  return cudaGetLastError();
}

} // namespace ecdump
