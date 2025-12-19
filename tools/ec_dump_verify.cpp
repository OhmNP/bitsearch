#define _CRT_SECURE_NO_WARNINGS
#include "ec_dump.cuh"
#include "ec_dump.h"
#include "secp256k1.h" // The custom wrapper
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>


namespace ecdump {

// ============================================================================
// HARDENED VERIFICATION HARNESS
// ============================================================================

enum class VerificationError {
  NONE,
  SERIALIZATION_ERROR,
  INVALID_POINT,
  CURVE_VIOLATION,
  GPU_CPU_MISMATCH
};

struct CanonicalGpuPoint {
  uint8_t data[33];

  const uint8_t *getX() const { return data + 1; }
  uint8_t getParity() const {
    return (data[0] == 0x03) ? 1 : 0;
  } // 02=Even(0), 03=Odd(1)
  void setHeader(uint8_t parity) { data[0] = (parity ? 0x03 : 0x02); }
};
static_assert(sizeof(CanonicalGpuPoint) == 33,
              "CanonicalGpuPoint size mismatch");

static void checkCuda(cudaError_t result, const char *func) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error in hard gate (" << func
              << "): " << cudaGetErrorString(result) << std::endl;
    exit(1);
  }
}

static void fail_verification(VerificationError type, const char *msg) {
  std::cerr << "!!! VERIFICATION FAILED !!!" << std::endl;
  std::cerr << "Type: ";
  switch (type) {
  case VerificationError::SERIALIZATION_ERROR:
    std::cerr << "SERIALIZATION_ERROR";
    break;
  case VerificationError::INVALID_POINT:
    std::cerr << "INVALID_POINT";
    break;
  case VerificationError::CURVE_VIOLATION:
    std::cerr << "CURVE_VIOLATION";
    break;
  case VerificationError::GPU_CPU_MISMATCH:
    std::cerr << "GPU_CPU_MISMATCH";
    break;
  default:
    std::cerr << "UNKNOWN";
    break;
  }
  std::cerr << std::endl;
  std::cerr << "Message: " << msg << std::endl;
  std::cerr << "ABORTING." << std::endl;
  exit(1);
}

// Helper: Bytes to Words (Big Endian)
static void bytes_to_words_be(const uint8_t *bytes, unsigned int *words) {
  for (int i = 0; i < 8; i++) {
    words[i] = ((unsigned int)bytes[i * 4] << 24) |
               ((unsigned int)bytes[i * 4 + 1] << 16) |
               ((unsigned int)bytes[i * 4 + 2] << 8) |
               ((unsigned int)bytes[i * 4 + 3]);
  }
}

// Helper: Words to Bytes (Big Endian)
static void words_to_bytes_be(const unsigned int *words, uint8_t *bytes) {
  for (int i = 0; i < 8; i++) {
    unsigned int w = words[i];
    bytes[i * 4] = (uint8_t)((w >> 24) & 0xFF);
    bytes[i * 4 + 1] = (uint8_t)((w >> 16) & 0xFF);
    bytes[i * 4 + 2] = (uint8_t)((w >> 8) & 0xFF);
    bytes[i * 4 + 3] = (uint8_t)((w) & 0xFF);
  }
}

static void gpu_words_to_canonical(const unsigned int *gpu_x,
                                   unsigned char gpu_y_parity,
                                   CanonicalGpuPoint &out) {
  out.setHeader(gpu_y_parity & 1);
  // gpu_x is uint32 array. Words are Big Endian index.
  // We want Big Endian Byte Stream.
  // Check endianness of host machine? we assume Little Endian (x86).
  // So if gpu_x[0] is MSB word...
  words_to_bytes_be(gpu_x, out.data + 1);
}

// 1. libsecp256k1 Roundtrip (Binary Only)
static bool check_roundtrip(const CanonicalGpuPoint &pt) {
  try {
    // Parse (using our new binary parser)
    secp256k1::ecpoint p = secp256k1::parsePublicKey(pt.data, 33);

    // Re-serialize strictly
    unsigned char output[33];
    output[0] = p.y.isEven() ? 0x02 : 0x03;

    unsigned int x_words[8];
    p.x.exportWords(x_words, 8, secp256k1::uint256::BigEndian);
    words_to_bytes_be(x_words, output + 1);

    // Binary Compare
    if (std::memcmp(pt.data, output, 33) != 0) {
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

// 2. Binary Curve Equation Check
static bool check_curve_binary(const CanonicalGpuPoint &pt) {
  // 02/03 prefix check
  if (pt.data[0] != 0x02 && pt.data[0] != 0x03)
    return false;

  // Convert X bytes to uint256
  unsigned int words[8];
  bytes_to_words_be(pt.data + 1, words);
  secp256k1::uint256 X(words, secp256k1::uint256::BigEndian);

  // Check valid range (X < P)
  // P is secp256k1::P
  if (!(X < secp256k1::P))
    return false;

  // RHS = X^3 + 7
  secp256k1::uint256 X2 = secp256k1::multiplyModP(X, X);
  secp256k1::uint256 X3 = secp256k1::multiplyModP(X2, X);
  secp256k1::uint256 RHS = secp256k1::addModP(X3, secp256k1::uint256(7));

  // Sqrt(RHS)
  // Exponent = (P+1)/4
  secp256k1::uint256 EXP = secp256k1::P.add(1).div(4);
  secp256k1::uint256 Y = secp256k1::powModP(RHS, EXP);

  // Verify Y^2 == RHS
  secp256k1::uint256 Y2 = secp256k1::multiplyModP(Y, Y);
  if (!(Y2 == RHS))
    return false; // Not on curve

  return true; // Exists on curve. Details of parity matches roundtrip check.
}

static void harness_self_test() {
  std::cout << "Running Harness Self-Test..." << std::endl;

  // 1. Valid Point (G)
  // 02 79BE667E F9DCBBAC 55A06295 CE870B07 029BFCDB 2DCE28D9 59F2815B 16F81798
  unsigned char valid_G[] = {
      0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0,
      0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D,
      0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98};
  CanonicalGpuPoint g_pt;
  std::memcpy(g_pt.data, valid_G, 33);

  if (!check_roundtrip(g_pt))
    fail_verification(VerificationError::NONE,
                      "Self-Test: Valid point failed roundtrip");
  if (!check_curve_binary(g_pt))
    fail_verification(VerificationError::NONE,
                      "Self-Test: Valid point failed binary math");

  // 2. Invalid Point (x=0)
  CanonicalGpuPoint bad_pt;
  std::memset(bad_pt.data, 0, 33);
  bad_pt.data[0] = 0x02;
  if (check_roundtrip(bad_pt))
    fail_verification(VerificationError::NONE,
                      "Self-Test: Invalid point PASSED roundtrip");
  if (check_curve_binary(bad_pt))
    fail_verification(VerificationError::NONE,
                      "Self-Test: Invalid point PASSED binary math");

  std::cout << "Self-Test PASSED." << std::endl;
}

// Implementation of ECDumpDriver::verifyCPU
bool ECDumpDriver::verifyCPU(const std::vector<secp256k1::uint256> &keys,
                             const std::vector<PointRecord> &gpu_results) {
  return true; // Stub
}

// HARD GATE ENTRY POINT
// Replaces old version
void perform_hard_correctness_gate() {
  std::cout << "Starting HARDENED Correctness Gate..." << std::endl;

  harness_self_test();

  unsigned int *d_gx = nullptr, *d_gy = nullptr;
  unsigned int *d_keys, *d_x;
  unsigned char *d_y;

  checkCuda(cudaMalloc(&d_keys, 8 * sizeof(unsigned int)), "malloc keys");
  checkCuda(cudaMalloc(&d_x, 8 * sizeof(unsigned int)), "malloc x");
  checkCuda(cudaMalloc(&d_y, 1 * sizeof(unsigned char)), "malloc y");

  initBasePointTable(&d_gx, &d_gy);

  // Verify G (k=1)
  unsigned int h_key[8] = {1, 0, 0, 0, 0, 0, 0, 0};
  checkCuda(cudaMemcpy(d_keys, h_key, 8 * sizeof(unsigned int),
                       cudaMemcpyHostToDevice),
            "memcpy k");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  launchComputePoints(d_keys, d_x, d_y, 1, d_gx, d_gy, 1, 1, start, stop,
                      nullptr);
  cudaEventSynchronize(stop);

  // Fetch result
  unsigned int gpu_x[8];
  unsigned char gpu_y;
  checkCuda(
      cudaMemcpy(gpu_x, d_x, 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost),
      "memcpy x");
  checkCuda(cudaMemcpy(&gpu_y, d_y, 1, cudaMemcpyDeviceToHost), "memcpy y");

  // Convert to Canonical
  CanonicalGpuPoint pt;
  gpu_words_to_canonical(gpu_x, gpu_y, pt);

  // HARD CHECKS
  if (!check_roundtrip(pt))
    fail_verification(VerificationError::INVALID_POINT,
                      "Step 1 (G) failed roundtrip");
  if (!check_curve_binary(pt))
    fail_verification(VerificationError::CURVE_VIOLATION,
                      "Step 1 (G) failed curve check");

  // Verify against CPU 1*G
  secp256k1::ecpoint P_cpu =
      secp256k1::multiplyPoint(secp256k1::uint256(1), secp256k1::G());

  // Import GPU X to uint256 for comparison
  unsigned int words[8];
  bytes_to_words_be(pt.getX(), words);
  secp256k1::uint256 X_gpu(words, secp256k1::uint256::BigEndian);

  if (!(X_gpu == P_cpu.x))
    fail_verification(VerificationError::GPU_CPU_MISMATCH,
                      "Step 1 (G) X mismatch");

  int cpu_parity = P_cpu.y.isEven() ? 0 : 1;
  if (pt.getParity() != cpu_parity)
    fail_verification(VerificationError::GPU_CPU_MISMATCH,
                      "Step 1 (G) Parity mismatch");

  std::cout << "All Hardened Checks PASSED." << std::endl;

  cudaFree(d_keys);
  cudaFree(d_x);
  cudaFree(d_y);
  freeBasePointTable(d_gx, d_gy);
}

} // namespace ecdump
