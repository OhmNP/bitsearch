#ifndef _EC_DUMP_TYPES_H
#define _EC_DUMP_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

namespace ecdump {

// Key family types
enum class KeyFamily {
  CONTROL,     // Uniform random in [1, n-1]
  CONSECUTIVE, // start_k + i
  MASKED,      // (fixed_high << t) | i
  STRIDE,      // start_k + i * stride
  HD           // H(seed || counter) mod n
};

// Convert family enum to string
inline const char *familyToString(KeyFamily family) {
  switch (family) {
  case KeyFamily::CONTROL:
    return "control";
  case KeyFamily::CONSECUTIVE:
    return "consecutive";
  case KeyFamily::MASKED:
    return "masked";
  case KeyFamily::STRIDE:
    return "stride";
  case KeyFamily::HD:
    return "hd";
  default:
    return "unknown";
  }
}

// Parse family string to enum
inline KeyFamily stringToFamily(const std::string &str) {
  if (str == "control")
    return KeyFamily::CONTROL;
  if (str == "consecutive")
    return KeyFamily::CONSECUTIVE;
  if (str == "masked")
    return KeyFamily::MASKED;
  if (str == "stride")
    return KeyFamily::STRIDE;
  if (str == "hd")
    return KeyFamily::HD;
  throw std::string("Unknown key family: " + str);
}

// Point record structure (41 bytes on disk)
struct PointRecord {
  uint32_t batch_id;
  uint32_t index_in_batch;
  uint8_t x[32];    // Big-endian
  uint8_t y_parity; // 0 or 1
};

// Batch telemetry structure
struct BatchTelemetry {
  uint64_t timestamp;
  uint32_t batch_id;
  KeyFamily family;
  std::string start_k;
  uint32_t mask_bits;
  uint64_t stride;
  uint64_t num_keys;
  double kernel_ms;
  double h2d_ms;
  double d2h_ms;
  double cpu_prep_ms;
  double cpu_wait_gpu_ms;
  uint64_t sampled_points;
  uint32_t matches; // Always 0 for this tool
};

// Experiment types for cryptanalysis
enum class ExperimentType {
  NONE,         // Standard point dump (P = k·G)
  DIFFERENTIAL, // Differential EC analysis (ΔP = P′ − P)
  ENDOMORPHISM, // Endomorphism projection (Δφ = Pφ − λ·P)
  ENDO_DIFF     // Endomorphism Differential (Δφ(k,δ) = φ((k+δ)G) - φ(kG))
};

// Differential experiment record (60 bytes)
// CRITICAL: All metadata is REQUIRED for statistical falsification
struct DifferentialRecord {
  uint8_t delta_P_x[32];       // ΔP.x full field element (big-endian)
  uint32_t delta_P_x_mod_m1;   // ΔP.x mod (2¹⁶−1) = 65535
  uint32_t delta_P_x_mod_m2;   // ΔP.x mod (2³²−5) = 4294967291
  uint32_t delta_value;        // The δ used (NOT delta_id)
  uint8_t scalar_family_id;    // Family enum value
  uint8_t mask_bits_or_stride; // Family-specific parameter
  uint32_t batch_id;           // REQUIRED for cross-batch tests
  uint32_t index_in_batch;     // REQUIRED for locality correlation
  uint8_t reserved[6];         // Padding to 60 bytes
};
// Total: 32 + 4 + 4 + 4 + 1 + 1 + 4 + 4 + 6 = 60 bytes

// Differential experiment configuration
struct DifferentialConfig {
  std::vector<uint64_t> delta_set; // {1, 2, 65536, ...}
  bool export_modular;             // Export modular reductions

  DifferentialConfig() : export_modular(true) {}
};

// Constants
const uint64_t MAX_BATCH_KEYS_DEFAULT = 100000000ULL; // 100M keys safety limit
const uint32_t VERIFY_POINT_COUNT = 1024;    // Number of points to verify
const uint32_t DEFAULT_BATCH_SIZE = 1048576; // 1M keys per batch
const size_t POINT_RECORD_SIZE = 41;         // Size of one point record on disk
const size_t DIFFERENTIAL_RECORD_SIZE = 60;  // Size of differential record

// Endomorphism experiment record (108 bytes)
struct EndomorphismRecord {
  uint32_t batch_id;
  uint32_t index_in_batch;
  uint8_t x[32];        // Base point P.x (Big-Endian)
  uint8_t beta_x[32];   // Endomorphism P_phi.x (Big-Endian)
  uint8_t lambda_x[32]; // Lambda point P_lambda.x (Big-Endian)
  uint32_t flags;       // flags:
                        // Bit 0: P.y parity
                        // Bit 1: P_phi.y parity
                        // Bit 2: P_lambda.y parity
                        // Bit 3: Infinity match (1 if P_phi == P_lambda)
};

struct EndomorphismConfig {
  bool export_full_coords; // Always true for this struct
  EndomorphismConfig() : export_full_coords(true) {}
};

const size_t ENDOMORPHISM_RECORD_SIZE = 108;

// Endomorphism Differential Record (60 bytes)
struct EndoDiffRecord {
  uint8_t delta_phi_x[32]; // Δφ.x (Big-Endian)
  uint32_t delta_value;
  uint32_t batch_id;
  uint32_t index_in_batch;
  uint32_t flags;       // Bit 0: P1.y parity, Bit 1: P2.y parity
  uint8_t reserved[12]; // Padding to 60 bytes
};

const size_t ENDO_DIFF_RECORD_SIZE = 60;

} // namespace ecdump

#endif // _EC_DUMP_TYPES_H
