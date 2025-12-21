#ifndef _EC_DUMP_H
#define _EC_DUMP_H

#include "ec_dump_types.h"
#include "secp256k1.h"
#include <memory>
#include <string>
#include <vector>

namespace ecdump {

// Configuration structure for CLI parameters
struct ECDumpConfig {
  KeyFamily family;
  secp256k1::uint256 start_k;
  uint32_t mask_bits;
  uint64_t stride;
  uint64_t batch_keys;
  uint32_t batches;
  std::string out_bin;
  std::string telemetry_file;
  double sample_rate; // 0.0 to 1.0, or if > 1.0, treated as reservoir size
  uint64_t seed;
  bool verify;
  int device;
  bool dry_run;
  bool force;

  // Experiment configuration
  ExperimentType experiment;
  DifferentialConfig diff_config;

  ECDumpConfig()
      : family(KeyFamily::CONSECUTIVE), start_k(1), mask_bits(0), stride(1),
        batch_keys(DEFAULT_BATCH_SIZE), batches(1), sample_rate(1.0), seed(0),
        verify(false), device(0), dry_run(false), force(false),
        experiment(ExperimentType::NONE) {}
};

// Abstract base class for key generators
class KeyGenerator {
public:
  virtual ~KeyGenerator() {}

  // Generate a batch of keys
  virtual void generateBatch(uint32_t batch_id, uint64_t batch_size,
                             std::vector<secp256k1::uint256> &keys_out) = 0;

  // Get the starting key for a batch (for telemetry)
  virtual secp256k1::uint256 getBatchStartKey(uint32_t batch_id,
                                              uint64_t batch_size) = 0;
};

// Factory function to create key generator
std::unique_ptr<KeyGenerator> createKeyGenerator(const ECDumpConfig &config);

// I/O classes (defined in ec_dump_io.cpp)
class PointFileWriter {
public:
  explicit PointFileWriter(const std::string &filename);
  ~PointFileWriter();
  void writePoints(const std::vector<PointRecord> &records);
  void flush();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class TelemetryWriter {
public:
  explicit TelemetryWriter(const std::string &filename);
  ~TelemetryWriter();
  void writeBatch(const BatchTelemetry &telem);
  void flush();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Differential experiment file writer
class DifferentialFileWriter {
public:
  explicit DifferentialFileWriter(const std::string &filename);
  ~DifferentialFileWriter();
  void writeRecords(const std::vector<DifferentialRecord> &records);
  void flush();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Endomorphism experiment file writer
class EndomorphismFileWriter {
public:
  explicit EndomorphismFileWriter(const std::string &filename);
  ~EndomorphismFileWriter();
  void writeRecords(const std::vector<EndomorphismRecord> &records);
  void flush();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Endomorphism Differential experiment file writer
class EndoDiffFileWriter {
public:
  explicit EndoDiffFileWriter(const std::string &filename);
  ~EndoDiffFileWriter();
  void writeRecords(const std::vector<EndoDiffRecord> &records);
  void flush();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// I/O helper functions
// I/O helper functions
void writeManifest(const ECDumpConfig &config, uint64_t total_keys);
void convertGPUOutput(uint32_t batch_id,
                      const std::vector<unsigned int> &xCoords,
                      const std::vector<unsigned char> &yParity,
                      std::vector<PointRecord> &records_out,

                      double sample_rate);

void convertGPUEndoDiffOutput(uint32_t batch_id, uint32_t delta_value,
                              const std::vector<unsigned int> &delta_phi_x,
                              const std::vector<uint32_t> &flags,
                              std::vector<EndoDiffRecord> &records_out);

// Hard Correctness Gate
void perform_hard_correctness_gate();

// Main driver class
class ECDumpDriver {
public:
  ECDumpDriver(const ECDumpConfig &config);
  ~ECDumpDriver();

  // Run the dump process
  void run();

private:
  ECDumpConfig config_;
  std::unique_ptr<KeyGenerator> key_gen_;

  // Initialize CUDA device
  void initDevice();

  // Process a single batch
  void processBatch(uint32_t batch_id);

  // Verify GPU results against CPU reference
  bool verifyCPU(const std::vector<secp256k1::uint256> &keys,
                 const std::vector<PointRecord> &gpu_results);

  // Write manifest file
  void writeManifest();

  // Validate configuration
  void validateConfig();
};

} // namespace ecdump

#endif // _EC_DUMP_H
