#include "ec_dump.cuh"
#include "ec_dump.h"
#include "ec_dump_types.h"
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

// Forward declarations from ec_dump_io.cpp
namespace ecdump {
class PointFileWriter;
class TelemetryWriter;
void writeManifest(const ECDumpConfig &config, uint64_t total_keys);
void convertGPUOutput(uint32_t batch_id,
                      const std::vector<unsigned int> &xCoords,
                      const std::vector<unsigned char> &yParity,
                      std::vector<PointRecord> &records_out,
                      double sample_rate);
} // namespace ecdump

namespace ecdump {

ECDumpDriver::ECDumpDriver(const ECDumpConfig &config) : config_(config) {
  validateConfig();
  key_gen_ = createKeyGenerator(config_);
}

ECDumpDriver::~ECDumpDriver() {}

void ECDumpDriver::validateConfig() {
  // Check for huge runs without --force
  uint64_t total_keys = config_.batch_keys * config_.batches;
  if (total_keys > MAX_BATCH_KEYS_DEFAULT && !config_.force) {
    std::cerr << "ERROR: Total keys (" << total_keys
              << ") exceeds safety limit (" << MAX_BATCH_KEYS_DEFAULT << ")"
              << std::endl;
    std::cerr << "Expected output file size: ~"
              << (total_keys * POINT_RECORD_SIZE / (1024 * 1024)) << " MB"
              << std::endl;
    std::cerr << "Use --force to bypass this check" << std::endl;
    throw std::runtime_error("Safety check failed");
  }

  // Validate family-specific parameters
  if (config_.family == KeyFamily::MASKED && config_.mask_bits == 0) {
    throw std::runtime_error("--mask-bits required for masked family");
  }

  if (config_.family == KeyFamily::STRIDE && config_.stride == 0) {
    throw std::runtime_error("--stride required for stride family");
  }

  // Validate output paths
  if (!config_.dry_run && config_.out_bin.empty()) {
    throw std::runtime_error("--out-bin required (unless --dry-run)");
  }

  if (!config_.dry_run && config_.telemetry_file.empty()) {
    throw std::runtime_error("--telemetry required (unless --dry-run)");
  }

  std::cout << "Configuration validated:" << std::endl;
  std::cout << "  Family: " << familyToString(config_.family) << std::endl;
  std::cout << "  Batch keys: " << config_.batch_keys << std::endl;
  std::cout << "  Batches: " << config_.batches << std::endl;
  std::cout << "  Total keys: " << total_keys << std::endl;
  std::cout << "  Sample rate: " << config_.sample_rate << std::endl;
  std::cout << "  Device: " << config_.device << std::endl;
  std::cout << "  Verify: " << (config_.verify ? "yes" : "no") << std::endl;
  std::cout << "  Dry run: " << (config_.dry_run ? "yes" : "no") << std::endl;
  std::cout << std::endl;
}

void ECDumpDriver::initDevice() {
  int deviceCount;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get CUDA device count");
  }

  if (config_.device >= deviceCount) {
    throw std::runtime_error("Invalid device index");
  }

  err = cudaSetDevice(config_.device);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device");
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, config_.device);

  std::cout << "Using CUDA device " << config_.device << ": " << prop.name
            << std::endl;
  std::cout << "  Compute capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "  Global memory: " << (prop.totalGlobalMem / (1024 * 1024))
            << " MB" << std::endl;
  std::cout << std::endl;
}

void ECDumpDriver::processBatch(uint32_t batch_id) {
  auto batch_start = std::chrono::high_resolution_clock::now();

  // Generate keys on CPU
  auto cpu_start = std::chrono::high_resolution_clock::now();
  std::vector<secp256k1::uint256> keys;
  key_gen_->generateBatch(batch_id, config_.batch_keys, keys);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  double cpu_prep_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  // Convert keys to GPU format (8 x uint32 per key, little-endian)
  std::vector<unsigned int> h_keys(config_.batch_keys * 8);
  for (size_t i = 0; i < keys.size(); i++) {
    keys[i].exportWords(&h_keys[i * 8], 8, secp256k1::uint256::LittleEndian);
  }

  // Allocate device memory
  unsigned int *d_keys = nullptr;
  unsigned int *d_xCoords = nullptr;
  unsigned char *d_yParity = nullptr;
  unsigned int *d_gxPtr = nullptr;
  unsigned int *d_gyPtr = nullptr;

  cudaError_t err;

  // Allocate and copy keys
  err = cudaMalloc(&d_keys, config_.batch_keys * 8 * sizeof(unsigned int));
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate device memory for keys");
  }

  // Allocate output buffers
  err = cudaMalloc(&d_xCoords, config_.batch_keys * 8 * sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(d_keys);
    throw std::runtime_error(
        "Failed to allocate device memory for x coordinates");
  }

  err = cudaMalloc(&d_yParity, config_.batch_keys * sizeof(unsigned char));
  if (err != cudaSuccess) {
    cudaFree(d_keys);
    cudaFree(d_xCoords);
    throw std::runtime_error("Failed to allocate device memory for y parity");
  }

  // Differential experiment: allocate additional device memory
  unsigned int *d_delta_P_x = nullptr;
  uint32_t *d_delta_P_x_mod_m1 = nullptr;
  uint32_t *d_delta_P_x_mod_m2 = nullptr;

  if (config_.experiment == ExperimentType::DIFFERENTIAL) {
    err =
        cudaMalloc(&d_delta_P_x, config_.batch_keys * 8 * sizeof(unsigned int));
    if (err != cudaSuccess) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      throw std::runtime_error(
          "Failed to allocate device memory for delta_P_x");
    }

    err =
        cudaMalloc(&d_delta_P_x_mod_m1, config_.batch_keys * sizeof(uint32_t));
    if (err != cudaSuccess) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      cudaFree(d_delta_P_x);
      throw std::runtime_error("Failed to allocate device memory for mod_m1");
    }

    err =
        cudaMalloc(&d_delta_P_x_mod_m2, config_.batch_keys * sizeof(uint32_t));
    if (err != cudaSuccess) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      cudaFree(d_delta_P_x);
      cudaFree(d_delta_P_x_mod_m1);
      throw std::runtime_error("Failed to allocate device memory for mod_m2");
    }
  }

  // Initialize base point table (only once per driver, but for simplicity doing
  // per batch)
  err = initBasePointTable(&d_gxPtr, &d_gyPtr);
  if (err != cudaSuccess) {
    cudaFree(d_keys);
    cudaFree(d_xCoords);
    cudaFree(d_yParity);
    if (config_.experiment == ExperimentType::DIFFERENTIAL) {
      cudaFree(d_delta_P_x);
      cudaFree(d_delta_P_x_mod_m1);
      cudaFree(d_delta_P_x_mod_m2);
    }
    throw std::runtime_error("Failed to initialize base point table");
  }

  // Create CUDA events for timing
  cudaEvent_t h2d_start, h2d_stop, kernel_start, kernel_stop, d2h_start,
      d2h_stop;
  cudaEventCreate(&h2d_start);
  cudaEventCreate(&h2d_stop);
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  cudaEventCreate(&d2h_start);
  cudaEventCreate(&d2h_stop);

  // H2D transfer
  cudaEventRecord(h2d_start);
  err = cudaMemcpy(d_keys, h_keys.data(),
                   config_.batch_keys * 8 * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
  cudaEventRecord(h2d_stop);
  cudaEventSynchronize(h2d_stop);

  if (err != cudaSuccess) {
    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_xCoords);
    cudaFree(d_yParity);
    freeBasePointTable(d_gxPtr, d_gyPtr);
    throw std::runtime_error("H2D transfer failed");
  }

  float h2d_ms;
  cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);

  // Launch kernel (differential or baseline mode)
  int blocks = 256;
  int threads = 256;

  if (config_.experiment == ExperimentType::DIFFERENTIAL) {
    // Differential mode: validate delta_set and launch differential kernel
    if (config_.diff_config.delta_set.empty()) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      cudaFree(d_delta_P_x);
      cudaFree(d_delta_P_x_mod_m1);
      cudaFree(d_delta_P_x_mod_m2);
      freeBasePointTable(d_gxPtr, d_gyPtr);
      throw std::runtime_error("Differential experiment requires --delta-set");
    }

    uint32_t delta = config_.diff_config.delta_set[0];
    uint8_t family_id = static_cast<uint8_t>(config_.family);
    uint8_t param = (config_.family == KeyFamily::MASKED)
                        ? static_cast<uint8_t>(config_.mask_bits)
                    : (config_.family == KeyFamily::STRIDE)
                        ? static_cast<uint8_t>(config_.stride & 0xFF)
                        : 0;

    err = launchDifferentialKernel(
        d_keys, delta, d_delta_P_x, d_delta_P_x_mod_m1, d_delta_P_x_mod_m2,
        batch_id, family_id, param, config_.batch_keys, d_gxPtr, d_gyPtr,
        blocks, threads, kernel_start, kernel_stop);
  } else {
    // Baseline mode: launch standard point computation kernel
    err = launchComputePoints(d_keys, d_xCoords, d_yParity, config_.batch_keys,
                              d_gxPtr, d_gyPtr, blocks, threads, kernel_start,
                              kernel_stop);
  }

  cudaEventSynchronize(kernel_stop);

  if (err != cudaSuccess) {
    cudaFree(d_keys);
    cudaFree(d_xCoords);
    cudaFree(d_yParity);
    freeBasePointTable(d_gxPtr, d_gyPtr);
    throw std::runtime_error(std::string("Kernel launch failed: ") +
                             cudaGetErrorString(err));
  }

  float kernel_ms;
  cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);

  // D2H transfer
  std::vector<unsigned int> h_xCoords(config_.batch_keys * 8);
  std::vector<unsigned char> h_yParity(config_.batch_keys);

  cudaEventRecord(d2h_start);
  err = cudaMemcpy(h_xCoords.data(), d_xCoords,
                   config_.batch_keys * 8 * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(h_yParity.data(), d_yParity,
                     config_.batch_keys * sizeof(unsigned char),
                     cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(d2h_stop);
  cudaEventSynchronize(d2h_stop);

  if (err != cudaSuccess) {
    cudaFree(d_keys);
    cudaFree(d_xCoords);
    cudaFree(d_yParity);
    freeBasePointTable(d_gxPtr, d_gyPtr);
    throw std::runtime_error("D2H transfer failed");
  }

  float d2h_ms;
  cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);

  // Differential mode: transfer differential results
  std::vector<unsigned int> h_delta_P_x;
  std::vector<uint32_t> h_delta_P_x_mod_m1;
  std::vector<uint32_t> h_delta_P_x_mod_m2;

  if (config_.experiment == ExperimentType::DIFFERENTIAL) {
    h_delta_P_x.resize(config_.batch_keys * 8);
    h_delta_P_x_mod_m1.resize(config_.batch_keys);
    h_delta_P_x_mod_m2.resize(config_.batch_keys);

    cudaEventRecord(d2h_start);

    err = cudaMemcpy(h_delta_P_x.data(), d_delta_P_x,
                     config_.batch_keys * 8 * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      cudaFree(d_delta_P_x);
      cudaFree(d_delta_P_x_mod_m1);
      cudaFree(d_delta_P_x_mod_m2);
      freeBasePointTable(d_gxPtr, d_gyPtr);
      throw std::runtime_error("D2H transfer failed for delta_P_x");
    }

    err = cudaMemcpy(h_delta_P_x_mod_m1.data(), d_delta_P_x_mod_m1,
                     config_.batch_keys * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      cudaFree(d_delta_P_x);
      cudaFree(d_delta_P_x_mod_m1);
      cudaFree(d_delta_P_x_mod_m2);
      freeBasePointTable(d_gxPtr, d_gyPtr);
      throw std::runtime_error("D2H transfer failed for mod_m1");
    }

    err = cudaMemcpy(h_delta_P_x_mod_m2.data(), d_delta_P_x_mod_m2,
                     config_.batch_keys * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      cudaFree(d_delta_P_x);
      cudaFree(d_delta_P_x_mod_m1);
      cudaFree(d_delta_P_x_mod_m2);
      freeBasePointTable(d_gxPtr, d_gyPtr);
      throw std::runtime_error("D2H transfer failed for mod_m2");
    }

    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);
  }

  // Cleanup device memory
  cudaFree(d_keys);
  cudaFree(d_xCoords);
  cudaFree(d_yParity);
  if (config_.experiment == ExperimentType::DIFFERENTIAL) {
    cudaFree(d_delta_P_x);
    cudaFree(d_delta_P_x_mod_m1);
    cudaFree(d_delta_P_x_mod_m2);
  }
  freeBasePointTable(d_gxPtr, d_gyPtr);

  cudaEventDestroy(h2d_start);
  cudaEventDestroy(h2d_stop);
  cudaEventDestroy(kernel_start);
  cudaEventDestroy(kernel_stop);
  cudaEventDestroy(d2h_start);
  cudaEventDestroy(d2h_stop);

  // Convert GPU output to point records
  std::vector<PointRecord> records;
  convertGPUOutput(batch_id, h_xCoords, h_yParity, records,
                   config_.sample_rate);

  // Verify if requested (first batch only)
  if (config_.verify && batch_id == 0) {
    if (!verifyCPU(keys, records)) {
      throw std::runtime_error("Verification failed!");
    }
  }

  // Write to file (unless dry run)
  if (!config_.dry_run) {
    // This is inefficient - we should keep the file open across batches
    // For now, append mode would be better, but let's keep it simple
    static std::unique_ptr<PointFileWriter> point_writer;
    static std::unique_ptr<TelemetryWriter> telem_writer;

    if (batch_id == 0) {
      point_writer = std::make_unique<PointFileWriter>(config_.out_bin);
      telem_writer = std::make_unique<TelemetryWriter>(config_.telemetry_file);
    }
    // Write output (baseline or differential mode)
    auto batch_end = std::chrono::high_resolution_clock::now();
    double batch_total_ms =
        std::chrono::duration<double, std::milli>(batch_end - batch_start)
            .count();

    if (config_.experiment == ExperimentType::DIFFERENTIAL) {
      // Differential mode: convert and write differential records
      std::vector<DifferentialRecord> diff_records;
      diff_records.reserve(config_.batch_keys);

      uint32_t delta = config_.diff_config.delta_set[0];
      uint8_t family_id = static_cast<uint8_t>(config_.family);
      uint8_t param = (config_.family == KeyFamily::MASKED)
                          ? static_cast<uint8_t>(config_.mask_bits)
                      : (config_.family == KeyFamily::STRIDE)
                          ? static_cast<uint8_t>(config_.stride & 0xFF)
                          : 0;

      for (uint64_t i = 0; i < config_.batch_keys; i++) {
        DifferentialRecord rec;

        // Copy delta_P_x (32 bytes, convert to big-endian byte array)
        for (int j = 0; j < 8; j++) {
          uint32_t word = h_delta_P_x[i * 8 + j];
          rec.delta_P_x[j * 4 + 0] = (word >> 24) & 0xFF;
          rec.delta_P_x[j * 4 + 1] = (word >> 16) & 0xFF;
          rec.delta_P_x[j * 4 + 2] = (word >> 8) & 0xFF;
          rec.delta_P_x[j * 4 + 3] = word & 0xFF;
        }

        rec.delta_P_x_mod_m1 = h_delta_P_x_mod_m1[i];
        rec.delta_P_x_mod_m2 = h_delta_P_x_mod_m2[i];
        rec.delta_value = delta;
        rec.scalar_family_id = family_id;
        rec.mask_bits_or_stride = param;
        rec.batch_id = batch_id;
        rec.index_in_batch = static_cast<uint32_t>(i);
        memset(rec.reserved, 0, 6);

        diff_records.push_back(rec);
      }

      // Write differential records
      if (!config_.dry_run) {
        static DifferentialFileWriter diff_writer(config_.out_bin);
        diff_writer.writeRecords(diff_records);
      }

      std::cout << "Batch " << batch_id << ": " << config_.batch_keys
                << " differential records in " << std::fixed
                << std::setprecision(2) << batch_total_ms << " ms ("
                << std::setprecision(0)
                << (config_.batch_keys / (batch_total_ms / 1000.0))
                << " keys/sec, kernel: " << std::fixed << std::setprecision(2)
                << kernel_ms << " ms)" << std::endl;
    } else {
      // Baseline mode: convert and write point records
      // The records vector is already populated by convertGPUOutput above
      // std::vector<PointRecord> records;
      // convertGPUOutput(batch_id, h_xCoords, h_yParity, records,
      //                  config_.sample_rate);

      // Write output
      if (!config_.dry_run) {
        static PointFileWriter point_writer(config_.out_bin);
        point_writer.writePoints(records);
      }

      std::cout << "Batch " << batch_id << ": " << config_.batch_keys
                << " keys in " << std::fixed << std::setprecision(2)
                << batch_total_ms << " ms (" << std::setprecision(0)
                << (config_.batch_keys / (batch_total_ms / 1000.0))
                << " keys/sec, kernel: " << std::fixed << std::setprecision(2)
                << kernel_ms << " ms, sampled: " << records.size() << " points)"
                << std::endl;
    }

    // Emit telemetry
    BatchTelemetry telem;
    telem.timestamp = static_cast<uint64_t>(std::time(nullptr));
    telem.batch_id = batch_id;
    telem.family = config_.family;
    telem.start_k =
        "0x" +
        key_gen_->getBatchStartKey(batch_id, config_.batch_keys).toString(16);
    telem.mask_bits = config_.mask_bits;
    telem.stride = config_.stride;
    telem.num_keys = config_.batch_keys;
    telem.kernel_ms = kernel_ms;
    telem.h2d_ms = h2d_ms;
    telem.d2h_ms = d2h_ms;
    telem.cpu_prep_ms = cpu_prep_ms;
    telem.cpu_wait_gpu_ms = 0.0; // Not implemented yet
    telem.sampled_points = records.size();
    telem.matches = 0;

    if (!config_.dry_run) {
      static std::unique_ptr<TelemetryWriter> telem_writer;
      if (batch_id == 0) {
        telem_writer =
            std::make_unique<TelemetryWriter>(config_.telemetry_file);
      }
      telem_writer->writeBatch(telem);
    }
  }
}

void ECDumpDriver::writeManifest() {
  uint64_t total_keys = config_.batch_keys * config_.batches;
  ecdump::writeManifest(config_, total_keys);
}

void ECDumpDriver::run() {
  std::cout << "=== BitCrack EC Dump ===" << std::endl;
  std::cout << std::endl;

  initDevice();

  // Hard Correctness Gate - Unskippable
  perform_hard_correctness_gate();
  std::cout << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (uint32_t i = 0; i < config_.batches; i++) {
    processBatch(i);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double total_sec =
      std::chrono::duration<double>(end_time - start_time).count();

  uint64_t total_keys = config_.batch_keys * config_.batches;
  double avg_keys_per_sec = total_keys / total_sec;

  std::cout << std::endl;
  std::cout << "=== Summary ===" << std::endl;
  std::cout << "Total keys processed: " << total_keys << std::endl;
  std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_sec
            << " seconds" << std::endl;
  std::cout << "Average throughput: " << std::fixed << std::setprecision(0)
            << avg_keys_per_sec << " keys/sec" << std::endl;

  if (!config_.dry_run) {
    writeManifest();
    std::cout << "Output written to: " << config_.out_bin << std::endl;
    std::cout << "Telemetry written to: " << config_.telemetry_file
              << std::endl;
    std::cout << "Manifest written to: manifest.json" << std::endl;
  }
}

} // namespace ecdump
