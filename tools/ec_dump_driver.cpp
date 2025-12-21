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

  // Endomorphism experiment
  unsigned int *d_beta_x = nullptr;
  unsigned int *d_lambda_x = nullptr;
  unsigned int *d_flags = nullptr;
  unsigned int *d_lambda_keys = nullptr;

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
  } else if (config_.experiment == ExperimentType::ENDOMORPHISM) {
    err = cudaMalloc(&d_beta_x, config_.batch_keys * 8 * sizeof(unsigned int));
    if (err != cudaSuccess)
      throw std::runtime_error("Failed to alloc d_beta_x");

    err =
        cudaMalloc(&d_lambda_x, config_.batch_keys * 8 * sizeof(unsigned int));
    if (err != cudaSuccess) {
      cudaFree(d_beta_x);
      throw std::runtime_error("Failed to alloc d_lambda_x");
    }

    err = cudaMalloc(&d_flags, config_.batch_keys * sizeof(unsigned int));
    if (err != cudaSuccess) {
      cudaFree(d_beta_x);
      cudaFree(d_lambda_x);
      throw std::runtime_error("Failed to alloc d_flags");
    }

    err = cudaMalloc(&d_lambda_keys,
                     config_.batch_keys * 8 * sizeof(unsigned int));
    if (err != cudaSuccess) {
      cudaFree(d_beta_x);
      cudaFree(d_lambda_x);
      cudaFree(d_flags);
      throw std::runtime_error("Failed to alloc d_lambda_keys");
    }
  } else if (config_.experiment == ExperimentType::ENDO_DIFF) {
    err =
        cudaMalloc(&d_delta_P_x, config_.batch_keys * 8 * sizeof(unsigned int));
    if (err != cudaSuccess) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      throw std::runtime_error("Failed to alloc d_delta_P_x for ENDO_DIFF");
    }
    err = cudaMalloc(&d_flags, config_.batch_keys * sizeof(uint32_t));
    if (err != cudaSuccess) {
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      cudaFree(d_delta_P_x);
      throw std::runtime_error("Failed to alloc d_flags for ENDO_DIFF");
    }
  }

  // Initialize base point table
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
    if (config_.experiment == ExperimentType::ENDOMORPHISM) {
      cudaFree(d_beta_x);
      cudaFree(d_lambda_x);
      cudaFree(d_flags);
      cudaFree(d_lambda_keys);
    }
    if (config_.experiment == ExperimentType::ENDO_DIFF) {
      cudaFree(d_delta_P_x);
      cudaFree(d_flags);
    }
    throw std::runtime_error("Failed to initialize base point table");
  }

  // Create CUDA events
  cudaEvent_t h2d_start, h2d_stop, kernel_start, kernel_stop, d2h_start,
      d2h_stop;
  cudaEventCreate(&h2d_start);
  cudaEventCreate(&h2d_stop);
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  cudaEventCreate(&d2h_start);
  cudaEventCreate(&d2h_stop);

  // Prepare Lambda Keys (if Endo)
  std::vector<unsigned int> h_lambda_keys;
  if (config_.experiment == ExperimentType::ENDOMORPHISM) {
    h_lambda_keys.resize(config_.batch_keys * 8);
    for (size_t i = 0; i < keys.size(); i++) {
      secp256k1::uint256 k_lam =
          secp256k1::multiplyModN(keys[i], secp256k1::LAMBDA);
      k_lam.exportWords(&h_lambda_keys[i * 8], 8,
                        secp256k1::uint256::LittleEndian);
    }
  }

  // H2D transfer
  cudaEventRecord(h2d_start);
  err = cudaMemcpy(d_keys, h_keys.data(),
                   config_.batch_keys * 8 * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);

  if (config_.experiment == ExperimentType::ENDOMORPHISM) {
    cudaMemcpy(d_lambda_keys, h_lambda_keys.data(),
               config_.batch_keys * 8 * sizeof(unsigned int),
               cudaMemcpyHostToDevice);
  }

  cudaEventRecord(h2d_stop);
  cudaEventSynchronize(h2d_stop);

  if (err != cudaSuccess) {
    cudaFree(d_keys);
    cudaFree(d_xCoords);
    cudaFree(d_yParity);
    if (config_.experiment == ExperimentType::ENDOMORPHISM) {
      cudaFree(d_beta_x);
      cudaFree(d_lambda_x);
      cudaFree(d_flags);
      cudaFree(d_lambda_keys);
    }
    if (config_.experiment == ExperimentType::ENDO_DIFF) {
      cudaFree(d_delta_P_x);
      cudaFree(d_flags);
    }
    freeBasePointTable(d_gxPtr, d_gyPtr);
    throw std::runtime_error("H2D transfer failed");
  }

  float h2d_ms;
  cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);

  // Launch kernel
  int blocks = 256;
  int threads = 256;

  if (config_.experiment == ExperimentType::DIFFERENTIAL) {
    if (config_.diff_config.delta_set.empty()) {
      // Cleanup logic omitted for brevity in rewrite logic, but should be here?
      // Let's assume user provides delta-set or throw before cleanup.
      // But wait, I must cleanup.
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
  } else if (config_.experiment == ExperimentType::ENDOMORPHISM) {
    err = launchEndomorphismKernel(d_keys, d_lambda_keys, d_xCoords, d_beta_x,
                                   d_lambda_x, d_flags, d_gxPtr, d_gyPtr,
                                   config_.batch_keys, blocks, threads,
                                   kernel_start, kernel_stop);
  } else if (config_.experiment == ExperimentType::ENDO_DIFF) {
    uint32_t delta = config_.diff_config.delta_set.empty()
                         ? 1
                         : config_.diff_config.delta_set[0];
    err = launchEndoDiffKernel(d_keys, delta, d_delta_P_x, d_flags, d_gxPtr,
                               d_gyPtr, config_.batch_keys, blocks, threads,
                               kernel_start, kernel_stop);
  } else {
    err = launchComputePoints(d_keys, d_xCoords, d_yParity, config_.batch_keys,
                              d_gxPtr, d_gyPtr, blocks, threads, kernel_start,
                              kernel_stop);
  }

  cudaEventSynchronize(kernel_stop);

  if (err != cudaSuccess) {
    cudaFree(d_keys);
    cudaFree(d_xCoords);
    cudaFree(d_yParity);
    if (config_.experiment == ExperimentType::ENDO_DIFF) {
      cudaFree(d_delta_P_x);
      cudaFree(d_flags);
    }
    if (config_.experiment == ExperimentType::ENDOMORPHISM) {
      cudaFree(d_beta_x);
      cudaFree(d_lambda_x);
      cudaFree(d_flags);
      cudaFree(d_lambda_keys);
    }
    freeBasePointTable(d_gxPtr, d_gyPtr);
    throw std::runtime_error(std::string("Kernel launch failed: ") +
                             cudaGetErrorString(err));
  }

  float kernel_ms;
  cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);

  // D2H transfer
  std::vector<unsigned int> h_xCoords(config_.batch_keys * 8);
  std::vector<unsigned char> h_yParity(config_.batch_keys);

  std::vector<unsigned int> h_beta_x;
  std::vector<unsigned int> h_lambda_x;
  std::vector<unsigned int> h_flags;

  if (config_.experiment == ExperimentType::ENDOMORPHISM) {
    h_beta_x.resize(config_.batch_keys * 8);
    h_lambda_x.resize(config_.batch_keys * 8);
    h_flags.resize(config_.batch_keys);
  }

  std::vector<unsigned int> h_endo_diff_x;
  std::vector<uint32_t> h_endo_diff_flags;
  if (config_.experiment == ExperimentType::ENDO_DIFF) {
    h_endo_diff_x.resize(config_.batch_keys * 8);
    h_endo_diff_flags.resize(config_.batch_keys);
  }

  // Differential host buffers will be handled after specific transfer

  cudaEventRecord(d2h_start);

  if (config_.experiment == ExperimentType::ENDO_DIFF) {
    cudaMemcpy(h_endo_diff_x.data(), d_delta_P_x,
               config_.batch_keys * 8 * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_endo_diff_flags.data(), d_flags,
               config_.batch_keys * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
  } else if (config_.experiment == ExperimentType::ENDOMORPHISM) {
    cudaMemcpy(h_xCoords.data(), d_xCoords,
               config_.batch_keys * 8 * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta_x.data(), d_beta_x,
               config_.batch_keys * 8 * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lambda_x.data(), d_lambda_x,
               config_.batch_keys * 8 * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flags.data(), d_flags,
               config_.batch_keys * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
  } else {
    err = cudaMemcpy(h_xCoords.data(), d_xCoords,
                     config_.batch_keys * 8 * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
      err = cudaMemcpy(h_yParity.data(), d_yParity,
                       config_.batch_keys * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost);
    }
  }

  cudaEventRecord(d2h_stop);
  cudaEventSynchronize(d2h_stop);

  if (err != cudaSuccess) {
    cudaFree(d_keys);
    cudaFree(d_xCoords);
    cudaFree(d_yParity);
    if (config_.experiment == ExperimentType::ENDO_DIFF) {
      cudaFree(d_delta_P_x);
      cudaFree(d_flags);
    }
    if (config_.experiment == ExperimentType::ENDOMORPHISM) {
      cudaFree(d_beta_x);
      cudaFree(d_lambda_x);
      cudaFree(d_flags);
      cudaFree(d_lambda_keys);
    }
    freeBasePointTable(d_gxPtr, d_gyPtr);
    throw std::runtime_error("D2H transfer failed");
  }

  float d2h_ms;
  cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);

  // Differential mode extra transfer
  std::vector<unsigned int> h_delta_P_x;
  std::vector<uint32_t> h_delta_P_x_mod_m1;
  std::vector<uint32_t> h_delta_P_x_mod_m2;

  if (config_.experiment == ExperimentType::DIFFERENTIAL) {
    h_delta_P_x.resize(config_.batch_keys * 8);
    h_delta_P_x_mod_m1.resize(config_.batch_keys);
    h_delta_P_x_mod_m2.resize(config_.batch_keys);

    cudaEventRecord(d2h_start); // Reuse event? fine.
    err = cudaMemcpy(h_delta_P_x.data(), d_delta_P_x,
                     config_.batch_keys * 8 * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
      err = cudaMemcpy(h_delta_P_x_mod_m1.data(), d_delta_P_x_mod_m1,
                       config_.batch_keys * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
      err = cudaMemcpy(h_delta_P_x_mod_m2.data(), d_delta_P_x_mod_m2,
                       config_.batch_keys * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);

    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);
    if (err != cudaSuccess) {
      // Cleanup...
      cudaFree(d_keys);
      cudaFree(d_xCoords);
      cudaFree(d_yParity);
      cudaFree(d_delta_P_x);
      cudaFree(d_delta_P_x_mod_m1);
      cudaFree(d_delta_P_x_mod_m2);
      freeBasePointTable(d_gxPtr, d_gyPtr);
      throw std::runtime_error("D2H differential transfer failed");
    }
  }

  // Cleanup Device Memory
  cudaFree(d_keys);
  cudaFree(d_xCoords);
  cudaFree(d_yParity);
  if (config_.experiment == ExperimentType::DIFFERENTIAL) {
    cudaFree(d_delta_P_x);
    cudaFree(d_delta_P_x_mod_m1);
    cudaFree(d_delta_P_x_mod_m2);
  }
  if (config_.experiment == ExperimentType::ENDOMORPHISM) {
    cudaFree(d_beta_x);
    cudaFree(d_lambda_x);
    cudaFree(d_flags);
    cudaFree(d_lambda_keys);
  }
  if (config_.experiment == ExperimentType::ENDO_DIFF) {
    cudaFree(d_delta_P_x);
    cudaFree(d_flags);
  }
  freeBasePointTable(d_gxPtr, d_gyPtr);

  cudaEventDestroy(h2d_start);
  cudaEventDestroy(h2d_stop);
  cudaEventDestroy(kernel_start);
  cudaEventDestroy(kernel_stop);
  cudaEventDestroy(d2h_start);
  cudaEventDestroy(d2h_stop);

  // Convert
  std::vector<PointRecord> records;
  convertGPUOutput(batch_id, h_xCoords, h_yParity, records,
                   config_.sample_rate);

  // Verify
  if (config_.verify && batch_id == 0 &&
      config_.experiment != ExperimentType::ENDOMORPHISM &&
      config_.experiment != ExperimentType::ENDO_DIFF) {
    if (!verifyCPU(keys, records)) {
      throw std::runtime_error("Verification failed!");
    }
  }

  // Write
  if (!config_.dry_run) {
    static std::unique_ptr<PointFileWriter> point_writer;
    static std::unique_ptr<TelemetryWriter> telem_writer;
    static std::unique_ptr<EndomorphismFileWriter> endo_writer;
    static std::unique_ptr<EndoDiffFileWriter> endo_diff_writer;

    if (batch_id == 0) {
      if (config_.experiment == ExperimentType::ENDOMORPHISM) {
        endo_writer = std::make_unique<EndomorphismFileWriter>(config_.out_bin);
      } else if (config_.experiment == ExperimentType::ENDO_DIFF) {
        endo_diff_writer =
            std::make_unique<EndoDiffFileWriter>(config_.out_bin);
      } else {
        point_writer = std::make_unique<PointFileWriter>(config_.out_bin);
      }
      telem_writer = std::make_unique<TelemetryWriter>(config_.telemetry_file);
    }

    auto batch_end = std::chrono::high_resolution_clock::now();
    double batch_total_ms =
        std::chrono::duration<double, std::milli>(batch_end - batch_start)
            .count();

    if (config_.experiment == ExperimentType::DIFFERENTIAL) {
      // Differential Write
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
      static DifferentialFileWriter diff_writer(config_.out_bin);
      diff_writer.writeRecords(diff_records);

      std::cout << "Batch " << batch_id << " (Diff)" << std::endl;

    } else if (config_.experiment == ExperimentType::ENDOMORPHISM) {
      std::vector<EndomorphismRecord> endo_records;
      endo_records.reserve(config_.batch_keys);
      for (uint64_t i = 0; i < config_.batch_keys; i++) {
        EndomorphismRecord rec;
        rec.batch_id = batch_id;
        rec.index_in_batch = static_cast<uint32_t>(i);
        rec.flags = h_flags[i];
        for (int j = 0; j < 8; j++) {
          uint32_t w = h_xCoords[i * 8 + j];
          rec.x[j * 4 + 0] = (w >> 24) & 0xFF;
          rec.x[j * 4 + 1] = (w >> 16) & 0xFF;
          rec.x[j * 4 + 2] = (w >> 8) & 0xFF;
          rec.x[j * 4 + 3] = w & 0xFF;
          w = h_beta_x[i * 8 + j];
          rec.beta_x[j * 4 + 0] = (w >> 24) & 0xFF;
          rec.beta_x[j * 4 + 1] = (w >> 16) & 0xFF;
          rec.beta_x[j * 4 + 2] = (w >> 8) & 0xFF;
          rec.beta_x[j * 4 + 3] = w & 0xFF;
          w = h_lambda_x[i * 8 + j];
          rec.lambda_x[j * 4 + 0] = (w >> 24) & 0xFF;
          rec.lambda_x[j * 4 + 1] = (w >> 16) & 0xFF;
          rec.lambda_x[j * 4 + 2] = (w >> 8) & 0xFF;
          rec.lambda_x[j * 4 + 3] = w & 0xFF;
        }
        endo_records.push_back(rec);
      }
      endo_writer->writeRecords(endo_records);
      std::cout << "Batch " << batch_id << " (Endo)" << std::endl;

    } else if (config_.experiment == ExperimentType::ENDO_DIFF) {
      std::vector<EndoDiffRecord> ed_records;
      uint32_t delta = config_.diff_config.delta_set.empty()
                           ? 1
                           : config_.diff_config.delta_set[0];
      convertGPUEndoDiffOutput(batch_id, delta, h_endo_diff_x,
                               h_endo_diff_flags, ed_records);
      endo_diff_writer->writeRecords(ed_records);
      std::cout << "Batch " << batch_id << " (EndoDiff)" << std::endl;
    } else {
      point_writer->writePoints(records);
      std::cout << "Batch " << batch_id << " (Baseline)" << std::endl;
    }

    // Telemetry
    BatchTelemetry telem;
    telem.timestamp = static_cast<uint64_t>(std::time(nullptr));
    telem.batch_id = batch_id;
    telem.family = config_.family;
    telem.num_keys = config_.batch_keys;
    telem.kernel_ms = kernel_ms;
    telem.h2d_ms = h2d_ms;
    telem.d2h_ms = d2h_ms;
    telem.cpu_prep_ms = cpu_prep_ms;
    telem.sampled_points = records.size();

    telem_writer->writeBatch(telem);
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
