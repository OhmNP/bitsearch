#include "CudaKeySearchDevice.h"
#include "AddressUtil.h"
#include "Logger.h"
#include "cudabridge.h"
#include "util.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

void CudaKeySearchDevice::cudaCall(cudaError_t err) {
  if (err) {
    std::string errStr = cudaGetErrorString(err);

    throw KeySearchException(errStr);
  }
}

CudaKeySearchDevice::CudaKeySearchDevice(int device, int threads,
                                         int pointsPerThread, int blocks) {
  cuda::CudaDeviceInfo info;
  try {
    info = cuda::getDeviceInfo(device);
    _deviceName = info.name;
  } catch (cuda::CudaException ex) {
    throw KeySearchException(ex.msg);
  }

  if (threads <= 0 || threads % 32 != 0) {
    throw KeySearchException("The number of threads must be a multiple of 32");
  }

  if (pointsPerThread <= 0) {
    throw KeySearchException("At least 1 point per thread required");
  }

  // Specifying blocks on the commandline is depcreated but still supported. If
  // there is no value for blocks, devide the threads evenly among the
  // multi-processors
  if (blocks == 0) {
    if (threads % info.mpCount != 0) {
      throw KeySearchException("The number of threads must be a multiple of " +
                               util::format("%d", info.mpCount));
    }

    _threads = threads / info.mpCount;

    _blocks = info.mpCount;

    while (_threads > 512) {
      _threads /= 2;
      _blocks *= 2;
    }
  } else {
    _threads = threads;
    _blocks = blocks;
  }

  _iterations = 0;

  _device = device;

  _pointsPerThread = pointsPerThread;

  cudaCall(cudaSetDevice(_device));
  cudaCall(cudaEventCreate(&_startEvent));
  cudaCall(cudaEventCreate(&_stopEvent));
  cudaCall(cudaEventCreate(&_memStartEvent));
  cudaCall(cudaEventCreate(&_memStopEvent));
  cudaCall(cudaStreamCreate(&_stream));
}

void CudaKeySearchDevice::init(const secp256k1::uint256 &start, int compression,
                               const secp256k1::uint256 &stride) {
  if (start.cmp(secp256k1::N) >= 0) {
    throw KeySearchException("Starting key is out of range");
  }

  _startExponent = start;

  _compression = compression;

  _stride = stride;

  cudaCall(cudaSetDevice(_device));

  // Block on kernel calls
  // cudaCall(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)); // Removed
  // for async

  // Use a larger portion of shared memory for L1 cache
  cudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

  generateStartingPoints();

  cudaCall(allocateChainBuf(_threads * _blocks * _pointsPerThread));

  // Set the incrementor
  secp256k1::ecpoint g = secp256k1::G();
  secp256k1::ecpoint p = secp256k1::multiplyPoint(
      secp256k1::uint256((uint64_t)_threads * _blocks * _pointsPerThread) *
          _stride,
      g);

  cudaCall(_resultList.init(sizeof(CudaDeviceResult), 16));

  cudaCall(setIncrementorPoint(p.x, p.y));
}

void CudaKeySearchDevice::generateStartingPoints() {
  uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;
  uint64_t totalMemory = totalPoints * 40;

  std::vector<secp256k1::uint256> exponents;

  Logger::log(
      LogLevel::Info,
      "Generating " + util::formatThousands(totalPoints) +
          " starting points (" +
          util::format("%.1f", (double)totalMemory / (double)(1024 * 1024)) +
          "MB)");

  // Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
  secp256k1::uint256 privKey = _startExponent;

  exponents.push_back(privKey);

  for (uint64_t i = 1; i < totalPoints; i++) {
    privKey = privKey.add(_stride);
    exponents.push_back(privKey);
  }

  cudaCall(_deviceKeys.init(_blocks, _threads, _pointsPerThread, exponents));

  // Show progress in 10% increments
  double pct = 10.0;
  for (int i = 1; i <= 256; i++) {
    cudaCall(_deviceKeys.doStep());

    if (((double)i / 256.0) * 100.0 >= pct) {
      Logger::log(LogLevel::Info, util::format("%.1f%%", pct));
      pct += 10.0;
    }
  }

  Logger::log(LogLevel::Info, "Done");

  _deviceKeys.clearPrivateKeys();
}

void CudaKeySearchDevice::setTargets(const std::set<KeySearchTarget> &targets) {
  _targets.clear();

  for (std::set<KeySearchTarget>::iterator i = targets.begin();
       i != targets.end(); ++i) {
    hash160 h(i->value);
    _targets.push_back(h);
  }

  cudaCall(_targetLookup.setTargets(_targets));
}

void CudaKeySearchDevice::doStep() {
  uint64_t numKeys = (uint64_t)_blocks * _threads * _pointsPerThread;

  // Step 2: CPU Setup Timing (Real)
  auto startSetup = std::chrono::high_resolution_clock::now();

  // Real CPU work:
  // 1. Calculate reset flag
  bool reset = (_iterations < 2 && _startExponent.cmp(numKeys) <= 0);
  // 2. Any other CPU-side prep would be here (none for this kernel)

  auto endSetup = std::chrono::high_resolution_clock::now();
  _cpuSetupTime =
      std::chrono::duration<double, std::milli>(endSetup - startSetup).count();

  try {
    cudaCall(cudaEventRecord(_startEvent, _stream));
    callKeyFinderKernel(_blocks, _threads, _pointsPerThread, reset,
                        _compression);
    cudaCall(cudaEventRecord(_stopEvent, _stream));
  } catch (cuda::CudaException ex) {
    throw KeySearchException(ex.msg);
  }

  getResultsInternal();

  _iterations++;
}

uint64_t CudaKeySearchDevice::keysPerStep() {
  return (uint64_t)_blocks * _threads * _pointsPerThread;
}

std::string CudaKeySearchDevice::getDeviceName() { return _deviceName; }

void CudaKeySearchDevice::getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem) {
  cudaCall(cudaMemGetInfo(&freeMem, &totalMem));
}

void CudaKeySearchDevice::removeTargetFromList(const unsigned int hash[5]) {
  size_t count = _targets.size();

  while (count) {
    if (memcmp(hash, _targets[count - 1].h, 20) == 0) {
      _targets.erase(_targets.begin() + count - 1);
      return;
    }
    count--;
  }
}

bool CudaKeySearchDevice::isTargetInList(const unsigned int hash[5]) {
  size_t count = _targets.size();

  while (count) {
    if (memcmp(hash, _targets[count - 1].h, 20) == 0) {
      return true;
    }
    count--;
  }

  return false;
}

uint32_t CudaKeySearchDevice::getPrivateKeyOffset(int thread, int block,
                                                  int idx) {
  // Total number of threads
  int totalThreads = _blocks * _threads;

  int base = idx * totalThreads;

  // Global ID of the current thread
  int threadId = block * _threads + thread;

  return base + threadId;
}

void CudaKeySearchDevice::getResultsInternal() {
  // Step 1: Measure CPU wait time (blocking on GPU completion)
  auto startWait = std::chrono::high_resolution_clock::now();
  cudaCall(cudaEventSynchronize(_stopEvent));
  auto endWait = std::chrono::high_resolution_clock::now();
  double cpuWaitMs =
      std::chrono::duration<double, std::milli>(endWait - startWait).count();

  // Step 2: Measure kernel time (GPU) - must be after synchronize
  float kernelMs = 0;
  cudaCall(cudaEventElapsedTime(&kernelMs, _startEvent, _stopEvent));

  // Step 3: Measure zero-copy read cost (CPU consuming GPU data)
  auto startD2H = std::chrono::high_resolution_clock::now();
  int count = _resultList.size();
  auto endD2H = std::chrono::high_resolution_clock::now();
  double d2hMs =
      std::chrono::duration<double, std::milli>(endD2H - startD2H).count();

  TelemetryData t;
  t.kernelTimeMs = kernelMs;
  t.keysSearched = keysPerStep();
  t.batchId = _iterations;
  t.cpuSetupTimeMs = _cpuSetupTime;
  t.cpuValidationTimeMs = 0;
  t.matches = 0;
  t.gpuIdleTimeMs = 0; // Removed
  t.cpuWaitTimeMs = cpuWaitMs;
  t.deviceToHostMs = d2hMs;

  // Step 4: Hard Assertions
  if (t.keysSearched == 0) {
    Logger::log(LogLevel::Error, "ASSERTION FAILED: keysSearched == 0");
    exit(1);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    Logger::log(LogLevel::Error,
                std::string("ASSERTION FAILED: cudaGetLastError: ") +
                    cudaGetErrorString(err));
    exit(1);
  }

  float memMs = 0;

  if (count > 0) {
    unsigned char *ptr = new unsigned char[count * sizeof(CudaDeviceResult)];

    cudaCall(cudaEventRecord(_memStartEvent,
                             _stream)); // Keep for timeline completeness?

    auto startPayload = std::chrono::high_resolution_clock::now();
    _resultList.read(ptr, count);
    auto endPayload = std::chrono::high_resolution_clock::now();

    cudaCall(cudaEventRecord(_memStopEvent, _stream));

    // Mapped memory read is sync on CPU, so we measure it directly
    double payloadMs =
        std::chrono::duration<double, std::milli>(endPayload - startPayload)
            .count();
    t.deviceToHostMs += payloadMs;

    // CPU Idle/Wait was already captured in Step 2.
    // cpuWaitTimeMs here was for memStopEvent previously, which is redundant if
    // we sync'd earlier. However, read() *might* wait if we didn't sync
    // properly? We did.

    double transferBytes = count * sizeof(CudaDeviceResult);
    if (memMs > 0) {
      t.bandwidthGBs =
          (transferBytes / (1024.0 * 1024.0 * 1024.0)) / (memMs / 1000.0);
    }

    int actualCount = 0;
    auto startVal = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < count; i++) {
      struct CudaDeviceResult *rPtr = &((struct CudaDeviceResult *)ptr)[i];

      if (!isTargetInList(rPtr->digest)) {
        continue;
      }
      actualCount++;

      KeySearchResult minerResult;
      secp256k1::uint256 offset =
          (secp256k1::uint256((uint64_t)_blocks * _threads * _pointsPerThread *
                              _iterations) +
           secp256k1::uint256(
               getPrivateKeyOffset(rPtr->thread, rPtr->block, rPtr->idx))) *
          _stride;
      secp256k1::uint256 privateKey =
          secp256k1::addModN(_startExponent, offset);

      minerResult.privateKey = privateKey;
      minerResult.compressed = rPtr->compressed;
      memcpy(minerResult.hash, rPtr->digest, 20);
      minerResult.publicKey = secp256k1::ecpoint(
          secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian),
          secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian));

      removeTargetFromList(rPtr->digest);
      _results.push_back(minerResult);
    }

    auto endVal = std::chrono::high_resolution_clock::now();
    t.cpuValidationTimeMs =
        std::chrono::duration<double, std::milli>(endVal - startVal).count();
    t.matches = actualCount;

    delete[] ptr;
    _resultList.clear();

    if (actualCount) {
      cudaCall(_targetLookup.setTargets(_targets));
    }
  } else {
    // If no results, blocking sync on stopEvent is already done for kernel
    // timing We can assume wait time is negligible or captured in sync
    t.cpuWaitTimeMs = 0;
    t.deviceToHostMs = 0;
    t.bandwidthGBs = 0;
  }

  t.occupancy = 1.0;

  // Step 5: Output EXACT Telemetry (JSON) for debug
  printf("{\n");
  printf("  \"batch\": %llu,\n", t.batchId);
  printf("  \"num_keys\": %llu,\n", t.keysSearched);
  printf("  \"cpu_scalar_gen_ms\": %f,\n", t.cpuSetupTimeMs);
  printf("  \"kernel_ms\": %f\n", t.kernelTimeMs);
  printf("}\n");
  fflush(stdout);

  Telemetry::getInstance().update(t);
}

// Verify a private key produces the public key and hash
bool CudaKeySearchDevice::verifyKey(const secp256k1::uint256 &privateKey,
                                    const secp256k1::ecpoint &publicKey,
                                    const unsigned int hash[5],
                                    bool compressed) {
  secp256k1::ecpoint g = secp256k1::G();

  secp256k1::ecpoint p = secp256k1::multiplyPoint(privateKey, g);

  if (!(p == publicKey)) {
    return false;
  }

  unsigned int xWords[8];
  unsigned int yWords[8];

  p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
  p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

  unsigned int digest[5];
  if (compressed) {
    Hash::hashPublicKeyCompressed(xWords, yWords, digest);
  } else {
    Hash::hashPublicKey(xWords, yWords, digest);
  }

  for (int i = 0; i < 5; i++) {
    if (digest[i] != hash[i]) {
      return false;
    }
  }

  return true;
}

size_t
CudaKeySearchDevice::getResults(std::vector<KeySearchResult> &resultsOut) {
  for (int i = 0; i < _results.size(); i++) {
    resultsOut.push_back(_results[i]);
  }
  _results.clear();

  return resultsOut.size();
}

secp256k1::uint256 CudaKeySearchDevice::getNextKey() {
  uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;

  return _startExponent +
         secp256k1::uint256(totalPoints) * _iterations * _stride;
}