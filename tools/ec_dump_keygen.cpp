#include "CryptoUtil.h"
#include "ec_dump.h"
#include <ctime>
#include <random>
#include <stdexcept>

namespace ecdump {

// Control family: uniform random in [1, n-1]
class ControlKeyGenerator : public KeyGenerator {
public:
  ControlKeyGenerator(uint64_t seed) {
    if (seed == 0) {
      seed = static_cast<uint64_t>(std::time(nullptr));
    }
    rng_.seed(seed);
  }

  void generateBatch(uint32_t batch_id, uint64_t batch_size,
                     std::vector<secp256k1::uint256> &keys_out) override {
    keys_out.resize(batch_size);

    // Generate random 256-bit values and reduce mod n
    for (uint64_t i = 0; i < batch_size; i++) {
      // Generate 4 random 64-bit values
      uint64_t r0 = rng_();
      uint64_t r1 = rng_();
      uint64_t r2 = rng_();
      uint64_t r3 = rng_();

      // Construct 256-bit value
      unsigned int words[8];
      words[0] = static_cast<uint32_t>(r0);
      words[1] = static_cast<uint32_t>(r0 >> 32);
      words[2] = static_cast<uint32_t>(r1);
      words[3] = static_cast<uint32_t>(r1 >> 32);
      words[4] = static_cast<uint32_t>(r2);
      words[5] = static_cast<uint32_t>(r2 >> 32);
      words[6] = static_cast<uint32_t>(r3);
      words[7] = static_cast<uint32_t>(r3 >> 32);

      secp256k1::uint256 key(words);

      // Reduce mod n (group order)
      while (!(key < secp256k1::N)) {
        key = key - secp256k1::N;
      }

      // Ensure key is not zero
      if (key.isZero()) {
        key = secp256k1::uint256(1);
      }

      keys_out[i] = key;
    }
  }

  secp256k1::uint256 getBatchStartKey(uint32_t batch_id,
                                      uint64_t batch_size) override {
    // For random keys, return 0 as placeholder
    return secp256k1::uint256(0);
  }

private:
  std::mt19937_64 rng_;
};

// Consecutive family: start_k + i
class ConsecutiveKeyGenerator : public KeyGenerator {
public:
  ConsecutiveKeyGenerator(const secp256k1::uint256 &start_k)
      : start_k_(start_k) {}

  void generateBatch(uint32_t batch_id, uint64_t batch_size,
                     std::vector<secp256k1::uint256> &keys_out) override {
    keys_out.resize(batch_size);

    secp256k1::uint256 offset = start_k_ + (batch_id * batch_size);

    for (uint64_t i = 0; i < batch_size; i++) {
      keys_out[i] = offset + i;

      // Check for overflow past n
      if (!(keys_out[i] < secp256k1::N)) {
        throw std::runtime_error(
            "Consecutive key generation exceeded group order");
      }
    }
  }

  secp256k1::uint256 getBatchStartKey(uint32_t batch_id,
                                      uint64_t batch_size) override {
    return start_k_ + (batch_id * batch_size);
  }

private:
  secp256k1::uint256 start_k_;
};

// Masked family: (fixed_high) | (random & mask)
class MaskedKeyGenerator : public KeyGenerator {
public:
  MaskedKeyGenerator(const secp256k1::uint256 &start_k, uint32_t mask_bits,
                     uint64_t seed)
      : fixed_high_(start_k), mask_bits_(mask_bits) {

    if (mask_bits > 256) {
      throw std::runtime_error("Mask bits must be <= 256");
    }

    if (seed == 0) {
      seed = static_cast<uint64_t>(std::time(nullptr));
    }
    rng_.seed(seed);
  }

  void generateBatch(uint32_t batch_id, uint64_t batch_size,
                     std::vector<secp256k1::uint256> &keys_out) override {
    keys_out.resize(batch_size);

    for (uint64_t i = 0; i < batch_size; i++) {
      // Generate random 256-bit value
      uint64_t r0 = rng_();
      uint64_t r1 = rng_();
      uint64_t r2 = rng_();
      uint64_t r3 = rng_();

      unsigned int words[8];
      words[0] = static_cast<uint32_t>(r0);
      words[1] = static_cast<uint32_t>(r0 >> 32);
      words[2] = static_cast<uint32_t>(r1);
      words[3] = static_cast<uint32_t>(r1 >> 32);
      words[4] = static_cast<uint32_t>(r2);
      words[5] = static_cast<uint32_t>(r2 >> 32);
      words[6] = static_cast<uint32_t>(r3);
      words[7] = static_cast<uint32_t>(r3 >> 32);

      // Apply Mask
      for (int b = 0; b < 256; b++) {
        if (b >= mask_bits_) {
          int word_idx = b / 32; // 0 is LSB word in this array layout?
          // ControlKeyGenerator uses words[0] as LSB parts of r0?
          // r0 is rng_().
          // words[0] = r0 & 0xFFFFFFFF.
          // words[1] = r0 >> 32.
          // So words[0] is bits 0-31. words[1] 32-63.
          // Correct.
          int bit_idx = b % 32;
          words[word_idx] &= ~(1U << bit_idx);
        }
      }

      secp256k1::uint256 key(words);

      // OR with fixed_high
      // Assuming fixed_high has 0s in mask region (or we just OR it)
      key = key + fixed_high_; // + or | ? User said `random & mask`. fixed_high
                               // is header.

      if (!(key < secp256k1::N)) {
        // Retry or just clamp? Masked random should be fine usually if high
        // bits are 0. If high bits are set, might overflow.
        while (!(key < secp256k1::N))
          key = key - secp256k1::N;
      }
      if (key.isZero())
        key = secp256k1::uint256(1);

      keys_out[i] = key;
    }
  }

  secp256k1::uint256 getBatchStartKey(uint32_t batch_id,
                                      uint64_t batch_size) override {
    return fixed_high_;
  }

private:
  secp256k1::uint256 fixed_high_;
  uint32_t mask_bits_;
  std::mt19937_64 rng_;
};

// Stride family: start_k + i * stride
class StrideKeyGenerator : public KeyGenerator {
public:
  StrideKeyGenerator(const secp256k1::uint256 &start_k, uint64_t stride)
      : start_k_(start_k), stride_(stride) {}

  void generateBatch(uint32_t batch_id, uint64_t batch_size,
                     std::vector<secp256k1::uint256> &keys_out) override {
    keys_out.resize(batch_size);

    secp256k1::uint256 base = start_k_ + (batch_id * batch_size * stride_);

    for (uint64_t i = 0; i < batch_size; i++) {
      keys_out[i] = base + (i * stride_);

      if (!(keys_out[i] < secp256k1::N)) {
        throw std::runtime_error("Stride key generation exceeded group order");
      }
    }
  }

  secp256k1::uint256 getBatchStartKey(uint32_t batch_id,
                                      uint64_t batch_size) override {
    return start_k_ + (batch_id * batch_size * stride_);
  }

private:
  secp256k1::uint256 start_k_;
  uint64_t stride_;
};

// HD family: H(seed || counter) mod n
class HDKeyGenerator : public KeyGenerator {
public:
  HDKeyGenerator(uint64_t seed) : seed_(seed) {}

  void generateBatch(uint32_t batch_id, uint64_t batch_size,
                     std::vector<secp256k1::uint256> &keys_out) override {
    keys_out.resize(batch_size);

    uint64_t base_counter = batch_id * batch_size;

    for (uint64_t i = 0; i < batch_size; i++) {
      uint64_t counter = base_counter + i;

      // Construct input: seed || counter (16 bytes total)
      // Format as 4 x uint32 words (16 bytes) for crypto::sha256
      unsigned int msg[16] = {0};

      // Pack seed and counter into first 4 words (16 bytes)
      // Little-endian packing
      msg[0] = static_cast<unsigned int>(seed_ & 0xFFFFFFFF);
      msg[1] = static_cast<unsigned int>((seed_ >> 32) & 0xFFFFFFFF);
      msg[2] = static_cast<unsigned int>(counter & 0xFFFFFFFF);
      msg[3] = static_cast<unsigned int>((counter >> 32) & 0xFFFFFFFF);

      // Set padding and length for SHA-256 (16 bytes = 128 bits)
      msg[4] = 0x80000000; // Padding bit
      msg[15] = 128;       // Message length in bits

      // Hash with SHA-256
      unsigned int sha256Digest[8];
      crypto::sha256Init(sha256Digest);
      crypto::sha256(msg, sha256Digest);

      // Convert to uint256 (little-endian from digest)
      unsigned int words[8];
      for (int j = 0; j < 8; j++) {
        words[j] = sha256Digest[j];
      }

      secp256k1::uint256 key(words);

      // Reduce mod n
      while (!(key < secp256k1::N)) {
        key = key - secp256k1::N;
      }

      // Ensure not zero
      if (key.isZero()) {
        key = secp256k1::uint256(1);
      }

      keys_out[i] = key;
    }
  }

  secp256k1::uint256 getBatchStartKey(uint32_t batch_id,
                                      uint64_t batch_size) override {
    // For HD keys, return seed as placeholder
    return secp256k1::uint256(seed_);
  }

private:
  uint64_t seed_;
};

// Factory function
std::unique_ptr<KeyGenerator> createKeyGenerator(const ECDumpConfig &config) {
  switch (config.family) {
  case KeyFamily::CONTROL:
    return std::make_unique<ControlKeyGenerator>(config.seed);

  case KeyFamily::CONSECUTIVE:
    return std::make_unique<ConsecutiveKeyGenerator>(config.start_k);

  case KeyFamily::MASKED:
    return std::make_unique<MaskedKeyGenerator>(config.start_k,
                                                config.mask_bits, config.seed);

  case KeyFamily::STRIDE:
    return std::make_unique<StrideKeyGenerator>(config.start_k, config.stride);

  case KeyFamily::HD:
    return std::make_unique<HDKeyGenerator>(config.seed);

  default:
    throw std::runtime_error("Unknown key family");
  }
}

} // namespace ecdump
