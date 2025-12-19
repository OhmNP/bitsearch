#include "ec_dump.h"
#include "ec_dump_types.h"
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ecdump {

// PointFileWriter::Impl - Private implementation
class PointFileWriter::Impl {
public:
  Impl(const std::string &filename) {
    file_.open(filename, std::ios::binary | std::ios::out);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open output file: " + filename);
    }
  }

  ~Impl() {
    if (file_.is_open()) {
      file_.close();
    }
  }

  void writePoint(const PointRecord &record) {
    // Write batch_id (4 bytes, little-endian)
    file_.write(reinterpret_cast<const char *>(&record.batch_id),
                sizeof(uint32_t));

    // Write index_in_batch (4 bytes, little-endian)
    file_.write(reinterpret_cast<const char *>(&record.index_in_batch),
                sizeof(uint32_t));

    // Write x coordinate (32 bytes, big-endian - already in correct format)
    file_.write(reinterpret_cast<const char *>(record.x), 32);

    // Write y parity (1 byte)
    file_.write(reinterpret_cast<const char *>(&record.y_parity), 1);
  }

  void writePoints(const std::vector<PointRecord> &records) {
    for (const auto &record : records) {
      writePoint(record);
    }
    file_.flush();
  }

  void flush() { file_.flush(); }

private:
  std::ofstream file_;
};

// PointFileWriter public interface
PointFileWriter::PointFileWriter(const std::string &filename)
    : impl_(std::make_unique<Impl>(filename)) {}

PointFileWriter::~PointFileWriter() = default;

void PointFileWriter::writePoints(const std::vector<PointRecord> &records) {
  impl_->writePoints(records);
}

void PointFileWriter::flush() { impl_->flush(); }

// TelemetryWriter::Impl - Private implementation
class TelemetryWriter::Impl {
public:
  Impl(const std::string &filename) {
    file_.open(filename, std::ios::out);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open telemetry file: " + filename);
    }
  }

  ~Impl() {
    if (file_.is_open()) {
      file_.close();
    }
  }

  void writeBatch(const BatchTelemetry &telem) {
    file_ << "{"
          << "\"timestamp\":" << telem.timestamp << ","
          << "\"batch_id\":" << telem.batch_id << ","
          << "\"family\":\"" << familyToString(telem.family) << "\","
          << "\"start_k\":\"" << telem.start_k << "\","
          << "\"mask_bits\":" << telem.mask_bits << ","
          << "\"stride\":" << telem.stride << ","
          << "\"num_keys\":" << telem.num_keys << ","
          << "\"kernel_ms\":" << std::fixed << std::setprecision(4)
          << telem.kernel_ms << ","
          << "\"h2d_ms\":" << std::fixed << std::setprecision(4) << telem.h2d_ms
          << ","
          << "\"d2h_ms\":" << std::fixed << std::setprecision(4) << telem.d2h_ms
          << ","
          << "\"cpu_prep_ms\":" << std::fixed << std::setprecision(4)
          << telem.cpu_prep_ms << ","
          << "\"cpu_wait_gpu_ms\":" << std::fixed << std::setprecision(4)
          << telem.cpu_wait_gpu_ms << ","
          << "\"sampled_points\":" << telem.sampled_points << ","
          << "\"matches\":" << telem.matches << "}\n";
    file_.flush();
  }

  void flush() { file_.flush(); }

private:
  std::ofstream file_;
};

// TelemetryWriter public interface
TelemetryWriter::TelemetryWriter(const std::string &filename)
    : impl_(std::make_unique<Impl>(filename)) {}

TelemetryWriter::~TelemetryWriter() = default;

void TelemetryWriter::writeBatch(const BatchTelemetry &telem) {
  impl_->writeBatch(telem);
}

void TelemetryWriter::flush() { impl_->flush(); }

// ============================================================================
// DifferentialFileWriter Implementation
// ============================================================================

class DifferentialFileWriter::Impl {
public:
  Impl(const std::string &filename) {
    file_.open(filename, std::ios::binary | std::ios::out);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open differential output file: " +
                               filename);
    }
  }

  ~Impl() {
    if (file_.is_open()) {
      file_.close();
    }
  }

  void writeRecord(const DifferentialRecord &record) {
    // Write 60-byte differential record in binary format
    file_.write(reinterpret_cast<const char *>(record.delta_P_x), 32);
    file_.write(reinterpret_cast<const char *>(&record.delta_P_x_mod_m1), 4);
    file_.write(reinterpret_cast<const char *>(&record.delta_P_x_mod_m2), 4);
    file_.write(reinterpret_cast<const char *>(&record.delta_value), 4);
    file_.write(reinterpret_cast<const char *>(&record.scalar_family_id), 1);
    file_.write(reinterpret_cast<const char *>(&record.mask_bits_or_stride), 1);
    file_.write(reinterpret_cast<const char *>(&record.batch_id), 4);
    file_.write(reinterpret_cast<const char *>(&record.index_in_batch), 4);
    file_.write(reinterpret_cast<const char *>(record.reserved), 6);
  }

  void writeRecords(const std::vector<DifferentialRecord> &records) {
    for (const auto &record : records) {
      writeRecord(record);
    }
    file_.flush();
  }

  void flush() { file_.flush(); }

private:
  std::ofstream file_;
};

// DifferentialFileWriter public interface
DifferentialFileWriter::DifferentialFileWriter(const std::string &filename)
    : impl_(std::make_unique<Impl>(filename)) {}

DifferentialFileWriter::~DifferentialFileWriter() = default;

void DifferentialFileWriter::writeRecords(
    const std::vector<DifferentialRecord> &records) {
  impl_->writeRecords(records);
}

void DifferentialFileWriter::flush() { impl_->flush(); }

// ============================================================================
// Manifest Writer
// ============================================================================

// Manifest writer
void writeManifest(const ECDumpConfig &config, uint64_t total_keys) {
  std::ofstream file("manifest.json");
  if (!file.is_open()) {
    std::cerr << "Warning: Failed to write manifest.json" << std::endl;
    return;
  }

  uint64_t timestamp = static_cast<uint64_t>(std::time(nullptr));

  file << "{\n"
       << "  \"version\": \"1.0\",\n"
       << "  \"timestamp\": " << timestamp << ",\n"
       << "  \"family\": \"" << familyToString(config.family) << "\",\n"
       << "  \"start_k\": \"0x" << config.start_k.toString(16) << "\",\n"
       << "  \"batch_keys\": " << config.batch_keys << ",\n"
       << "  \"batches\": " << config.batches << ",\n"
       << "  \"total_keys\": " << total_keys << ",\n"
       << "  \"sample_rate\": " << config.sample_rate << ",\n"
       << "  \"seed\": " << config.seed << ",\n"
       << "  \"device\": " << config.device << ",\n"
       << "  \"output_file\": \"" << config.out_bin << "\",\n"
       << "  \"telemetry_file\": \"" << config.telemetry_file << "\"\n"
       << "}\n";

  file.close();
}

// Convert GPU output to PointRecord structures
void convertGPUOutput(
    uint32_t batch_id,
    const std::vector<unsigned int> &xCoords, // 8 words per point, big-endian
    const std::vector<unsigned char> &yParity,
    std::vector<PointRecord> &records_out, double sample_rate = 1.0) {
  size_t numPoints = yParity.size();
  records_out.clear();

  if (sample_rate >= 1.0) {
    // No sampling, output all points
    records_out.reserve(numPoints);

    for (size_t i = 0; i < numPoints; i++) {
      PointRecord record;
      record.batch_id = batch_id;
      record.index_in_batch = static_cast<uint32_t>(i);

      // Copy x coordinate (already big-endian from GPU)
      for (int j = 0; j < 8; j++) {
        uint32_t word = xCoords[i * 8 + j];
        record.x[j * 4 + 0] = static_cast<uint8_t>((word >> 24) & 0xFF);
        record.x[j * 4 + 1] = static_cast<uint8_t>((word >> 16) & 0xFF);
        record.x[j * 4 + 2] = static_cast<uint8_t>((word >> 8) & 0xFF);
        record.x[j * 4 + 3] = static_cast<uint8_t>(word & 0xFF);
      }

      record.y_parity = yParity[i];

      records_out.push_back(record);
    }
  } else {
    // Simple downsampling: take every Nth point
    size_t step = static_cast<size_t>(1.0 / sample_rate);
    if (step < 1)
      step = 1;

    for (size_t i = 0; i < numPoints; i += step) {
      PointRecord record;
      record.batch_id = batch_id;
      record.index_in_batch = static_cast<uint32_t>(i);

      for (int j = 0; j < 8; j++) {
        uint32_t word = xCoords[i * 8 + j];
        record.x[j * 4 + 0] = static_cast<uint8_t>((word >> 24) & 0xFF);
        record.x[j * 4 + 1] = static_cast<uint8_t>((word >> 16) & 0xFF);
        record.x[j * 4 + 2] = static_cast<uint8_t>((word >> 8) & 0xFF);
        record.x[j * 4 + 3] = static_cast<uint8_t>(word & 0xFF);
      }

      record.y_parity = yParity[i];

      records_out.push_back(record);
    }
  }
}

} // namespace ecdump
