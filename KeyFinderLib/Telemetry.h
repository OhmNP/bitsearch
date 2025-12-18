#ifndef _TELEMETRY_H
#define _TELEMETRY_H

#include <stdint.h>
#include <string>

struct TelemetryData {
  uint64_t batchId;
  double kernelTimeMs;
  double hostToDeviceMs;
  double deviceToHostMs;
  double cpuValidationTimeMs;
  double cpuSetupTimeMs;
  double cpuWaitTimeMs;
  double totalTimeMs;
  uint64_t keysSearched;
  uint64_t matches;
  double bandwidthGBs;
  double occupancy;

  TelemetryData() = default;
};

class Telemetry {
public:
  static Telemetry &getInstance();

  void update(const TelemetryData &data);

  TelemetryData getLast();

  void getSummary(double &avgKeysSec, double &avgKernMs, double &avgWaitMs);

private:
  Telemetry() = default;
  TelemetryData lastData;

  // Aggregates
  uint64_t count;
  double totalKernMs;
  double totalSetupMs;
  double totalWaitMs;
  double totalValidationMs;
  double totalD2HMs;
  uint64_t totalKeys;
  uint64_t totalMatches;
};

#endif
