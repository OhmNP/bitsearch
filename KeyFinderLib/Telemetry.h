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
  double totalTimeMs;
  uint64_t keysSearched;

  TelemetryData()
      : batchId(0), kernelTimeMs(0.0), hostToDeviceMs(0.0), deviceToHostMs(0.0),
        cpuValidationTimeMs(0.0), totalTimeMs(0.0), keysSearched(0) {}
};

class Telemetry {
public:
  static Telemetry &getInstance() {
    static Telemetry instance;
    return instance;
  }

  void update(const TelemetryData &data) { lastData = data; }

  TelemetryData getLast() { return lastData; }

private:
  Telemetry() {}
  TelemetryData lastData;
};

#endif
