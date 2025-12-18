#include "Telemetry.h"

#include <iostream>

Telemetry &Telemetry::getInstance() {
  static Telemetry instance;
  // printf("Telemetry::getInstance: %p\n", &instance);
  return instance;
}

void Telemetry::update(const TelemetryData &data) {
  lastData = data;

  // Aggregation
  count++;
  totalKernMs += data.kernelTimeMs;
  totalSetupMs += data.cpuSetupTimeMs;
  totalWaitMs += data.cpuWaitTimeMs;
  totalValidationMs += data.cpuValidationTimeMs;
  totalD2HMs += data.deviceToHostMs;
  totalKeys += data.keysSearched;
  totalMatches += data.matches;
}

TelemetryData Telemetry::getLast() { return lastData; }

void Telemetry::getSummary(double &avgKeysSec, double &avgKernMs,
                           double &avgWaitMs) {
  if (count == 0)
    return;
  avgKernMs = totalKernMs / count;
  avgWaitMs = totalWaitMs / count;
  avgKeysSec = (double)totalKeys / (totalKernMs / 1000.0);
}
