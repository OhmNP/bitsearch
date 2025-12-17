#ifndef _KEY_FINDER_TYPES
#define _KEY_FINDER_TYPES

#include "secp256k1.h"
#include <stdint.h>
#include <string>
#include "Telemetry.h"


namespace PointCompressionType {
enum Value { COMPRESSED = 0, UNCOMPRESSED = 1, BOTH = 2 };
}

typedef struct hash160 {

  unsigned int h[5];

  hash160(const unsigned int hash[5]) {
    memcpy(h, hash, sizeof(unsigned int) * 5);
  }
} hash160;

typedef struct {
  double speed;
  uint64_t total;
  uint64_t totalTime;
  uint64_t freeMemory;
  uint64_t deviceMemory;
  std::string deviceName;
  int targets;
  secp256k1::uint256 nextKey;
  TelemetryData telemetry;
} KeySearchStatus;

class KeySearchTarget {

public:
  unsigned int value[5];

  KeySearchTarget() { memset(value, 0, sizeof(value)); }

  KeySearchTarget(const unsigned int h[5]) {
    for (int i = 0; i < 5; i++) {
      value[i] = h[i];
    }
  }

  bool operator==(const KeySearchTarget &t) const {
    for (int i = 0; i < 5; i++) {
      if (value[i] != t.value[i]) {
        return false;
      }
    }

    return true;
  }

  bool operator<(const KeySearchTarget &t) const {
    for (int i = 0; i < 5; i++) {
      if (value[i] < t.value[i]) {
        return true;
      } else if (value[i] > t.value[i]) {
        return false;
      }
    }

    return false;
  }

  bool operator>(const KeySearchTarget &t) const {
    for (int i = 0; i < 5; i++) {
      if (value[i] > t.value[i]) {
        return true;
      } else if (value[i] < t.value[i]) {
        return false;
      }
    }

    return false;
  }
};

#endif