#include "ec_dump.h"
#include "ec_dump_types.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

void printUsage(const char *progName) {
  std::cout << "BitCrack EC Dump - Raw elliptic curve point generator"
            << std::endl;
  std::cout << std::endl;
  std::cout << "Usage: " << progName << " [OPTIONS]" << std::endl;
  std::cout << std::endl;
  std::cout << "Required:" << std::endl;
  std::cout << "  --family FAMILY          Key family: control, consecutive, "
               "masked, stride, hd"
            << std::endl;
  std::cout << "  --out-bin PATH           Binary point output file"
            << std::endl;
  std::cout << "  --telemetry PATH         JSON telemetry output file"
            << std::endl;
  std::cout << std::endl;
  std::cout << "Family-specific:" << std::endl;
  std::cout << "  --start-k HEX            Starting key (for "
               "consecutive/masked/stride)"
            << std::endl;
  std::cout << "  --mask-bits N            Number of variable bits (for masked "
               "family)"
            << std::endl;
  std::cout << "  --stride S               Stride value (for stride family)"
            << std::endl;
  std::cout << "  --seed SEED              PRNG seed (for control/hd families, "
               "default: time-based)"
            << std::endl;
  std::cout << std::endl;
  std::cout << "Batch control:" << std::endl;
  std::cout
      << "  --batch-keys N           Keys per GPU batch (default: 1048576)"
      << std::endl;
  std::cout << "  --batches M              Number of batches (default: 1)"
            << std::endl;
  std::cout << "  --sample-rate R          Sample rate 0.0-1.0 or reservoir "
               "size (default: 1.0)"
            << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout
      << "  --verify                 Verify first batch against CPU reference"
      << std::endl;
  std::cout << "  --device IDX             CUDA device index (default: 0)"
            << std::endl;
  std::cout << "  --dry-run                Simulate without writing files"
            << std::endl;
  std::cout << "  --force                  Bypass safety checks for large runs"
            << std::endl;
  std::cout << "  --help                   Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Cryptanalysis Experiments:" << std::endl;
  std::cout << "  --experiment TYPE        Experiment type: diff (differential)"
            << std::endl;
  std::cout
      << "  --delta-set VALUES       Comma-separated delta values for diff "
         "(e.g., 1,2,4,8)"
      << std::endl;
  std::cout << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  # 1M consecutive keys starting from 0x1" << std::endl;
  std::cout << "  " << progName << " --family consecutive --start-k 0x1 \\"
            << std::endl;
  std::cout << "    --batch-keys 1048576 --batches 1 \\" << std::endl;
  std::cout << "    --out-bin out.bin --telemetry telem.json --verify"
            << std::endl;
  std::cout << std::endl;
  std::cout << "  # 10M masked keys with 24 variable bits" << std::endl;
  std::cout << "  " << progName
            << " --family masked --start-k 0x1000000000000 \\" << std::endl;
  std::cout << "    --mask-bits 24 --batch-keys 1048576 --batches 10 \\"
            << std::endl;
  std::cout << "    --out-bin masked.bin --telemetry masked_telem.json"
            << std::endl;
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  try {
    ecdump::ECDumpConfig config;
    bool family_set = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "--help" || arg == "-h") {
        printUsage(argv[0]);
        return 0;
      } else if (arg == "--family") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --family requires an argument" << std::endl;
          return 1;
        }
        config.family = ecdump::stringToFamily(argv[++i]);
        family_set = true;
      } else if (arg == "--start-k") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --start-k requires an argument" << std::endl;
          return 1;
        }
        config.start_k = secp256k1::uint256(argv[++i]);
      } else if (arg == "--mask-bits") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --mask-bits requires an argument" << std::endl;
          return 1;
        }
        config.mask_bits = std::atoi(argv[++i]);
      } else if (arg == "--stride") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --stride requires an argument" << std::endl;
          return 1;
        }
        config.stride = std::strtoull(argv[++i], nullptr, 10);
      } else if (arg == "--batch-keys") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --batch-keys requires an argument" << std::endl;
          return 1;
        }
        config.batch_keys = std::strtoull(argv[++i], nullptr, 10);
      } else if (arg == "--batches") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --batches requires an argument" << std::endl;
          return 1;
        }
        config.batches = std::atoi(argv[++i]);
      } else if (arg == "--out-bin") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --out-bin requires an argument" << std::endl;
          return 1;
        }
        config.out_bin = argv[++i];
      } else if (arg == "--telemetry") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --telemetry requires an argument" << std::endl;
          return 1;
        }
        config.telemetry_file = argv[++i];
      } else if (arg == "--sample-rate") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --sample-rate requires an argument" << std::endl;
          return 1;
        }
        config.sample_rate = std::atof(argv[++i]);
      } else if (arg == "--seed") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --seed requires an argument" << std::endl;
          return 1;
        }
        config.seed = std::strtoull(argv[++i], nullptr, 10);
      } else if (arg == "--verify") {
        config.verify = true;
      } else if (arg == "--device") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --device requires an argument" << std::endl;
          return 1;
        }
        config.device = std::atoi(argv[++i]);
      } else if (arg == "--dry-run") {
        config.dry_run = true;
      } else if (arg == "--force") {
        config.force = true;
      } else if (arg == "--experiment") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --experiment requires an argument" << std::endl;
          return 1;
        }
        std::string exp_type = argv[++i];
        if (exp_type == "diff" || exp_type == "differential") {
          config.experiment = ecdump::ExperimentType::DIFFERENTIAL;
        } else if (exp_type == "endo" || exp_type == "endomorphism") {
          config.experiment = ecdump::ExperimentType::ENDOMORPHISM;
        } else if (exp_type == "endo_diff" || exp_type == "structure") {
          config.experiment = ecdump::ExperimentType::ENDO_DIFF;
        } else {
          std::cerr << "Error: Unknown experiment type: " << exp_type
                    << std::endl;
          return 1;
        }
      } else if (arg == "--delta-set") {
        if (i + 1 >= argc) {
          std::cerr << "Error: --delta-set requires an argument" << std::endl;
          return 1;
        }
        // Parse comma-separated delta values
        std::string deltas_str = argv[++i];
        std::stringstream ss(deltas_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
          config.diff_config.delta_set.push_back(std::stoull(token));
        }
      } else {
        std::cerr << "Error: Unknown argument: " << arg << std::endl;
        printUsage(argv[0]);
        return 1;
      }
    }

    // Validate required arguments
    if (!family_set) {
      std::cerr << "Error: --family is required" << std::endl;
      printUsage(argv[0]);
      return 1;
    }

    // Create and run driver
    ecdump::ECDumpDriver driver(config);
    driver.run();

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  } catch (const std::string &e) {
    std::cerr << "Error: " << e << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown error occurred" << std::endl;
    return 1;
  }
}
