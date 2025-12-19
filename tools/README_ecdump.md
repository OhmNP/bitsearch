# BitCrack EC Dump

A CUDA-based tool for generating raw elliptic-curve point data (P = k·G) for large-scale structural testing of secp256k1 private-key families.

## Purpose

This tool generates **raw point data only** - it does NOT:
- Compute SHA-256 or RIPEMD-160 hashes
- Derive Bitcoin addresses
- Perform address matching or validation
- Use Base58 encoding

It is designed for offline statistical analysis and structural testing of secp256k1 key families.

## Building

### Requirements
- CUDA Toolkit 10.1 or later
- C++11 compatible compiler
- GNU Make

### Build Instructions

```bash
# Build with CUDA support
make BUILD_CUDA=1

# The binary will be created at: bin/bitcrack-ecdump
```

## Usage

### Basic Syntax

```bash
bitcrack-ecdump --family FAMILY --out-bin OUTPUT --telemetry TELEMETRY [OPTIONS]
```

### Required Arguments

- `--family FAMILY` - Key family type: `control`, `consecutive`, `masked`, `stride`, or `hd`
- `--out-bin PATH` - Binary point output file
- `--telemetry PATH` - JSON telemetry output file

### Family-Specific Arguments

- `--start-k HEX` - Starting key (hex) for consecutive/masked/stride families
- `--mask-bits N` - Number of variable low bits for masked family
- `--stride S` - Stride value for stride family
- `--seed SEED` - PRNG seed for control/hd families (default: time-based)

### Batch Control

- `--batch-keys N` - Keys per GPU batch (default: 1048576)
- `--batches M` - Number of batches to process (default: 1)
- `--sample-rate R` - Sample rate 0.0-1.0 or reservoir size (default: 1.0 = all points)

### Options

- `--verify` - Verify first batch against CPU reference (libsecp256k1)
- `--device IDX` - CUDA device index (default: 0)
- `--dry-run` - Simulate without writing files (for performance testing)
- `--force` - Bypass safety checks for large runs (>100M keys)
- `--help` - Show help message

## Key Families

### Control
Uniform random keys in [1, n-1] using seedable PRNG.

```bash
bitcrack-ecdump --family control --seed 42 \
  --batch-keys 1048576 --batches 10 \
  --out-bin control.bin --telemetry control_telem.json
```

### Consecutive
Sequential keys: start_k, start_k+1, start_k+2, ...

```bash
bitcrack-ecdump --family consecutive --start-k 0x1 \
  --batch-keys 1048576 --batches 1 \
  --out-bin consecutive.bin --telemetry consecutive_telem.json --verify
```

### Masked
Keys with fixed high bits and variable low bits: (fixed_high) | i

```bash
bitcrack-ecdump --family masked --start-k 0x8000000000000000 \
  --mask-bits 32 --batch-keys 1048576 --batches 100 \
  --out-bin masked.bin --telemetry masked_telem.json
```

### Stride
Keys with fixed stride: start_k, start_k+stride, start_k+2*stride, ...

```bash
bitcrack-ecdump --family stride --start-k 0x1 --stride 1000 \
  --batch-keys 1048576 --batches 10 \
  --out-bin stride.bin --telemetry stride_telem.json
```

### HD (Hierarchical Deterministic)
Keys derived from H(seed || counter) mod n

```bash
bitcrack-ecdump --family hd --seed 123456789 \
  --batch-keys 1048576 --batches 10 \
  --out-bin hd.bin --telemetry hd_telem.json
```

## Output Formats

### Binary Point File

Each point record is 41 bytes:

| Offset | Size | Field | Format |
|--------|------|-------|--------|
| 0 | 4 | batch_id | uint32_t (little-endian) |
| 4 | 4 | index_in_batch | uint32_t (little-endian) |
| 8 | 32 | x coordinate | 32 bytes (big-endian) |
| 40 | 1 | y parity | 0 or 1 |

**Total**: 41 bytes per point

### Telemetry JSON

One JSON object per line (JSONL format), one line per batch:

```json
{
  "timestamp": 1700000000,
  "batch_id": 0,
  "family": "consecutive",
  "start_k": "0x1",
  "mask_bits": 0,
  "stride": 1,
  "num_keys": 1048576,
  "kernel_ms": 48.123,
  "h2d_ms": 0.001,
  "d2h_ms": 0.0012,
  "cpu_prep_ms": 0.0001,
  "cpu_wait_gpu_ms": 0.002,
  "sampled_points": 1048576,
  "matches": 0
}
```

### Manifest JSON

Written to `manifest.json` at the end of the run:

```json
{
  "version": "1.0",
  "timestamp": 1700000000,
  "family": "consecutive",
  "start_k": "0x1",
  "batch_keys": 1048576,
  "batches": 10,
  "total_keys": 10485760,
  "sample_rate": 1.0,
  "seed": 0,
  "device": 0,
  "output_file": "out.bin",
  "telemetry_file": "telemetry.json"
}
```

## Sampling

For large runs, use `--sample-rate` to reduce output size:

```bash
# Sample 0.1% of points (1 in 1000)
bitcrack-ecdump --family control --seed 42 \
  --batch-keys 10485760 --batches 100 \
  --sample-rate 0.001 \
  --out-bin control_sample.bin --telemetry control_telem.json
```

## Safety Checks

The tool refuses to run if `batch_keys * batches > 100,000,000` without `--force`:

```bash
# This will fail without --force
bitcrack-ecdump --family consecutive --start-k 0x1 \
  --batch-keys 10000000 --batches 100 \
  --out-bin huge.bin --telemetry huge_telem.json

# Add --force to bypass
bitcrack-ecdump --family consecutive --start-k 0x1 \
  --batch-keys 10000000 --batches 100 \
  --out-bin huge.bin --telemetry huge_telem.json --force
```

## Verification

Use `--verify` to validate GPU results against CPU reference (libsecp256k1) for the first 1024 points:

```bash
bitcrack-ecdump --family consecutive --start-k 0x1 \
  --batch-keys 1048576 --batches 1 \
  --out-bin test.bin --telemetry test_telem.json --verify
```

Output:
```
Verifying 1024 points against CPU reference...
Verification PASSED: 1024/1024 points match CPU reference
```

## Performance Tuning

- **Batch size**: Larger batches improve GPU utilization but use more memory
- **Sampling**: Reduces output size and I/O overhead
- **Dry run**: Use `--dry-run` to test performance without writing files

Example dry run:
```bash
bitcrack-ecdump --family control --seed 42 \
  --batch-keys 8388608 --batches 10 --dry-run
```

## Parsing Output

Use the included Python script to parse binary output:

```bash
# Convert to CSV
python tools/parse_ecdump.py out.bin --csv out.csv

# Extract first 1000 points
python tools/parse_ecdump.py out.bin --sample 1000 --csv sample.csv

# Compute statistics
python tools/parse_ecdump.py out.bin --stats
```

## Troubleshooting

### CUDA Out of Memory
Reduce `--batch-keys` to a smaller value (e.g., 524288 or 262144).

### Slow Performance
- Ensure you're using a CUDA-capable GPU
- Check GPU utilization with `nvidia-smi`
- Try increasing `--batch-keys` for better GPU utilization

### Verification Failures
This indicates a bug in the GPU kernel. Please report with:
- Full command line
- GPU model and CUDA version
- First failing key value

## License

Same as BitCrack (MIT License)

## Contact

For questions or bug reports, contact: bitcrack.project@gmail.com
