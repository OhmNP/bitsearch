# Statistical Test Harness - Installation & Usage Guide

## Python Dependencies

The statistical test harness requires Python 3.7+ with the following packages:

```bash
pip install numpy scipy pandas
```

## Files Created

1. **`parse_experiments.py`** - Binary record parser
   - Parses 60-byte differential records
   - Validates record format
   - Exports to CSV

2. **`crypto_analysis.py`** - Statistical test suite
   - 6 cryptanalysis-grade tests
   - JSON output with p-values
   - Automatic pass/fail determination

## Installation

### Option 1: Install Dependencies
```cmd
pip install numpy scipy pandas
```

### Option 2: Use Conda
```cmd
conda install numpy scipy pandas
```

## Usage

### Parse Differential Records
```cmd
cd C:\Users\parim\Desktop\projects\BitCrack\bin
python ..\tools\parse_experiments.py test_diff.bin --csv test_diff.csv --limit 10
```

### Run Statistical Tests
```cmd
python ..\tools\crypto_analysis.py test_diff.bin --output test_diff_analysis.json
```

## Test Descriptions

### 1. χ² Uniformity Test
- **Purpose**: Tests if ΔP.x values are uniformly distributed
- **Method**: Chi-square test with 256 bins
- **Pass**: p-value > 0.01

### 2. Collision Rate Test
- **Purpose**: Compares collision rate against birthday paradox
- **Method**: Binomial test
- **Pass**: p-value > 0.01

### 3. Rank Test (Sliding Window)
- **Purpose**: Tests rank of binary submatrices
- **Method**: T-test against expected rank
- **Pass**: p-value > 0.01

### 4. Serial Correlation Test
- **Purpose**: Measures correlation between consecutive values
- **Method**: Autocorrelation analysis
- **Pass**: |correlation| < 0.1

### 5. Differential Bias Test
- **Purpose**: Tests for bit bias in ΔP.x
- **Method**: Binomial test per bit position
- **Pass**: No biased bits (p < 0.01)

### 6. Small-Modulus Lattice Test
- **Purpose**: Detects structure in modular reductions
- **Method**: Collision rate + autocorrelation
- **Pass**: Rates match expected, low autocorrelation

## Expected Output

```json
{
  "num_records": 1024,
  "tests": {
    "chi_square": {
      "test": "chi_square_uniformity",
      "statistic": 255.3,
      "p_value": 0.512,
      "pass": true
    },
    "collision_rate": {
      "test": "collision_rate",
      "observed_rate": 0.0,
      "p_value": 1.0,
      "pass": true
    },
    ...
  },
  "overall_pass": true
}
```

## Failure Protocol

**STOP immediately if**:
- Any p-value < 0.01
- Deviation reproducible across batches
- Deviation survives family change
- Deviation strengthens with larger N

## Next Steps

1. Install dependencies: `pip install numpy scipy pandas`
2. Run tests on test_diff.bin (1024 records)
3. If all pass, scale to 10⁷ records
4. Document results

## Troubleshooting

**ModuleNotFoundError: No module named 'numpy'**
- Solution: `pip install numpy scipy pandas`

**File not found**
- Ensure you're in the `bin` directory
- Use relative path: `..\tools\crypto_analysis.py`

**Insufficient data**
- Minimum 256 records recommended
- Some tests require larger datasets
