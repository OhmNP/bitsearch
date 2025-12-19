"""
Statistical analysis for differential EC experiments.

Implements 6 cryptanalysis-grade statistical tests for secp256k1 differential data.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any
import pandas as pd


class DifferentialAnalyzer:
    """Statistical test suite for differential EC data."""
    
    def __init__(self, significance_level: float = 0.01):
        """
        Initialize analyzer.
        
        Args:
            significance_level: Threshold for test failure (default: 0.01)
        """
        self.alpha = significance_level
    
    def chi_square_uniformity(self, data: np.ndarray) -> Dict[str, Any]:
        """
        χ² test for uniform distribution of ΔP.x values.
        
        Tests if the distribution of x-coordinates is uniform across bins.
        
        Args:
            data: Array of delta_P_x_int values
            
        Returns:
            Dictionary with test results
        """
        # Use 256 bins for byte-level uniformity
        bins = 256
        observed, _ = np.histogram(data % 256, bins=bins, range=(0, 256))
        expected = len(data) / bins
        
        # χ² statistic
        chi2_stat = np.sum((observed - expected)**2 / expected)
        dof = bins - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        return {
            'test': 'chi_square_uniformity',
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'dof': dof,
            'pass': p_value > self.alpha,
            'expected_mean': expected,
            'observed_range': (int(observed.min()), int(observed.max()))
        }
    
    def collision_rate_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Birthday paradox collision rate test.
        
        Compares observed collision rate against expected rate for random data.
        
        Args:
            data: Array of delta_P_x_int values
            
        Returns:
            Dictionary with test results
        """
        n = len(data)
        unique = len(np.unique(data))
        collision_rate = 1 - (unique / n)
        
        # Expected collision rate for 256-bit space (approximation)
        # For small n relative to 2^256, expected ≈ n²/(2 * 2^256) ≈ 0
        # For practical purposes, we expect very few collisions
        expected_rate = 0.0  # Negligible for 256-bit space
        
        # Use binomial test
        # H0: collision rate = expected_rate
        collisions = n - unique
        # scipy 1.16+ uses binomtest instead of binom_test
        try:
            p_value = stats.binomtest(collisions, n, expected_rate, alternative='greater').pvalue
        except AttributeError:
            # Fallback for older scipy versions
            p_value = stats.binom_test(collisions, n, expected_rate, alternative='greater')
        
        return {
            'test': 'collision_rate',
            'observed_rate': float(collision_rate),
            'expected_rate': float(expected_rate),
            'unique_values': int(unique),
            'total_values': int(n),
            'collisions': int(collisions),
            'p_value': float(p_value),
            'pass': p_value > self.alpha
        }
    
    def rank_test_sliding_window(self, data: np.ndarray, 
                                  window_size: int = 256) -> Dict[str, Any]:
        """
        Rank test over sliding windows.
        
        Tests if the rank of submatrices is as expected for random data.
        
        Args:
            data: Array of delta_P_x_int values
            window_size: Size of sliding window
            
        Returns:
            Dictionary with test results
        """
        if len(data) < window_size:
            return {
                'test': 'rank_test_sliding_window',
                'error': 'Insufficient data for window size',
                'pass': False
            }
        
        # Convert to binary matrix
        # Take lower 8 bits of each value
        binary_data = np.array([x & 0xFF for x in data], dtype=np.uint8)
        
        ranks = []
        num_windows = len(data) - window_size + 1
        
        for i in range(min(num_windows, 100)):  # Limit to 100 windows for performance
            window = binary_data[i:i+window_size]
            # Convert to binary matrix (8 bits per value)
            matrix = np.unpackbits(window.reshape(-1, 1), axis=1)
            rank = np.linalg.matrix_rank(matrix)
            ranks.append(rank)
        
        mean_rank = np.mean(ranks)
        expected_rank = min(window_size, 8)  # Expected rank for random binary matrix
        
        # Use t-test to compare mean rank against expected
        t_stat, p_value = stats.ttest_1samp(ranks, expected_rank)
        
        return {
            'test': 'rank_test_sliding_window',
            'mean_rank': float(mean_rank),
            'expected_rank': float(expected_rank),
            'std_rank': float(np.std(ranks)),
            'num_windows': len(ranks),
            'p_value': float(p_value),
            'pass': p_value > self.alpha
        }
    
    def serial_correlation_test(self, data: np.ndarray, 
                                max_lag: int = 10) -> Dict[str, Any]:
        """
        Measure correlation between ΔP.x[i] and ΔP.x[i+k].
        
        Tests mixing speed of group action.
        
        Args:
            data: Array of delta_P_x_int values
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary with test results
        """
        # Normalize data to [0, 1] for correlation
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)
        
        correlations = []
        for lag in range(1, max_lag + 1):
            if len(data) > lag:
                try:
                    # Manual correlation calculation (compatible with numpy 2.2)
                    x = data_norm[:-lag]
                    y = data_norm[lag:]
                    
                    if len(x) > 1 and len(y) > 1:
                        mean_x = np.mean(x)
                        mean_y = np.mean(y)
                        std_x = np.std(x)
                        std_y = np.std(y)
                        
                        if std_x > 1e-10 and std_y > 1e-10:
                            corr = np.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)
                        else:
                            corr = 0.0
                    else:
                        corr = 0.0
                except (ValueError, IndexError):
                    corr = 0.0
                correlations.append(corr)
        
        if not correlations:
            correlations = [0.0]
        
        max_corr = np.max(np.abs(correlations))
        
        # For random data, correlation should be near 0
        # Use threshold of 0.1 for significance
        threshold = 0.1
        
        return {
            'test': 'serial_correlation',
            'lag_1_correlation': float(correlations[0]) if correlations else 0.0,
            'max_correlation': float(max_corr),
            'correlations': [float(c) for c in correlations],
            'threshold': threshold,
            'pass': max_corr < threshold
        }
    
    def differential_bias_test(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Test for bit bias in ΔP.x values.
        
        For each bit position j: Pr(bit_j(ΔP.x) = 1) should be ≈ 0.5
        
        Args:
            data: Array of delta_P_x_int values
            
        Returns:
            Dictionary with test results
        """
        n = len(data)
        bit_biases = []
        biased_bits = []
        
        # Test lower 64 bits (sufficient for bias detection)
        for bit_pos in range(64):
            bit_values = (data >> bit_pos) & 1
            prob_one = np.mean(bit_values)
            bias = abs(prob_one - 0.5)
            bit_biases.append(bias)
            
            # Binomial test: H0: p = 0.5
            ones = np.sum(bit_values)
            try:
                p_value = stats.binomtest(ones, n, 0.5).pvalue
            except AttributeError:
                p_value = stats.binom_test(ones, n, 0.5)
            
            if p_value < self.alpha:
                biased_bits.append(bit_pos)
        
        max_bias = np.max(bit_biases)
        
        return {
            'test': 'differential_bias',
            'max_bias': float(max_bias),
            'mean_bias': float(np.mean(bit_biases)),
            'biased_bits': biased_bits,
            'num_biased_bits': len(biased_bits),
            'pass': len(biased_bits) == 0
        }
    
    def small_modulus_lattice_test(self, data: np.ndarray,
                                   mod_m1: np.ndarray,
                                   mod_m2: np.ndarray) -> Dict[str, Any]:
        """
        Small-modulus lattice leakage detection.
        
        Tests for structure in small-modulus reductions.
        
        Args:
            data: Array of delta_P_x_int values
            mod_m1: Array of delta_P_x_mod_m1 values (mod 65535)
            mod_m2: Array of delta_P_x_mod_m2 values (mod 4294967291)
            
        Returns:
            Dictionary with test results
        """
        # GCD collision test
        unique_m1 = len(np.unique(mod_m1))
        unique_m2 = len(np.unique(mod_m2))
        n = len(data)
        
        collision_rate_m1 = 1 - (unique_m1 / n)
        collision_rate_m2 = 1 - (unique_m2 / n)
        
        # Expected collision rates (birthday paradox)
        expected_m1 = 1 - np.exp(-n**2 / (2 * 65535))
        expected_m2 = 1 - np.exp(-n**2 / (2 * 4294967291))
        
        # Autocorrelation of residues
        if len(mod_m1) > 1:
            autocorr_m1 = np.corrcoef(mod_m1[:-1], mod_m1[1:])[0, 1]
        else:
            autocorr_m1 = 0.0
        
        if len(mod_m2) > 1:
            autocorr_m2 = np.corrcoef(mod_m2[:-1], mod_m2[1:])[0, 1]
        else:
            autocorr_m2 = 0.0
        
        # Pass if collision rates are close to expected and autocorrelation is low
        collision_ok = (abs(collision_rate_m1 - expected_m1) < 0.1 and
                       abs(collision_rate_m2 - expected_m2) < 0.1)
        autocorr_ok = (abs(autocorr_m1) < 0.1 and abs(autocorr_m2) < 0.1)
        
        return {
            'test': 'small_modulus_lattice',
            'collision_rate_m1': float(collision_rate_m1),
            'expected_rate_m1': float(expected_m1),
            'collision_rate_m2': float(collision_rate_m2),
            'expected_rate_m2': float(expected_m2),
            'autocorr_m1': float(autocorr_m1),
            'autocorr_m2': float(autocorr_m2),
            'pass': collision_ok and autocorr_ok
        }
    
    def run_all_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all statistical tests on differential data.
        
        Args:
            df: DataFrame from parse_experiments.py
            
        Returns:
            Dictionary with all test results and overall pass/fail
        """
        data = df['delta_P_x_int'].values
        mod_m1 = df['delta_P_x_mod_m1'].values
        mod_m2 = df['delta_P_x_mod_m2'].values
        
        results = {
            'num_records': len(df),
            'tests': {}
        }
        
        # Run all tests
        results['tests']['chi_square'] = self.chi_square_uniformity(data)
        results['tests']['collision_rate'] = self.collision_rate_test(data)
        results['tests']['rank_test'] = self.rank_test_sliding_window(data)
        results['tests']['serial_correlation'] = self.serial_correlation_test(data)
        results['tests']['differential_bias'] = self.differential_bias_test(data)
        results['tests']['small_modulus_lattice'] = self.small_modulus_lattice_test(
            data, mod_m1, mod_m2)
        
        # Overall pass/fail
        all_pass = all(test.get('pass', False) for test in results['tests'].values())
        results['overall_pass'] = all_pass
        
        return results


if __name__ == '__main__':
    import sys
    import json
    from parse_experiments import parse_differential_file
    
    if len(sys.argv) < 2:
        print("Usage: python crypto_analysis.py <differential_file.bin> [--output results.json]")
        sys.exit(1)
    
    filename = sys.argv[1]
    output_file = 'analysis_results.json'
    
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    # Parse differential file
    print(f"Parsing {filename}...")
    df = parse_differential_file(filename)
    
    # Run statistical tests
    print("\nRunning statistical tests...")
    analyzer = DifferentialAnalyzer(significance_level=0.01)
    results = analyzer.run_all_tests(df)
    
    # Print results
    print("\n=== Statistical Test Results ===")
    print(f"Total records: {results['num_records']}")
    print(f"\nOverall: {'PASS ✓' if results['overall_pass'] else 'FAIL ✗'}\n")
    
    for test_name, test_result in results['tests'].items():
        status = 'PASS ✓' if test_result.get('pass', False) else 'FAIL ✗'
        print(f"{test_name}: {status}")
        if 'p_value' in test_result:
            print(f"  p-value: {test_result['p_value']:.6f}")
    
    # Save results
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy_types(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {output_file}")
