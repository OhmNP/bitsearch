"""
Binary parser for differential experiment records.

Parses 60-byte differential records from bitcrack-ecdump differential experiments.
"""

import struct
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def parse_differential_file(filename: str) -> pd.DataFrame:
    """
    Parse 60-byte differential records from binary file.
    
    Record format (60 bytes):
    - delta_P_x (32 bytes): ΔP.x full field element (big-endian)
    - delta_P_x_mod_m1 (4 bytes): ΔP.x mod (2^16-1) = 65535 (little-endian)
    - delta_P_x_mod_m2 (4 bytes): ΔP.x mod (2^32-5) = 4294967291 (little-endian)
    - delta_value (4 bytes): The δ used (little-endian)
    - scalar_family_id (1 byte): Family enum value
    - mask_bits_or_stride (1 byte): Family-specific parameter
    - batch_id (4 bytes): Batch ID (little-endian)
    - index_in_batch (4 bytes): Index within batch (little-endian)
    - reserved (6 bytes): Padding
    
    Args:
        filename: Path to binary differential file
        
    Returns:
        DataFrame with columns: delta_P_x, delta_P_x_mod_m1, delta_P_x_mod_m2,
                                delta_value, family, param, batch_id, index
    """
    records = []
    path = Path(filename)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    file_size = path.stat().st_size
    if file_size % 60 != 0:
        raise ValueError(f"File size {file_size} is not a multiple of 60 bytes")
    
    num_records = file_size // 60
    print(f"Parsing {num_records} differential records from {filename}")
    
    with open(filename, 'rb') as f:
        for i in range(num_records):
            data = f.read(60)
            if len(data) < 60:
                break
            
            # Parse fields
            delta_P_x = data[0:32]  # Big-endian 256-bit integer
            delta_P_x_mod_m1 = struct.unpack('<I', data[32:36])[0]
            delta_P_x_mod_m2 = struct.unpack('<I', data[36:40])[0]
            delta_value = struct.unpack('<I', data[40:44])[0]
            scalar_family_id = data[44]
            mask_bits_or_stride = data[45]
            batch_id = struct.unpack('<I', data[46:50])[0]
            index_in_batch = struct.unpack('<I', data[50:54])[0]
            
            records.append({
                'delta_P_x': delta_P_x.hex(),
                'delta_P_x_int': int.from_bytes(delta_P_x, byteorder='big'),
                'delta_P_x_mod_m1': delta_P_x_mod_m1,
                'delta_P_x_mod_m2': delta_P_x_mod_m2,
                'delta_value': delta_value,
                'family': scalar_family_id,
                'param': mask_bits_or_stride,
                'batch_id': batch_id,
                'index': index_in_batch
            })
    
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} records successfully")
    return df


def validate_record_format(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate differential records for correctness.
    
    Args:
        df: DataFrame from parse_differential_file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_records': len(df),
        'unique_delta_values': df['delta_value'].nunique(),
        'unique_batches': df['batch_id'].nunique(),
        'family_distribution': df['family'].value_counts().to_dict(),
        'delta_P_x_range': {
            'min': df['delta_P_x_int'].min(),
            'max': df['delta_P_x_int'].max()
        },
        'mod_m1_range': {
            'min': df['delta_P_x_mod_m1'].min(),
            'max': df['delta_P_x_mod_m1'].max(),
            'expected_max': 65534  # 2^16 - 2
        },
        'mod_m2_range': {
            'min': df['delta_P_x_mod_m2'].min(),
            'max': df['delta_P_x_mod_m2'].max(),
            'expected_max': 4294967290  # 2^32 - 6
        }
    }
    
    # Validate modular reductions are within expected ranges
    results['mod_m1_valid'] = df['delta_P_x_mod_m1'].max() < 65535
    results['mod_m2_valid'] = df['delta_P_x_mod_m2'].max() < 4294967291
    
    return results


def export_to_csv(df: pd.DataFrame, output_file: str, limit: int = None):
    """
    Export differential records to CSV for analysis.
    
    Args:
        df: DataFrame from parse_differential_file
        output_file: Output CSV filename
        limit: Optional limit on number of records to export
    """
    if limit:
        df = df.head(limit)
    
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} records to {output_file}")



def parse_endomorphism_file(filename: str) -> pd.DataFrame:
    """
    Parse 108-byte endomorphism records from binary file.
    
    Record format (108 bytes):
    - batch_id (4 bytes): Batch ID (little-endian)
    - index_in_batch (4 bytes): Index within batch (little-endian)
    - x (32 bytes): P.x (big-endian)
    - beta_x (32 bytes): P_phi.x (big-endian)
    - lambda_x (32 bytes): P_lambda.x (big-endian)
    - flags (4 bytes): Flags (little-endian)
    
    Args:
        filename: Path to binary endomorphism file
        
    Returns:
        DataFrame with columns
    """
    records = []
    path = Path(filename)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    file_size = path.stat().st_size
    if file_size % 108 != 0:
        raise ValueError(f"File size {file_size} is not a multiple of 108 bytes")
    
    num_records = file_size // 108
    print(f"Parsing {num_records} endomorphism records from {filename}")
    
    with open(filename, 'rb') as f:
        for i in range(num_records):
            data = f.read(108)
            if len(data) < 108:
                break
            
            # Parse fields
            batch_id = struct.unpack('<I', data[0:4])[0]
            index_in_batch = struct.unpack('<I', data[4:8])[0]
            x = data[8:40]
            beta_x = data[40:72]
            lambda_x = data[72:104]
            flags = struct.unpack('<I', data[104:108])[0]
            
            records.append({
                'batch_id': batch_id,
                'index': index_in_batch,
                'x': x.hex(),
                'beta_x': beta_x.hex(),
                'lambda_x': lambda_x.hex(),
                'flags': flags,
                'match': (flags >> 3) & 1  # Bit 3 is match flag
            })
    
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} records successfully")
    return df

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parse_experiments.py <file.bin> [--csv output.csv] [--limit N]")
        sys.exit(1)
    
    filename = sys.argv[1]
    csv_output = None
    limit = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--csv' and i + 1 < len(sys.argv):
            csv_output = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--limit' and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # Detect type by size
    path = Path(filename)
    size = path.stat().st_size
    
    if size > 0 and size % 108 == 0:
        print(f"Detected Endomorphism format (multiple of 108 bytes)")
        df = parse_endomorphism_file(filename)
    elif size > 0 and size % 60 == 0:
        print(f"Detected Differential format (multiple of 60 bytes)")
        df = parse_differential_file(filename)
        # Validate differential
        validation = validate_record_format(df)
        print("\nValidation Results:")
        print(f"  Total records: {validation['total_records']}")
        print(f"  Unique deltas: {validation['unique_delta_values']}")
    else:
        print(f"Unknown file format (size {size} not multiple of 60 or 108)")
        sys.exit(1)
    
    # Show first few records
    print("\nFirst 5 records:")
    print(df.head().to_string())
    
    # Export to CSV if requested
    if csv_output:
        export_to_csv(df, csv_output, limit)
