#!/usr/bin/env python3
"""
BitCrack EC Dump Parser

Parses binary point output files from bitcrack-ecdump and provides
various analysis and conversion utilities.
"""

import struct
import sys
import argparse
from typing import Iterator, Tuple

# Point record format: 41 bytes
# [batch_id:4][index:4][x:32][y_parity:1]
RECORD_SIZE = 41

class PointRecord:
    """Represents a single point record from the binary file."""
    
    def __init__(self, batch_id: int, index: int, x: bytes, y_parity: int):
        self.batch_id = batch_id
        self.index = index
        self.x = x  # 32 bytes, big-endian
        self.y_parity = y_parity  # 0 or 1
    
    def x_hex(self) -> str:
        """Return x coordinate as hex string."""
        return self.x.hex()
    
    def __str__(self) -> str:
        return f"Point(batch={self.batch_id}, idx={self.index}, x=0x{self.x_hex()}, y_parity={self.y_parity})"


def read_points(filename: str) -> Iterator[PointRecord]:
    """
    Generator that yields PointRecord objects from a binary file.
    
    Args:
        filename: Path to binary point file
        
    Yields:
        PointRecord objects
    """
    with open(filename, 'rb') as f:
        while True:
            data = f.read(RECORD_SIZE)
            if len(data) < RECORD_SIZE:
                break
            
            # Unpack: batch_id (4 bytes LE), index (4 bytes LE), x (32 bytes), y_parity (1 byte)
            batch_id = struct.unpack('<I', data[0:4])[0]
            index = struct.unpack('<I', data[4:8])[0]
            x = data[8:40]
            y_parity = data[40]
            
            yield PointRecord(batch_id, index, x, y_parity)


def count_points(filename: str) -> int:
    """Count total number of points in file."""
    import os
    file_size = os.path.getsize(filename)
    if file_size % RECORD_SIZE != 0:
        print(f"Warning: File size ({file_size}) is not a multiple of record size ({RECORD_SIZE})", 
              file=sys.stderr)
    return file_size // RECORD_SIZE


def convert_to_csv(filename: str, output_csv: str, max_points: int = None):
    """
    Convert binary point file to CSV.
    
    Args:
        filename: Input binary file
        output_csv: Output CSV file
        max_points: Maximum number of points to convert (None = all)
    """
    with open(output_csv, 'w') as out:
        # Write header
        out.write("batch_id,index_in_batch,x_hex,y_parity\n")
        
        count = 0
        for point in read_points(filename):
            out.write(f"{point.batch_id},{point.index},{point.x_hex()},{point.y_parity}\n")
            count += 1
            
            if max_points and count >= max_points:
                break
        
        print(f"Converted {count} points to {output_csv}")


def compute_stats(filename: str):
    """
    Compute and print statistics about the point data.
    
    Args:
        filename: Input binary file
    """
    total_points = count_points(filename)
    print(f"Total points: {total_points:,}")
    
    # Count by batch
    batch_counts = {}
    y_parity_counts = {0: 0, 1: 0}
    
    # Sample first and last points
    first_point = None
    last_point = None
    
    for i, point in enumerate(read_points(filename)):
        if i == 0:
            first_point = point
        last_point = point
        
        batch_counts[point.batch_id] = batch_counts.get(point.batch_id, 0) + 1
        y_parity_counts[point.y_parity] = y_parity_counts.get(point.y_parity, 0) + 1
    
    print(f"\nBatches: {len(batch_counts)}")
    print(f"Points per batch: {total_points // len(batch_counts) if batch_counts else 0:,}")
    
    print(f"\nY parity distribution:")
    print(f"  Even (0): {y_parity_counts[0]:,} ({100.0 * y_parity_counts[0] / total_points:.2f}%)")
    print(f"  Odd (1):  {y_parity_counts[1]:,} ({100.0 * y_parity_counts[1] / total_points:.2f}%)")
    
    if first_point:
        print(f"\nFirst point:")
        print(f"  {first_point}")
    
    if last_point:
        print(f"\nLast point:")
        print(f"  {last_point}")


def extract_sample(filename: str, output_csv: str, sample_size: int):
    """
    Extract a sample of points to CSV.
    
    Args:
        filename: Input binary file
        output_csv: Output CSV file
        sample_size: Number of points to extract
    """
    convert_to_csv(filename, output_csv, max_points=sample_size)


def validate_file(filename: str) -> bool:
    """
    Validate file integrity.
    
    Args:
        filename: Input binary file
        
    Returns:
        True if valid, False otherwise
    """
    import os
    file_size = os.path.getsize(filename)
    
    if file_size % RECORD_SIZE != 0:
        print(f"ERROR: File size ({file_size}) is not a multiple of record size ({RECORD_SIZE})")
        return False
    
    # Try to read all records
    try:
        count = 0
        for point in read_points(filename):
            count += 1
            
            # Validate y_parity
            if point.y_parity not in (0, 1):
                print(f"ERROR: Invalid y_parity ({point.y_parity}) at record {count}")
                return False
        
        print(f"File is valid: {count:,} records")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Parse and analyze BitCrack EC dump binary files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire file to CSV
  %(prog)s out.bin --csv out.csv
  
  # Extract first 1000 points
  %(prog)s out.bin --sample 1000 --csv sample.csv
  
  # Compute statistics
  %(prog)s out.bin --stats
  
  # Validate file integrity
  %(prog)s out.bin --validate
        """
    )
    
    parser.add_argument('input', help='Input binary point file')
    parser.add_argument('--csv', metavar='FILE', help='Convert to CSV file')
    parser.add_argument('--sample', type=int, metavar='N', help='Extract N points to CSV')
    parser.add_argument('--stats', action='store_true', help='Compute and print statistics')
    parser.add_argument('--validate', action='store_true', help='Validate file integrity')
    
    args = parser.parse_args()
    
    # Check if file exists
    import os
    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        return 1
    
    # Execute requested operation
    if args.validate:
        if not validate_file(args.input):
            return 1
    
    if args.stats:
        compute_stats(args.input)
    
    if args.csv:
        if args.sample:
            extract_sample(args.input, args.csv, args.sample)
        else:
            convert_to_csv(args.input, args.csv)
    
    # If no operation specified, show stats by default
    if not (args.csv or args.stats or args.validate):
        compute_stats(args.input)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
