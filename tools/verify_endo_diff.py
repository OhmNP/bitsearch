import struct
import sys
import os

# Secp256k1 Constants
P = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
N = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
BETA = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
LAMBDA = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
GX = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
GY = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8

# Initial point G
G = (GX, GY)

# Modular arithmetic helpers
def inverse_mod(a, m):
    if a == 0: return 0
    lm, hm = 1, 0
    low, high = a % m, m
    while low > 1:
        ratio = high // low
        nm, new = hm - lm * ratio, high - low * ratio
        lm, low, hm, high = nm, new, lm, low
    return lm % m

def point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if y1 != y2: return None # Infinity
        # Double
        lam = (3 * x1 * x1 * inverse_mod(2 * y1, P)) % P
    else:
        lam = ((y2 - y1) * inverse_mod(x2 - x1, P)) % P
    
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)

def point_mul(k, p):
    r = None
    while k > 0:
        if k % 2 == 1:
            r = point_add(r, p)
        p = point_add(p, p)
        k //= 2
    return r

def main():
    if len(sys.argv) < 2:
        print("Usage: verify_endo_diff.py <file.bin>")
        sys.exit(1)

    filename = sys.argv[1]
    
    # Record size: 60 bytes
    # struct EndoDiffRecord {
    #   uint8_t delta_phi_x[32]; // Output Δφ.x (Big-Endian)
    #   uint32_t delta_value;    // Input δ
    #   uint32_t batch_id;
    #   uint32_t index_in_batch;
    #   uint32_t flags;
    #   uint8_t reserved[12];
    # };
    RECORD_SIZE = 60

    file_size = os.path.getsize(filename)
    num_records = file_size // RECORD_SIZE
    print(f"File: {filename}, Size: {file_size}, Records: {num_records}")

    if num_records == 0:
        print("Error: No records found.")
        sys.exit(1)

    with open(filename, "rb") as f:
        data = f.read(RECORD_SIZE) # Read first record
        
        delta_phi_x_bytes = data[0:32]
        delta_val = struct.unpack("<I", data[32:36])[0]
        
        delta_phi_x = int.from_bytes(delta_phi_x_bytes, byteorder='big')
        
        print(f"Record 0: delta={delta_val}, delta_phi_x={hex(delta_phi_x)}")
        
        # Calculate expected: P = (lambda * delta) * G
        scalar = (LAMBDA * delta_val) % N
        expected_P = point_mul(scalar, G)
        
        if expected_P is None:
            expected_x = 0 # Infinity? Wait, delta_phi_x is 0 if infinity?
            # Kernel sets 0 if P2 == P1 (delta=0).
            # If delta=0, scalar=0, point_mul returns None.
            # But x coord of None is undefined. Kernel outputs 0.
            # Let's assume 0 for check.
            pass
        else:
            expected_x = expected_P[0]
            
        print(f"Expected: delta_phi_x={hex(expected_x)}")
        
        if expected_x == delta_phi_x:
            print("MATCH: Output matches expected value (lambda * delta * G).x")
            
            # Check all records (they should all be identical for same delta)
            f.seek(0)
            all_data = f.read()
            match_count = 0
            for i in range(num_records):
                offset = i * RECORD_SIZE
                rec_bytes = all_data[offset : offset + RECORD_SIZE]
                rec_x_bytes = rec_bytes[0:32]
                rec_delta = struct.unpack("<I", rec_bytes[32:36])[0]
                
                rec_x = int.from_bytes(rec_x_bytes, byteorder='big')
                if rec_x == expected_x and rec_delta == delta_val:
                    match_count += 1
            
            print(f"Verified {match_count}/{num_records} records match the expected constant value.")
            if match_count == num_records:
                print("SUCCESS: All records are correct.")
                sys.exit(0)
            else:
                print("FAILURE: Some records mismatch.")
                sys.exit(1)
                
        else:
            print("FAILURE: Mismatch in first record.")
            print(f"Got:      {hex(delta_phi_x)}")
            print(f"Expected: {hex(expected_x)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
