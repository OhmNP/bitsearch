# Building bitcrack-ecdump with Visual Studio 2022

## Prerequisites

1. **Visual Studio 2022** with C++ development tools
2. **CUDA Toolkit 12.6** (or compatible version)
3. **Windows 10/11 x64**

## Build Steps

### Option 1: Using Visual Studio IDE

1. Open `BitCrack.sln` in Visual Studio 2022
2. In Solution Explorer, you should see the new **ECDump** project
3. Right-click on **ECDump** and select "Set as Startup Project" (optional)
4. Select build configuration:
   - **Debug|x64** for debugging
   - **Release|x64** for optimized build
5. Build the solution: **Build → Build Solution** (or press `Ctrl+Shift+B`)
6. The executable will be created at: `bin\bitcrack-ecdump.exe`

### Option 2: Using MSBuild Command Line

Open **Developer Command Prompt for VS 2022** and run:

```cmd
cd C:\Users\parim\Desktop\projects\BitCrack

REM Build Release configuration
msbuild BitCrack.sln /p:Configuration=Release /p:Platform=x64 /t:ECDump

REM Or build Debug configuration
msbuild BitCrack.sln /p:Configuration=Debug /p:Platform=x64 /t:ECDump
```

## Project Configuration

The ECDump project is configured with:

- **Configuration Type**: Application (.exe)
- **Platform Toolset**: v143 (Visual Studio 2022)
- **CUDA Compute Capability**: sm_86 (RTX 3080 Ti)
- **C++ Standard**: C++17
- **Output**: `bin\bitcrack-ecdump.exe`

### Dependencies

The project automatically links against:
- secp256k1lib (static library)
- cudaUtil (static library)
- CryptoUtil (static library)
- util (static library)
- Logger (static library)
- CUDA runtime (cudart_static.lib)

## Testing the Build

After building, test the executable:

```cmd
cd bin

REM Show help
bitcrack-ecdump.exe --help

REM Run a small test (1024 keys, with verification)
bitcrack-ecdump.exe --family consecutive --start-k 0x1 ^
  --batch-keys 1024 --batches 1 ^
  --out-bin test.bin --telemetry test.json --verify
```

Expected output:
```
=== BitCrack EC Dump ===

Configuration validated:
  Family: consecutive
  Batch keys: 1024
  ...

Using CUDA device 0: NVIDIA GeForce RTX 3080 Ti
  ...

Verifying 1024 points against CPU reference...
Verification PASSED: 1024/1024 points match CPU reference

Batch 0: 1024 keys in X.XX ms (XXXXX keys/sec, kernel: X.XX ms, sampled: 1024 points)

=== Summary ===
Total keys processed: 1,024
...
```

## Troubleshooting

### "CUDA 12.6.props not found"

If you have a different CUDA version installed:

1. Open `tools\ECDump.vcxproj` in a text editor
2. Find lines with `CUDA 12.6.props` and `CUDA 12.6.targets`
3. Change `12.6` to your CUDA version (e.g., `12.0`, `11.8`, etc.)
4. Save and reload the project in Visual Studio

### "Cannot find CUDA installation"

Ensure the CUDA Toolkit is installed and the `CUDA_PATH` environment variable is set:

```cmd
echo %CUDA_PATH%
REM Should output something like: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

### Compute Capability Mismatch

If you have a different GPU, update the compute capability:

1. Open `tools\ECDump.vcxproj`
2. Find `<CodeGeneration>compute_86,sm_86</CodeGeneration>`
3. Change to your GPU's compute capability:
   - RTX 4090/4080: `compute_89,sm_89`
   - RTX 3090/3080: `compute_86,sm_86`
   - RTX 2080 Ti: `compute_75,sm_75`
   - GTX 1080 Ti: `compute_61,sm_61`

### Build Errors

If you encounter build errors:

1. **Clean and rebuild**: Right-click solution → Clean Solution, then rebuild
2. **Check dependencies**: Ensure all dependency projects build successfully first
3. **Update include paths**: Verify that `BitCrack.props` has correct paths
4. **Check CUDA version**: Ensure CUDA Toolkit version matches project settings

## Next Steps

After successful build:

1. Read `tools\README_ecdump.md` for usage guide
2. Try different key families (control, masked, stride, hd)
3. Use `tools\parse_ecdump.py` to analyze output files
4. Experiment with batch sizes and sampling rates

## Performance Notes

- **Batch size**: Start with 1M keys (`--batch-keys 1048576`)
- **GPU memory**: Monitor with `nvidia-smi` during runs
- **Optimization**: Use Release build for best performance (2-3x faster than Debug)
