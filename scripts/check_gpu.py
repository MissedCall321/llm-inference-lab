"""Check CUDA visibility and execute real GPU operations before downloading a model."""

import torch


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Check the Windows NVIDIA driver and Docker GPU access.")

    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name}")
    print(f"Total VRAM: {properties.total_memory / 1024**3:.2f} GiB")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    result = x @ x.T
    if not torch.isfinite(result).all().item():
        raise SystemExit("GPU computation produced non-finite values.")
    torch.cuda.synchronize()
    print("PASS: CUDA allocation and FP16 matrix multiplication succeeded.")


if __name__ == "__main__":
    main()
