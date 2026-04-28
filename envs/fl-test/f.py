

# flashattn test 

import torch
from flash_attn import flash_attn_func

def main():
    print("=" * 60)
    print("FlashAttention Environment Test")
    print("=" * 60)

    # Basic torch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA runtime version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Create test tensors
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64

    print("\nCreating test tensors...")

    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    k = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    v = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    print("Running FlashAttention kernel...")

    # Run FlashAttention
    out = flash_attn_func(q, k, v)

    print("\nSUCCESS!")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output device: {out.device}")

    print("\nFlashAttention is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()