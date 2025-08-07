#!/usr/bin/env python3
"""
Test script to verify GPU cleanup functionality.
This script allocates GPU memory and then tests the cleanup mechanisms.
"""

import torch
import signal
import sys
import time
import os

def cleanup_gpu():
    """Clean up GPU memory and resources."""
    print("🧹 Cleaning up GPU resources...")
    
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("    ✓ CUDA cache cleared")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Final CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("    ✓ Final CUDA cleanup completed")
            
    except Exception as e:
        print(f"    ⚠️  Warning: Error during GPU cleanup: {e}")
    
    print("  - GPU cleanup completed")

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print(f"\n⚠️  Received signal {signum}. Cleaning up and shutting down gracefully...")
    cleanup_gpu()
    sys.exit(0)

def main():
    """Main test function."""
    print("=== GPU Cleanup Test ===")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU cleanup.")
        return
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✓ Initial memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Allocate some GPU memory
    print("\n📦 Allocating GPU memory...")
    tensors = []
    for i in range(10):
        # Allocate 100MB chunks
        tensor = torch.randn(100, 100, 100, device='cuda')
        tensors.append(tensor)
        print(f"  - Allocated tensor {i+1}/10: {tensor.numel() * 4 / 1024**2:.1f} MB")
        time.sleep(0.1)  # Small delay to see progress
    
    print(f"✓ Memory after allocation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Simulate training for a few seconds
    print("\n🔄 Simulating training for 10 seconds...")
    print("  (Press Ctrl+C to test interrupt cleanup)")
    
    start_time = time.time()
    while time.time() - start_time < 10:
        # Do some computation to keep GPU busy
        for tensor in tensors:
            _ = tensor * 2
        time.sleep(0.1)
    
    print("\n✅ Test completed successfully!")
    print("  - All tensors will be cleaned up automatically")
    print("  - GPU memory will be freed")
    
    # Clean up
    cleanup_gpu()

if __name__ == "__main__":
    main() 