import torch
import cola_kernels

# Try importing the C++ extension
try:
    import cola_kernels._C
    print("Successfully imported CUDA/C++ kernels")
except ImportError as e:
    print(f"Failed to import kernels: {e}")

# Print torch CUDA availability 
print(f"CUDA available: {torch.cuda.is_available()}")