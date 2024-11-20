import torch
import cola_kernels
import numpy as np

torch.manual_seed(42) #1000


import torch

def is_identity_product(A, A_Inv, tol=0.1):
    identity = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    product = torch.matmul(A, A_Inv)
    return torch.allclose(product, identity, atol=tol, rtol=tol)


# Define matrix A
n = 2048

A = torch.randn(n, n)
A = torch.matmul(A, A.T) + torch.eye(n) * 1e-3
A = A.to("cuda")

A3 = A.detach().clone()

R1 = cola_kernels.ops.fuse_kernel_2(A)

# Verify if A @ A_Inv is close to identity
result = is_identity_product(R1, A3)
print("Checking if A @ A_Inv = Identity: ", result)


# Define matrix A
# n = 2048

# A = torch.randn(n, n)
# A = torch.matmul(A, A.T) + torch.eye(n) * 1e-3
# A = A.to("cuda")

# # A2 = A.detach().clone()

# A3 = A.detach().clone()

# # Compute results
# R1 = cola_kernels.ops.fuse_kernel_2(A)
# R1 = R1.to("cpu")

# A_np = A.cpu().numpy()
# L = np.linalg.cholesky(A_np)  # A = L @ L.T
# L_inv = np.linalg.inv(L)  # Inverse of the lower triangular matrix
# R1 = L_inv.T @ L_inv  # A_inv = (L^T)^(-1) @ L^(-1)
# R1 = torch.from_numpy(R1)

# R2 = torch.linalg.inv(A2)

# Check if they are close
# tolerance = 0.1
# is_close = torch.isclose(R1, R2, atol=tolerance)
# all_close = is_close.all()

# print("All values close within tolerance:", all_close.item())

# if not all_close:
#     # Count the number of values outside the tolerance
#     not_close = ~is_close  # Negate is_close to get values outside the tolerance
#     num_not_close = not_close.sum().item()  # Count the number of True values in not_close

#     print(f"Number of values not within the tolerance: {num_not_close}")

#     # Calculate and print differences
#     abs_diff = torch.abs(R1 - R2)
#     max_abs_diff = torch.max(abs_diff).item()
#     mean_abs_diff = torch.mean(abs_diff).item()
#     relative_diff = torch.abs((R1 - R2) / (R2 + 1e-6))  # Avoid division by zero
#     max_rel_diff = torch.max(relative_diff).item()
    
#     print(f"Maximum absolute difference: {max_abs_diff}")
#     print(f"Mean absolute difference: {mean_abs_diff}")
#     print(f"Maximum relative difference: {max_rel_diff}")

# Verify if A @ A_Inv is close to identity
# result = is_identity_product(R1, A3)
# print("Checking if A @ A_Inv = Identity: ", result)
# result = is_identity_product(R2, A3)
# print("R2: ", result)
# A = A.to("cpu")
# result = is_identity_product(A, A3)
# print("A: ", result)
