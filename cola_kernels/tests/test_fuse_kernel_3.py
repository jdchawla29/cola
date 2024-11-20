import torch
import cola_kernels
import numpy as np

torch.manual_seed(42) #1000


n = 2048
A1 = torch.randn(n, n).to("cuda")
A2 = A1.detach().clone().cpu().numpy()

U1, S1, Vt1 = cola_kernels.ops.fuse_kernel_3(A1)
U2, S2, Vt2 = np.linalg.svd(A2)

S1, _ = torch.sort(S1, descending=True)
print("Comparing Sigma generated with numpy: ", torch.allclose(S1.cpu(), torch.from_numpy(S2), rtol=1e-02, atol=1e-02, equal_nan=False))

# Define matrix A
# n = 3

# A = torch.randn(33, 100)
# A = A.to("cuda")
# A2 = A.detach().clone()
# A3 = A.cpu().detach().clone().numpy()

# print(A)
# u, s, vt = cola_kernels.ops.fuse_kernel_3(A)
# print("U: ", u)
# print("S: ", s)
# print("VT: ", vt)

# U, S, Vh = torch.linalg.svd(A2)
# # Create a diagonal matrix for singular values with appropriate shape
# S_diag = torch.zeros_like(A2, dtype=A2.dtype)  # Same shape as A2
# S_diag[:S.size(0), :S.size(0)] = torch.diag(S)

# # Reconstruct A2
# A2_reconstructed = U @ S_diag @ Vh

# print("Original A2: ", A2)
# print("Reconstructed A2: ", A2_reconstructed)

# # Verify reconstruction
# print("Reconstruction successful: ", torch.allclose(A2, A2_reconstructed, atol=1e-6))


# # Perform SVD decomposition
# U, S, Vh = np.linalg.svd(A3)

# # Convert singular values (1D array) to a diagonal matrix
# S_diag = np.zeros((U.shape[1], Vh.shape[0]))
# np.fill_diagonal(S_diag, S)

# # Reconstruct the matrix
# A3_reconstructed = U @ S_diag @ Vh

# # Print results
# print("Original A3: ", A3)
# print("Reconstructed A3: ", A3_reconstructed)

# # Verify reconstruction
# print("Reconstruction successful: ", np.allclose(A3, A3_reconstructed, atol=1e-6))



# Step 1: Define a matrix A
# A = np.array([[4, 1], [2, 3], [1, 5]])
# A1 = torch.from_numpy(A).to(dtype=torch.float32, device="cuda")

# Step 2: Compute A^T A and A A^T
# A_T_A = np.dot(A.T, A)  # Compute A^T A
# A_A_T = np.dot(A, A.T)  # Compute A A^T

# # Step 3: Compute eigenvalues and eigenvectors
# eigvals_A_T_A, eigvecs_A_T_A = np.linalg.eig(A_T_A)  # Eigenvalues and Eigenvectors of A^T A
# eigvals_A_A_T, eigvecs_A_A_T = np.linalg.eig(A_A_T)  # Eigenvalues and Eigenvectors of A A^T

# # Print the results
# print("A^T A:\n", A_T_A)
# print("Eigenvalues of A^T A:", eigvals_A_T_A)
# print("Eigenvectors of A^T A:\n", eigvecs_A_T_A)

# print("\nA A^T:\n", A_A_T)
# print("Eigenvalues of A A^T:", eigvals_A_A_T)
# print("Eigenvectors of A A^T:\n", eigvecs_A_A_T)

# u, s, vt = cola_kernels.ops.fuse_kernel_3(A1)

# print(torch.matmul(u, u.T))
# print("U: ", u)
# print("S: ", s)
# print("VT: ", vt)

# S_mat = torch.zeros(3, 2, device=A1.device)
# S_mat[:s.size(0), :s.size(0)] = torch.diag(s)  # Create a diagonal matrix from singular values
# A_reconstructed = torch.matmul(u, torch.matmul(S_mat, vt)) 
# print(A_reconstructed)


# u, s, vt = np.linalg.svd(A)
# print(np.matmul(vt, vt.T))
# print("U: ", u)
# print("S: ", s)
# print("VT: ", vt)

