"""

This file is computes the inverse of a matrix using Cholesky algorithm
implemented in CoLA kernels. The code was testing in cuda3 machine and the max size N 4096. 
Beyond this it is very slow.


"""
import torch
import cola_kernels

N = 1000
A = torch.randn((N, N)).to("cuda")
A = torch.matmul(A, A.T).to("cuda") + torch.eye(N).to("cuda") * 1e-3
A2 = A.detach().clone()

D = cola_kernels.ops.fuse_kernel_2(A2)
print(D.shape)