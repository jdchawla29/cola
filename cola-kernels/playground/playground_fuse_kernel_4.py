"""

This file is computes the eigen values and vectors using Arnoldi algorithm 
implemented in CoLA kernels. The code was tested in cuda3 machine and the max size 
is 20000 for N and 5000 for iterations. Beyond this we ran into CUDA out of memory issues


"""
import torch
import cola_kernels

N = 1000
iterations = 100
A = torch.randn(N, N).to(device="cuda")

H, Q = cola_kernels.ops.fuse_kernel_4(A, iterations)
H, Q  = H.reshape(iterations, iterations+1), Q.reshape(iterations+1, A.shape[0])
H, Q = H.T, Q.T
H, Q = H[:-1], Q[:,:-1]
e, v = torch.linalg.eig(H)
Q = Q.to(v.dtype)
ev = Q @ v
print(e.shape)
print(ev.shape)