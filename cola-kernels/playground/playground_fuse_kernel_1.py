"""

This file is computes the A * B + C implemented in CoLA kernels. 
The code was tested in cuda3 machine and the max size ask Rahul Raman
for N. Beyond this we ran into CUDA out of memory issues


"""
import torch
import cola_kernels

N = 1024
A = torch.randn((N, N), device='cuda')
B = torch.randn((N, N), device='cuda')
C = torch.randn((N, N), device='cuda')

D = cola_kernels.ops.fuse_kernel_1(A, B, C)
print(D.shape)