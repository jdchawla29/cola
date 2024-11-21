"""

This file is computes the estimates the diagonals using Hutchinson algorithm 
implemented in CoLA kernels. The code was testing in cuda3 machine and the max size 
is 20000 for N. Beyond this we ran into CUDA out of memory issues


"""
import torch
import cola_kernels

N = 5000
A = torch.eye(N, N).to(device="cuda:1")
V = torch.zeros(N).to(device="cuda:1")

D = cola_kernels.ops.fuse_kernel_5(A, V)
print(torch.sum(D, dim=0))

