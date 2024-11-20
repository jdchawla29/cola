import cola_kernels
import torch

n = 1000 
max_iters = 100 
A = torch.randn(n,n, device='cuda:1')

H, Q = cola_kernels.ops.fuse_kernel_4(A, max_iters)

print(H.shape)
print(Q.shape)


n = 1000

A = torch.eye(n, device='cuda:1')

estimated_trace = cola_kernels.ops.fuse_kernel_5(A, diag=torch.zeros(n, device='cuda:1')).sum()
print(estimated_trace)