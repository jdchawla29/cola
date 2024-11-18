import torch
import torch.profiler as profiler
import cola_kernels

n = 1024
A = torch.randn((n, n))
A = torch.matmul(A, A.T) + torch.eye(n) * 1e-3
A = A.to("cuda")
A2 = A.detach().clone()

with profiler.profile(activities=[profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU]) as prof:
    L = torch.linalg.cholesky(A)
    D = torch.cholesky_inverse(L)
    
print(prof.key_averages().table(sort_by="cuda_time_total"))

with profiler.profile(activities=[profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU]) as prof:
    D = cola_kernels.ops.fuse_kernel_2(A2)

print(prof.key_averages().table(sort_by="cuda_time_total"))