import torch
import torch.profiler as profiler
import cola_kernels

n = 8192
A = torch.randn((n, n))
A3 = A.clone()
A = A.to("cuda")
A2 = A.detach().clone()

with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
profile_memory=True, record_shapes=True) as prof:
    S, V, D = torch.linalg.svd(A)

print(prof.key_averages().table())

# with open("profiler_output.txt", "w") as f:
#     f.write(prof.key_averages().table(sort_by="cuda_time_total"))

with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
profile_memory=True, record_shapes=True) as prof:
    S, V, D = cola_kernels.ops.fuse_kernel_3(A2)

print(prof.key_averages().table())

# with open("profiler_output2.txt", "w") as f:
#     f.write(prof.key_averages().table(sort_by="cuda_time_total"))

# with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
# profile_memory=True, record_shapes=True) as prof:
#     S, V, D = torch.linalg.svd(A3)

# print(prof.key_averages().table())
