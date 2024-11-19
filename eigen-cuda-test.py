import torch
import cola_kernels
from datetime import datetime

torch.manual_seed(42) #1000

n = 100

A = torch.randn(n, n).to(device="cuda")
H, Q = cola_kernels.ops.fuse_kernel_4(A)

def count_nan_and_zeros(tensor):
    # Count NaN values using torch.isnan
    nan_count = torch.sum(torch.isnan(tensor)).item()  # Sum the boolean mask to count NaN values

    # Count zero values
    zero_count = torch.sum(tensor == 0).item()  # Sum the boolean mask to count zero values

    # Print the counts
    print(f"Number of NaN values: {nan_count}")
    print(f"Number of zero values: {zero_count}")


H  = H.reshape(101, 100)
H = H[:-1]
Q = Q.reshape(100, 101)
Q = Q[:,:-1]
e, v = torch.linalg.eig(H)
Q = Q.to(v.dtype)
ev = Q @ v
print(e.shape)
print(ev.shape)

start_time = datetime.now()
te, tv = torch.linalg.eig(A)
end_time = datetime.now()
time_difference = end_time - start_time
print("Total Seconds:", time_difference.total_seconds())
print(te.shape)
print(tv.shape)

sorted_tensor1, _ = torch.sort(torch.abs(e))
sorted_tensor2, _ = torch.sort(torch.abs(te))
min_length = min(len(sorted_tensor1), len(sorted_tensor2))
diff = sorted_tensor1[:min_length] - sorted_tensor2[:min_length]
print(diff)
threshold = 1e-04
count_greater = torch.sum(torch.abs(diff) > threshold).item()
print(count_greater)
