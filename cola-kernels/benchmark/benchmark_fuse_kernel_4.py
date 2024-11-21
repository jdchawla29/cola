import torch
from datetime import datetime
from torch.profiler import profiler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cola_kernels
import cola
from cola import Auto
from cola.linalg import Arnoldi

torch.manual_seed(42)

def eigen_pytorch(A_torch):
    start_time = datetime.now()
    torch_e, torch_v = torch.linalg.eig(A_torch)
    end_time = datetime.now()
    diff = end_time - start_time
    return torch_e, torch_v, diff.total_seconds()

def eigen_cola_gpu(A_cola, iters):
    start_time = datetime.now()
    A_cola = cola.ops.Dense(A_cola)
    Q_cola, H_cola, info = Arnoldi(max_iters=iters,pbar=True)(A_cola)
    Q_cola, H_cola = Q_cola.to_dense(), H_cola.to_dense()
    Q_cola, H_cola = Q_cola[:, :-1], H_cola[:-1]
    e_cola, v_cola = torch.linalg.eig(H_cola.to_dense())
    Q_cola = Q_cola.to(v_cola.dtype)
    ev_cola = Q_cola @ v_cola
    end_time = datetime.now()
    diff = end_time - start_time
    return e_cola, ev_cola, diff.total_seconds()

    
def eigen_cola_kernels(A, iters):
    start_time = datetime.now()
    H, Q = cola_kernels.ops.fuse_kernel_4(A, iters)
    H, Q  = H.reshape(iters, iters+1), Q.reshape(iters+1, A.shape[0])
    H, Q = H.T, Q.T
    H, Q = H[:-1], Q[:,:-1]
    e, v = torch.linalg.eig(H)
    Q = Q.to(v.dtype)
    ev = Q @ v
    end_time = datetime.now()
    diff = end_time - start_time
    return e, ev, diff.total_seconds()


def eigen_cola_cpu(A_cpu, iters):
    start_time = datetime.now()
    A_cola_cpu = cola.ops.Dense(A_cpu)
    Q_cpu_cola, H_cpu_cola, info = Arnoldi(max_iters=iters,pbar=True)(A_cola_cpu)
    Q_cpu_cola, H_cpu_cola = Q_cpu_cola.to_dense(), H_cpu_cola.to_dense()
    Q_cpu_cola, H_cpu_cola = Q_cpu_cola[:, :-1], H_cpu_cola[:-1]
    e_cpu_cola, v_cpu_cola = torch.linalg.eig(H_cpu_cola.to_dense())
    Q_cpu_cola = Q_cpu_cola.to(v_cpu_cola.dtype)
    ev_cpu_cola = Q_cpu_cola @ v_cpu_cola
    end_time = datetime.now()
    diff = end_time - start_time
    e_cpu_cola, ev_cpu_cola = e_cpu_cola.to("cuda:1"), ev_cpu_cola.to("cuda:1")
    return e_cpu_cola, ev_cpu_cola, diff.total_seconds()

def diff(a, b, tol=1e-03):
    assert a.size() == b.size(), "Tensors are not same size"
    sorted_tensor1, _ = torch.sort(torch.abs(a))
    sorted_tensor2, _ = torch.sort(torch.abs(b))
    min_length = min(len(sorted_tensor1), len(sorted_tensor2))
    diff = sorted_tensor1 - sorted_tensor2
    count_greater = torch.sum(torch.abs(diff) > tol).item()
    return count_greater


def compare(torch_e, e_cola, e, e_cola_cpu, torch_v, ev_cola, ev, ev_cola_cpu):
    print("eigen values different between torch and cola gpu: " + str(diff(torch_e, e_cola)))
    print("eigen values different between torch and cola kernel: " + str(diff(torch_e, e)))
    print("eigen values different between cola gpu and cola kernel: " + str(diff(e_cola, e)))
    print("eigen values different between torch and cola cpu: " + str(diff(torch_e, e_cola_cpu)))

    print("eigen vector different between torch and cola gpu: " + str(diff(torch_v, ev_cola, 1e-02)))
    print("eigen vector different between torch and cola kernel: " + str(diff(torch_v, ev, 1e-02)))
    print("eigen vector different between cola gpu and cola kernel: " + str(diff(ev_cola, ev, 1e-02)))
    print("eigen vector different between torch and cola cpu: " + str(diff(torch_v, ev_cola_cpu)))


configs = [(100, 100), (500, 500), (1000, 1000), (2000, 2000), (5000, 5000), (10000, 1000), (20000, 2000)]

pytorch_time = []
cola_python_gpu_time = []
cola_kernel_time = []
cola_python_cpu_time = []

def run():
    for config in configs: 
        n, iters = config
        A_cpu = torch.randn(n, n)
        A = A_cpu.clone().to(device="cuda:1")

        torch_e, torch_v, t_torch = eigen_pytorch(A)
        e_cola, ev_cola, t_cola = eigen_cola_gpu(A, iters)
        e, ev, t = eigen_cola_kernels(A, iters)
        e_cola_cpu, ev_cola_cpu, t_cola_cpu = eigen_cola_cpu(A_cpu, iters)

        print(f"n: {n}, pytorch:{t_torch} sec, cola:{t_cola} sec, cola-kernel:{t} sec, cola-cpu:{t_cola_cpu} sec")

        if n == 100 or n == 500 or n == 1000:
            compare(torch_e, e_cola, e, e_cola_cpu, torch_v, ev_cola, ev, ev_cola_cpu)
        pytorch_time.append(t_torch)
        cola_python_gpu_time.append(t_cola)
        cola_kernel_time.append(t)
        cola_python_cpu_time.append(t_cola_cpu)

        print("pytorch: ", pytorch_time)
        print("cola: " , cola_python_gpu_time)
        print("cola kernel: ", cola_kernel_time)
        print("cola cpu: ",cola_python_cpu_time)

run()
  
def draw_graphs():
    df = pd.DataFrame({
        'Configurations': [f"{n, i}" for (n, i) in configs],
        'pytorch': np.array(pytorch_time),
        'cola cuda': np.array(cola_python_gpu_time),
        'cola kernel': np.array(cola_kernel_time),
        'cola cpu': np.array(cola_python_cpu_time),
    })
    ax = df.plot(x='Configurations', y=['pytorch', 'cola cuda', 'cola kernel', 'cola cpu'], kind='bar', logy=True)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.2f')
    plt.xlabel('size(n, iternations)')
    plt.ylabel('Time in seconds (log scale)')
    plt.title(f'Eigen value calculations using Arnoldi algorithm')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cola-kernels/benchmark/speed.png")

    df = pd.DataFrame({
        'Configurations': [f"{n, i}" for (n, i) in configs],
        'cola cpu/cola kernel': np.array(cola_python_cpu_time) / np.array(cola_kernel_time),
        'cola cuda/cola kernel': np.array(cola_python_gpu_time) / np.array(cola_kernel_time),
    })
    ax = df.plot(x='Configurations', y=['cola cpu/cola kernel', 'cola cuda/cola kernel'], kind='bar')
    plt.xlabel('size(n, iternations)')
    plt.ylabel('Speedup')
    plt.title(f'Eigen value calculations using Arnoldi algorithm')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cola-kernels/benchmark/speedup.png")

# # with profiler.profile(
# #     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
# #     record_shapes=True,
# #     with_stack=True
# # ) as prof:
# #   H, Q = cola_kernels.ops.fuse_kernel_4(A, iters)
# # print(prof.key_averages().table(sort_by="cuda_time_total"))