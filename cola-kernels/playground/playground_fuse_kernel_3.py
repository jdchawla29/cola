"""
This file is computes the SVD implemented in CoLA kernels. 
The code was tested in cuda3 machine and the max size 
is 20000 for N. 
Beyond this we ran into CUDA out of memory issues

"""

import torch
import cola_kernels
import numpy as np

torch.manual_seed(42)


n = 8192
A1 = torch.randn(n, n).to("cuda")

U1, S1, Vt1 = cola_kernels.ops.fuse_kernel_3(A1)
print(U1.shape)