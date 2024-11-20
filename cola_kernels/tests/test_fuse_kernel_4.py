import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import cola_kernels
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import cola
from cola import Auto
from cola.linalg import Arnoldi

def torch_reference_fuse_kernel_4(a):
    e, v = torch.linalg.eig(a)
    return e, v

def cola_reference_fuse_kernel_4(a):
    A_cola = cola.ops.Dense(a)
    Q_cola, H_cola, info = Arnoldi(max_iters=iters,pbar=True)(A_cola)
    Q_cola, H_cola = Q_cola.to_dense(), H_cola.to_dense()
    Q_cola, H_cola = Q_cola[:, :-1], H_cola[:-1]
    e_cola, v_cola = torch.linalg.eig(H_cola.to_dense())
    Q_cola = Q_cola.to(v_cola.dtype)
    ev_cola = Q_cola @ v_cola
    return e_cola, ev_cola

def cola_fuse_kernel_4(a, iters):
    H, Q = cola_kernels.ops.fuse_kernel_4(a, iters)
    H, Q  = H.reshape(iters, iters+1), Q.reshape(iters+1, A.shape[0])
    H, Q = H.T, Q.T
    H, Q = H[:-1], Q[:,:-1]
    e, v = torch.linalg.eig(H)
    Q = Q.to(v.dtype)
    ev = Q @ v
    return e, ev


class TestFuseKernel4(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn((size,size), device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn((size,size), device=device, requires_grad=False)

        return [
            [make_nondiff_tensor(10), 10],
            [make_nondiff_tensor(100), 100],
            [make_nondiff_tensor(500), 500],
            [make_nondiff_tensor(1000), 1000],
        ]

    def sort(e, v):
        return torch.sort(torch.abs(e)), torch.sort(torch.abs(v))

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            e_result, v_result = cola_fuse_kernel_4(*args)
            torch_e, torch_ev = torch_reference_fuse_kernel_4(*args)
            cola_e, cola_ev = cola_reference_fuse_kernel_4(*args)

            e_result, v_result = sort(e_result, v_result)
            torch_e, torch_ev = sort(torch_e, torch_ev)
            cola_e, cola_ev = sort(cola_e, cola_ev)

            torch.testing.assert_close(e_result, torch_e, msg="eigen values from cola kernels does not pytorch eigen values", atol=1e-4)
            torch.testing.assert_close(e_result, cola_e, msg="eigen values from cola kernels does not cola eigen values", atol=1e-4)
            torch.testing.assert_close(v_result, torch_ev, msg="eigen vectors from cola kernels does not pytorch eigen values", atol=1e-3)
            torch.testing.assert_close(v_result, cola_ev, msg="eigen vectors from cola kernels does not cola eigen values", atol=1e-3)

    def test_correctness_cpu(self):
        self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    def _test_gradients(self, device):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = cola_kernels.ops.fuse_kernel_1(*args)
            grad_out = torch.randn_like(out)
            result = torch.autograd.grad(out, diff_tensors, grad_out)

            out = reference_fuse_kernel_1(*args)
            expected = torch.autograd.grad(out, diff_tensors, grad_out)
            
            torch.testing.assert_close(result, expected)

    def test_gradients_cpu(self):
        self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.cola_kernels.fuse_kernel_1.default, args)

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")

if __name__ == "__main__":
    unittest.main()