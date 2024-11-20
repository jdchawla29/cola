import torch
from torch.testing._internal.common_utils import TestCase
import unittest
import cola_kernels
import cola
from cola.linalg import Arnoldi

def torch_reference_fuse_kernel_4(a):
    e, v = torch.linalg.eig(a)
    return e, v

def cola_reference_fuse_kernel_4(a, iters):
    A_cola = cola.ops.Dense(a)
    Q_cola, H_cola, info = Arnoldi(max_iters=iters,pbar=False)(A_cola)
    Q_cola, H_cola = Q_cola.to_dense(), H_cola.to_dense()
    Q_cola, H_cola = Q_cola[:, :-1], H_cola[:-1]
    e_cola, v_cola = torch.linalg.eig(H_cola.to_dense())
    Q_cola = Q_cola.to(v_cola.dtype)
    ev_cola = Q_cola @ v_cola
    return e_cola, ev_cola

def cola_fuse_kernel_4(a, iters):
    H, Q = cola_kernels.ops.fuse_kernel_4(a, iters)
    H, Q  = H.reshape(iters, iters+1), Q.reshape(iters+1, a.shape[0])
    H, Q = H.T, Q.T
    H, Q = H[:-1], Q[:,:-1]
    e, v = torch.linalg.eig(H)
    Q = Q.to(v.dtype)
    ev = Q @ v
    return e, ev

def sort(e, v):
    e_sort, _ = torch.sort(torch.abs(e))
    v_sort, _ = torch.sort(torch.abs(v))
    return e_sort, v_sort

class TestFuseKernel4(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn((size,size), device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(size):
            return torch.randn((size,size), device=device, requires_grad=False)

        return [
            [make_nondiff_tensor(10), 10],
            [make_nondiff_tensor(100), 100],
            [make_nondiff_tensor(500), 500],
            [make_nondiff_tensor(1000), 1000],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for (a, iters) in samples:
            e_result, v_result = cola_fuse_kernel_4(a, iters)
            torch_e, torch_ev = torch_reference_fuse_kernel_4(a)
            cola_e, cola_ev = cola_reference_fuse_kernel_4(a, iters)

            e_result, v_result = sort(e_result, v_result)
            torch_e, torch_ev = sort(torch_e, torch_ev)
            cola_e, cola_ev = sort(cola_e, cola_ev)
            torch.testing.assert_close(e_result, torch_e, msg="eigen values from cola kernels does not match pytorch eigen values", atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(e_result, cola_e, msg="eigen values from cola kernels does not matchncola eigen values", atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(v_result, torch_ev, msg="eigen vectors from cola kernels does match not pytorch eigen values", atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(v_result, cola_ev, msg="eigen vectors from cola kernels does match not cola eigen values", atol=1e-2, rtol=1e-2)


    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda:1")


if __name__ == "__main__":
    unittest.main()