import torch
from torch import Tensor

__all__ = ["fuse_kernel_1", "fuse_kernel_2", "fuse_kernel_3", "fuse_kernel_4", "fuse_kernel_5"]


def fuse_kernel_1(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.cola_kernels.fuse_kernel_1.default(a, b, c)

def fuse_kernel_2(a: Tensor) -> Tensor:
    """Performs inverse of A efficiently"""
    return torch.ops.cola_kernels.fuse_kernel_2.default(a)

def fuse_kernel_3(a: Tensor) -> Tensor:
    """Performs SVD of A efficiently"""
    return torch.ops.cola_kernels.fuse_kernel_3.default(a)

def fuse_kernel_4(a: Tensor, max_iters: int = 100) -> Tensor:
    """Performs Arnoldi of A efficiently"""
    return torch.ops.cola_kernels.fuse_kernel_4.default(a)

def fuse_kernel_5(mat: Tensor, 
                  diag: Tensor,
                  batch_size: int = 100,
                  tolerance: float = 3e-2,
                  max_iter: int = 1000,
                  diagonal_offset: int = 0,
                  use_rademacher: bool = False,
                  ) -> Tensor:
    """Performs Hutch Diagonal estimation of A efficiently"""
    return torch.ops.cola_kernels.fuse_kernel_5.default(
        mat, diag, batch_size, tolerance, max_iter, diagonal_offset, use_rademacher
        )

@torch.library.register_fake("cola_kernels::fuse_kernel_1")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.shape == c.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(c.dtype == torch.float)
    torch._check(a.device == b.device)
    torch._check(a.device == c.device)
    return torch.empty_like(a)

def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b, grad_c = None, None, None
    if ctx.needs_input_grad[0]:
        grad_a = grad * b
    if ctx.needs_input_grad[1]:
        grad_b = grad * a
    if ctx.needs_input_grad[2]:
        grad_c = grad.detach().clone()
    return grad_a, grad_b, grad_c


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


torch.library.register_autograd(
    "cola_kernels::fuse_kernel_1", _backward, setup_context=_setup_context)


@torch.library.register_fake("cola_kernels::fuse_kernel_2")
def _(a):
    torch._check(a.dtype == torch.float)
    return torch.empty_like(a)


@torch.library.register_fake("cola_kernels::fuse_kernel_3")
def _(a):
    torch._check(a.dtype == torch.float)
    return torch.empty_like(a), torch.empty_like(a), torch.empty_like(a)

@torch.library.register_fake("cola_kernels::fuse_kernel_4")
def _(
    a: Tensor,
    max_iters: int
):
    torch._check(a.dtype == torch.float)
    return torch.empty_like(a), torch.empty_like(a)

@torch.library.register_fake("cola_kernels::fuse_kernel_5")
def _(
    mat: Tensor, 
    diag: Tensor,
    batch_size: int,
    tolerance: float,
    max_iterations: int,
    diagonal_offset: int,
    use_rademacher: bool
) -> Tensor:
    torch._check(mat.dtype == torch.float)
    torch._check(diag.dtype == torch.float)
    return torch.empty_like(diag)