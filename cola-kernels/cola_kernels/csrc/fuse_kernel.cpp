#include <torch/extension.h>

namespace cola_kernels {

at::Tensor fuse_kernel_1_cpu(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.sizes() == c.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(c.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(c.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor c_contig = c.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  const float* c_ptr = c_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c_ptr[i];
  }
  return result;
}

at::Tensor fuse_kernel_2_cpu(at::Tensor& a) {
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  float* a_ptr = a_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  // TODO:
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i];
  }
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fuse_kernel_3_cpu(at::Tensor& a) {
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  float* a_ptr = a_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  // TODO:
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i];
  }
  return std::make_tuple(result, result, result);
}

std::tuple<at::Tensor, at::Tensor> fuse_kernel_4_cpu(at::Tensor& a, int64_t max_iters) {
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  float* a_ptr = a_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  // TODO:
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i];
  }
  return std::make_tuple(result, result);
}

at::Tensor fuse_kernel_5_cpu(
  at::Tensor& a,
  at::Tensor& b,
  int64_t batch_size,
  double tolerance,
  int64_t max_iterations,
  int64_t diagonal_offset,
  bool use_rademacher
) {
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  float* a_ptr = a_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  // TODO:
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i];
  }
  return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(cola_kernels, m) {
  m.def("fuse_kernel_1(Tensor a, Tensor b, Tensor c) -> Tensor");
  m.def("fuse_kernel_2(Tensor a) -> Tensor");
  m.def("fuse_kernel_3(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("fuse_kernel_4(Tensor a, int max_iters) -> (Tensor, Tensor)");
  m.def("fuse_kernel_5(Tensor a, Tensor b, int batch_size, float tolerance, int max_iterations, int diagonal_offset, bool use_rademacher) -> Tensor");
}

TORCH_LIBRARY_IMPL(cola_kernels, CPU, m) {
  m.impl("fuse_kernel_1", &fuse_kernel_1_cpu);
  m.impl("fuse_kernel_2", &fuse_kernel_2_cpu);
  m.impl("fuse_kernel_3", &fuse_kernel_3_cpu);
  m.impl(TORCH_SELECTIVE_NAME("fuse_kernel_4"), [](
        at::Tensor& a, 
        int64_t max_iters) {
            return fuse_kernel_4_cpu(a, max_iters);
    });
   m.impl(TORCH_SELECTIVE_NAME("fuse_kernel_5"), [](
        at::Tensor& mat, 
        at::Tensor& diag,
        int64_t batch_size,
        double tolerance,
        int64_t max_iterations,
        int64_t diagonal_offset,
        bool use_rademacher) {
            return fuse_kernel_5_cpu(mat, diag, batch_size, tolerance, max_iterations, diagonal_offset, use_rademacher);
    });
}


}