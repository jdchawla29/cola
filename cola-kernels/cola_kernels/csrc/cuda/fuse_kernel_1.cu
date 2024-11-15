#include <stdio.h>
#include <stdlib.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace cola_kernels {

__global__ void fuse_kernel_1(int numel, const float* a, const float* b, const float* c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c[idx];
}

at::Tensor fuse_kernel_1_cuda(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.sizes() == c.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(c.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(c.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor c_contig = c.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  const float* c_ptr = c_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int numel = a_contig.numel();
  fuse_kernel_1<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c_ptr, result_ptr);
  return result;
}

#define MATIDX(i,j,N) (j * N + i)

__global__ void decompose_cholesky_mm_kernel_device(float *a, int N) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    cg::grid_group grid = cg::this_grid();

    // Loop over k
    for (int k = 0; k < N; k++) {

        // Compute diagonal element in the first thread
        if (thread_id == 0) {
            //printf("sqrt: %f\n", a[MATIDX(k, k, N)]);
            a[MATIDX(k, k, N)] = sqrt(a[MATIDX(k, k, N)]);

            // Update column elements by dividing by the diagonal
            for (int j = k + 1; j < N; j++) {
                //printf("div: %f %f\n", a[MATIDX(j, k, N)], a[MATIDX(k, k, N)]);
                a[MATIDX(j, k, N)] /= a[MATIDX(k, k, N)];
            }
        }

        grid.sync(); // Synchronize threads after updating the diagonal

        // Update the rest of the matrix, only threads that handle i > k
        int i = thread_id + k + 1; // Global row index
        if (i < N) {
            for (int j = i; j < N; j++) {
                //printf("parallel: %f %f %f\n", a[MATIDX(i, j, N)], a[MATIDX(i, k, N)], a[MATIDX(j, k, N)]);
                a[MATIDX(j, i, N)] -= a[MATIDX(i, k, N)] * a[MATIDX(j, k, N)];
            }
        }

        grid.sync(); // Synchronize threads after updating the matrix
    }

    // Zero out the upper triangular part of the matrix after decomposition (only for thread 0)
    if (thread_id == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < i; j++) {
                a[MATIDX(j, i, N)] = 0;
            }
        }
    }
    // int row = thread_id / N;
    // int col = thread_id % N;

    // if (row < N && col < N && col > row) {
    //     a[MATIDX(row, col, N)] = 0;
   // }
}
__global__
void inverse_lower_mm_kernel_device(float *a, float *aInv, int N) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    cg::grid_group grid = cg::this_grid();

    if (threadIdx.x == 0) {
        // Initialize `aInv` to zero
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                aInv[MATIDX(i, j, N)] = 0;
            }
        }
    }

    // int i = thread_id / N; // Row index
    // int j = thread_id % N; // Column index

    // // Check if the thread corresponds to a lower triangular element (including diagonal)
    // if (i < N && j <= i) {
    //     aInv[MATIDX(i, j, N)] = 0;
    // }

   grid.sync();

    // Compute the elements of the lower inverse matrix
    for (int j = 0; j < N; j++) {
        for (int i = j + 1; i < N; i++) {

            if (thread_id == 0) {
                aInv[MATIDX(i, j, N)] = -a[MATIDX(i, j, N)] / 
                                                    (a[MATIDX(j, j, N)] * a[MATIDX(i, i, N)]);
            }
            grid.sync();

            int k = thread_id + j + 1;
            if (k < i) {
                atomicAdd((float*)&aInv[MATIDX(i, j, N)], 
                          -a[MATIDX(i, k, N)] * aInv[MATIDX(k, j, N)] / a[MATIDX(i, i, N)]);
            }
        }
    }
    grid.sync();

    if (thread_id == 0) {
        // Set the inverse of the diagonal elements and copy results to `aInv`
        for (int i = 0; i < N; i++) {
            aInv[MATIDX(i, i, N)] = 1.0 / a[MATIDX(i, i, N)];
            for (int j = 0; j <= i; j++) {
                // Set only the lower triangular values in `aInv`
                aInv[MATIDX(i, j, N)] = aInv[MATIDX(i, j, N)];
            }
        }
    }
}

__global__
void multiply_lower_mm_kernel_device(float *a, float *aInv, int N) {

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    cg::grid_group grid = cg::this_grid();

    // Perform multiplication directly in `aInv`
    for (int j = 0; j < N; j++) {
        for (int i = j; i < N; i++) {

            if (thread_id == 0) {
                aInv[MATIDX(i, j, N)] *= aInv[MATIDX(i, i, N)];
            }
            grid.sync();

            int k = thread_id + i + 1;
            if (k < N) {
                atomicAdd((float*)&aInv[MATIDX(i, j, N)], 
                          aInv[MATIDX(k, j, N)] * aInv[MATIDX(k, i, N)]);
            }
        }
    }
    grid.sync();

    if (thread_id == 0) {
        // Copy the results into the full lower and symmetric upper triangle of `aInv`
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                aInv[MATIDX(j, i, N)] = aInv[MATIDX(i, j, N)];
            }
        }
    }
    // int i = thread_id / N; // Row index
    // int j = thread_id % N; // Column index

    // // Ensure the thread is within bounds and in the upper triangular part (j <= i)
    // if (i < N && j <= i) {
    //     aInv[MATIDX(j, i, N)] = aInv[MATIDX(i, j, N)];
    // }
}


at::Tensor fuse_kernel_2_cuda(at::Tensor& a){
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    float* a_ptr = a_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    int N = static_cast<int>(a.size(0));
    // printf("N = %d\n", N);
    // TEST
    // at::Tensor a_cpu = a.cpu();  
    // // Ensure it is contiguous on the CPU
    // at::Tensor a_contig_cpu = a_cpu.contiguous();
    // float* a_ptr_cpu = a_contig_cpu.data_ptr<float>();
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         float value = a_ptr_cpu[i * 3 + j];
    //         printf("A[%d, %d] = %f\n", i, j, value);
    //     }
    // }
    //OVER

    // decompose_cholesky_mm_kernel_device<<< 1, N>>>(a_ptr, N);

    // cudaDeviceSynchronize();
    //Kernel launch configuration for cooperative kernel
    dim3 blockSize(256);  // Number of threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);  // Number of blocks
    // Kernel launch with cooperative groups
    void* args1[] = { &a_ptr, &N };
    cudaLaunchCooperativeKernel((void*)decompose_cholesky_mm_kernel_device, gridSize, blockSize, args1, 0, 0);
    cudaDeviceSynchronize();


    // printf("AFTER decompose\n");
    // a_cpu = a.cpu();  
    // // Ensure it is contiguous on the CPU
    // a_contig_cpu = a_cpu.contiguous();
    // a_ptr_cpu = a_contig_cpu.data_ptr<float>();
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         float value = a_ptr_cpu[i * 3 + j];
    //         printf("A[%d, %d] = %f\n", i, j, value);
    //     }
    // }
    // inverse_lower_mm_kernel_device<<< 1, N>>>(a_ptr, result_ptr, N);
    // Kernel launch with cooperative groups
    void* args2[] = { &a_ptr, &result_ptr, &N };
    cudaLaunchCooperativeKernel((void*)inverse_lower_mm_kernel_device, gridSize, blockSize, args2, 0, 0);
    cudaDeviceSynchronize();
    // printf("AFTER inverse\n");
    // a_cpu = a.cpu();  
    // // Ensure it is contiguous on the CPU
    // a_contig_cpu = a_cpu.contiguous();
    // a_ptr_cpu = a_contig_cpu.data_ptr<float>();
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         float value = a_ptr_cpu[i * 3 + j];
    //         printf("A[%d, %d] = %f\n", i, j, value);
    //     }
    // }
    // multiply_lower_mm_kernel_device<<< 1, N>>>(a_ptr, result_ptr, N);
    cudaLaunchCooperativeKernel((void*)multiply_lower_mm_kernel_device, gridSize, blockSize, args2, 0, 0);
    cudaDeviceSynchronize();
    // printf("AFTER multiply\n");
    // a_cpu = a.cpu();  
    // // Ensure it is contiguous on the CPU
    // a_contig_cpu = a_cpu.contiguous();
    // a_ptr_cpu = a_contig_cpu.data_ptr<float>();
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         float value = a_ptr_cpu[i * 3 + j];
    //         printf("A[%d, %d] = %f\n", i, j, value);
    //     }
    // }

    return result;

}


TORCH_LIBRARY_IMPL(cola_kernels, CUDA, m) {
  m.impl("fuse_kernel_1", &fuse_kernel_1_cuda);
  m.impl("fuse_kernel_2", &fuse_kernel_2_cuda);
}

}

