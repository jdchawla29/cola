#include <stdio.h>
#include <stdlib.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <iostream>

#include <time.h>
#include <math.h>

namespace cg = cooperative_groups;

namespace cola_kernels
{

    __global__ void fuse_kernel_1(int numel, const float *a, const float *b, const float *c, float *result)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel)
            result[idx] = a[idx] * b[idx] + c[idx];
    }

    at::Tensor fuse_kernel_1_cuda(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c)
    {
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
        const float *a_ptr = a_contig.data_ptr<float>();
        const float *b_ptr = b_contig.data_ptr<float>();
        const float *c_ptr = c_contig.data_ptr<float>();
        float *result_ptr = result.data_ptr<float>();

        int numel = a_contig.numel();
        fuse_kernel_1<<<(numel + 255) / 256, 256>>>(numel, a_ptr, b_ptr, c_ptr, result_ptr);
        return result;
    }

#define MATIDX(i, j, N) (j * N + i)

    __global__ void decompose_cholesky_mm_kernel_device(float *a, int N)
    {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
        cg::grid_group grid = cg::this_grid();

        // Loop over k
        for (int k = 0; k < N; k++)
        {

            // Compute diagonal element in the first thread
            if (thread_id == 0)
            {
                // printf("sqrt: %f\n", a[MATIDX(k, k, N)]);
                a[MATIDX(k, k, N)] = sqrt(a[MATIDX(k, k, N)]);

                // Update column elements by dividing by the diagonal
                for (int j = k + 1; j < N; j++)
                {
                    // printf("div: %f %f\n", a[MATIDX(j, k, N)], a[MATIDX(k, k, N)]);
                    a[MATIDX(j, k, N)] /= a[MATIDX(k, k, N)];
                }
            }

            grid.sync(); // Synchronize threads after updating the diagonal

            // Update the rest of the matrix, only threads that handle i > k
            int i = thread_id + k + 1; // Global row index
            if (i < N)
            {
                for (int j = i; j < N; j++)
                {
                    // printf("parallel: %f %f %f\n", a[MATIDX(i, j, N)], a[MATIDX(i, k, N)], a[MATIDX(j, k, N)]);
                    a[MATIDX(j, i, N)] -= a[MATIDX(i, k, N)] * a[MATIDX(j, k, N)];
                }
            }

            grid.sync(); // Synchronize threads after updating the matrix
        }

        // Zero out the upper triangular part of the matrix after decomposition (only for thread 0)
        if (thread_id == 0)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < i; j++)
                {
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
    __global__ void inverse_lower_mm_kernel_device(float *a, float *aInv, int N)
    {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
        cg::grid_group grid = cg::this_grid();

        if (threadIdx.x == 0)
        {
            // Initialize `aInv` to zero
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j <= i; j++)
                {
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
        for (int j = 0; j < N; j++)
        {
            for (int i = j + 1; i < N; i++)
            {

                if (thread_id == 0)
                {
                    aInv[MATIDX(i, j, N)] = -a[MATIDX(i, j, N)] /
                                            (a[MATIDX(j, j, N)] * a[MATIDX(i, i, N)]);
                }
                grid.sync();

                int k = thread_id + j + 1;
                if (k < i)
                {
                    atomicAdd((float *)&aInv[MATIDX(i, j, N)],
                              -a[MATIDX(i, k, N)] * aInv[MATIDX(k, j, N)] / a[MATIDX(i, i, N)]);
                }
            }
        }
        grid.sync();

        if (thread_id == 0)
        {
            // Set the inverse of the diagonal elements and copy results to `aInv`
            for (int i = 0; i < N; i++)
            {
                aInv[MATIDX(i, i, N)] = 1.0 / a[MATIDX(i, i, N)];
                for (int j = 0; j <= i; j++)
                {
                    // Set only the lower triangular values in `aInv`
                    aInv[MATIDX(i, j, N)] = aInv[MATIDX(i, j, N)];
                }
            }
        }
    }

    __global__ void multiply_lower_mm_kernel_device(float *a, float *aInv, int N)
    {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
        cg::grid_group grid = cg::this_grid();

        // Perform multiplication directly in `aInv`
        for (int j = 0; j < N; j++)
        {
            for (int i = j; i < N; i++)
            {

                if (thread_id == 0)
                {
                    aInv[MATIDX(i, j, N)] *= aInv[MATIDX(i, i, N)];
                }
                grid.sync();

                int k = thread_id + i + 1;
                if (k < N)
                {
                    atomicAdd((float *)&aInv[MATIDX(i, j, N)],
                              aInv[MATIDX(k, j, N)] * aInv[MATIDX(k, i, N)]);
                }
            }
        }
        grid.sync();

        if (thread_id == 0)
        {
            // Copy the results into the full lower and symmetric upper triangle of `aInv`
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j <= i; j++)
                {
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

    at::Tensor fuse_kernel_2_cuda(at::Tensor &a)
    {
        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
        at::Tensor a_contig = a.contiguous();
        at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
        float *a_ptr = a_contig.data_ptr<float>();
        float *result_ptr = result.data_ptr<float>();

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
        // OVER

        // decompose_cholesky_mm_kernel_device<<< 1, N>>>(a_ptr, N);

        // cudaDeviceSynchronize();
        // Kernel launch configuration for cooperative kernel
        dim3 blockSize(256);                                // Number of threads per block
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x); // Number of blocks
        // Kernel launch with cooperative groups
        void *args1[] = {&a_ptr, &N};
        cudaLaunchCooperativeKernel((void *)decompose_cholesky_mm_kernel_device, gridSize, blockSize, args1, 0, 0);
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
        void *args2[] = {&a_ptr, &result_ptr, &N};
        cudaLaunchCooperativeKernel((void *)inverse_lower_mm_kernel_device, gridSize, blockSize, args2, 0, 0);
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
        cudaLaunchCooperativeKernel((void *)multiply_lower_mm_kernel_device, gridSize, blockSize, args2, 0, 0);
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

    __global__ void computeSquareRoot(float *s_ptr, const float *eigenvalues_ptr, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            float lambda = eigenvalues_ptr[idx];
            s_ptr[idx] = sqrtf(fmaxf(lambda, 0.0f));
        }
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor> fuse_kernel_3_cuda(at::Tensor &a)
    {
        // Ensure input tensor is float and CUDA-based
        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

        // Make tensor contiguous and get sizes
        at::Tensor a_contig = a.contiguous();
        int m = a_contig.size(0); // Rows
        int n = a_contig.size(1); // Columns
        int lda = m;

        float *a_ptr = a_contig.data_ptr<float>();

        // Allocate output tensors
        at::Tensor u = torch::empty({m, m}, a_contig.options());
        at::Tensor s = torch::empty({std::min(m, n)}, a_contig.options());
        at::Tensor vt = torch::empty({n, n}, a_contig.options());
        float *u_ptr = u.data_ptr<float>();
        float *s_ptr = s.data_ptr<float>();
        float *vt_ptr = vt.data_ptr<float>();

        // cuBLAS and cuSOLVER handles
        cublasHandle_t cublasH;
        cublasCreate(&cublasH);
        cusolverDnHandle_t cusolverH;
        cusolverDnCreate(&cusolverH);

        float alpha = 1.0f, beta = 0.0f;
        at::Tensor aat = torch::empty({m, m}, a_contig.options());
        at::Tensor ata = torch::empty({n, n}, a_contig.options());
        float *aat_ptr = aat.data_ptr<float>();
        float *ata_ptr = ata.data_ptr<float>();

        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, m, n, &alpha, a_ptr, lda, a_ptr, lda, &beta, aat_ptr, m);
        cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha, a_ptr, lda, a_ptr, lda, &beta, ata_ptr, n);

        at::Tensor u_eigenvalues = torch::empty({m}, a_contig.options());
        float *u_eigenvalues_ptr = u_eigenvalues.data_ptr<float>();
        int u_lwork = 0;
        cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    m, aat_ptr, m, u_eigenvalues_ptr, &u_lwork);

        float *u_work;
        cudaMalloc(&u_work, u_lwork * sizeof(float));
        int *devInfo;
        cudaMalloc(&devInfo, sizeof(int));
        cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                         m, aat_ptr, m, u_eigenvalues_ptr, u_work, u_lwork, devInfo);

        // Normalize and sort U's eigenvectors and eigenvalues
        // normalize_and_sort_eigen(aat_ptr, u_eigenvalues_ptr, m, true);

        // Repeat for V (using A^T A)
        at::Tensor v_eigenvalues = torch::empty({n}, a_contig.options());
        float *v_eigenvalues_ptr = v_eigenvalues.data_ptr<float>();
        int v_lwork = 0;
        cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    n, ata_ptr, n, v_eigenvalues_ptr, &v_lwork);

        float *v_work;
        cudaMalloc(&v_work, v_lwork * sizeof(float));
        cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                         n, ata_ptr, n, v_eigenvalues_ptr, v_work, v_lwork, devInfo);

        // normalize_and_sort_eigen(ata_ptr, v_eigenvalues_ptr, n, false);

        // Step 3: Form Sigma
        int size = std::min(m, n);
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        computeSquareRoot<<<gridSize, blockSize>>>(s_ptr, u_eigenvalues_ptr, size);

        // cudaMemcpy(u_ptr, aat_ptr, m * m * sizeof(float), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(vt_ptr, ata_ptr, n * n * sizeof(float), cudaMemcpyDeviceToDevice);

        // Cleanup
        cudaFree(u_work);
        cudaFree(v_work);
        cudaFree(devInfo);
        cublasDestroy(cublasH);
        cusolverDnDestroy(cusolverH);

        return std::make_tuple(aat, s, ata);
    }

#define index(i, j, N) ((i) * (N)) + (j)

    __global__ void sum(float *array, float *out, int size)
    {
        __shared__ float sharedMem[1024]; // Shared memory for block reduction

        int tid = threadIdx.x;
        int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

        sharedMem[tid] = (globalIdx < size) ? array[globalIdx] : 0.0;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                sharedMem[tid] += sharedMem[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0)
        {
            out[blockIdx.x] = sharedMem[0];
        }
    }

    __global__ void sum_of_squares(float *array, float *out, int size)
    {
        __shared__ float sharedMem[1024]; // Shared memory for block reduction

        int tid = threadIdx.x;
        int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        sharedMem[tid] = (globalIdx < size) ? array[globalIdx] * array[globalIdx] : 0.0;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                sharedMem[tid] += sharedMem[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0)
        {
            out[blockIdx.x] = sharedMem[0];
        }
    }

    __global__ void norm_cal(float *v, float norm, int N)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < N)
        {
            v[tid] /= norm;
        }
    }

    __global__ void update_arr(float *Q, float *V, int N, int M, int i)
    {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (row_idx < N)
        {
            Q[index(row_idx, i, M)] = V[row_idx];
        }
    }

    __global__ void matrix_vector_mul(float *A, float *B, float *C, int N, int M, int i)
    {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (row_idx < N)
        {
            float dot_product = 0.0;
            for (int col_idx = 0; col_idx < N; col_idx++)
            {
                dot_product += A[index(row_idx, col_idx, N)] * B[index(col_idx, i, M)];
            }
            C[row_idx] = dot_product;
        }
    }

    __global__ void update_new_vector(float A, float *B, float *C, int N, int M, int i)
    {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (row_idx < N)
        {
            C[row_idx] -= A * B[index(row_idx, i, M)];
        }
    }

    __global__ void angle_cal(float *Q, float *V, float *ang, int N, int M, int i)
    {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (row_idx < N)
        {
            ang[row_idx] = Q[index(row_idx, i, M)] * V[row_idx];
        }
    }

    std::tuple<at::Tensor, at::Tensor> fuse_kernel_4_cuda(at::Tensor &a)
    {

        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
        at::Tensor a_contig = a.contiguous();
        float *A_d = a_contig.data_ptr<float>();

        int max_iters = 100;
        time_t start_t, end_t;
        double diff_t;
        int idx = 0;
        float norm, angle;
        srand(21);
        int random_number = rand();
        int N = a.sizes()[0];
        int blocks = (N + 1023) / 1024;
        float tol = 0.0000001;
        float limit = 1;

        // cpu data
        float *new_vector, *new_vector_r;
        new_vector = (float *)calloc(N, sizeof(float));
        new_vector_r = (float *)calloc(blocks, sizeof(float));

        // cuda data
        float *new_vector_d, *new_vector_d_r, *H_d, *Q_d, *h_vec_d, *ang_d;
        cudaMalloc((void **)&new_vector_d, N * sizeof(float));
        cudaMalloc((void **)&new_vector_d_r, blocks * sizeof(float));
        cudaMalloc((void **)&H_d, (max_iters + 1) * max_iters * sizeof(float));
        cudaMalloc((void **)&Q_d, N * (max_iters + 1) * sizeof(float));
        cudaMalloc((void **)&h_vec_d, (max_iters + 1) * sizeof(float));
        cudaMalloc((void **)&ang_d, N * sizeof(float));

        for (int i = 0; i < N; i++)
        {
            // new_vector[i] = float(float(i), 0.0f);
            new_vector[i] = ((float)rand() / RAND_MAX);
        }

        time(&start_t);

        // cuda mem copy and set
        cudaMemcpy(new_vector_d, new_vector, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(H_d, 0, max_iters * (max_iters + 1) * sizeof(float));
        cudaMemset(Q_d, 0, N * (max_iters + 1) * sizeof(float));

        // norm calcuations
        sum_of_squares<<<blocks, 1024>>>(new_vector_d, new_vector_d_r, N);
        cudaMemcpy(new_vector_r, new_vector_d_r, blocks * sizeof(float), cudaMemcpyDeviceToHost);
        norm = 0;
        for (int i = 0; i < blocks; i++)
        {
            norm = norm + new_vector_r[i];
        }
        norm = sqrt(norm);
        norm_cal<<<blocks, 1024>>>(new_vector_d, norm, N);

        // update Q
        update_arr<<<blocks, 1024>>>(Q_d, new_vector_d, N, (max_iters + 1), 0);

        // iterations
        while (idx < max_iters && norm > tol * limit)
        {
            matrix_vector_mul<<<blocks, 1024>>>(A_d, Q_d, new_vector_d, N, (max_iters + 1), idx);
            cudaMemset(h_vec_d, 0, (max_iters + 1) * sizeof(float));
            for (int j = 0; j < idx + 1; j++)
            {
                cudaMemset(ang_d, 0, N * sizeof(float));
                cudaMemset(new_vector_d_r, 0, blocks * sizeof(float));
                memset(new_vector_r, 0, blocks * sizeof(float));
                angle_cal<<<blocks, 1024>>>(Q_d, new_vector_d, ang_d, N, (max_iters + 1), j);
                sum<<<blocks, 1024>>>(ang_d, new_vector_d_r, N);
                cudaMemcpy(new_vector_r, new_vector_d_r, blocks * sizeof(float), cudaMemcpyDeviceToHost);
                angle = 0;
                for (int i = 0; i < blocks; i++)
                {
                    angle = angle + new_vector_r[i];
                }
                cudaMemcpy(&h_vec_d[j], &angle, sizeof(float), cudaMemcpyHostToDevice);
                update_new_vector<<<blocks, 1024>>>(angle, Q_d, new_vector_d, N, (max_iters + 1), j);
            }
            cudaMemset(new_vector_d_r, 0, blocks * sizeof(float));
            memset(new_vector_r, 0, blocks * sizeof(float));
            sum_of_squares<<<blocks, 1024>>>(new_vector_d, new_vector_d_r, N);
            cudaMemcpy(new_vector_r, new_vector_d_r, blocks * sizeof(float), cudaMemcpyDeviceToHost);
            norm = 0;
            for (int i = 0; i < blocks; i++)
            {
                norm = norm + new_vector_r[i];
            }
            norm = sqrt(norm);
            if (std::abs(norm) < tol / 2.0)
            {
                norm = tol / 2.0;
            }
            norm_cal<<<blocks, 1024>>>(new_vector_d, norm, N);
            cudaMemcpy(&h_vec_d[idx + 1], &norm, sizeof(float), cudaMemcpyHostToDevice);
            update_arr<<<((max_iters + 1) + 1023) / 1024, 1024>>>(H_d, h_vec_d, (max_iters + 1), max_iters, idx);
            update_arr<<<blocks, 1024>>>(Q_d, new_vector_d, N, (max_iters + 1), (idx + 1));
            cudaMemcpy(&limit, &H_d[index(0, 0, max_iters)], sizeof(float), cudaMemcpyDeviceToHost);
            idx += 1;
        }

        torch::Tensor rH = torch::from_blob(H_d, {(max_iters + 1) * max_iters}, torch::kFloat).cuda();
        torch::Tensor rQ = torch::from_blob(Q_d, {N * (max_iters + 1)}, torch::kFloat).cuda();

        time(&end_t);
        diff_t = difftime(end_t, start_t);
        printf("Elapsed wall-clock time: %f seconds\n", diff_t);
        return std::make_tuple(rH, rQ);
    }

    TORCH_LIBRARY_IMPL(cola_kernels, CUDA, m)
    {
        m.impl("fuse_kernel_1", &fuse_kernel_1_cuda);
        m.impl("fuse_kernel_2", &fuse_kernel_2_cuda);
        m.impl("fuse_kernel_3", &fuse_kernel_3_cuda);
        m.impl("fuse_kernel_4", &fuse_kernel_4_cuda);
    }

}
