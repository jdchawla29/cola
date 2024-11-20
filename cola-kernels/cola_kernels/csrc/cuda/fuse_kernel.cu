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


#include <curand_kernel.h>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <iomanip>

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

#define Index(i, j, N) (j * N + i)


    __global__ void decompose_cholesky(float *a, int N) {
        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        cg::grid_group grid = cg::this_grid();

        for (int k = 0; k < N; k++) {
            if (thread_id == 0) {
                a[Index(k, k, N)] = sqrt(a[Index(k, k, N)]);
            }
            grid.sync();

            int i = k + 1 + thread_id;
            if(i < N) {
                a[Index(i, k, N)] /= a[Index(k, k, N)];
            }
            grid.sync();

            // Update the submatrix
            if(i < N) {
                for (int j = i; j < N; j++) {
                    a[Index(j, i, N)] -= a[Index(i, k, N)] * a[Index(j, k, N)];
                }
            }
            grid.sync();
        }

        if (thread_id < N) {
            for (int j = thread_id + 1; j < N; j++) {
                a[Index(thread_id, j, N)] = 0.0f;
            }
        }
    }

    __global__ void inverse_lower(float *a, float *aInv, int N)
    {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        cg::grid_group grid = cg::this_grid();

        // Compute the elements of the lower inverse matrix
        for (int j = 0; j < N; j++)
        {
            for (int i = j + 1; i < N; i++)
            {

                if (thread_id == 0)
                {
                    aInv[Index(i, j, N)] = -a[Index(i, j, N)] /
                                            (a[Index(j, j, N)] * a[Index(i, i, N)]);
                }
                grid.sync();

                int k = thread_id + j + 1;
                if (k < i)
                {
                    atomicAdd((float *)&aInv[Index(i, j, N)],
                              -a[Index(i, k, N)] * aInv[Index(k, j, N)] / a[Index(i, i, N)]);
                }
            }
        }
        grid.sync();

        if (thread_id < N) {
            aInv[Index(thread_id, thread_id, N)] = 1.0 / a[Index(thread_id, thread_id, N)];
        }
    }

    __global__ void multiply_lower(float *a, float *aInv, int N)
    {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        cg::grid_group grid = cg::this_grid();

        // Perform multiplication directly in aInv
        for (int j = 0; j < N; j++)
        {
            for (int i = j; i < N; i++)
            {

                if (thread_id == 0)
                {
                    aInv[Index(i, j, N)] *= aInv[Index(i, i, N)];
                }
                grid.sync();

                int k = thread_id + i + 1;
                if (k < N)
                {
                    atomicAdd((float *)&aInv[Index(i, j, N)],
                              aInv[Index(k, j, N)] * aInv[Index(k, i, N)]);
                }
            }
        }
        grid.sync();


        if (thread_id < N) {
            for (int j = 0; j <= thread_id; j++) {
                aInv[Index(j, thread_id, N)] = aInv[Index(thread_id, j, N)];
            }
        }
    }

    at::Tensor fuse_kernel_2_cuda(at::Tensor &a)
    {
        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
        at::Tensor a_contig = a.contiguous();
        at::Tensor result = torch::zeros(a_contig.sizes(), a_contig.options());
        float *a_ptr = a_contig.data_ptr<float>();
        float *result_ptr = result.data_ptr<float>();

        int N = static_cast<int>(a.size(0));

        dim3 blockSize(256);                                
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x); 
        // Kernel launch with cooperative groups
        void *args1[] = {&a_ptr, &N};
        cudaLaunchCooperativeKernel((void *)decompose_cholesky, gridSize, blockSize, args1, 0, 0);
        cudaDeviceSynchronize();

        void *args2[] = {&a_ptr, &result_ptr, &N};
        cudaLaunchCooperativeKernel((void *)inverse_lower, gridSize, blockSize, args2, 0, 0);
        cudaDeviceSynchronize();

        cudaLaunchCooperativeKernel((void *)multiply_lower, gridSize, blockSize, args2, 0, 0);
        cudaDeviceSynchronize();

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

        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

        at::Tensor a_contig = a.contiguous();
        int m = a_contig.size(0);
        int n = a_contig.size(1);
        int lda = m;

        float *a_ptr = a_contig.data_ptr<float>();


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

        // U (using A A^T)
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


        // V (using A^T A)
        at::Tensor v_eigenvalues = torch::empty({n}, a_contig.options());
        float *v_eigenvalues_ptr = v_eigenvalues.data_ptr<float>();
        int v_lwork = 0;
        cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                    n, ata_ptr, n, v_eigenvalues_ptr, &v_lwork);

        float *v_work;
        cudaMalloc(&v_work, v_lwork * sizeof(float));
        cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                         n, ata_ptr, n, v_eigenvalues_ptr, v_work, v_lwork, devInfo);


        // Sigma
        int size = std::min(m, n);
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        computeSquareRoot<<<gridSize, blockSize>>>(s_ptr, u_eigenvalues_ptr, size);


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

    __global__ void update_arr(float *Q, float *V, int N, int i)
    {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (row_idx < N)
        {
            Q[index(i, row_idx, N)] = V[row_idx];
        }
    }

    __global__ void matrix_vector_mul(float *A, float *B, float *C, int N, int i)
    {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (row_idx < N)
        {
            float dot_product = 0.0;
            for (int col_idx = 0; col_idx < N; col_idx++)
            {
                dot_product += A[index(row_idx, col_idx, N)] * B[index(i, col_idx, N)];
            }
            C[row_idx] = dot_product;
        }
    }

    __global__ void update_new_vector(float A, float *B, float *C, int N, int i)
    {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (row_idx < N)
        {
            C[row_idx] -= A * B[index(i,row_idx, N)];
        }
    }

    __global__ void angle_cal(float *Q, float *V, float *ang, int N, int i)
    {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (row_idx < N)
        {
            ang[row_idx] = Q[index(i, row_idx, N)] * V[row_idx];
        }
    }

    std::tuple<at::Tensor, at::Tensor> fuse_kernel_4_cuda(at::Tensor &a, int64_t max_iters=100)
    {

        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
        at::Tensor a_contig = a.contiguous();
        float *A_d = a_contig.data_ptr<float>();

        // int max_iters = 100;
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
            new_vector[i] = ((float)rand() / RAND_MAX);
        }
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
        update_arr<<<blocks, 1024>>>(Q_d, new_vector_d, N, 0);

        // iterations
        while (idx < max_iters && norm > tol * limit)
        {
            matrix_vector_mul<<<blocks, 1024>>>(A_d, Q_d, new_vector_d, N, idx);
            cudaMemset(h_vec_d, 0, (max_iters + 1) * sizeof(float));
            for (int j = 0; j < idx + 1; j++)
            {
                cudaMemset(ang_d, 0, N * sizeof(float));
                cudaMemset(new_vector_d_r, 0, blocks * sizeof(float));
                memset(new_vector_r, 0, blocks * sizeof(float));
                angle_cal<<<blocks, 1024>>>(Q_d, new_vector_d, ang_d, N, j);
                sum<<<blocks, 1024>>>(ang_d, new_vector_d_r, N);
                cudaMemcpy(new_vector_r, new_vector_d_r, blocks * sizeof(float), cudaMemcpyDeviceToHost);
                angle = 0;
                for (int i = 0; i < blocks; i++)
                {
                    angle = angle + new_vector_r[i];
                }
                cudaMemcpy(&h_vec_d[j], &angle, sizeof(float), cudaMemcpyHostToDevice);
                update_new_vector<<<blocks, 1024>>>(angle, Q_d, new_vector_d, N, j);
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
            update_arr<<<((max_iters + 1) + 1023) / 1024, 1024>>>(H_d, h_vec_d, (max_iters + 1), idx);
            update_arr<<<blocks, 1024>>>(Q_d, new_vector_d, N, (idx + 1));
            cudaMemcpy(&limit, &H_d[index(0, 0, max_iters)], sizeof(float), cudaMemcpyDeviceToHost);
            idx += 1;
        }
        torch::Tensor rH = torch::from_blob(H_d, {(max_iters + 1) * max_iters}, torch::kFloat).to(torch::Device(torch::kCUDA, 1));
        torch::Tensor rQ = torch::from_blob(Q_d, {N * (max_iters + 1)}, torch::kFloat).to(torch::Device(torch::kCUDA, 1));

        cudaFree(new_vector_d);
        cudaFree(new_vector_d_r);
        cudaFree(H_d);
        cudaFree(Q_d);
        cudaFree(h_vec_d);
        cudaFree(ang_d);
        return std::make_tuple(rH, rQ);
    }

    // Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            cudaDeviceReset(); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

// Error checking macro for cuBLAS calls
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            cudaDeviceReset(); \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while (0)

// Constants for CUDA kernel configuration
constexpr int BLOCK_SIZE = 256;     // Number of threads per block
constexpr int MAX_BLOCKS = 65535;   // Maximum number of blocks

// Initialize CUDA random number generator states
__global__ void setup_curand_kernel(curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(clock64(), idx, 0, &state[idx]);
}

// Generate random vectors for Hutchinson estimation
__global__ void generate_random_vector(curandState *state, float *z, int n, int bs, bool is_rademacher) {
    // Each thread gets its own random state in shared memory
    __shared__ curandState localState[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Copy state to shared memory for faster access
    localState[threadIdx.x] = state[tid];
    
    // Generate random values for the entire matrix
    int total_elements = n * bs;
    for (int i = tid; i < total_elements; i += gridDim.x * blockDim.x) {
        float rand_val = curand_normal(&localState[threadIdx.x]);
        // Either Rademacher (+1/-1) or normal distribution
        z[i] = is_rademacher ? ((rand_val >= 0.0f) ? 1.0f : -1.0f) : rand_val;
    }
    
    // Save state back to global memory
    state[tid] = localState[threadIdx.x];
}

__global__ void compute_diagonal_estimate(
    const float* __restrict__ Az,     // Matrix-vector product Az
    const float* __restrict__ z,      // Random vector z
    float* __restrict__ diag_sum,     // Running sum of diagonal estimates
    float* __restrict__ diag_sumsq,   // Running sum of squares for variance estimation
    int n,                           // Matrix dimension
    int bs,                          // Batch size for random vectors
    int k,                           // Diagonal offset (0 for main diagonal)
    int iter                         // Current iteration number
) {
    // Shared memory for temporary storage of sums and squared sums
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sumsq[BLOCK_SIZE];
    
    // Calculate thread ID and grid stride
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Process diagonal elements in strided fashion
    for (int i = tid; i < n - abs(k); i += stride) {
        // Initialize accumulators for this diagonal element
        shared_sum[threadIdx.x] = 0.0f;
        shared_sumsq[threadIdx.x] = 0.0f;
        
        // Process batch of random vectors
        #pragma unroll 4
        for (int b = 0; b < bs; b++) {
            int idx = i + (k >= 0 ? k : 0);  // Adjust index for off-diagonal elements
            float est = Az[i + b * n] * z[idx + b * n];  // Compute estimate for this batch
            shared_sum[threadIdx.x] += est;              // Accumulate sum
            shared_sumsq[threadIdx.x] += est * est;      // Accumulate sum of squares
        }
        
        // Normalize by batch size
        shared_sum[threadIdx.x] /= bs;
        shared_sumsq[threadIdx.x] /= bs;
        
        __syncthreads();
        
        // Update running statistics
        if (iter == 0) {
            // First iteration: just store the values
            diag_sum[i] = shared_sum[threadIdx.x];
            diag_sumsq[i] = shared_sumsq[threadIdx.x];
        } else {
            // Subsequent iterations: update mean and sum of squares
            float old_mean = diag_sum[i];
            float delta = shared_sum[threadIdx.x] - old_mean;
            diag_sum[i] += delta / (iter + 1);                    // Update mean
            diag_sumsq[i] += delta * (shared_sum[threadIdx.x] - diag_sum[i]); // Update sum of squares
        }
    }
}

__global__ void compute_relative_error_kernel(
    const float* diag_sum,      // Running sum of diagonal estimates
    const float* diag_sumsq,    // Running sum of squares for variance estimation
    float* max_error,           // Output: maximum relative error across all elements
    int n,                      // Number of diagonal elements
    int total_iters            // Total number of iterations completed
) {
    // Shared memory for block-wise reduction of maximum error
    __shared__ float shared_max_error[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local maximum error for this thread
    float local_max_error = 0.0f;
    
    // Process elements with grid-stride loop
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float mean = diag_sum[i];
        float variance = diag_sumsq[i] / (total_iters - 1);
        float stderr = sqrtf(variance / total_iters);
        // Relative error with minimum denominator of 0.1 to avoid division by zero
        float rel_error = stderr / fmaxf(fabsf(mean), 0.1f);
        local_max_error = fmaxf(local_max_error, rel_error);
    }
    
    // Store local maximum in shared memory
    shared_max_error[threadIdx.x] = local_max_error;
    __syncthreads();
    
    // Parallel reduction to find maximum error within the block
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max_error[threadIdx.x] = fmaxf(
                shared_max_error[threadIdx.x],
                shared_max_error[threadIdx.x + s]
            );
        }
        __syncthreads();
    }
    
    // First thread in block updates global maximum using atomic operation
    if (threadIdx.x == 0) {
        atomicMax((int*)max_error, __float_as_int(shared_max_error[0]));
    }
}

    at::Tensor fuse_kernel_5_cuda(
        at::Tensor &mat,      // Input matrix (host)
        at::Tensor &diag,          // Output diagonal (host)
        int64_t bs = 100,            // Batch size
        double tol = 3e-2f,       // Tolerance
        int64_t max_iters = 10000,   // Maximum iterations
        int64_t k = 0,               // Diagonal offset
        bool use_rademacher = false  // Use rademacher instead of normal distribution
    ) {

        TORCH_CHECK(mat.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(mat.device().type() == at::DeviceType::CUDA);
        TORCH_CHECK(diag.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(diag.device().type() == at::DeviceType::CUDA);


        at::Tensor mat_contig = mat.contiguous();
        at::Tensor diag_contig = diag.contiguous();

        int n = mat_contig.size(0); // Matrix dimension

        float* matrix = mat_contig.data_ptr<float>();
        float* diagonal = diag_contig.data_ptr<float>();


    // Create CUDA stream for asynchronous operations
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Initialize cuBLAS handle and set stream
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        
        // Custom deleter for CUDA memory management
        struct CudaDeleter {
            void operator()(void* p) { cudaFree(p); }
        };
        // printf("CUDAstream created\n");
        // Smart pointers for CUDA memory management
        std::unique_ptr<float, CudaDeleter> d_matrix;      // Device matrix
        std::unique_ptr<float, CudaDeleter> d_z;           // Random vectors
        std::unique_ptr<float, CudaDeleter> d_Az;          // Matrix-vector products
        std::unique_ptr<float, CudaDeleter> d_diag_sum;    // Running sum of estimates
        std::unique_ptr<float, CudaDeleter> d_diag_sumsq;  // Running sum of squares
        std::unique_ptr<curandState, CudaDeleter> d_rand_state;  // RNG states
        
        float *ptr;
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&ptr, n * n * sizeof(float)));
        d_matrix.reset(ptr);
        CUDA_CHECK(cudaMalloc(&ptr, n * bs * sizeof(float)));
        d_z.reset(ptr);
        CUDA_CHECK(cudaMalloc(&ptr, n * bs * sizeof(float)));
        d_Az.reset(ptr);
        CUDA_CHECK(cudaMalloc(&ptr, (n - abs(k)) * sizeof(float)));
        d_diag_sum.reset(ptr);
        CUDA_CHECK(cudaMalloc(&ptr, (n - abs(k)) * sizeof(float)));
        d_diag_sumsq.reset(ptr);
        

        // Allocate RNG states
        curandState *rand_ptr;
        CUDA_CHECK(cudaMalloc(&rand_ptr, BLOCK_SIZE * sizeof(curandState)));
        d_rand_state.reset(rand_ptr);
        
        // Copy matrix to device and initialize accumulators
        // CUDA_CHECK(cudaMemcpyAsync(d_matrix.get(), matrix, n * n * sizeof(float),
        //                         cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_matrix.get(), matrix, n * n * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(d_diag_sum.get(), 0, (n - abs(k)) * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_diag_sumsq.get(), 0, (n - abs(k)) * sizeof(float), stream));
        
        // Configure kernel grid
        int num_blocks = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        setup_curand_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_rand_state.get());
        
        // cuBLAS matrix multiplication constants
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Convergence control variables
        float rel_error = tol + 1.0f;
        float *d_max_error;
        CUDA_CHECK(cudaMalloc(&d_max_error, sizeof(float)));
        
        // Main iteration loop
        int iter;
        for (iter = 0; iter < max_iters && rel_error > tol; iter++) {
            // printf("Iter: %d\n", iter);
            // Generate batch of random vectors
            generate_random_vector<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_rand_state.get(), d_z.get(), n, bs, use_rademacher);
            
            // Compute matrix-vector products using cuBLAS
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, bs, n,
                                    &alpha,
                                    d_matrix.get(), n,
                                    d_z.get(), n,
                                    &beta,
                                    d_Az.get(), n));
            
            // Update diagonal estimates
            compute_diagonal_estimate<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_Az.get(), d_z.get(), d_diag_sum.get(), d_diag_sumsq.get(), 
                n, bs, k, iter);
            
            // Check convergence every 10 iterations
            if ((iter + 1) % 10 == 0 && iter > 0) {
                CUDA_CHECK(cudaMemsetAsync(d_max_error, 0, sizeof(float), stream));
                compute_relative_error_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    d_diag_sum.get(), d_diag_sumsq.get(), d_max_error, 
                    n - abs(k), iter + 1);
                CUDA_CHECK(cudaMemcpyAsync(&rel_error, d_max_error, sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
        
        // Copy results back to host
        // CUDA_CHECK(cudaMemcpyAsync(diagonal, d_diag_sum.get(),
        //                         (n - abs(k)) * sizeof(float), 
        //                         cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(diagonal, d_diag_sum.get(),
                        (n - abs(k)) * sizeof(float), 
                        cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Print convergence information
        // std::cout << "Completed after " << iter << " iterations\n";
        // std::cout << "Final relative error: " << rel_error << "\n";
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_max_error));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUBLAS_CHECK(cublasDestroy(handle));

        return diag;
    }

    TORCH_LIBRARY_IMPL(cola_kernels, CUDA, m)
    {
        m.impl("fuse_kernel_1", &fuse_kernel_1_cuda);
        m.impl("fuse_kernel_2", &fuse_kernel_2_cuda);
        m.impl("fuse_kernel_3", &fuse_kernel_3_cuda);
        m.impl("fuse_kernel_4", &fuse_kernel_4_cuda);
        m.impl("fuse_kernel_5", &fuse_kernel_5_cuda);
    }

}
