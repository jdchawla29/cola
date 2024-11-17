#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>
#include <iostream>
#include <time.h>

typedef std::complex<float> cfloat;
#define index(i, j, N) ((i) * (N)) + (j)

__global__ void sum_reduction(cuFloatComplex *v, cuFloatComplex *v_r, int N);
__global__ void norm_cal(cuFloatComplex *v, cuFloatComplex *v_r, int N);
__global__ void norm_new_vector(cuFloatComplex *v, cfloat norm, int n);
__global__ void update_angle(cuFloatComplex *h_vec, cfloat ang, int i);
__global__ void update_arr(cuFloatComplex *Q, cuFloatComplex *V, int N, int M, int i);
__global__ void matrix_vector_mul(float *A, cuFloatComplex *B, cuFloatComplex *C, int N, int M, int i);
__global__ void update_new_vector(cuFloatComplex *A, cuFloatComplex *B, cuFloatComplex *C, int N, int M, int i);
__global__ void update_new_vector_clip(cuFloatComplex *A, cuFloatComplex *norm, int N);
__global__ void angle_cal(cuFloatComplex *Q, cuFloatComplex *V, cuFloatComplex *ang, int N, int M, int i);
int main(int argc, char *argv[])
{
    time_t start_t, end_t;
    double diff_t;
    unsigned int N = atoi(argv[1]);
    int idx = 0;
    int max_iters = atoi(argv[2]);
    cfloat norm, angle;
    srand(21);
    int random_number = rand();

    // cpu data
    float *A;
    cfloat *new_vector, *new_vector_r, *H, *Q, *h_vec, *ang;
    A = (float *)calloc(N * N, sizeof(float));
    new_vector = (cfloat *)calloc(N, sizeof(cfloat));
    new_vector_r = (cfloat *)calloc((N + 255) / 256, sizeof(cfloat));
    H = (cfloat *)calloc((max_iters + 1) * max_iters, sizeof(cfloat));
    Q = (cfloat *)calloc(N * (max_iters + 1), sizeof(cfloat));
    h_vec = (cfloat *)calloc((max_iters + 1), sizeof(cfloat));
    ang = (cfloat *)calloc(N, sizeof(cfloat));
    // cuda data
    float *A_d;
    cuFloatComplex *new_vector_d, *new_vector_d_r, *H_d, *Q_d, *h_vec_d, *ang_d;
    cudaMalloc((void **)&A_d, N * N * sizeof(float));
    cudaMalloc((void **)&new_vector_d, N * sizeof(cuFloatComplex));
    cudaMalloc((void **)&new_vector_d_r, (N + 255) / 256 * sizeof(cuFloatComplex));
    cudaMalloc((void **)&H_d, (max_iters + 1) * max_iters * sizeof(cuFloatComplex));
    cudaMalloc((void **)&Q_d, N * (max_iters + 1) * sizeof(cuFloatComplex));
    cudaMalloc((void **)&h_vec_d, (max_iters + 1) * sizeof(cuFloatComplex));
    cudaMalloc((void **)&ang_d, N * sizeof(cuFloatComplex));

    // initialization
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // int random_int = rand() % 100;
            // A[index(i, j, N)] = (float)random_int;
            A[index(i, j, N)] = float(i * N + j), 0.0f;
        }
    }

    for (int i = 0; i < N; i++)
    {
        new_vector[i] = cfloat(float(i), 0.0f);
        // new_vector[i] = ((float)rand() / RAND_MAX);
    }

    time(&start_t);
    cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(new_vector_d, new_vector, N * sizeof(cfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(new_vector_d_r, new_vector_r, N * sizeof(cfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(H_d, H, max_iters * (max_iters + 1) * sizeof(cfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(Q_d, Q, N * (max_iters + 1) * sizeof(cfloat), cudaMemcpyHostToDevice);
    cudaMemcpy(ang_d, ang, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h_vec_d, h_vec, (max_iters + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // norm calcuations
    norm_cal<<<(N + 255) / 256, 256>>>(new_vector_d, new_vector_d_r, N);
    cudaMemcpy(new_vector_r, new_vector_d_r, (N + 255) / 256 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    norm = 0;
    for (int i = 0; i < (N + 255) / 256; i++)
    {
        norm = norm + new_vector_r[i];
    }
    norm = sqrt(norm);
    std::cout << "norm(" << norm.real() << ", " << norm.imag() << "i)" << std::endl;
    norm_new_vector<<<(N + 255) / 256, 256>>>(new_vector_d, norm, N);

    // update Q
    update_arr<<<(N + 255) / 256, 256>>>(Q_d, new_vector_d, N, (max_iters + 1), 0);
    while (idx < max_iters)
    {
        matrix_vector_mul<<<(N + 255) / 256, 256>>>(A_d, Q_d, new_vector_d, N, (max_iters + 1), idx);
        cudaMemset(h_vec_d, 0, (max_iters + 1) * sizeof(cuFloatComplex));
        for (int j = 0; j < idx + 1; j++)
        {
            cudaMemset(ang_d, 0, N * sizeof(cuFloatComplex));
            cudaMemset(new_vector_d_r, 0, (N + 255) / 256 * sizeof(cuFloatComplex));
            memset(new_vector_r, 0, (N + 255) / 256 * sizeof(cfloat));
            angle_cal<<<(N + 255) / 256, 256>>>(Q_d, new_vector_d, ang_d, N, (max_iters + 1), j);
            sum_reduction<<<(N + 255) / 256, 256>>>(ang_d, new_vector_d_r, N);
            cudaMemcpy(new_vector_r, new_vector_d_r, (N + 255) / 256 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
            angle = 0;
            for (int i = 0; i < (N + 255) / 256; i++)
            {
                angle = angle + new_vector_r[i];
            }
            std::cout << "angle(" << angle.real() << ", " << angle.imag() << "i)" << std::endl;
            update_angle<<<1, 1>>>(h_vec_d, angle, j);
            update_new_vector<<<(N + 255) / 256, 256>>>(h_vec_d, Q_d, new_vector_d, N, (max_iters + 1), j);
        }
        cudaMemset(new_vector_d_r, 0, (N + 255) / 256 * sizeof(cuFloatComplex));
        memset(new_vector_r, 0, (N + 255) / 256 * sizeof(cfloat));
        norm_cal<<<(N + 255) / 256, 256>>>(new_vector_d, new_vector_d_r, N);
        cudaMemcpy(new_vector_r, new_vector_d_r, (N + 255) / 256 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        norm = 0; // same as sum
        for (int i = 0; i < (N + 255) / 256; i++)
        {
            norm = norm + new_vector_r[i];
        }
        norm = sqrt(norm);
        std::cout << "norm(" << norm.real() << ", " << norm.imag() << "i)" << std::endl;
        if (std::abs(norm.real()) < 0.0000001 / 2.0)
        {
            norm.real(0.0000001 / 2.0);
            norm.imag(0.0f);
        }
        norm_new_vector<<<(N + 255) / 256, 256>>>(new_vector_d, norm, N);
        update_angle<<<1, 1>>>(h_vec_d, norm, (idx + 1));
        update_arr<<<((max_iters + 1) + 255) / 256, 256>>>(H_d, h_vec_d, (max_iters + 1), max_iters, idx);
        update_arr<<<(N + 255) / 256, 256>>>(Q_d, new_vector_d, N, (max_iters + 1), (idx + 1));
        idx += 1;
    }

    cudaMemcpy(H, H_d, (max_iters + 1) * max_iters * sizeof(cfloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q, Q_d, N * (max_iters + 1) * sizeof(cfloat), cudaMemcpyDeviceToHost);

    time(&end_t);
    diff_t = difftime(end_t, start_t);
    printf("Elapsed wall-clock time: %f seconds\n", diff_t);
}

// cuda functions
__global__ void clip(cuFloatComplex *v, cfloat ang, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        cuFloatComplex a = make_cuFloatComplex(ang.real(), ang.imag());
        v[idx] = cuCdivf(v[idx], a);
    }
}

__global__ void sum_reduction(cuFloatComplex *array, cuFloatComplex *out, int size)
{
    __shared__ cuFloatComplex sharedMem[256]; // Shared memory for block reduction

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory, square the elements
    sharedMem[tid] = (globalIdx < size) ? array[globalIdx] : make_cuFloatComplex(0.0f, 0.0f);
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sharedMem[tid] = cuCaddf(sharedMem[tid], sharedMem[tid + stride]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        out[blockIdx.x] = sharedMem[0];
    }
}

__global__ void norm_cal(cuFloatComplex *array, cuFloatComplex *out, int size)
{
    __shared__ cuFloatComplex sharedMem[256]; // Shared memory for block reduction

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Load data into shared memory, square the elements
    sharedMem[tid] = (globalIdx < size) ? cuCmulf(array[globalIdx], array[globalIdx]) : make_cuFloatComplex(0.0f, 0.0f);
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sharedMem[tid] = cuCaddf(sharedMem[tid], sharedMem[tid + stride]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        out[blockIdx.x] = sharedMem[0];
    }
}

__global__ void norm_new_vector(cuFloatComplex *v, cfloat n, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cuFloatComplex norm = make_cuFloatComplex(n.real(), n.imag());
    // Ensure that tid is within bounds
    if (tid < N)
    {
        // Divide each complex number by the norm
        v[tid] = cuCdivf(v[tid], norm);
    }
}

__global__ void update_angle(cuFloatComplex *h_vec, cfloat ang, int i)
{
    if (threadIdx.x == 0)
    {
        h_vec[i] = make_cuFloatComplex(ang.real(), ang.imag());
    }
}

__global__ void update_arr(cuFloatComplex *Q, cuFloatComplex *V, int N, int M, int i)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < N)
    {
        int idx = index(row_idx, i, M);
        Q[idx] = V[row_idx];
    }
}

__global__ void matrix_vector_mul(float *A, cuFloatComplex *B, cuFloatComplex *C, int N, int M, int i)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we only process rows in the valid range
    if (row_idx < N)
    {
        cuFloatComplex dot_product = make_cuFloatComplex(0.0f, 0.0f);
        // Compute the dot product of row_idx-th row of A with vector B
        for (int col_idx = 0; col_idx < N; col_idx++)
        {
            cuFloatComplex a = make_cuFloatComplex(A[index(row_idx, col_idx, N)], 0.0f);
            dot_product = cuCaddf(dot_product, cuCmulf(a, B[index(col_idx, i, M)]));
        }
        // Store the result in C
        C[row_idx] = dot_product;
    }
}

__global__ void update_new_vector(cuFloatComplex *A, cuFloatComplex *B, cuFloatComplex *C, int N, int M, int i)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < N)
    {
        int idx = index(row_idx, i, M);
        C[row_idx] = cuCsubf(C[row_idx], cuCmulf(A[i], B[idx]));
    }
}

__global__ void update_new_vector_clip(cuFloatComplex *A, cuFloatComplex *norm, int N)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we only process rows in the valid range
    if (row_idx < N)
    {
        A[row_idx] = cuCdivf(A[row_idx], norm[row_idx]);
    }
}

__global__ void angle_cal(cuFloatComplex *Q, cuFloatComplex *V, cuFloatComplex *ang, int N, int M, int i)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < N)
    {
        int idx = index(row_idx, i, M);
        ang[row_idx] = cuCmulf(cuConjf(Q[idx]), V[row_idx]);
    }
}

// cudaMemset(ang_d, 0, N * sizeof(cuFloatComplex));
// cudaMemset(new_vector_d_r, 0, (N + 255) / 256 * sizeof(cuFloatComplex));
// memset(new_vector_r, 0, (N + 255) / 256 * sizeof(cfloat));

// cudaMemcpy(new_vector, new_vector_d, N * sizeof(cfloat), cudaMemcpyDeviceToHost);
// for (int i = 0; i < N; i++)
// {
//     std::cout << "(" << new_vector[i].real() << ", " << new_vector[i].imag() << "i)" << std::endl;
// }

// for (int i = 0; i < max_iters + 1; i++)
// {
//     for (int j = 0; j < max_iters; j++)
//     {
//         std::cout << "(" << H[index(i, j, max_iters)].real() << ", " << H[index(i, j, max_iters)].imag() << "i)";
//     }
//     printf("\n");
// }

// for (int i = 0; i < N; i++)
// {
//     for (int j = 0; j < max_iters + 1; j++)
//     {
//         std::cout << "(" << Q[index(i, j, max_iters+1)].real() << ", " << Q[index(i, j, max_iters+1)].imag() << "i)";
//     }
//     printf("\n");
// }