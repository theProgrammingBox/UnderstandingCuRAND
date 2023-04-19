#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

void PrintMatrixf32(float* arr, uint32_t rows, uint32_t cols, const char* label)
{
    printf("%s:\n", label);
    for (uint32_t i = 0; i < rows; i++)
    {
        for (uint32_t j = 0; j < cols; j++)
            printf("%8.3f ", arr[i * cols + j]);
        printf("\n");
    }
    printf("\n");
}

__global__ void generate_random_matrix(float* matrix, int rows, int cols, unsigned int seed, curandState* state, uint32_t offset)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;

    if (row < rows && col < cols)
    {
        curand_init(seed, idx + offset, 0, &state[idx]);
        matrix[idx] = curand_uniform(&state[idx]);
    }
}

int main()
{
    const uint32_t rows = 16;
    const uint32_t cols = 16;
    uint32_t seed = 10;

    float* h_matrix = new float[rows * cols];
    float* d_matrix;
    curandState* d_states;

    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMalloc(&d_states, rows * cols * sizeof(curandState));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 1; i <= 2; ++i)
    {
        curandGenerateUniform(gen, d_matrix, rows * cols);
        cudaMemcpy(h_matrix, d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
        printf("cuRAND Random Matrix (call #%d):\n", i);
        PrintMatrixf32(h_matrix, rows, cols, "");
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("cuRAND kernel execution time: %f ms\n", elapsedTime);

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    cudaEventRecord(start, 0);

    uint32_t offset = 0;
    for (int i = 1; i <= 2; ++i)
    {
        generate_random_matrix << <gridDim, blockDim >> > (d_matrix, rows, cols, seed, d_states, offset);
        cudaMemcpy(h_matrix, d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Custom cuRAND Random Matrix (call #%d):\n", i);
        PrintMatrixf32(h_matrix, rows, cols, "");
        offset += rows * cols;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Custom cuRAND kernel execution time: %f ms\n", elapsedTime);

    cudaFree(d_matrix);
    cudaFree(d_states);
    delete[] h_matrix;
    curandDestroyGenerator(gen);

    return 0;
}
