#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>

#define WIDTH 1024
#define HEIGHT 1024
#define KERNEL_SIZE 3

__global__ void convolution(const float* image, const float* kernel, float* result, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float value = 0.0f;
    for (int ky = -KERNEL_SIZE / 2; ky <= KERNEL_SIZE / 2; ++ky) {
        for (int kx = -KERNEL_SIZE / 2; kx <= KERNEL_SIZE / 2; ++kx) {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);
            value += image[iy * width + ix] * kernel[(ky + KERNEL_SIZE / 2) * KERNEL_SIZE + (kx + KERNEL_SIZE / 2)];
        }
    }
    result[y * width + x] = value;
}

void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << msg << " Error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(msg);
    }
}

void cudaConvolution(const float* h_image, const float* h_kernel, float* h_result, int width, int height) {
    float *d_image, *d_kernel, *d_result;

    size_t imageSize = width * height * sizeof(float);
    size_t kernelSize = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    checkCudaError(cudaMalloc(&d_image, imageSize), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_kernel, kernelSize), "Failed to allocate device memory for kernel");
    checkCudaError(cudaMalloc(&d_result, imageSize), "Failed to allocate device memory for result");

    checkCudaError(cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice), "Failed to copy image to device");
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice), "Failed to copy kernel to device");

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convolution<<<gridSize, blockSize>>>(d_image, d_kernel, d_result, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    checkCudaError(cudaMemcpy(h_result, d_result, imageSize, cudaMemcpyDeviceToHost), "Failed to copy result to host");

    checkCudaError(cudaFree(d_image), "Failed to free device memory for image");
    checkCudaError(cudaFree(d_kernel), "Failed to free device memory for kernel");
    checkCudaError(cudaFree(d_result), "Failed to free device memory for result");
}

void logCudaDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA Devices: " << deviceCount << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << "Device " << i << ": " << props.name << std::endl;
        std::cout << "  Compute capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "  Total memory: " << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    }
}

void performCudaImageTransformation(const float* h_image, float* h_result, int width, int height) {
    float* d_image, *d_result;
    size_t imageSize = width * height * sizeof(float);

    checkCudaError(cudaMalloc(&d_image, imageSize), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_result, imageSize), "Failed to allocate device memory for result");

    checkCudaError(cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice), "Failed to copy image to device");

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    // Dummy transformation kernel launch
    convolution<<<gridSize, blockSize>>>(d_image, d_image, d_result, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    checkCudaError(cudaMemcpy(h_result, d_result, imageSize, cudaMemcpyDeviceToHost), "Failed to copy result to host");

    checkCudaError(cudaFree(d_image), "Failed to free device memory for image");
    checkCudaError(cudaFree(d_result), "Failed to free device memory for result");
}

void performCudaEdgeDetection(const float* h_image, float* h_result, int width, int height) {
    float* d_image, *d_result;
    size_t imageSize = width * height * sizeof(float);

    checkCudaError(cudaMalloc(&d_image, imageSize), "Failed to allocate device memory for image");
    checkCudaError(cudaMalloc(&d_result, imageSize), "Failed to allocate device memory for result");

    checkCudaError(cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice), "Failed to copy image to device");

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    // Dummy edge detection kernel launch
    convolution<<<gridSize, blockSize>>>(d_image, d_image, d_result, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    checkCudaError(cudaMemcpy(h_result, d_result, imageSize, cudaMemcpyDeviceToHost), "Failed to copy result to host");

    checkCudaError(cudaFree(d_image), "Failed to free device memory for image");
    checkCudaError(cudaFree(d_result), "Failed to free device memory for result");
}