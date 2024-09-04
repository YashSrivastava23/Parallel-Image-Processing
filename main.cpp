#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>

// Function declarations
void cudaConvolution(const float* h_image, const float* h_kernel, float* h_result, int width, int height);
void edgeDetection(const float* image, float* result, int width, int height);
void logCudaDeviceProperties();
void logOpenMPInfo();
void performCudaImageTransformation(const float* h_image, float* h_result, int width, int height);
void performOpenMPImageTransformation(const float* image, float* result, int width, int height);
void computeImageStatistics(const float* image, int width, int height, float& minVal, float& maxVal, float& meanVal);
void performCudaEdgeDetection(const float* h_image, float* h_result, int width, int height);
void performOpenMPEdgeDetection(const float* image, float* result, int width, int height);
void applyCustomFilter(const float* image, float* result, int width, int height, const float* filter, int filterSize);

#define WIDTH 1024
#define HEIGHT 1024
#define KERNEL_SIZE 3
#define CUSTOM_FILTER_SIZE 5

void saveImage(const float* image, int width, int height, const char* filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                file << image[y * width + x] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

int main() {
    // Initialize image and kernel
    float* image = new float[WIDTH * HEIGHT];
    float* result = new float[WIDTH * HEIGHT];
    float kernel[KERNEL_SIZE * KERNEL_SIZE] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    float customFilter[CUSTOM_FILTER_SIZE * CUSTOM_FILTER_SIZE] = {
        1, 2, 3, 2, 1,
        2, 4, 5, 4, 2,
        3, 5, 6, 5, 3,
        2, 4, 5, 4, 2,
        1, 2, 3, 2, 1
    };

    // Fill image with dummy data
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        image[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Log CUDA and OpenMP properties
    logCudaDeviceProperties();
    logOpenMPInfo();

    // CUDA Convolution
    auto start = std::chrono::high_resolution_clock::now();
    cudaConvolution(image, kernel, result, WIDTH, HEIGHT);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "CUDA Convolution Time: " << duration.count() << " seconds" << std::endl;

    saveImage(result, WIDTH, HEIGHT, "cuda_convolution_result.txt");

    // OpenMP Edge Detection
    start = std::chrono::high_resolution_clock::now();
    edgeDetection(image, result, WIDTH, HEIGHT);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenMP Edge Detection Time: " << duration.count() << " seconds" << std::endl;

    saveImage(result, WIDTH, HEIGHT, "omp_edge_detection_result.txt");

    // CUDA Image Transformation
    start = std::chrono::high_resolution_clock::now();
    performCudaImageTransformation(image, result, WIDTH, HEIGHT);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CUDA Image Transformation Time: " << duration.count() << " seconds" << std::endl;

    saveImage(result, WIDTH, HEIGHT, "cuda_image_transformation_result.txt");

    // OpenMP Image Transformation
    start = std::chrono::high_resolution_clock::now();
    performOpenMPImageTransformation(image, result, WIDTH, HEIGHT);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenMP Image Transformation Time: " << duration.count() << " seconds" << std::endl;

    saveImage(result, WIDTH, HEIGHT, "omp_image_transformation_result.txt");

    // Compute Image Statistics
    float minVal, maxVal, meanVal;
    computeImageStatistics(image, WIDTH, HEIGHT, minVal, maxVal, meanVal);
    std::cout << "Image Statistics - Min: " << minVal << ", Max: " << maxVal << ", Mean: " << meanVal << std::endl;

    // CUDA Edge Detection
    start = std::chrono::high_resolution_clock::now();
    performCudaEdgeDetection(image, result, WIDTH, HEIGHT);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CUDA Edge Detection Time: " << duration.count() << " seconds" << std::endl;

    saveImage(result, WIDTH, HEIGHT, "cuda_edge_detection_result.txt");

    // OpenMP Edge Detection
    start = std::chrono::high_resolution_clock::now();
    performOpenMPEdgeDetection(image, result, WIDTH, HEIGHT);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenMP Edge Detection Time: " << duration.count() << " seconds" << std::endl;

    saveImage(result, WIDTH, HEIGHT, "omp_edge_detection_result.txt");

    // Apply Custom Filter using OpenMP
    start = std::chrono::high_resolution_clock::now();
    applyCustomFilter(image, result, WIDTH, HEIGHT, customFilter, CUSTOM_FILTER_SIZE);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "OpenMP Custom Filter Application Time: " << duration.count() << " seconds" << std::endl;

    saveImage(result, WIDTH, HEIGHT, "omp_custom_filter_result.txt");

    delete[] image;
    delete[] result;

    return 0;
}