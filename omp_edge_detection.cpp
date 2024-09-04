#include <iostream>
#include <omp.h>
#include <cmath>
#include <stdexcept>
#include <cfloat>

#define WIDTH 1024
#define HEIGHT 1024
#define SOBEL_X {-1, 0, 1, -2, 0, 2, -1, 0, 1}
#define SOBEL_Y {-1, -2, -1, 0, 0, 0, 1, 2, 1}

void edgeDetection(const float* image, float* result, int width, int height) {
    int sobel_x[9] = SOBEL_X;
    int sobel_y[9] = SOBEL_Y;

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    float pixel = image[iy * width + ix];
                    gx += pixel * sobel_x[(ky + 1) * 3 + (kx + 1)];
                    gy += pixel * sobel_y[(ky + 1) * 3 + (kx + 1)];
                }
            }
            result[y * width + x] = sqrt(gx * gx + gy * gy);
        }
    }
}

void logOpenMPInfo() {
    int maxThreads = omp_get_max_threads();
    std::cout << "OpenMP Max Threads: " << maxThreads << std::endl;
    std::cout << "OpenMP Nested Parallelism: " << omp_get_nested() << std::endl;
}

void performOpenMPImageTransformation(const float* image, float* result, int width, int height) {
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            result[y * width + x] = image[y * width + x] * 0.5f; // Example transformation
        }
    }
}

void computeImageStatistics(const float* image, int width, int height, float& minVal, float& maxVal, float& meanVal) {
    minVal = FLT_MAX;
    maxVal = FLT_MIN;
    float sum = 0.0f;

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:sum)
    for (int i = 0; i < width * height; ++i) {
        float pixel = image[i];
        if (pixel < minVal) minVal = pixel;
        if (pixel > maxVal) maxVal = pixel;
        sum += pixel;
    }

    meanVal = sum / (width * height);
}

void performOpenMPEdgeDetection(const float* image, float* result, int width, int height) {
    int sobel_x[9] = SOBEL_X;
    int sobel_y[9] = SOBEL_Y;

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    float pixel = image[iy * width + ix];
                    gx += pixel * sobel_x[(ky + 1) * 3 + (kx + 1)];
                    gy += pixel * sobel_y[(ky + 1) * 3 + (kx + 1)];
                }
            }
            result[y * width + x] = sqrt(gx * gx + gy * gy);
        }
    }
}

void applyCustomFilter(const float* image, float* result, int width, int height, const float* filter, int filterSize) {
    int halfSize = filterSize / 2;

    #pragma omp parallel for collapse(2)
    for (int y = halfSize; y < height - halfSize; ++y) {
        for (int x = halfSize; x < width - halfSize; ++x) {
            float value = 0.0f;
            for (int ky = -halfSize; ky <= halfSize; ++ky) {
                for (int kx = -halfSize; kx <= halfSize; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    value += image[iy * width + ix] * filter[(ky + halfSize) * filterSize + (kx + halfSize)];
                }
            }
            result[y * width + x] = value;
        }
    }
}
