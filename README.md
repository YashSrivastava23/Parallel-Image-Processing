# Parallel Image Processing Project

## Overview
This project implements parallel image processing techniques using CUDA and OpenMP to accelerate computationally intensive tasks such as convolution, edge detection, and image transformations. The code is written in C++ and demonstrates how GPU acceleration and multi-threading can significantly improve performance in image processing applications.

## Features

### CUDA Implementations:
- **Convolution:** Performs convolution on images using CUDA kernels for GPU acceleration.
- **Edge Detection:** Implements edge detection using CUDA.
- **Image Transformation:** Applies transformations to images using CUDA.

### OpenMP Implementations:
- **Edge Detection:** Performs edge detection using OpenMP for CPU-based parallelism.
- **Image Transformation:** Applies transformations to images using OpenMP.
- **Custom Filters:** Supports application of custom filters using OpenMP.
- **Image Statistics:** Computes statistical data (min, max, mean) of images using OpenMP.

### Performance Logging:
- **Execution Time:** Measures and logs execution time for various operations.
- **Device Properties:** Logs CUDA device properties and OpenMP settings.

### Modular Design:
- **Code Structure:** Organized code structure for easy extension and maintenance.
- **Separation of Concerns:** Separated concerns between CUDA and OpenMP functionalities.

## Technologies Used
- **Programming Language:** C++
- **Parallel Computing:**
  - **CUDA:** For GPU-based parallel processing.
  - **OpenMP:** For multi-threaded CPU processing.
- **Libraries and APIs:**
  - `<cuda_runtime.h>` and `<device_launch_parameters.h>`: For CUDA operations.
  - `<omp.h>`: For OpenMP operations.

- **Standard Libraries:**
  - `<iostream>`, `<fstream>`: For input/output operations.
  - `<chrono>`: For measuring execution time.
  - `<cmath>`: For mathematical functions.
  - `<cstdlib>`, `<vector>`, `<stdexcept>`: For general utilities.

## Detailed Description

### `main.cpp`
- **Purpose:** Serves as the entry point of the application. It initializes data, calls processing functions, measures execution time, and saves results.
- **Key Functions:**
  - `saveImage()`: Saves image data to a text file.
  - Timing and logging execution times for each operation.
- **Processing Steps:**
  - Initializes dummy image data and kernels.
  - Logs CUDA and OpenMP properties.
  - Performs CUDA convolution and saves results.
  - Performs OpenMP edge detection and saves results.
  - Performs image transformations using both CUDA and OpenMP.
  - Computes image statistics.
  - Applies custom filters using OpenMP.

### `cuda_convolution.cu`
- **Purpose:** Contains CUDA kernels and functions for GPU-accelerated image processing.
- **Key Components:**
  - **Kernels:**
    - `convolution()`: Performs convolution on the GPU.
  - **Functions:**
    - `cudaConvolution()`: Manages memory and launches the convolution kernel.
    - `performCudaImageTransformation()`: Applies transformations using CUDA.
    - `performCudaEdgeDetection()`: Placeholder for CUDA edge detection.
  - **Utilities:**
    - `checkCudaError()`: Error checking for CUDA calls.
    - `logCudaDeviceProperties()`: Logs information about available CUDA devices.

### `omp_edge_detection.cpp`
- **Purpose:** Implements image processing functions using OpenMP for parallel execution on the CPU.
- **Key Functions:**
  - `edgeDetection()`: Performs edge detection using the Sobel operator.
  - `logOpenMPInfo()`: Logs OpenMP configuration.
  - `performOpenMPImageTransformation()`: Applies transformations using OpenMP.
  - `computeImageStatistics()`: Calculates min, max, and mean pixel values.
  - `performOpenMPEdgeDetection()`: Another implementation of edge detection.
  - `applyCustomFilter()`: Applies a custom filter to the image.

## Customization

### Adjust Image Size:
- Modify `#define WIDTH` and `#define HEIGHT` in the source files to change the dimensions.

### Use Real Images:
- Replace the dummy data generation in `main.cpp` with actual image loading logic. You can use libraries like OpenCV or stb_image for handling image files.

### Modify Kernels and Filters:
- Update the convolution kernel in `cuda_convolution.cu`.
- Change the Sobel operators or create new filters in `omp_edge_detection.cpp`.

### Adjust Parallelism Parameters:
- For OpenMP, set the number of threads using environment variables or OpenMP API functions.
- For CUDA, adjust block and grid sizes to optimize performance.

## Performance Evaluation

### Execution Time:
- The program measures and outputs the execution time for each processing step.
- Compare CUDA and OpenMP implementations to evaluate performance gains.

### Hardware Utilization:
- Ensure that your GPU is capable of running CUDA applications.
- Monitor CPU and GPU usage to identify bottlenecks.

## Future Enhancements

### Implement Full CUDA Edge Detection:
- Currently, the CUDA edge detection uses a placeholder. Implement a proper Sobel operator in CUDA.

### Image I/O Support:
- Integrate image reading and writing capabilities to handle standard image formats (e.g., PNG, JPEG).

### Optimization:
- Optimize memory usage and kernel performance.
- Experiment with different parallelism strategies.

### Additional Algorithms:
- Implement more complex image processing techniques such as blurring, sharpening, or noise reduction.

## Troubleshooting

### Common Issues:
- **Compilation Errors:** Ensure that all include paths and library links are correctly specified.
- **CUDA Errors:** Verify that your GPU supports CUDA and that the CUDA toolkit is properly installed.
- **OpenMP Errors:** Make sure OpenMP is enabled in your compiler settings (e.g., `-fopenmp` flag for GCC).

### Debugging Tips:
- Use CUDA's built-in debugging tools like `cuda-memcheck`.
- For OpenMP, you can set the environment variable `OMP_DISPLAY_ENV=TRUE` to display OpenMP settings.
