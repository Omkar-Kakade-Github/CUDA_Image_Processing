#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// CUDA kernel for RGB to grayscale conversion
__global__ void rgbToGrayscale(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Calculate the index for the current pixel
        int grayOffset = y * width + x;

        // Get the RGB values
        unsigned char r = input[grayOffset * 3];
        unsigned char g = input[grayOffset * 3 + 1];
        unsigned char b = input[grayOffset * 3 + 2];

        // Convert to grayscale
        output[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

int main() {
    // Load the input image using OpenCV
    cv::Mat image = cv::imread("/home/omkar/Brendan/Projects/CUDA_Grayscaling/input.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // Allocate memory for the input and output images on the host
    unsigned char* h_input = image.data;
    unsigned char* h_output = new unsigned char[width * height];

    // Allocate memory on the device
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * sizeof(unsigned char));

    // Copy the input image to the device
    cudaMemcpy(d_input, h_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    rgbToGrayscale<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the output grayscale image
    cv::Mat grayscaleImage(height, width, CV_8UC1, h_output);
    cv::imwrite("output_grayscale.jpg", grayscaleImage);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] h_output;

    std::cout << "Grayscale conversion complete! Output saved as output_grayscale.jpg" << std::endl;

    return 0;
}
