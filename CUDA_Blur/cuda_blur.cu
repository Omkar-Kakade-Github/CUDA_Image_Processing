#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel for blurring an RGB image
__global__ void cuda_blur(unsigned char *input_image, unsigned char *output_image, int image_width, int image_height, int blur_radius) {
    // Calculate the pixel coordinates this thread is responsible for
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel coordinates are within the image bounds
    if (pixel_x >= image_width || pixel_y >= image_height) {
        return;
    }

    // Variables to accumulate the sum of RGB values in the neighborhood
    int red_sum = 0;
    int green_sum = 0;
    int blue_sum = 0;
    int pixel_count = 0;

    // Iterate over the neighborhood defined by the blur radius
    for (int offset_y = -blur_radius; offset_y <= blur_radius; offset_y++) {
        for (int offset_x = -blur_radius; offset_x <= blur_radius; offset_x++) {
            // Calculate the coordinates of the neighboring pixel
            int neighbor_x = pixel_x + offset_x;
            int neighbor_y = pixel_y + offset_y;

            // Check if the neighboring pixel is within the image bounds
            if (neighbor_x >= 0 && neighbor_x < image_width && neighbor_y >= 0 && neighbor_y < image_height) {
                // Calculate the index of the neighboring pixel in the input image array
                int neighbor_index = (neighbor_y * image_width + neighbor_x) * 3;

                // Accumulate the RGB values of the neighboring pixel
                red_sum += input_image[neighbor_index];
                green_sum += input_image[neighbor_index + 1];
                blue_sum += input_image[neighbor_index + 2];
                pixel_count++;
            }
        }
    }

    // Calculate the index of the current pixel in the output image array
    int output_index = (pixel_y * image_width + pixel_x) * 3;

    // Compute the average RGB values and write them to the output image
    output_image[output_index] = red_sum / pixel_count;         // Red channel
    output_image[output_index + 1] = green_sum / pixel_count;   // Green channel
    output_image[output_index + 2] = blue_sum / pixel_count;    // Blue channel
}

int main() {
    // Load the input image
    int width, height, channels;
    unsigned char *input_image = stbi_load("/home/omkar/Brendan/Projects/CUDA_Blur/input.jpg", &width, &height, &channels, 3);
    if (!input_image) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Allocate memory for the output image
    unsigned char *output_image = (unsigned char *)malloc(width * height * 3);
    if (!output_image) {
        std::cerr << "Error: Could not allocate memory for output image!" << std::endl;
        stbi_image_free(input_image);
        return -1;
    }

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, width * height * 3);
    cudaMalloc((void **)&d_output, width * height * 3);

    // Copy input image to device
    cudaMemcpy(d_input, input_image, width * height * 3, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Set blur radius
    int radius = 5;

    // Launch the CUDA kernel
    cuda_blur<<<gridDim, blockDim>>>(d_input, d_output, width, height, radius);

    // Copy the result back to the host
    cudaMemcpy(output_image, d_output, width * height * 3, cudaMemcpyDeviceToHost);

    // Save the output image
    stbi_write_jpg("output_blurred.jpg", width, height, 3, output_image, 100);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    stbi_image_free(input_image);
    free(output_image);

    std::cout << "Blurred image saved as output_blurred.jpg" << std::endl;
    return 0;
}
