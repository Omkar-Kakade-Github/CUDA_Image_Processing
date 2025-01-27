cmake_minimum_required(VERSION 3.8)
project(CUDA_Grayscaling)

# Enable CUDA support
enable_language(CUDA)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Specify CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75) # Example: 75 for Turing (e.g., RTX 20-series), 86 for Ampere (e.g., RTX 30-series)

# Add the executable
add_executable(rgb_to_grayscale rgb_to_grayscale.cu)

# Link OpenCV libraries
target_link_libraries(rgb_to_grayscale ${OpenCV_LIBS})
