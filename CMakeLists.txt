cmake_minimum_required(VERSION 3.13.1)

# Set project name and version
project(SstealzZ-project-1 VERSION 1.0 LANGUAGES CXX)

# Define the executable target, here 'app' is the executable target
add_executable(app main.cpp)

# Include the Edge Impulse SDK and model files
include(edge-impulse-sdk/cmake/utils.cmake)
add_subdirectory(edge-impulse-sdk/cmake/zephyr)

# Find all model source files in the tflite-model directory
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "tflite-model" "*.cpp")

# Add include directories to the project (for example SDK and classifier)
target_include_directories(app PRIVATE . ./edge-impulse-sdk ./edge-impulse-sdk/classifier ./model-parameters ./tflite-model)

# Link the necessary source files
target_sources(app PRIVATE ${MODEL_SOURCE})

# You may need to link additional libraries or add further options depending on your environment
