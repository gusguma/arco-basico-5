cmake_minimum_required(VERSION 3.23)
project(arco_basico_5 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(arco_basico_5 main.cu)

set_target_properties(arco_basico_5 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
