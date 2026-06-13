cmake_minimum_required(VERSION 3.18)
project(CUDAPathTracerBenchmark LANGUAGES CXX CUDA)

# Wymagaj standardu C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Konfiguracja kompilatora CUDA (architektura automatyczna lub ogólna)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3") # Maksymalna optymalizacja w CUDA

# 1. Wyszukiwanie pakietu OpenMP (dla wersji CPU)
find_package(OpenMP REQUIRED)

# 2. Wyszukiwanie pakietów OpenGL, GLEW i GLFW
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW3 REQUIRED)

# Definicja pliku wykonywalnego (kernel.cu automatycznie pociągnie pliki .cuh)
add_executable(cuda_benchmark kernel.cu)

# Wskazanie katalogu bieżącego jako inclusion path
target_include_directories(cuda_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 3. Linkowanie wszystkich wymaganych bibliotek sprzętowych i graficznych
target_link_libraries(cuda_benchmark PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    OpenMP::OpenMP_CXX
)

# Ustawienie flag dla kompilacji NVCC (wyłączenie ostrzeżeń o architekturze)
set_target_properties(cuda_benchmark PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)