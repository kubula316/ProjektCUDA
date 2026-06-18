ifeq ($(OS),Windows_NT)
    TARGET = ProjektCUDA.exe
    NVCC_FLAGS = -O3 -std=c++14 -Xcompiler /openmp
    INCLUDES = -IDependencies/include
    LIB_PATHS = -LDependencies/lib
    LIBS = opengl32.lib glfw3.lib glew32.lib
    CLEAN_CMD = del /Q $(TARGET) *.obj *.exp *.lib 2>nul
    RUN_CMD = $(TARGET)
else
    TARGET = ProjektCUDA
    NVCC_FLAGS = -O3 -std=c++14 -Xcompiler -fopenmp
    INCLUDES =
    LIB_PATHS =
    LIBS = -lGLEW -lglfw -lGL
    CLEAN_CMD = rm -f $(TARGET) *.o
    RUN_CMD = ./$(TARGET)
endif

SRC = kernel.cu
HEADERS = vec3.cuh physics.cuh renderer.cuh camera.cuh


all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	nvcc $(NVCC_FLAGS) $(INCLUDES) $(LIB_PATHS) $(SRC) -o $(TARGET) $(LIBS)

run: $(TARGET)
	$(RUN_CMD)

clean:
	-$(CLEAN_CMD)