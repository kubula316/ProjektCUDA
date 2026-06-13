NVCC = nvcc
NVCC_FLAGS = -O3 -Xcompiler -fopenmp -lglfw -lGL -lGLEW

TARGET = cuda_benchmark
SRCS = kernel.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(SRCS) -o $(TARGET) $(NVCC_FLAGS)

clean:
	rm -f $(TARGET)