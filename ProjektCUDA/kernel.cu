#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <math.h>

// ==========================================
// 1. MATEMATYKA I FIZYKA (Stary kod)
// ==========================================
struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    __host__ __device__ float length() const { return sqrt(x * x + y * y + z * z); }
    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return Vec3(x / len, y / len, z / len);
    }
};

__host__ __device__ float dot(const Vec3& u, const Vec3& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}
    __host__ __device__ Vec3 point_at_parameter(float t) const { return origin + direction * t; }
};

__device__ float hit_sphere(const Vec3& center, float radius, const Ray& r) {
    Vec3 oc = r.origin - center;
    float a = dot(r.direction, r.direction);
    float b = 2.0f * dot(oc, r.direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return -1.0f;
    else return (-b - sqrt(discriminant)) / (2.0f * a);
}

// ==========================================
// 2. KERNEL CUDA (Rysowanie klatki)
// ==========================================
__global__ void render(unsigned char* fb, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    // Przekształcenie u, v (w Ray Tracingu często oś Y rośnie w górę, w OpenGL w dół, stąd mała zmiana)
    float u = (float(x) / float(width)) * 4.0f - 2.0f;
    float v = (float(height - y) / float(height)) * 3.0f - 1.5f;

    Ray r(Vec3(0.0f, 0.0f, 0.0f), Vec3(u, v, -1.0f));
    Vec3 pixel_color;

    float t_hit = hit_sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, r);

    if (t_hit > 0.0f) {
        Vec3 P = r.point_at_parameter(t_hit);
        Vec3 N = (P - Vec3(0.0f, 0.0f, -1.0f)).normalize();
        Vec3 light_dir = Vec3(1.0f, 1.0f, -0.5f).normalize();

        float light_intensity = dot(N, light_dir);
        if (light_intensity < 0.0f) light_intensity = 0.0f;

        float ambient = 0.2f;
        float brightness = ambient + (1.0f - ambient) * light_intensity;
        pixel_color = Vec3(255.0f, 0.0f, 0.0f) * brightness;
    }
    else {
        Vec3 unit_direction = r.direction.normalize();
        float t = 0.5f * (unit_direction.y + 1.0f);
        pixel_color = Vec3(255.0f, 255.0f, 255.0f) * (1.0f - t) + Vec3(127.0f, 178.0f, 255.0f) * t;
    }

    int pixel_index = (y * width + x) * 3;
    fb[pixel_index + 0] = (unsigned char)pixel_color.x;
    fb[pixel_index + 1] = (unsigned char)pixel_color.y;
    fb[pixel_index + 2] = (unsigned char)pixel_color.z;
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ==========================================
// 3. GŁÓWNA PĘTLA PROGRAMU I OKIENKO
// ==========================================
int main() {
    int width = 800;
    int height = 600;

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(width, height, "Ray Tracer 3D na ZYWO!", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) return -1;

    // Przygotowanie pamięci na obraz (Host i Device)
    size_t fb_size = width * height * 3 * sizeof(unsigned char);
    std::vector<unsigned char> h_fb(width * height * 3);
    unsigned char* d_fb;
    cudaMalloc((void**)&d_fb, fb_size);

    dim3 blocks(16, 16);
    dim3 grid(ceil_div(width, blocks.x), ceil_div(height, blocks.y));

    // Tworzenie "naklejki" (tekstury) w OpenGL
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    // Ustawienia tekstury (żeby piksele się nie rozmazywały przy skalowaniu)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Główna pętla programu
    while (!glfwWindowShouldClose(window)) {

        // 1. CUDA generuje obraz!
        render << <grid, blocks >> > (d_fb, width, height);
        cudaDeviceSynchronize();

        // 2. Kopiujemy gotowy obraz z karty graficznej na procesor
        cudaMemcpy(h_fb.data(), d_fb, fb_size, cudaMemcpyDeviceToHost);

        // 3. Wrzucamy obraz z procesora do tekstury OpenGL
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, h_fb.data());

        // 4. Rysujemy pełnoekranowy kwadrat z naszą teksturą
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f); // Lewy górny
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f); // Prawy górny
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f); // Prawy dolny
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f); // Lewy dolny
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(d_fb);
    glfwTerminate();
    return 0;
}