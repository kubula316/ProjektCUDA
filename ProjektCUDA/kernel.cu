#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <cstdio>

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

// KERNEL: Zaktualizowana "kamera" odbierająca pozycję gracza
__global__ void render(unsigned char* fb, int width, int height, float time, Vec3 cam_pos) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    float u = (float(x) / float(width)) * 4.0f - 2.0f;
    float v = (float(height - y) / float(height)) * 3.0f - 1.5f;

    // NOWOŚĆ: Promienie wylatują z aktualnej pozycji kamery!
    Vec3 direction(u, v, -1.0f);
    Ray r(cam_pos, direction);

    Vec3 pixel_color;

    // Testowa kula (zostaje w miejscu Z = -1.0)
    float t_hit = hit_sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, r);

    if (t_hit > 0.0f) {
        Vec3 P = r.point_at_parameter(t_hit);
        Vec3 N = (P - Vec3(0.0f, 0.0f, -1.0f)).normalize();
        Vec3 light_dir = Vec3(sin(time), 1.0f, cos(time)).normalize();

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
    GLFWwindow* window = glfwCreateWindow(width, height, "Ray Tracer CUDA", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) return -1;

    size_t fb_size = width * height * 3 * sizeof(unsigned char);
    std::vector<unsigned char> h_fb(width * height * 3);
    unsigned char* d_fb;
    cudaMalloc((void**)&d_fb, fb_size);

    dim3 blocks(16, 16);
    dim3 grid(ceil_div(width, blocks.x), ceil_div(height, blocks.y));

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Zmienne czasowe i pozycja kamery
    double last_time = glfwGetTime();
    double last_frame_time = glfwGetTime();
    int nb_frames = 0;

    // Kamera zaczyna w punkcie 0,0,0
    Vec3 camera_pos(0.0f, 0.0f, 0.0f);

    while (!glfwWindowShouldClose(window)) {

        double current_time = glfwGetTime();

        // Obliczanie Delta Time (ile sekund trwała poprzednia klatka)
        float delta_time = (float)(current_time - last_frame_time);
        last_frame_time = current_time;

        // ---------------------------------------------------------
        // STEROWANIE KAMERĄ (Prędkość to np. 2.5 jednostki na sekundę)
        // ---------------------------------------------------------
        float camera_speed = 2.5f * delta_time;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera_pos.z -= camera_speed; // Do przodu (w głąb ekranu, u nas to -Z)
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera_pos.z += camera_speed; // Do tyłu
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera_pos.x -= camera_speed; // W lewo
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera_pos.x += camera_speed; // W prawo
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera_pos.y += camera_speed; // W górę
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera_pos.y -= camera_speed; // W dół

        // Zamykanie programu klawiszem ESCAPE
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

        // Wywołanie kernela z nowym parametrem kamery
        render << <grid, blocks >> > (d_fb, width, height, (float)current_time, camera_pos);
        cudaDeviceSynchronize();

        cudaMemcpy(h_fb.data(), d_fb, fb_size, cudaMemcpyDeviceToHost);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, h_fb.data());
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Liczenie FPS
        nb_frames++;
        if (current_time - last_time >= 1.0) {
            double frame_time = 1000.0 / double(nb_frames);
            double fps = double(nb_frames);
            char title[256];
            snprintf(title, sizeof(title), "Ray Tracer CUDA | Tryb: GPU | %.1f FPS | %.2f ms/klatke | Poz: %.1f, %.1f, %.1f",
                fps, frame_time, camera_pos.x, camera_pos.y, camera_pos.z);
            glfwSetWindowTitle(window, title);
            nb_frames = 0;
            last_time += 1.0;
        }
    }

    cudaFree(d_fb);
    glfwTerminate();
    return 0;
}