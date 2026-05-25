#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <cstdio>

// ==========================================
// 1. MATEMATYKA (Struktury Wektorów i Promieni)
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

__host__ __device__ float dot(const Vec3& u, const Vec3& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }

__host__ __device__ Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}
    __host__ __device__ Vec3 point_at_parameter(float t) const { return origin + direction * t; }
};

// ==========================================
// 2. FIZYKA ZDERZEŃ (Sfera i Nieskończona Podłoga)
// ==========================================
__device__ float hit_sphere(const Vec3& center, float radius, const Ray& r) {
    Vec3 oc = r.origin - center;
    float a = dot(r.direction, r.direction);
    float b = 2.0f * dot(oc, r.direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return -1.0f;
    else return (-b - sqrt(discriminant)) / (2.0f * a);
}

__device__ float hit_plane(float plane_y, const Ray& r) {
    if (abs(r.direction.y) < 0.0001f) return -1.0f;
    float t = (plane_y - r.origin.y) / r.direction.y;
    return t > 0.0f ? t : -1.0f;
}

// ==========================================
// 3. KERNEL CUDA (Rdzeń silnika)
// ==========================================
__global__ void render(unsigned char* fb, int width, int height, float time,
    Vec3 cam_pos, Vec3 cam_forward, Vec3 cam_right, Vec3 cam_up) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    float u = (float(x) / float(width)) * 4.0f - 2.0f;
    float v = (float(height - y) / float(height)) * 3.0f - 1.5f;

    // FOV i perspektywa
    float focal_length = 2.0f;
    Vec3 direction = (cam_right * u) + (cam_up * v) + (cam_forward * focal_length);
    Ray r(cam_pos, direction.normalize());

    // Ustawienia sceny
    Vec3 sphere_center(0.0f, 0.0f, -2.0f);
    float sphere_radius = 0.5f;
    float floor_y = -0.5f;

    // Fizyczna żarówka w stałym miejscu przestrzeni
    Vec3 light_pos(2.0f, 2.0f, -1.0f);

    // Zmienne do śledzenia trafień
    float t_min = 99999.0f;
    int hit_object = 0;
    Vec3 hit_point;
    Vec3 normal;
    Vec3 base_color;

    // Test 1: Podłoga
    float t_plane = hit_plane(floor_y, r);
    if (t_plane > 0.0f && t_plane < t_min) {
        t_min = t_plane;
        hit_object = 2;
        hit_point = r.point_at_parameter(t_plane);
        normal = Vec3(0.0f, 1.0f, 0.0f);

        int ix = floorf(hit_point.x * 2.0f);
        int iz = floorf(hit_point.z * 2.0f);
        if (abs(ix + iz) % 2 == 0) base_color = Vec3(220.0f, 220.0f, 220.0f);
        else base_color = Vec3(60.0f, 60.0f, 60.0f);
    }

    // Test 2: Kula
    float t_sphere = hit_sphere(sphere_center, sphere_radius, r);
    if (t_sphere > 0.0f && t_sphere < t_min) {
        t_min = t_sphere;
        hit_object = 1;
        hit_point = r.point_at_parameter(t_sphere);
        normal = (hit_point - sphere_center).normalize();
        base_color = Vec3(255.0f, 50.0f, 50.0f);
    }

    // Kolorowanie piksela
    Vec3 pixel_color;
    if (hit_object > 0) {
        Vec3 light_dir = (light_pos - hit_point).normalize();
        float distance_to_light = (light_pos - hit_point).length();

        Ray shadow_ray(hit_point + normal * 0.001f, light_dir);

        bool in_shadow = false;
        float t_shadow = hit_sphere(sphere_center, sphere_radius, shadow_ray);
        if (t_shadow > 0.0f && t_shadow < distance_to_light) {
            in_shadow = true;
        }

        float light_intensity = dot(normal, light_dir);
        if (light_intensity < 0.0f) light_intensity = 0.0f;
        if (in_shadow) light_intensity = 0.0f;

        // Tłumienie światła z odległością (Falloff)
        float attenuation = 1.0f / (1.0f + 0.1f * distance_to_light * distance_to_light);
        light_intensity *= attenuation;

        float ambient = 0.15f;
        float brightness = ambient + (1.0f - ambient) * light_intensity;

        pixel_color = base_color * brightness;
    }
    else {
        // Czy promień z kamery trafił prosto w naszą żarówkę?
        float t_light = hit_sphere(light_pos, 0.15f, r);

        if (t_light > 0.0f) {
            pixel_color = Vec3(255.0f, 255.0f, 255.0f); // Rysujemy białą żarówkę
        }
        else {
            // Tło - niebo
            float t = 0.5f * (r.direction.y + 1.0f);
            pixel_color = Vec3(255.0f, 255.0f, 255.0f) * (1.0f - t) + Vec3(127.0f, 178.0f, 255.0f) * t;
        }
    }

    int pixel_index = (y * width + x) * 3;
    fb[pixel_index + 0] = (unsigned char)pixel_color.x;
    fb[pixel_index + 1] = (unsigned char)pixel_color.y;
    fb[pixel_index + 2] = (unsigned char)pixel_color.z;
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ==========================================
// 4. GŁÓWNA FUNKCJA (Obsługa okna, myszy FPS)
// ==========================================
int main() {
    int width = 800;
    int height = 600;

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(width, height, "Ray Tracer CUDA 3D", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) return -1;

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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

    Vec3 camera_pos(0.0f, 0.0f, 1.5f);
    float yaw = -90.0f;
    float pitch = 0.0f;

    double last_mouse_x = 0.0, last_mouse_y = 0.0;
    bool first_mouse = true;

    double last_time = glfwGetTime();
    double last_frame_time = glfwGetTime();
    int nb_frames = 0;

    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        float delta_time = (float)(current_time - last_frame_time);
        last_frame_time = current_time;

        // OBSŁUGA MYSZY
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);

        if (first_mouse) {
            last_mouse_x = mouse_x;
            last_mouse_y = mouse_y;
            first_mouse = false;
        }

        float offset_x = (float)(mouse_x - last_mouse_x);
        float offset_y = (float)(last_mouse_y - mouse_y);
        last_mouse_x = mouse_x;
        last_mouse_y = mouse_y;

        float sensitivity = 0.15f;
        yaw += offset_x * sensitivity;
        pitch += offset_y * sensitivity;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        float yaw_rad = yaw * 3.14159265f / 180.0f;
        float pitch_rad = pitch * 3.14159265f / 180.0f;

        Vec3 forward;
        forward.x = cos(pitch_rad) * cos(yaw_rad);
        forward.y = sin(pitch_rad);
        forward.z = cos(pitch_rad) * sin(yaw_rad);
        Vec3 cam_forward = forward.normalize();

        Vec3 world_up(0.0f, 1.0f, 0.0f);
        Vec3 cam_right = cross(cam_forward, world_up).normalize();
        Vec3 cam_up = cross(cam_right, cam_forward).normalize();

        // OBSŁUGA KLAWIATURY
        float camera_speed = 3.0f * delta_time;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera_pos = camera_pos + cam_forward * camera_speed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera_pos = camera_pos - cam_forward * camera_speed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera_pos = camera_pos - cam_right * camera_speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera_pos = camera_pos + cam_right * camera_speed;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera_pos = camera_pos + world_up * camera_speed;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera_pos = camera_pos - world_up * camera_speed;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

        // RENDEROWANIE CUDA
        render << <grid, blocks >> > (d_fb, width, height, (float)current_time, camera_pos, cam_forward, cam_right, cam_up);
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

        // FPS I DIAGNOSTYKA
        nb_frames++;
        if (current_time - last_time >= 1.0) {
            double frame_time = 1000.0 / double(nb_frames);
            double fps = double(nb_frames);
            char title[256];
            snprintf(title, sizeof(title), "Ray Tracer CUDA | Tryb: GPU | %.1f FPS | %.2f ms | Cam: %.1f, %.1f, %.1f",
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