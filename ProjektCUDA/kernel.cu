#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cstdio>

#include "vec3.cuh"
#include "physics.cuh"
#include "renderer.cuh"
#include "camera.cuh" // <--- Ładujemy nasz nowy moduł sterowania!

int main() {
    int width = 1280; // Zmieniłem na 720p dla wygody prezentacji w oknie
    int height = 720;

    bool use_gpu = true;
    bool space_was_pressed = false;

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(width, height, "Path Tracer CUDA", NULL, NULL);
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

    // ==========================================
    // INICJALIZACJA NASZEJ KAMERY
    // ==========================================
    Camera cam(Vec3(0.0f, 0.0f, 1.5f));

    double last_time = glfwGetTime();
    double last_frame_time = glfwGetTime();
    int nb_frames = 0;

    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        float delta_time = (float)(current_time - last_frame_time);
        last_frame_time = current_time;

        // Wyjście i przełącznik trybów (ESC i Spacja)
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
        bool space_is_pressed = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
        if (space_is_pressed && !space_was_pressed) use_gpu = !use_gpu;
        space_was_pressed = space_is_pressed;

        // ==========================================
        // CAŁE STEROWANIE W JEDNEJ LINIJCE!
        // ==========================================
        cam.process_input(window, delta_time);

        // RENDEROWANIE (Przekazujemy wektory prosto z obiektu cam)
        if (use_gpu) {
            render << <grid, blocks >> > (d_fb, width, height, (float)current_time,
                cam.position, cam.forward, cam.right, cam.up);
            cudaDeviceSynchronize();
            cudaMemcpy(h_fb.data(), d_fb, fb_size, cudaMemcpyDeviceToHost);
        }
        else {
            render_cpu(h_fb.data(), width, height, (float)current_time,
                cam.position, cam.forward, cam.right, cam.up);
        }

        // RYSOWANIE
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
            snprintf(title, sizeof(title), "Path Tracer CUDA | Tryb: %s | %.1f FPS | %.2f ms",
                use_gpu ? "GPU" : "CPU", fps, frame_time);
            glfwSetWindowTitle(window, title);
            nb_frames = 0; last_time += 1.0;
        }
    }

    cudaFree(d_fb);
    glfwTerminate();
    return 0;
}