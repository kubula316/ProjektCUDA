#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <fstream> 

#include "vec3.cuh"
#include "physics.cuh"
#include "renderer.cuh"
#include "camera.cuh"

int main() {
    int width = 1280;
    int height = 720;

    bool use_gpu = true;
    bool space_was_pressed = false;
    bool b_was_pressed = false;

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(width, height, "Path Tracer CUDA Benchmark", NULL, NULL);
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
    // INICJALIZACJA SCENY I PARAMETRÓW TESTU
    // ==========================================
    int num_spheres = 64;      // liczba obiektów przed kompilacją
    int shadow_samples = 32;    // jakość cieni (np. 1, 4, 16, 64)

    std::vector<Sphere> h_spheres(num_spheres);

    h_spheres[0].center = Vec3(-2.5f, 0.5f, -3.0f); h_spheres[0].radius = 1.0f; h_spheres[0].mat.color = Vec3(255.0f, 50.0f, 50.0f);
    h_spheres[1].center = Vec3(0.0f, 0.5f, -2.0f); h_spheres[1].radius = 1.0f; h_spheres[1].mat.color = Vec3(50.0f, 255.0f, 50.0f);
    h_spheres[2].center = Vec3(2.5f, 0.5f, -3.0f); h_spheres[2].radius = 1.0f; h_spheres[2].mat.color = Vec3(50.0f, 50.0f, 255.0f);

    for (int i = 3; i < num_spheres; i++) {
        h_spheres[i].center = Vec3(((rand() % 100) / 100.0f) * 20.0f - 10.0f, 0.2f, ((rand() % 100) / 100.0f) * 20.0f - 15.0f);
        h_spheres[i].radius = ((rand() % 100) / 100.0f) * 0.3f + 0.1f;
        h_spheres[i].mat.color = Vec3(rand() % 255, rand() % 255, rand() % 255);
    }

    cudaMemcpyToSymbol(c_spheres_data, h_spheres.data(), num_spheres * sizeof(Sphere));

    Camera cam(Vec3(0.0f, 1.5f, 4.0f));

    double last_time = glfwGetTime();
    double last_frame_time = glfwGetTime();
    int nb_frames = 0;

    while (!glfwWindowShouldClose(window)) {
        double current_time = glfwGetTime();
        float delta_time = (float)(current_time - last_frame_time);
        last_frame_time = current_time;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

        bool space_is_pressed = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
        if (space_is_pressed && !space_was_pressed) use_gpu = !use_gpu;
        space_was_pressed = space_is_pressed;

        // ==========================================
        // SYSTEM BENCHMARKOWANIA
        // ==========================================
        bool b_is_pressed = (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS);
        if (b_is_pressed && !b_was_pressed) {
            std::cout << "\n--- ROZPOCZYNAM BENCHMARK ---" << std::endl;
            std::cout << "Sfery: " << num_spheres << " | Probki Cieni: " << shadow_samples << std::endl;

            // Parametry testu
            int gpu_test_frames = 100;
            int cpu_test_frames = 10;

            // GPU TEST
            std::cout << "Testuje GPU (" << gpu_test_frames << " klatek)... ";
            double start_gpu = glfwGetTime();
            for (int i = 0; i < gpu_test_frames; i++) {
                render << <grid, blocks >> > (d_fb, width, height, (float)glfwGetTime(), cam.position, cam.forward, cam.right, cam.up, num_spheres, shadow_samples);
                cudaDeviceSynchronize();
            }
            double gpu_ms = ((glfwGetTime() - start_gpu) / (double)gpu_test_frames) * 1000.0;
            std::cout << "OK!" << std::endl;

            // CPU TEST
            std::cout << "Testuje CPU (" << cpu_test_frames << " klatek, to zajmie chwile)... ";
            double start_cpu = glfwGetTime();
            for (int i = 0; i < cpu_test_frames; i++) {
                render_cpu(h_fb.data(), width, height, (float)glfwGetTime(), cam.position, cam.forward, cam.right, cam.up, h_spheres.data(), num_spheres, shadow_samples);
            }
            double cpu_ms = ((glfwGetTime() - start_cpu) / (double)cpu_test_frames) * 1000.0;
            std::cout << "OK!" << std::endl;

            // Zapis do CSV
            std::ofstream csv_file("benchmark_results.csv", std::ios_base::app);
            csv_file.seekp(0, std::ios::end);
            if (csv_file.tellp() == 0) {
                // Nowy nagłówek uwzględnia próbki cieni
                csv_file << "Rozdzielczosc,Liczba_Sfer,Shadow_Samples,Czas_GPU_ms,FPS_GPU,Czas_CPU_ms,FPS_CPU,Przyspieszenie_GPU\n";
            }

            float gpu_fps = 1000.0f / gpu_ms;
            float cpu_fps = 1000.0f / cpu_ms;
            float speedup = cpu_ms / gpu_ms;

            csv_file << width << "x" << height << ","
                << num_spheres << ","
                << shadow_samples << ","
                << gpu_ms << "," << gpu_fps << ","
                << cpu_ms << "," << cpu_fps << ","
                << speedup << "x\n";

            std::cout << ">> Wyniki zapisano do CSV! GPU bylo szybsze " << speedup << " razy." << std::endl;
        }
        b_was_pressed = b_is_pressed;

        cam.process_input(window, delta_time);

        if (use_gpu) {
            render << <grid, blocks >> > (d_fb, width, height, (float)current_time,
                cam.position, cam.forward, cam.right, cam.up, num_spheres, shadow_samples); // Przekazujemy zmienną
            cudaDeviceSynchronize();
            cudaMemcpy(h_fb.data(), d_fb, fb_size, cudaMemcpyDeviceToHost);
        }
        else {
            render_cpu(h_fb.data(), width, height, (float)current_time,
                cam.position, cam.forward, cam.right, cam.up, h_spheres.data(), num_spheres, shadow_samples); // Przekazujemy zmienną
        }

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

        nb_frames++;
        if (current_time - last_time >= 1.0) {
            double frame_time = 1000.0 / double(nb_frames);
            double fps = double(nb_frames);
            char title[256];
            // Aktualizacja tytułu okna, żeby pokazywał jakość cieni
            snprintf(title, sizeof(title), "Path Tracer CUDA | %d Sfer, %d Probki | %s | %.1f FPS | [B] Benchmark",
                num_spheres, shadow_samples, use_gpu ? "GPU" : "CPU", fps);
            glfwSetWindowTitle(window, title);
            nb_frames = 0; last_time += 1.0;
        }
    }

    cudaFree(d_fb);
    glfwTerminate();
    return 0;
}