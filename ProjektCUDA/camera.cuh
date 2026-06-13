#pragma once
#include <GLFW/glfw3.h>
#include "vec3.cuh"
#include <math.h>

// ==========================================
// KAMERA/INPUTY
// ==========================================
class Camera {
public:
    Vec3 position;
    Vec3 forward;
    Vec3 right;
    Vec3 up;

    float yaw;
    float pitch;
    float speed;
    float sensitivity;

    bool first_mouse;
    double last_mouse_x;
    double last_mouse_y;

    Camera(Vec3 start_pos) {
        position = start_pos;
        yaw = -90.0f;
        pitch = 0.0f;
        speed = 3.0f;
        sensitivity = 0.15f;
        first_mouse = true;
        last_mouse_x = 0.0;
        last_mouse_y = 0.0;
        update_vectors();
    }

    void update_vectors() {
        float yaw_rad = yaw * 3.14159265f / 180.0f;
        float pitch_rad = pitch * 3.14159265f / 180.0f;

        Vec3 front;
        front.x = cos(pitch_rad) * cos(yaw_rad);
        front.y = sin(pitch_rad);
        front.z = cos(pitch_rad) * sin(yaw_rad);
        forward = front.normalize();

        Vec3 world_up(0.0f, 1.0f, 0.0f);
        right = cross(forward, world_up).normalize();
        up = cross(right, forward).normalize();
    }

    void process_input(GLFWwindow* window, float delta_time) {
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

        yaw += offset_x * sensitivity;
        pitch += offset_y * sensitivity;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        update_vectors();
        float velocity = speed * delta_time;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) position = position + forward * velocity;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) position = position - forward * velocity;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) position = position - right * velocity;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) position = position + right * velocity;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) position = position + Vec3(0.0f, 1.0f, 0.0f) * velocity;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) position = position - Vec3(0.0f, 1.0f, 0.0f) * velocity;
    }
};