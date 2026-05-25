#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "vec3.cuh"
#include "physics.cuh"

// ==========================================
// 3. KERNEL CUDA (Rdzeñ silnika)
// ==========================================
__global__ void render(unsigned char* fb, int width, int height, float time,
    Vec3 cam_pos, Vec3 cam_forward, Vec3 cam_right, Vec3 cam_up) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    float u = (float(x) / float(width)) * 4.0f - 2.0f;
    float v = (float(height - y) / float(height)) * 3.0f - 1.5f;

    float focal_length = 2.0f;
    Vec3 direction = (cam_right * u) + (cam_up * v) + (cam_forward * focal_length);
    Ray r(cam_pos, direction.normalize());

    float floor_y = -0.5f;
    float wall_z = -4.0f;
    Vec3 sphere_center(0.0f, 0.0f, -1.5f);
    float sphere_radius = 0.5f;

    Vec3 light_pos(sin(time * 0.5f) * 8.0f, 6.0f, -15.0f);

    float t_min = 99999.0f;
    int hit_object = 0;
    Vec3 hit_point, normal, base_color;

    float t_plane = hit_floor(floor_y, r);
    if (t_plane > 0.0f && t_plane < t_min) {
        t_min = t_plane; hit_object = 2; hit_point = r.point_at_parameter(t_plane); normal = Vec3(0.0f, 1.0f, 0.0f);
        int ix = floorf(hit_point.x * 4.0f); int iz = floorf(hit_point.z * 4.0f);
        if (abs(ix + iz) % 2 == 0) base_color = Vec3(200.0f, 200.0f, 200.0f); else base_color = Vec3(80.0f, 80.0f, 80.0f);
    }

    float t_wall = hit_wall(wall_z, r);
    if (t_wall > 0.0f && t_wall < t_min) {
        t_min = t_wall; hit_object = 3; hit_point = r.point_at_parameter(t_wall);
        normal = r.direction.z < 0.0f ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(0.0f, 0.0f, -1.0f);
        base_color = Vec3(100.0f, 150.0f, 200.0f);
    }

    float t_sphere = hit_sphere(sphere_center, sphere_radius, r);
    if (t_sphere > 0.0f && t_sphere < t_min) {
        t_min = t_sphere; hit_object = 1; hit_point = r.point_at_parameter(t_sphere);
        normal = (hit_point - sphere_center).normalize(); base_color = Vec3(255.0f, 50.0f, 50.0f);
    }

    Vec3 pixel_color;
    if (hit_object > 0) {
        Vec3 light_dir = (light_pos - hit_point).normalize();
        float distance_to_light = (light_pos - hit_point).length();
        Ray shadow_ray(hit_point + normal * 0.001f, light_dir);
        bool in_shadow = false;
        float t_shadow_sph = hit_sphere(sphere_center, sphere_radius, shadow_ray);
        if (t_shadow_sph > 0.0f && t_shadow_sph < distance_to_light) in_shadow = true;
        float t_shadow_wall = hit_wall(wall_z, shadow_ray);
        if (t_shadow_wall > 0.0f && t_shadow_wall < distance_to_light) in_shadow = true;

        float light_intensity = dot(normal, light_dir);
        if (light_intensity < 0.0f) light_intensity = 0.0f;
        if (in_shadow) light_intensity = 0.0f;

        float ambient = 0.15f;
        float brightness = ambient + (1.0f - ambient) * light_intensity;
        pixel_color = base_color * brightness;
    }
    else {
        float t_light = hit_sphere(light_pos, 2.0f, r);
        if (t_light > 0.0f) { pixel_color = Vec3(255.0f, 255.0f, 220.0f); }
        else {
            float t = 0.5f * (r.direction.y + 1.0f);
            pixel_color = Vec3(255.0f, 255.0f, 255.0f) * (1.0f - t) + Vec3(127.0f, 178.0f, 255.0f) * t;
        }
    }

    int pixel_index = (y * width + x) * 3;
    fb[pixel_index + 0] = (unsigned char)pixel_color.x;
    fb[pixel_index + 1] = (unsigned char)pixel_color.y;
    fb[pixel_index + 2] = (unsigned char)pixel_color.z;
}

// ==========================================
// WERSJA CPU
// ==========================================
inline void render_cpu(unsigned char* fb, int width, int height, float time,
    Vec3 cam_pos, Vec3 cam_forward, Vec3 cam_right, Vec3 cam_up) {
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = (float(x) / float(width)) * 4.0f - 2.0f;
            float v = (float(height - y) / float(height)) * 3.0f - 1.5f;

            float focal_length = 2.0f;
            Vec3 direction = (cam_right * u) + (cam_up * v) + (cam_forward * focal_length);
            Ray r(cam_pos, direction.normalize());

            float floor_y = -0.5f;
            float wall_z = -4.0f;
            Vec3 sphere_center(0.0f, 0.0f, -1.5f);
            float sphere_radius = 0.5f;

            Vec3 light_pos(sin(time * 0.5f) * 8.0f, 6.0f, -15.0f);

            float t_min = 99999.0f;
            int hit_object = 0;
            Vec3 hit_point, normal, base_color;

            float t_plane = hit_floor(floor_y, r);
            if (t_plane > 0.0f && t_plane < t_min) {
                t_min = t_plane; hit_object = 2; hit_point = r.point_at_parameter(t_plane); normal = Vec3(0.0f, 1.0f, 0.0f);
                int ix = floorf(hit_point.x * 4.0f); int iz = floorf(hit_point.z * 4.0f);
                if (abs(ix + iz) % 2 == 0) base_color = Vec3(200.0f, 200.0f, 200.0f); else base_color = Vec3(80.0f, 80.0f, 80.0f);
            }

            float t_wall = hit_wall(wall_z, r);
            if (t_wall > 0.0f && t_wall < t_min) {
                t_min = t_wall; hit_object = 3; hit_point = r.point_at_parameter(t_wall);
                normal = r.direction.z < 0.0f ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(0.0f, 0.0f, -1.0f); base_color = Vec3(100.0f, 150.0f, 200.0f);
            }

            float t_sphere = hit_sphere(sphere_center, sphere_radius, r);
            if (t_sphere > 0.0f && t_sphere < t_min) {
                t_min = t_sphere; hit_object = 1; hit_point = r.point_at_parameter(t_sphere);
                normal = (hit_point - sphere_center).normalize(); base_color = Vec3(255.0f, 50.0f, 50.0f);
            }

            Vec3 pixel_color;
            if (hit_object > 0) {
                Vec3 light_dir = (light_pos - hit_point).normalize();
                float distance_to_light = (light_pos - hit_point).length();
                Ray shadow_ray(hit_point + normal * 0.001f, light_dir);
                bool in_shadow = false;
                if (hit_sphere(sphere_center, sphere_radius, shadow_ray) > 0.0f && hit_sphere(sphere_center, sphere_radius, shadow_ray) < distance_to_light) in_shadow = true;
                if (hit_wall(wall_z, shadow_ray) > 0.0f && hit_wall(wall_z, shadow_ray) < distance_to_light) in_shadow = true;

                float light_intensity = dot(normal, light_dir);
                if (light_intensity < 0.0f) light_intensity = 0.0f;
                if (in_shadow) light_intensity = 0.0f;
                pixel_color = base_color * (0.15f + (1.0f - 0.15f) * light_intensity);
            }
            else {
                if (hit_sphere(light_pos, 2.0f, r) > 0.0f) pixel_color = Vec3(255.0f, 255.0f, 220.0f);
                else { float t = 0.5f * (r.direction.y + 1.0f); pixel_color = Vec3(255.0f, 255.0f, 255.0f) * (1.0f - t) + Vec3(127.0f, 178.0f, 255.0f) * t; }
            }

            int pixel_index = (y * width + x) * 3;
            fb[pixel_index + 0] = (unsigned char)pixel_color.x;
            fb[pixel_index + 1] = (unsigned char)pixel_color.y;
            fb[pixel_index + 2] = (unsigned char)pixel_color.z;
        }
    }
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }