#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "vec3.cuh"
#include "physics.cuh"

#define MAX_SPHERES 256
__constant__ char c_spheres_data[MAX_SPHERES * sizeof(Sphere)];

// ==========================================
// GENERATOR LICZB LOSOWYCH
// ==========================================
__host__ __device__ inline float rand_float(unsigned int* seed) {
    *seed = (*seed ^ 61) ^ (*seed >> 16);
    *seed *= 9;
    *seed = *seed ^ (*seed >> 4);
    *seed *= 0x27d4eb2d;
    *seed = *seed ^ (*seed >> 15);
    return (*seed) / 4294967295.0f;
}

// ==========================================
// KERNEL CUDA 
// ==========================================
__global__ void render(unsigned char* fb, int width, int height, float time,
    Vec3 cam_pos, Vec3 cam_forward, Vec3 cam_right, Vec3 cam_up,
    int num_spheres, int shadow_samples) { // <--- DODANY PARAMETR

    const Sphere* c_spheres = (const Sphere*)c_spheres_data;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    float u = (float(x) / float(width)) * 4.0f - 2.0f;
    float v = (float(height - y) / float(height)) * 3.0f - 1.5f;

    float focal_length = 2.0f;
    Vec3 direction = (cam_right * u) + (cam_up * v) + (cam_forward * focal_length);
    Ray r(cam_pos, direction.normalize());

    float floor_y = -0.5f;
    float orbit_radius = 15.0f;
    float orbit_speed = 0.8f;
    Vec3 light_pos(sin(time * orbit_speed) * orbit_radius, 8.0f, cos(time * orbit_speed) * orbit_radius);

    float t_min = 99999.0f;
    float closest_t = t_min;
    int hit_index = -1;
    Vec3 hit_point, normal, base_color;

    float t_plane = hit_floor(floor_y, r);
    if (t_plane > 0.0f && t_plane < closest_t) {
        closest_t = t_plane;
        hit_index = -2;
        hit_point = r.point_at_parameter(t_plane);
        normal = Vec3(0.0f, 1.0f, 0.0f);
        int ix = floorf(hit_point.x * 4.0f);
        int iz = floorf(hit_point.z * 4.0f);
        if (abs(ix + iz) % 2 == 0) base_color = Vec3(200.0f, 200.0f, 200.0f); else base_color = Vec3(80.0f, 80.0f, 80.0f);
    }

    for (int i = 0; i < num_spheres; i++) {
        Sphere s = c_spheres[i];
        if (i >= 3) s.center.y += sin(time * 2.0f + s.center.x * 2.0f) * 0.4f;

        float t;
        if (hit_sphere_obj(s, r, 0.001f, closest_t, t)) {
            closest_t = t;
            hit_index = i;
            hit_point = r.point_at_parameter(t);
            normal = (hit_point - s.center).normalize();
            base_color = s.mat.color;
        }
    }

    Vec3 pixel_color;
    if (hit_index != -1) {
        float light_radius = 2.0f;
        float total_light_intensity = 0.0f;
        unsigned int seed = (unsigned int)(x + y * width);
        float inv_samples = 1.0f / (float)shadow_samples;

        for (int i = 0; i < shadow_samples; i++) {
            float rx = rand_float(&seed) * 2.0f - 1.0f;
            float rz = rand_float(&seed) * 2.0f - 1.0f;

            Vec3 sample_light_pos = light_pos + Vec3(rx, 0.0f, rz) * light_radius;
            Vec3 light_vec = sample_light_pos - hit_point;
            float intensity = dot(normal, light_vec);

            if (intensity <= 0.0f) continue;

            float distance_to_light = light_vec.length();
            float inv_dist = 1.0f / distance_to_light;
            Vec3 light_dir = light_vec * inv_dist;
            intensity *= inv_dist;

            Ray shadow_ray(hit_point + normal * 0.001f, light_dir);
            bool in_shadow = false;

            for (int j = 0; j < num_spheres; j++) {
                Sphere s = c_spheres[j];
                if (j >= 3) s.center.y += sin(time * 2.0f + s.center.x * 2.0f) * 0.4f;

                float t_shadow;
                if (hit_sphere_obj(s, shadow_ray, 0.001f, distance_to_light, t_shadow)) {
                    in_shadow = true;
                    break;
                }
            }

            if (!in_shadow) {
                total_light_intensity += intensity;
            }
        }

        float light_intensity = total_light_intensity * inv_samples;
        float ambient = 0.15f;
        float brightness = ambient + (1.0f - ambient) * light_intensity;
        pixel_color = base_color * brightness;
    }
    else {
        float temp_t;
        Sphere light_sphere = { light_pos, 2.0f, {Vec3(0,0,0)} };
        if (hit_sphere_obj(light_sphere, r, 0.001f, 9999.0f, temp_t)) {
            pixel_color = Vec3(255.0f, 255.0f, 220.0f);
        }
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
    Vec3 cam_pos, Vec3 cam_forward, Vec3 cam_right, Vec3 cam_up,
    Sphere* spheres, int num_spheres, int shadow_samples) { // <--- DODANY PARAMETR
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = (float(x) / float(width)) * 4.0f - 2.0f;
            float v = (float(height - y) / float(height)) * 3.0f - 1.5f;

            float focal_length = 2.0f;
            Vec3 direction = (cam_right * u) + (cam_up * v) + (cam_forward * focal_length);
            Ray r(cam_pos, direction.normalize());

            float floor_y = -0.5f;
            float orbit_radius = 15.0f;
            float orbit_speed = 0.8f;
            Vec3 light_pos(sin(time * orbit_speed) * orbit_radius, 8.0f, cos(time * orbit_speed) * orbit_radius);

            float t_min = 99999.0f;
            float closest_t = t_min;
            int hit_index = -1;
            Vec3 hit_point, normal, base_color;

            float t_plane = hit_floor(floor_y, r);
            if (t_plane > 0.0f && t_plane < closest_t) {
                closest_t = t_plane; hit_index = -2;
                hit_point = r.point_at_parameter(t_plane);
                normal = Vec3(0.0f, 1.0f, 0.0f);
                int ix = floorf(hit_point.x * 4.0f); int iz = floorf(hit_point.z * 4.0f);
                if (abs(ix + iz) % 2 == 0) base_color = Vec3(200.0f, 200.0f, 200.0f); else base_color = Vec3(80.0f, 80.0f, 80.0f);
            }

            for (int i = 0; i < num_spheres; i++) {
                Sphere s = spheres[i];
                if (i >= 3) s.center.y += sin(time * 2.0f + s.center.x * 2.0f) * 0.4f;
                float t;
                if (hit_sphere_obj(s, r, 0.001f, closest_t, t)) {
                    closest_t = t;
                    hit_index = i;
                    hit_point = r.point_at_parameter(t);
                    normal = (hit_point - s.center).normalize();
                    base_color = s.mat.color;
                }
            }

            Vec3 pixel_color;
            if (hit_index != -1) {
                float total_light_intensity = 0.0f;
                unsigned int seed = (unsigned int)(x + y * width);

                for (int i = 0; i < shadow_samples; i++) {
                    Vec3 sample_light_pos = light_pos + Vec3((rand_float(&seed) * 2 - 1) * 2.0f, 0, (rand_float(&seed) * 2 - 1) * 2.0f);
                    Vec3 light_vec = sample_light_pos - hit_point;
                    float intensity = dot(normal, light_vec);
                    if (intensity > 0.0f) {
                        float dist = light_vec.length();
                        intensity *= (1.0f / dist);
                        Ray shadow_ray(hit_point + normal * 0.001f, light_vec * (1.0f / dist));
                        bool in_shadow = false;
                        for (int j = 0; j < num_spheres; j++) {
                            Sphere s = spheres[j];
                            if (j >= 3) s.center.y += sin(time * 2.0f + s.center.x * 2.0f) * 0.4f;
                            float t_sh;
                            if (hit_sphere_obj(s, shadow_ray, 0.001f, dist, t_sh)) { in_shadow = true; break; }
                        }
                        if (!in_shadow) total_light_intensity += intensity;
                    }
                }
                float light_intensity = total_light_intensity / (float)shadow_samples;
                pixel_color = base_color * (0.15f + (1.0f - 0.15f) * light_intensity);
            }
            else {
                float temp_t;
                Sphere light_sphere = { light_pos, 2.0f, {Vec3(0,0,0)} };
                if (hit_sphere_obj(light_sphere, r, 0.001f, 9999.0f, temp_t)) {
                    pixel_color = Vec3(255.0f, 255.0f, 220.0f);
                }
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
    }
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }