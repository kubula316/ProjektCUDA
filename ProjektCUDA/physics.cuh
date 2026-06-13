#pragma once
#include "vec3.cuh"

// ==========================================
// STRUKTURY DANYCH OBIEKTÓW
// ==========================================
struct Material {
    Vec3 color;
};

struct Sphere {
    Vec3 center;
    float radius;
    Material mat;
};

// ==========================================
// FIZYKA ZDERZEÑ
// ==========================================
__host__ __device__ inline bool hit_sphere_obj(const Sphere& s, const Ray& r, float t_min, float t_max, float& out_t) {
    Vec3 oc = r.origin - s.center;
    float a = dot(r.direction, r.direction);
    float b = 2.0f * dot(oc, r.direction);
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant > 0.0f) {
        float temp = (-b - sqrt(discriminant)) / (2.0f * a);
        if (temp < t_max && temp > t_min) {
            out_t = temp;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / (2.0f * a);
        if (temp < t_max && temp > t_min) {
            out_t = temp;
            return true;
        }
    }
    return false;
}

__host__ __device__ inline float hit_floor(float floor_y, const Ray& r) {
    if (abs(r.direction.y) < 0.0001f) return -1.0f;
    float t = (floor_y - r.origin.y) / r.direction.y;
    return t > 0.0f ? t : -1.0f;
}