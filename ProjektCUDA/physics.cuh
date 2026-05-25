#pragma once
#include "vec3.cuh"

// ==========================================
// 2. FIZYKA ZDERZEÑ (Architektura Pokoju)
// ==========================================
__host__ __device__ inline float hit_sphere(const Vec3& center, float radius, const Ray& r) {
    Vec3 oc = r.origin - center;
    float a = dot(r.direction, r.direction);
    float b = 2.0f * dot(oc, r.direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return -1.0f;
    else return (-b - sqrt(discriminant)) / (2.0f * a);
}

__host__ __device__ inline float hit_floor(float floor_y, const Ray& r) {
    if (abs(r.direction.y) < 0.0001f) return -1.0f;
    float t = (floor_y - r.origin.y) / r.direction.y;
    return t > 0.0f ? t : -1.0f;
}

__host__ __device__ inline float hit_wall(float wall_z, const Ray& r) {
    if (abs(r.direction.z) < 0.0001f) return -1.0f;
    float t = (wall_z - r.origin.z) / r.direction.z;
    if (t <= 0.0f) return -1.0f;

    Vec3 p = r.point_at_parameter(t);
    if (p.x > -1.5f && p.x < 1.5f && p.y < 2.5f) {
        return -1.0f;
    }
    return t;
}