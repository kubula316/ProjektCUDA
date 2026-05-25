#pragma once
#include <cuda_runtime.h>
#include <math.h>

// ==========================================
// 1. MATEMATYKA 
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

__host__ __device__ inline float dot(const Vec3& u, const Vec3& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
__host__ __device__ inline Vec3 cross(const Vec3& u, const Vec3& v) { return Vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x); }

struct Ray {
    Vec3 origin;
    Vec3 direction;
    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {}
    __host__ __device__ Vec3 point_at_parameter(float t) const { return origin + direction * t; }
};