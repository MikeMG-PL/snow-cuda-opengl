#pragma once

#include <crt/host_defines.h>
#include <Eigen/Dense>

class Grid {
public:
    Eigen::Vector3i idx;
    Eigen::Vector3f force, velocity, velocity_star;
    float mass;

    __host__ __device__ Grid() {}
    __host__ __device__ Grid(const Eigen::Vector3i _idx)
        : idx(_idx), mass(0.0f), force(0.0f, 0.0f, 0.0f), velocity(0.0f, 0.0f, 0.0f), velocity_star(0.0f, 0.0f, 0.0f)
    {}
    __host__ __device__ virtual ~Grid() {}

    __host__ __device__ void reset();
    __host__ __device__ void update_velocity();
};