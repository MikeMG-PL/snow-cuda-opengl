#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <crt/host_defines.h>
#include <Eigen/Dense>

#include <thrust/pair.h>
#include <thrust/tuple.h>

#include "constant.h"

class Particle
{
public:
    float volume, mass;
    Eigen::Vector3f position, velocity;

    float hardening;
    float lambda, mu;
    float compression, stretch;

    Eigen::Matrix3f def_elastic, def_plastic;

    __host__ __device__ Particle() {}
    __host__ __device__ Particle(const Eigen::Vector3f&, const Eigen::Vector3f&, float, float, float, float, float, float);

    __host__ __device__ ~Particle() = default;

    __host__ friend std::ostream& operator<<(std::ostream&, const Particle&);

    __host__ __device__ void update_position();
    __host__ __device__ void update_velocity(const Eigen::Vector3f&, const Eigen::Vector3f&);
    __host__ __device__ void update_deformation_gradient(const Eigen::Matrix3f&);
    __host__ __device__ void apply_boundary_collision();
    __host__ __device__ const Eigen::Matrix3f energy_derivative() const;

private:
    __host__ __device__ const thrust::pair<float, float> compute_hardening() const;
};

#endif  // PARTICLE_H_
