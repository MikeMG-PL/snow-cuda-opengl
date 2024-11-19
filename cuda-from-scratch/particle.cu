#include "particle.h"

#include <crt/host_defines.h>

#include "constant.h"
#include "algebra.h"

__host__ __device__ Particle::Particle(const Eigen::Vector3f& _position, const Eigen::Vector3f& _velocity, float _mass,
    float _hardening, float young, float poisson, float _compression, float _stretch)
    : mass(_mass), position(_position), velocity(_velocity),
    hardening(_hardening),
    compression(_compression), stretch(_stretch)
{

    lambda = (poisson * young) / ((1.0f + poisson) * (1.0f - 2.0f * poisson));
    mu = young / (2.0f * (1.0f + poisson));

    def_elastic.setIdentity();
    def_plastic.setIdentity();
}

__host__ std::ostream& operator<<(std::ostream& os, const Particle& p)
{
    unsigned short x = p.position(0) * 65535.0f / (grid_bound_x * particle_diameter),
        y = p.position(1) * 65535.0f / (grid_bound_y * particle_diameter),
        z = p.position(2) * 65535.0f / (grid_bound_z * particle_diameter);

    os.write(reinterpret_cast<char*>(&x), sizeof(unsigned short));
    os.write(reinterpret_cast<char*>(&y), sizeof(unsigned short));
    os.write(reinterpret_cast<char*>(&z), sizeof(unsigned short));

    return os;
}

__host__ __device__ void Particle::update_position()
{
    position += step * velocity;
}

__host__ __device__ void Particle::update_velocity(const Eigen::Vector3f& velocity_pic, const Eigen::Vector3f& velocity_flip)
{
    velocity = (1 - damping) * velocity_pic + damping * velocity_flip;
}

__host__ __device__ void Particle::update_deformation_gradient(const Eigen::Matrix3f& velocity_gradient)
{
    def_elastic = (Eigen::Matrix3f::Identity() + (step * velocity_gradient)) * def_elastic;

    Eigen::Matrix3f force_all(def_elastic * def_plastic);

    Eigen::Matrix3f u, s, v;

    algebra::svd_3x3(def_elastic, u, s, v);

    // clip values
    auto e = s.diagonal().array();
    e = e.min(1 + stretch).max(1 - compression);

    Eigen::Matrix3f u_tmp(u), v_tmp(v);
    u_tmp.array().rowwise() *= e.transpose();
    v_tmp.array().rowwise() /= e.transpose();

    def_plastic = v_tmp * u.transpose() * force_all;
    def_elastic = u_tmp * v.transpose();
}

__host__ __device__ const thrust::pair<float, float> Particle::compute_hardening() const
{
    float factor = expf(hardening * (1 - algebra::det(def_plastic)));
    return thrust::make_pair(mu * factor, lambda * factor);
}

__host__ __device__ const Eigen::Matrix3f Particle::energy_derivative() const
{
    Eigen::Matrix3f u, s, v;

    algebra::svd_3x3(def_elastic, u, s, v);

    float _mu, _lambda;
    thrust::tie(_mu, _lambda) = compute_hardening();
    float je = algebra::det(def_elastic);

    Eigen::Matrix3f tmp(2.0f * _mu * (def_elastic - u * v.transpose()) * def_elastic.transpose());

    tmp.diagonal().array() += (_lambda * je * (je - 1));

    return volume * tmp;
}
