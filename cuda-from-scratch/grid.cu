#include "constant.h"
#include "grid.h"

__host__ __device__ void Grid::reset()
{
    mass = 0.0f;
    force.setZero();
    velocity.setZero();
    velocity_star.setZero();
}

__host__ __device__ void Grid::update_velocity()
{
    if (mass > 0.0f) {
        Eigen::Vector3f const g = { gravity[0], gravity[1], gravity[2] };
        float const inv_mass = 1.0f / mass;
        force += (mass * g);
        velocity *= inv_mass;
        velocity_star = velocity + step * inv_mass * force;
    }
}
