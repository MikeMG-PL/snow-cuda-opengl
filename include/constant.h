#ifndef CONSTANT_H_
#define CONSTANT_H_

#include <Eigen/Dense>

#define TIMESTEP 1e-4f
#define GRAVITY Eigen::Vector3f(-6.9f, -6.48f, 0.0f)
#define PARTICLE_DIAM 0.005f

#define ALPHA 0.95f
#define STICKY_WALL 0.9f
#define FRICTION 1.0f

#define GRID_BOUND_X 130
#define GRID_BOUND_Y 50
#define GRID_BOUND_Z 100

#define BOX_BOUNDARY_1 (0.0 * PARTICLE_DIAM)
#define BOX_BOUNDARY_2 (GRID_BOUND_X * PARTICLE_DIAM)

#define G2P 2

#endif  // CONSTANT_H_
