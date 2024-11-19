#ifndef MPM_SOLVER_H_
#define MPM_SOLVER_H_

#include <vector>
#include <glad/glad.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <cuda_gl_interop.h>

#include "grid.h"
#include "constant.h"
#include "particle.h"

class MPMSolver
{
public:
    __host__ explicit MPMSolver(const std::vector<Particle>&);
    __host__ MPMSolver(const std::vector<Particle>&, const std::vector<Grid>&);
    __host__ ~MPMSolver() {}

    __host__ void reset_grid();
    __host__ void perform_initial_transfer();
    __host__ void transfer_data();
    __host__ void compute_volumes();
    __host__ void update_velocities();
    __host__ void body_collisions();
    __host__ void update_deformation_gradient();
    __host__ void update_particle_velocities();
    __host__ void particle_body_collisions();
    __host__ void update_particle_positions();

    __host__ void simulate();
    __host__ void bind_gl_buffer(const GLuint);
    __host__ void write_gl_buffer();
    __host__ void write_to_file(const std::string&);

private:
    thrust::device_vector<Particle> particles;
    thrust::device_vector<Grid> grids;
    struct cudaGraphicsResource* vbo_resource;
    bool initial_transfer = true;
};

#endif  // MPM_SOLVER_H_
