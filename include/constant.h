#pragma once

__constant__ constexpr int G2P = 2;
__constant__ constexpr int grid_bound_x = 130;
__constant__ constexpr int grid_bound_y = 50;
__constant__ constexpr int grid_bound_z = 100;

__constant__ constexpr float step = 0.0001f;
__constant__ constexpr float particle_diameter = 0.005f;
__constant__ constexpr float damping = 0.95f;
__constant__ constexpr float stickiness = 0.9f;
__constant__ constexpr float friction = 1.0f;
__constant__ constexpr float box_boundary_1 = 0.0f * particle_diameter;
__constant__ constexpr float box_boundary_2 = grid_bound_x * particle_diameter;

__constant__ static const float gravity[3] = { -6.9f, -6.48f, 0.0f };
