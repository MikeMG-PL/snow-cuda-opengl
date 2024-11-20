#pragma once


#include <vector>

#include <Eigen/Dense>

class PointCloud
{
public:
    std::vector<Eigen::Vector3f> positions;
    explicit PointCloud(const std::string& fname) : PointCloud(fname, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 1.0f) {}
    PointCloud(const std::string&, const Eigen::Vector3f&, const float);
};
