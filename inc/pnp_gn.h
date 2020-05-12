#ifndef PNP_GN
#define PNP_GN
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
void pnpGaussNewton(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points3d,
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& points2d,
    const Eigen::Matrix3d& K, Eigen::Matrix3d& R, Eigen::Vector3d& t
);
#endif