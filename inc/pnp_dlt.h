#ifndef PNP_DLT
#define PNP_DLT
#include <vector>
#include <Eigen/Dense>
void solvePnPbyDLT(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pts3d,
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& pts2d,
    const Eigen::Matrix3d& K, Eigen::Matrix3d& R, Eigen::Vector3d& t);
#endif
