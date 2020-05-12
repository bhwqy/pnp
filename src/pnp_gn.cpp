#include <inc/pnp_gn.h>
#include <iostream>
#include <cmath>
#include <Eigen/Geometry>
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

// compute so3 exponential map
Eigen::Matrix3d exp(const Eigen::Vector3d& a) {
    double theta = a.norm();
    Eigen::Vector3d b;
    b = a / theta;
    Eigen::Matrix3d bup;
    bup << 0, -b(2, 0), b(1, 0),
        b(2, 0), 0, -b(0, 0),
        -b(1, 0), b(0, 0), 0;

    Eigen::Matrix3d R;
    R = std::cos(theta) * Eigen::Matrix3d::Identity() 
        + (1 - std::cos(theta)) * b * b.transpose() 
        + std::sin(theta) * bup;
    return R;
}

// compute se3 exponential map
Eigen::Matrix4d exp(const Vector6d& a) {
    Eigen::Vector3d rho, phi, b;
    rho = a.block<3, 1>(0, 0);
    phi = a.block<3, 1>(3, 0);
    Eigen::Matrix3d J;
    
    double theta = phi.norm();
    b = phi / theta;
    Eigen::Matrix3d bup;
    bup << 0, -b(2, 0), b(1, 0),
        b(2, 0), 0, -b(0, 0),
        -b(1, 0), b(0, 0), 0;

    J = std::sin(theta) / theta * Eigen::Matrix3d::Identity()
        + (1 - std::sin(theta) / theta) * b * b.transpose()
        + (1 - std::cos(theta)) / theta * bup;
    
    Eigen::Matrix4d T;
    Eigen::Matrix3d R = exp(phi);
    Eigen::Vector3d t = J * rho;
    T << R(0, 0), R(0, 1), R(0, 2), t(0, 0),
        R(1, 0), R(1, 1), R(1, 2), t(1, 0),
        R(2, 0), R(2, 1), R(2, 2), t(2, 0),
        0, 0, 0, 1;
    return T;
}

void pnpGaussNewton(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& points3d,
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& points2d,
    const Eigen::Matrix3d& K, Eigen::Matrix3d& R, Eigen::Vector3d& t
) {
    const int iterations = 10;
    double cost = 0, lastcost = 0;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    for (int iter = 0; iter < iterations; ++iter) {
        Matrix6d H = Matrix6d::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < points3d.size(); ++i) {
            Eigen::Vector3d pc = R * points3d[i] + t;
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
            Eigen::Vector2d e = points2d[i] - proj;
            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }
        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (std::isnan(dx[0])) {
            std::cout << "result is nan!\n";
            break;
        }

        if (iter > 0 && cost >= lastcost) {
            std::cout << "cost: " << cost << ", lastcost: " << lastcost << "\n";
            break;
        }

        Eigen::Matrix4d T;
        T << R(0, 0), R(0, 1), R(0, 2), t(0, 0),
            R(1, 0), R(1, 1), R(1, 2), t(1, 0),
            R(2, 0), R(2, 1), R(2, 2), t(2, 0),
            0, 0, 0, 1;
        T = exp(dx) * T;
        R = T.block<3, 3>(0, 0);
        t = T.block<3, 1>(0, 3);

        lastcost = cost;
        std::cout << "iteration " << iter << " cost: " << cost << "\n";
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }
    std::cout << "R:\n" << R << "\nt:\n" << t << "\n";
}
