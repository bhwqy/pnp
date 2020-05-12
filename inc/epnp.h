#ifndef EPNP
#define EPNP
#include <vector>
#include <Eigen/Dense>
class EPNPSolver {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 3d points image points
    EPNPSolver(const Eigen::Matrix3d& K,
        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pts3d,
        const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& pts2d);
    EPNPSolver(const EPNPSolver&) = delete;
    EPNPSolver& operator=(const EPNPSolver&) = delete;
    void computePose(Eigen::Matrix3d& R, Eigen::Vector3d& t);  
private:

    void chooseControlPoints();
    void computeBarycentricCoordinates();
    void computeEigenVectors();
    void computeL();
    void computeRho();
    void solveN2(Eigen::Matrix3d& R, Eigen::Vector3d& t);
    void solveN3(Eigen::Matrix3d& R, Eigen::Vector3d& t);
    void solveN4(Eigen::Matrix3d& R, Eigen::Vector3d& t);
    void computeCameraControlPoints(const Eigen::Vector4d& betas);
    bool isGoodBetas();
    void optimizeBeta(Eigen::Vector4d& betas);
    void computeRt(
        const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pts3d_camera,
        Eigen::Matrix3d& R, Eigen::Vector3d& t);
    double reprojectionError(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    
    int n;
    double fx, fy, cx, cy;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts3d;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts2d;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> alphas;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pcs;
    Eigen::Vector3d cws[4], ccs[4]; // control points
    Eigen::Matrix<double, 12, 4> eigen_vectors;
    Eigen::Matrix<double, 6, 10> L;
    Eigen::Matrix<double, 6, 1> rho;
};

#endif