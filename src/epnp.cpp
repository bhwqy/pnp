#include <algorithm>
#include <cmath>
#include "../inc/epnp.h"

EPNPSolver::EPNPSolver(const Eigen::Matrix3d& K,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pts3d,
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& pts2d
) {
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);
    n = std::min(pts2d.size(), pts3d.size());
    for (int i = 0; i < n; ++i) {
        // TODO see OpenCV
        this->pts3d.push_back(pts3d[i]);
        this->pts2d.push_back(pts2d[i]);
    }
}

void EPNPSolver::computePose(Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    chooseControlPoints();
    computeBarycentricCoordinates();

    // control points under camera frame
    computeEigenVectors();
    computeL();
    computeRho();

    Eigen::Matrix3d tmp_R;
    Eigen::Vector3d tmp_t;
    solveN2(tmp_R, tmp_t);
    double err2 = reprojectionError(tmp_R, tmp_t);
    R = tmp_R;
    t = tmp_t;

    double err3 = reprojectionError(tmp_R, tmp_t);
    if (err3 < err2) {
        R = tmp_R;
        t = tmp_t;
    }
    else
        err3 = err2;

    double err4 = reprojectionError(tmp_R, tmp_t);
    if (err4 < err3) {
        R = tmp_R;
        t = tmp_t;
    }
}

void EPNPSolver::chooseControlPoints() {
    using std::sqrt;
    // Take C0 as the reference points centroid:
    cws[0] << 0, 0, 0;
    for (int i = 0; i < n; i++)
        cws[0] += pts3d[i];
    cws[0] /= (double)n;

    // Take C1, C2, and C3 from PCA on the reference points:
    Eigen::MatrixXd A;
    A.resize(n, 3);

    for (int i = 0; i < n; i++)
        A.block<1, 3>(i, 0) = (pts3d.at(i) - cws[0]).transpose();

    Eigen::Matrix3d ATA = A.transpose() * A;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(ATA);
    Eigen::Vector3d D = es.eigenvalues();
    Eigen::MatrixXd V = es.eigenvectors();

    cws[1] = cws[0] + sqrt(D(0) / n) * V.block<3, 1>(0, 0);
    cws[2] = cws[0] + sqrt(D(1) / n) * V.block<3, 1>(0, 1);
    cws[3] = cws[0] + sqrt(D(2) / n) * V.block<3, 1>(0, 2);

}

void EPNPSolver::computeBarycentricCoordinates() {

    alphas.clear();
    alphas.resize(n);

    Eigen::Matrix4d C;
    for (int i = 0; i < 4; i++)
        C.block<3, 1>(0, i) = cws[i];
    C.block<1, 4>(3, 0) << 1.0, 1.0, 1.0, 1.0;
    Eigen::Matrix4d C_inv = C.inverse();

    // compute \alpha_ij for all points
    for (int i = 0; i < n; i++) {
        Eigen::Vector4d ptw;
        ptw << pts3d[i][0], pts3d[i][1], pts3d[i][2], 1.0;
        alphas[i] = C_inv * ptw;
    }  
}

void EPNPSolver::computeEigenVectors() {
    Eigen::MatrixXd M;
    M.resize(2 * n, 12);

    for (int i = 0; i < n; i++) {

        // get uv
        const double& u = pts2d.at(i) (0);
        const double& v = pts2d.at(i) (1);

        // idx
        const int id0 = 2 * i;
        const int id1 = id0 + 1;

        // the first line
        M(id0, 0) = alphas[i][0] * fx;
        M(id0, 1) = 0.0;
        M(id0, 2) = alphas[i][0] * (cx - u);

        M(id0, 3) = alphas[i][1] * fx;
        M(id0, 4) = 0.0;
        M(id0, 5) = alphas[i][1] * (cx - u);

        M(id0, 6) = alphas[i][2] * fx;
        M(id0, 7) = 0.0;
        M(id0, 8) = alphas[i][2] * (cx - u);

        M(id0, 9) = alphas[i][3] * fx;
        M(id0, 10) = 0.0;
        M(id0, 11) = alphas[i][3] * (cx - u);

        // for the second line
        M(id1, 0) = 0.0;
        M(id1, 1) = alphas[i][0] * fy;
        M(id1, 2) = alphas[i][0] * (cy - v);

        M(id1, 3) = 0.0;
        M(id1, 4) = alphas[i][1] * fy;
        M(id1, 5) = alphas[i][1] * (cy - v);

        M(id1, 6) = 0.0;
        M(id1, 7) = alphas[i][2] * fy;
        M(id1, 8) = alphas[i][2] * (cy - v);

        M(id1, 9) = 0.0;
        M(id1, 10) = alphas[i][3] * fy;
        M(id1, 11) = alphas[i][3] * (cy - v);
    }

    Eigen::Matrix<double, 12, 12> MTM = M.transpose() * M;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es(MTM);
    Eigen::MatrixXd e_vectors = es.eigenvectors();
    eigen_vectors = e_vectors.block<12, 4>(0, 0);
}

void EPNPSolver::computeL() {

    const int idx0[6]{ 0, 0, 0, 1, 1, 2 };
    const int idx1[6]{ 1, 2, 3, 2, 3, 3 };

    for (int i = 0; i < 6; i++) {
        const int idi = idx0[i] * 3;
        const int idj = idx1[i] * 3;

        // the first control point.
        const Eigen::Vector3d v1i = eigen_vectors.block<3, 1>(idi, 0);
        const Eigen::Vector3d v2i = eigen_vectors.block<3, 1>(idi, 1);
        const Eigen::Vector3d v3i = eigen_vectors.block<3, 1>(idi, 2);
        const Eigen::Vector3d v4i = eigen_vectors.block<3, 1>(idi, 3);

        // the second control point
        const Eigen::Vector3d v1j = eigen_vectors.block<3, 1>(idj, 0);
        const Eigen::Vector3d v2j = eigen_vectors.block<3, 1>(idj, 1);
        const Eigen::Vector3d v3j = eigen_vectors.block<3, 1>(idj, 2);
        const Eigen::Vector3d v4j = eigen_vectors.block<3, 1>(idj, 3);

        Eigen::Vector3d S1 = v1i - v1j;
        Eigen::Vector3d S2 = v2i - v2j;
        Eigen::Vector3d S3 = v3i - v3j;
        Eigen::Vector3d S4 = v4i - v4j;

        Eigen::Matrix<double, 1, 3> S1_T = S1.transpose();
        Eigen::Matrix<double, 1, 3> S2_T = S2.transpose();
        Eigen::Matrix<double, 1, 3> S3_T = S3.transpose();
        Eigen::Matrix<double, 1, 3> S4_T = S4.transpose();

        //[B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
        L(i, 0) = S1_T * S1;
        L(i, 1) = 2 * S1_T * S2;
        L(i, 2) = S2_T * S2;
        L(i, 3) = 2 * S1_T * S3;
        L(i, 4) = 2 * S2_T * S3;
        L(i, 5) = S3_T * S3;
        L(i, 6) = 2 * S1_T * S4;
        L(i, 7) = 2 * S2_T * S4;
        L(i, 8) = 2 * S3_T * S4;
        L(i, 9) = S4_T * S4;
    }
}

void EPNPSolver::computeRho() {
    const int idx0[6] = { 0, 0, 0, 1, 1, 2 };
    const int idx1[6] = { 1, 2, 3, 2, 3, 3 };
    for (int i = 0; i < 6; i++) {
        Eigen::Vector3d v01 = cws[idx0[i]] - cws[idx1[i]];
        rho(i, 0) = (v01.transpose() * v01);
    }
}

void EPNPSolver::solveN2(Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    const Eigen::Matrix<double, 6, 3>& L_approx = L.block(0, 0, 6, 3);
    Eigen::Vector3d b3 = L_approx.fullPivHouseholderQr().solve(rho);
    Eigen::Vector4d betas;
    if (b3[0] < 0) {
        betas[0] = std::sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? std::sqrt(-b3[2]) : 0.0;
    }
    else {
        betas[0] = std::sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? std::sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0)
        betas[0] = -betas[0];

    betas[2] = 0.0;
    betas[3] = 0.0;
    computeCameraControlPoints(betas);
    if (isGoodBetas() == false)
        betas = -betas;
    optimizeBeta(betas);
    computeCameraControlPoints(betas);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts3d_camera(n);
    for (int i = 0; i < n; i++) {
        pts3d_camera[i] =
            ccs[0] * alphas[i][0] + ccs[1] * alphas[i][1] +
            ccs[2] * alphas[i][2] + ccs[3] * alphas[i][3];
    }
    computeRt(pts3d_camera, R, t);
}

void EPNPSolver::solveN3(Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    const Eigen::Matrix<double, 6, 5>& L_approx = L.block(0, 0, 6, 5);
    Eigen::Matrix<double, 5, 1> b5 = L_approx.fullPivHouseholderQr().solve(rho);
    Eigen::Vector4d betas;
    if (b5[0] < 0) {
        betas[0] = sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    }
    else {
        betas[0] = sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0)
        betas[0] = -betas[0];
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;

    // Check betas.
    computeCameraControlPoints(betas);
    if (isGoodBetas() == false)
        betas = -betas;
    optimizeBeta(betas);
    computeCameraControlPoints(betas);

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts3d_camera(n);
    for (int i = 0; i < n; i++) {
        pts3d_camera[i] =
            ccs[0] * alphas[i][0] + ccs[1] * alphas[i][1] +
            ccs[2] * alphas[i][2] + ccs[3] * alphas[i][3];
    }
    computeRt(pts3d_camera, R, t);
}

void EPNPSolver::solveN4(Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    Eigen::Matrix<double, 6, 4> L_approx;
    L_approx.block(0, 0, 6, 2) = L.block(0, 0, 6, 2);
    L_approx.block(0, 2, 6, 1) = L.block(0, 3, 6, 1);
    L_approx.block(0, 3, 6, 1) = L.block(0, 6, 6, 1);
    Eigen::Vector4d b4 = L_approx.fullPivHouseholderQr().solve(rho);
    Eigen::Vector4d betas;
    if (b4[0] < 0) {
        betas[0] = sqrt(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    }
    else {
        betas[0] = sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
    computeCameraControlPoints(betas);
    if (isGoodBetas() == false)
        betas = -betas;
    optimizeBeta(betas);
    computeCameraControlPoints(betas);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts3d_camera(n);
    for (int i = 0; i < n; i++) {
        pts3d_camera[i] =
            ccs[0] * alphas[i][0] + ccs[1] * alphas[i][1] +
            ccs[2] * alphas[i][2] + ccs[3] * alphas[i][3];
    }
    computeRt(pts3d_camera, R, t);
}

void EPNPSolver::computeCameraControlPoints(const Eigen::Vector4d& betas) {
    Eigen::Matrix<double, 12, 1> vec =
        betas[0] * eigen_vectors.block<12, 1>(0, 0) +
        betas[1] * eigen_vectors.block<12, 1>(0, 1) +
        betas[2] * eigen_vectors.block<12, 1>(0, 2) +
        betas[3] * eigen_vectors.block<12, 1>(0, 3);

    for (int i = 0; i < 4; i++)
        ccs[i] = vec.block<3, 1>(i * 3, 0);
}

bool EPNPSolver::isGoodBetas() {
    int num_positive = 0;
    int num_negative = 0;
    for (int i = 0; i < 4; i++) {
        if (ccs[i][2] > 0)
            num_positive++;
        else
            num_negative++;
    }
    if (num_negative >= num_positive)
        return false;
    return true;
}

void EPNPSolver::optimizeBeta(Eigen::Vector4d& betas) {
    const int max_iter = 5;
    for (int nit = 0; nit < max_iter; nit++) {

        Eigen::Matrix<double, 6, 4> J;
        for (int i = 0; i < 6; i++) {
            // [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
            J(i, 0) = 2 * betas[0] * L(i, 0) + betas[1] * L(i, 1) + betas[2] * L(i, 3) + betas[3] * L(i, 6);
            J(i, 1) = betas[0] * L(i, 1) + 2 * betas[1] * L(i, 2) + betas[2] * L(i, 3) + betas[3] * L(i, 7);
            J(i, 2) = betas[0] * L(i, 3) + betas[1] * L(i, 4) + 2 * betas[2] * L(i, 5) + betas[3] * L(i, 8);
            J(i, 3) = betas[0] * L(i, 6) + betas[1] * L(i, 7) + betas[2] * L(i, 8) + 2 * betas[3] * L(i, 9);
        }

        Eigen::Matrix<double, 4, 6> J_T = J.transpose();
        Eigen::Matrix<double, 4, 4> H = J_T * J;

        // Compute residual
        // [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
        // [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
        Eigen::Matrix<double, 10, 1> bs;
        bs << betas[0] * betas[0], betas[0] * betas[1], betas[1] * betas[1], betas[0] * betas[2], betas[1] * betas[2],
            betas[2] * betas[2], betas[0] * betas[3], betas[1] * betas[3], betas[2] * betas[3], betas[3] * betas[3];
        Eigen::Matrix<double, 6, 1> residual = L * bs - rho;

        // Solve J^T * J \delta_beta = -J^T * residual;
        Eigen::Matrix<double, 4, 1> delta_betas = H.fullPivHouseholderQr().solve(-J_T * residual);

        // update betas;
        betas += delta_betas;
    }
}

void EPNPSolver::computeRt(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pts3d_camera,
    Eigen::Matrix3d& R, Eigen::Vector3d& t
) {
    // step 1. compute center points
    Eigen::Vector3d pcc(0.0, 0.0, 0.0);
    Eigen::Vector3d pcw(0.0, 0.0, 0.0);

    for (int i = 0; i < n; i++) {
        pcc += pts3d_camera[i];
        pcw += pts3d[i];
    }

    pcc /= (double)n;
    pcw /= (double)n;

    // step 2. remove centers.
    Eigen::MatrixXd Pc, Pw;
    Pc.resize(n, 3);
    Pw.resize(n, 3);

    for (int i = 0; i < n; i++) {
        Pc.block<1, 3>(i, 0) = (pts3d_camera[i] - pcc).transpose();
        Pw.block<1, 3>(i, 0) = (pts3d[i] - pcw).transpose();
    }

    // step 3. compute R.
    Eigen::Matrix3d W = Pc.transpose() * Pw;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    R = U * V.transpose();

    if (R.determinant() < 0) {
        R.block<1, 3>(2, 0) = -R.block<1, 3>(2, 0);
    }

    // step 3. compute t
    t = pcc - R * pcw;
}

double EPNPSolver::reprojectionError(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    double sum_err2 = 0.0;
    Eigen::Matrix3d K;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;
    for (size_t i = 0; i < n; i++) {
        const Eigen::Vector3d& ptw = pts3d[i];
        Eigen::Vector3d lamda_uv = K * (R * ptw + t);
        Eigen::Vector2d uv = lamda_uv.block(0, 0, 2, 1) / lamda_uv(2);
        Eigen::Vector2d e_uv = pts2d.at(i) - uv;
        sum_err2 += e_uv.transpose() * e_uv;
    }

    return sqrt(sum_err2 / (double)n);
}
