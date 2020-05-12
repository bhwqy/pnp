#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "inc/pnp_gn.h"
#include "inc/pnp_dlt.h"
#include "inc/epnp.h"

void find_feature_matches(
    const cv::Mat& img_1, const cv::Mat& img_2,
    std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2,
    std::vector<cv::DMatch>& matches
);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat& K);

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "usage: pnp img1 img2 depth1 depth2\n";
        return 1;
    }
    //-- 读取图像
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (img_1.empty() || img_2.empty())
        std::cout << "Can not load images!\n";

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "find " << matches.size() << " points in total!\n";

    // 建立3D点
    cv::Mat d1 = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (cv::DMatch m : matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)   // bad depth
            continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    std::cout << "3d-2d pairs: " << pts_3d.size() << "\n";
    {
        cv::Mat r, t;
        cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false);
        cv::Mat R;
        cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
        std::cout << "solve pnp in opencv:\nR =\n" << R << "\nt =\n" << t << "\n";
    }
    Eigen::Matrix3d K_eigen;
    K_eigen << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pts_3d_eigen;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    {        
        std::cout << "calling pnp by gauss newton\n";
        Eigen::Matrix3d R_eigen = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_eigen = Eigen::Vector3d::Zero();
        pnpGaussNewton(pts_3d_eigen, pts_2d_eigen, K_eigen, R_eigen, t_eigen);
        std::cout << "solve pnp by gauss newton:\nR =\n" << R_eigen << "\nt =\n" << t_eigen << "\n";
    }
    {
        std::cout << "calling pnp by dlt\n";
        Eigen::Matrix3d R_eigen = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_eigen = Eigen::Vector3d::Zero();
        solvePnPbyDLT(pts_3d_eigen, pts_2d_eigen, K_eigen, R_eigen, t_eigen);
        std::cout << "solve pnp by dlt:\nR =\n" << R_eigen << "\nt =\n" << t_eigen << "\n";
    }
    {
        std::cout << "calling epnp\n";
        Eigen::Matrix3d R_eigen = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_eigen = Eigen::Vector3d::Zero();
        EPNPSolver epnpsolver(K_eigen, pts_3d_eigen, pts_2d_eigen);
        epnpsolver.computePose(R_eigen, t_eigen);
        std::cout << "solve epnp:\nR =\n" << R_eigen << "\nt =\n" << t_eigen << "\n";
    }
    return 0;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches
) {
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K) {
    return cv::Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}
