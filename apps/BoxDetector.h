#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <unistd.h>
#include <config.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>


#include "Utils.h"
#include "ImgProc.h"
#include "PointCloudPlaneAligner.h"
#include "BoxRegistration.h"


using MyPoint = pcl::PointXYZI; // FIXME: pescato da un altro file... va spostato?

class BoxDetector {

public:
    float computeError(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& estimated,
        const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& groundTruth);

    std::vector<float> computePaiwiseDistanceSet(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& points);

    void findContoursAndDraw(cv::Mat& cdstP, cv::Mat& imageIntensities, std::vector<std::vector<cv::Point>>& contours,
        std::vector<cv::Vec4i>& hierarchy, cv::RNG& rng, cv::Mat& drawing, cv::Mat& drawingHull,
        pcl::visualization::PCLVisualizer::Ptr& viewer, pcl::PointCloud<MyPoint>::Ptr cloudAligned,
        std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& faceCentroids,
        std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& maf_centroids_preReg,
        std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& maf_centroids_postReg,
        std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& maf_faceCentroids_postReg,
        float longEdge, float shortEdge);

    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> loadGroundTruth(const std::string& filenameIn);

    void topPlaneExtraction(pcl::PointCloud<MyPoint>::Ptr cloudIn, pcl::PointCloud<MyPoint>::Ptr cloud1,
        pcl::PointCloud<MyPoint>::Ptr cloud2, pcl::PointCloud<MyPoint>::Ptr cloudNoGround,
        pcl::PointCloud<MyPoint>::Ptr cloudPlane, Eigen::VectorXf& planeCoeffs);

    void alignPointCloudWithPlane(pcl::PointCloud<MyPoint>::Ptr cloudIn, const Eigen::VectorXf& planeCoeffs,
        pcl::PointCloud<MyPoint>::Ptr cloudAligned);

    void buildImagesFromPointCloud(const pcl::PointCloud<MyPoint>::Ptr cloudIn, const pcl::PointCloud<MyPoint>::Ptr cloudPlane, const Eigen::VectorXf& planeCoeffs, cv::Mat& imageIntensities, cv::Mat& imageDepth, cv::Mat& imageIntensitiesPlane, cv::Mat& imageDepthPlane, cv::Mat& imageLambert, bool doLambert);

    void blurAndRemoveInvalidPixels(cv::Mat& inputImage, cv::Mat& outputImage, int filterSize, double sigmaColor, double sigmaSpace);

    void computeImageGradientAndOrientation(const cv::Mat& imageBlurred, cv::Mat& imageSobelX, cv::Mat& imageSobelY, const cv::Mat& imageNormalized, cv::Mat& imageGradient, cv::Mat& orientation);

    void applyColorMapAndThreshold(const cv::Mat& imageNormalized, const cv::Mat& orientation, cv::Mat& orientationColored);

private:

};
