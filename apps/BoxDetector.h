#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "BoxRegistration.h"

using MyPoint = pcl::PointXYZI; // FIXME: pescato da un altro file... va spostato?

class BoxDetector {

public:
float computeError(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &estimated, 
                   const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &groundTruth);

std::vector<float> computePaiwiseDistanceSet(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &points);

void findContoursAndDraw(cv::Mat &cdstP, cv::Mat &imageIntensities, std::vector<std::vector<cv::Point>> &contours, 
    std::vector<cv::Vec4i> &hierarchy,cv::RNG &rng, cv::Mat &drawing, cv::Mat &drawingHull, 
    pcl::visualization::PCLVisualizer::Ptr &viewer,pcl::PointCloud<MyPoint>::Ptr cloudAligned,
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &faceCentroids,
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &maf_centroids_preReg,
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &maf_centroids_postReg,
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &maf_faceCentroids_postReg,
    float longEdge, float shortEdge);

std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> loadGroundTruth(const std::string& filenameIn);

                            
private:

};
