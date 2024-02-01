/**
 * Pallet box detector using tof camera
 * @author Francesco Patander
 *
 *
 * distancePlaneThreshold
 * - dataset progetto coorsa (imballi con nastro adesivo): 0.05
 * - dataset acquisito in data 17-02-2022:
 *    - bianche: 0.14
 *
 * Per motivi di scadenze il filtro sui convex hull sulla dimensione delle scatole viene fatta in pixel immagine e non in coordinate mondo.
 * area pixel scatole convex hull:
 * - progetto coorsa e scatole bianche: 1000-6000 pixel
 * - scatole "fragile":
 * - scatole "rosse":
 */

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <rofl/geometry/hough_plane_detector.h>
#include <rofl/geometry/correspondence_graph_association.h>
#include <rofl/common/param_map.h>
#include <config.h>
#include "Utils.h"
#include "ImgProc.h"
#include "PointCloudPlaneAligner.h"
#include "BoxRegistration.h"
#include "BoxDetector.h"
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>


struct ArgumentList {
  std::string config;
  std::string pointCloud;
  std::string boxCenter;
};

bool ParseInputs(ArgumentList& args, int argc, char** argv);

using namespace rofl;

const int slider_max = 200;

/**
 * Gestione trackbar Hough Probabilistic Transform
 */
int houghP_threshold_value = 11; // 15
int houghP_minLL_value = 1;
int houghP_maxLG_value = 15;
float houghP_threshold = (float)houghP_threshold_value / slider_max;
float houghP_minLL = (float)houghP_minLL_value / slider_max;
float houghP_maxLG = (float)houghP_maxLG_value / slider_max;

static void on_houghP_threshold_trackbar(int, void*)
{
  houghP_threshold = (float)houghP_threshold_value / slider_max;
  //   std::cout<<"houghP_threshold_value: "<<houghP_threshold<<std::endl;
}

static void on_houghP_minLL_trackbar(int, void*)
{
  houghP_minLL = (float)houghP_minLL_value / slider_max;
  //   std::cout<<"houghP_minLL: "<<houghP_minLL<<std::endl;
}

static void on_houghP_maxLG_trackbar(int, void*)
{
  houghP_maxLG = (float)houghP_maxLG_value / slider_max;
  //   std::cout<<"houghP_maxLG: "<<houghP_maxLG<<std::endl;
}

float norm2(const Eigen::Vector2f& p1, const Eigen::Vector2f& p2)
{
  return (p1 - p2).norm();
}

void lineGradientFilter(const pcl::PointCloud<MyPoint>& cloud,
  const cv::Mat& in, const cv::Mat& mask,
  cv::Mat& gradientX, cv::Mat& gradientY,
  cv::Mat& magnitude, cv::Mat& op,
  float threshold = 0.1)
{

  //   const float distanceTh = 0.06;
  const float distanceTh = 0.08;
  const float distanceTh2 = distanceTh * distanceTh;

  cv::Mat oppositeMat = cv::Mat::zeros(in.rows, in.cols, CV_8UC1);
  cv::Mat scotchMat = cv::Mat::zeros(in.rows, in.cols, CV_8UC1);
  for (int r = 0; r < magnitude.rows; ++r) {
    for (int c = 0; c < magnitude.cols; ++c) {
      if (c == 123 && r == 33) {
        std::cout << " CHECK [" << c << " " << r << "]   intensity=" << in.at<float>(r, c)
          << " sgMask=" << (mask.at<float>(r, c) * 256)
          << std::endl;
      }

      if (in.at<float>(r, c) < 1) {
        continue;
      }

      const MyPoint& p0 = cloud.at(r * cloud.width + c);
      //       if(std::isnan(p0.intensity)) { continue; }

      if (mask.at<float>(r, c) < 0.25 / 256) {
        //         magnitude.at<float>(r,c)=0;
        continue;
      }

      const Eigen::Vector3f p0coord(p0.x, p0.y, p0.z);
      float gX = gradientX.at<float>(r, c);
      float gY = gradientY.at<float>(r, c);
      const float gN = magnitude.at<float>(r, c);

      //       if (gN < threshold) {
      // //         std::cout << "  A " << r << "," << c << " -> 0 (under threshold)" << std::endl;
      //         magnitude.at<float>(r,c) = 0;
      //         continue;
      //       }

      int maxLen = 20;
      int dx = std::ceil(maxLen * gX / gN);
      int dy = std::ceil(maxLen * gY / gN);

      cv::LineIterator it(magnitude, cv::Point(c, r), cv::Point(c - dx, r - dy));
      cv::Point bestOpposite = it.pos();
      float bestOppositeCost = gN * gN;

      bool greatestGradient = true;
      bool intoDarkness = false;
      for (int k = 0; k < it.count; ++k, ++it) {
        if (in.at<float>(it.pos()) < 1) {
          intoDarkness = true;
        }
        const MyPoint& p1 = cloud.at(it.pos().y * cloud.width + it.pos().x);
        //         if (std::isnan(p1.intensity)) { intoDarkness=true; break; }
        const Eigen::Vector3f p1coord(p1.x, p1.y, p1.z);
        float distance2 = (p0coord - p1coord).squaredNorm();
        float dgN = magnitude.at<float>(it.pos()) - gN;
        if (dgN > 0) {
          greatestGradient = false;
        }
        float gXpt = gradientX.at<float>(it.pos());
        float gYpt = gradientY.at<float>(it.pos());
        //         if(mask.at<float>(it.pos()) < 0.25/256) { gXpt = gYpt = 0;}
        float dgX = gX + gXpt;
        float dgY = gY + gYpt;
        float oppositeCost = dgX * dgX + dgY * dgY;

        if (c == 146 && r == 122) {
          std::cout << " " << std::setw(3) << k << "] " << it.pos()
            << " " << std::fixed << std::setprecision(3) << std::sqrt(distance2)
            << " " << std::sqrt(oppositeCost)
            << " " << ((oppositeCost < bestOppositeCost) ? "*" : "")
            << " " << (intoDarkness ? "[intoDarkness]" : "")
            << " " << ((distance2 > distanceTh2) ? "[DistanceReached]" : "")
            << std::endl;
        }
        if (intoDarkness) {
          break;
        }
        if (distance2 > distanceTh2) {
          break;
        }

        if (oppositeCost < bestOppositeCost) {
          bestOppositeCost = oppositeCost;
          bestOpposite = it.pos();
        }

        //         if (c == 247 && (r == 141 || r == 140)) {
        //         if (c == 273 && r == 146) {
        //         if (c == 225 && r == 67) {
      }

      if ((!intoDarkness) || (!greatestGradient)) {
        //         magnitude.at<float>(r,c)=0;
      }

      int nn = cv::norm(bestOpposite - cv::Point(c, r));
      //       if (bestOppositeCost < 0.05) {
      //         oppositeMat.at<unsigned char>(r,c) = 254-nn*254/maxLen;
      //       } else {
      //         oppositeMat.at<unsigned char>(r,c) = 255;
      //       }

      if (nn > 5 && bestOppositeCost < 0.05 && !intoDarkness) {
        cv::LineIterator it(magnitude, cv::Point(c, r), bestOpposite);
        for (int k = 0; k < it.count; ++k, ++it) {
          const unsigned char v = 255; // 255-10*k;
          scotchMat.at<unsigned char>(it.pos()) = std::max(v, scotchMat.at<unsigned char>(it.pos()));
        }
      }

      if (intoDarkness) {
        oppositeMat.at<unsigned char>(r, c) = 255;
      } else if (bestOppositeCost < 0.05) {
        oppositeMat.at<unsigned char>(r, c) = 253 - nn * 253 / maxLen;
      } else {
        oppositeMat.at<unsigned char>(r, c) = 254;
      }

      //       for(int k=0; k<it.count; ++k, ++it) {
      //         float i = in.at<float>(r,c);
      //         if (i < 1) { break; }
      //         float dgX = gX+gradientX.at<float>(it.pos());
      //         float dgY = gY+gradientY.at<float>(it.pos());
      //
      // //         std::cout << "   " << (dgX*dgX+dgY*dgY) << std::endl;
      // //         if(dgX*dgX+dgY*dgY < 0.0625) {
      // //           if(k < 4) break;
      // //
      // // //         if(dgX*dgX+dgY*dgY < 0.1) {
      // // //           std::cout << "  B " << r << "," << c << " -> 0 (under threshold)" << std::endl;
      // // //           std::cout << "  C " << it.pos().y << "," << it.pos().x << " -> 0 (under threshold)" << std::endl;
      // //           magnitude.at<float>(r,c) = 0;
      // //           magnitude.at<float>(it.pos()) = 0;
      // //
      // // //           magnitude.at<float>(r,c) = 1000*cv::norm(cv::Point(c,r)-it.pos());
      // // //           magnitude.at<float>(it.pos()) = 1000*cv::norm(cv::Point(c,r)-it.pos());
      // //           break;
      // //         }
      //           magnitude.at<float>(r,c) = 0;
      //           magnitude.at<float>(it.pos()) = 0;
      //       }
    }
  }

  cv::Mat morphTmp;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(0, 0));
  cv::erode(scotchMat, morphTmp, element);
  cv::dilate(morphTmp, scotchMat, element);
  cv::dilate(scotchMat, morphTmp, element);
  cv::erode(morphTmp, scotchMat, element);

  //   oppositeMat.copyTo(op);
  op = oppositeMat.mul(~scotchMat, 1.0 / 255);

  cv::namedWindow("scotch", cv::WINDOW_NORMAL);
  cv::imshow("scotch", scotchMat);
  cv::namedWindow("opposite1", cv::WINDOW_NORMAL);
  cv::imshow("opposite1", oppositeMat);
}

void setupWindows()
{
  cv::namedWindow("imageDepth", cv::WINDOW_NORMAL);
  cv::moveWindow("imageDepth", 0, 0);
  cv::namedWindow("imageDepthPlane", cv::WINDOW_NORMAL);
  cv::moveWindow("imageDepthPlane", 0, 0);
  cv::namedWindow("imageIntensities", cv::WINDOW_NORMAL);
  cv::moveWindow("imageIntensities", 400, 0);
  cv::namedWindow("imageIntensitiesPlane", cv::WINDOW_NORMAL);
  cv::moveWindow("imageIntensitiesPlane", 400, 0);
  cv::namedWindow("imageLambert", cv::WINDOW_NORMAL);
  cv::moveWindow("imageLambert", 800, 0);
  cv::namedWindow("imageNormalized", cv::WINDOW_NORMAL);
  cv::moveWindow("imageNormalized", 1200, 0);
  cv::namedWindow("imageBlurred", cv::WINDOW_NORMAL);
  cv::moveWindow("imageBlurred", 1600, 0);
  cv::namedWindow("imageGradient", cv::WINDOW_NORMAL);
  cv::moveWindow("imageGradient", 0, 350);
  cv::namedWindow("imageGradientSuppressed", cv::WINDOW_NORMAL);
  cv::moveWindow("imageGradientSuppressed", 400, 350);
  cv::namedWindow("imageOrientation", cv::WINDOW_NORMAL);
  cv::moveWindow("imageOrientation", 0, 675);
  cv::namedWindow("opposite", cv::WINDOW_NORMAL);
  cv::moveWindow("opposite", 800, 350);
  cv::namedWindow("oppositeTh", cv::WINDOW_NORMAL);
  cv::moveWindow("oppositeTh", 1200, 350);

  cv::namedWindow("Probabilistic Line Transform", cv::WINDOW_NORMAL);
  cv::moveWindow("Probabilistic Line Transform", 400, 675);
  cv::createTrackbar("houghP_threshold_value", "Probabilistic Line Transform",
    &houghP_threshold_value, slider_max, on_houghP_threshold_trackbar);
  cv::createTrackbar("houghP_minLineLength_value", "Probabilistic Line Transform",
    &houghP_minLL_value, slider_max, on_houghP_minLL_trackbar);
  cv::createTrackbar("houghP_maxLineGap_value", "Probabilistic Line Transform",
    &houghP_maxLG_value, slider_max, on_houghP_maxLG_trackbar);
  on_houghP_threshold_trackbar(houghP_threshold_value, 0);
  on_houghP_minLL_trackbar(houghP_minLL_value, 0);
  on_houghP_maxLG_trackbar(houghP_maxLG_value, 0);

  cv::namedWindow("findContours", cv::WINDOW_NORMAL);
  cv::moveWindow("findContours", 800, 675);
  cv::namedWindow("approximation", cv::WINDOW_NORMAL);
  cv::moveWindow("approximation", 800, 675);
  cv::namedWindow("hull", cv::WINDOW_NORMAL);
  cv::moveWindow("hull", 800, 675);
}

void plotPlane(const rofl::HoughPlaneDetector::PlaneParam& params, const pcl::PointCloud<MyPoint>& cloud, float distMax, pcl::ModelCoefficients& coefficients, float& x, float& y,
  float& z, int& n)
{
  coefficients.values.resize(4);
  coefficients.values[0] = params(0);
  coefficients.values[1] = params(1);
  coefficients.values[2] = params(2);
  coefficients.values[3] = params(3);

  x = 0.0f;
  y = 0.0f;
  z = 0.0f;
  n = 0;
  for (auto& p : cloud.points) {
    float dist = fabs(p.x * params(0) + p.y * params(1) + p.z * params(2) + params(3));
    if (dist < distMax) {
      x += p.x;
      y += p.y;
      z += p.z;
      n++;
    }
  }
  if (n > 0) {
    x = x / n;
    y = y / n;
    z = z / n;
  }
}

float evaluateAssociationDistance(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& input,
  const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& gt)
{

  rofl::CorrespondenceGraphAssociation corr;
  corr.setDistanceMin(0.0);
  corr.setTolerance(0.1);
  corr.insertSrc(input.begin(), input.end(), &norm2);
  corr.insertDst(gt.begin(), gt.end(), &norm2);
  std::vector<rofl::CorrespondenceGraphAssociation::AssociationHypothesis> associationsBest;
  corr.associate(associationsBest);
  const rofl::CorrespondenceGraphAssociation::AssociationHypothesis& hypothesis = associationsBest[0];

  int n = gt.size();
  float ret = 0;

  for (int i = 0; i < n; ++i) {
    std::cout << " * " << i << "/ " << n << " (" << hypothesis.associations[i].first << ", " << hypothesis.associations[i].second << ")" << std::endl;
    for (int j = i + 1; j < n; ++j) {
      std::cout << "!hypotesis.associations.size() = " << hypothesis.associations.size() << std::endl;
      std::cout << " _ " << j << "/ " << n << " (" << hypothesis.associations[j].first << ", " << hypothesis.associations[j].second << ")" << std::endl;

      // first e second sono i due punti nelle due nuvole
      // i e j sono gli indici di due associazioni
      //  i1 e i2 sono i due punti nelle due nuvole dell'associazione i
      const auto i1 = input[hypothesis.associations[i].first];
      const auto i2 = gt[hypothesis.associations[i].second];
      const auto j1 = input[hypothesis.associations[j].first];
      const auto j2 = gt[hypothesis.associations[j].second];
      float src = (i1 - j1).norm();
      float dst = (i2 - j2).norm();
      std::cout << "(" << i << ", " << j << ") src = " << src << " dts = " << dst << std::endl;
      ret += std::abs(dst - src);
    }
  }

  return ret / n;
}
//Aggiunte da Andrea
void printConsole(const std::string& str) {
  std::cout << str << std::endl;
}

struct Parameters {
  int thetaNum;
  int phiNum;
  int rhoNum;
  float thetaStep;
  float phiStep;
  float rhoStep;
  float thetaMin;
  float phiMin;
  float rhoMin;
  float thetaWin;
  float phiWin;
  float rhoWin;
  float distancePlaneThreshold;
  bool doLambert;
  bool doLineFilter;
  bool estimateError;
  float longEdge;
  float shortEdge;
};
bool fileExists(const std::string& filename) {
  std::ifstream file(filename);
  return file.good();
}
Parameters readConfigParam(const std::string& configFile) {
  Parameters params;

  try {
    boost::property_tree::ptree root;
    boost::property_tree::read_json(configFile, root);

    params.thetaNum = root.get<int>("thetaNum", 90);
    params.phiNum = root.get<int>("phiNum", 180);
    params.rhoNum = root.get<int>("rhoNum", 150);

    params.thetaStep = root.get<float>("thetaStep", M_PI / 180);
    params.phiStep = root.get<float>("phiStep", M_PI / 180);
    params.rhoStep = root.get<float>("rhoStep", 0.01);

    params.thetaMin = root.get<float>("thetaMin", M_PI / 2);
    params.phiMin = root.get<float>("phiMin", -M_PI / 2);
    params.rhoMin = root.get<float>("rhoMin", 0.5);

    params.thetaWin = root.get<float>("thetaWin", 5.0);
    params.phiWin = root.get<float>("phiWin", 5.0);
    params.rhoWin = root.get<float>("rhoWin", 5.0 * params.rhoStep);

    params.distancePlaneThreshold = root.get<float>("distancePlaneThreshold", 0.03);

    params.doLambert = root.get<bool>("doLambert", true);
    params.doLineFilter = root.get<bool>("doLineFilter", true);
    params.estimateError = root.get<bool>("estimateError", true);

    params.longEdge = root.get<float>("longEdge", 0.31);
    params.shortEdge = root.get<float>("shortEdge", 0.22);

    std::cout << "Lettura del file JSON completata con successo." << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Errore durante la lettura del file JSON: " << e.what() << std::endl;
  }

  return params;
}

int main(int argc, char** argv)
{
  ArgumentList args;
  if (!ParseInputs(args, argc, argv)) {
    exit(0);
  }
  if(args.config.empty()) {
    std::cout << "Config file not specified" << std::endl;
    exit(0);
  }
  if(args.pointCloud.empty()) {
    std::cout << "Point cloud file not specified" << std::endl;
    exit(0);
  }
  if(args.boxCenter.empty()) {
    std::cout << "Box center file not specified" << std::endl;
    exit(0);
  }
  
  std::cout << "Executes Box Detection algorithm" << std::endl;



  BoxDetector boxDetector;


#pragma region READ_CONF_PARAM
  // LOADING CONFIG
  if (!fileExists(args.config)) {
    throw std::runtime_error("File " + args.config+ " not exits");
  }
  Parameters params = readConfigParam(args.config);
  std::cout << "Box size: longEdge = " << params.longEdge << " shortEdge = " << params.shortEdge << std::endl;
#pragma endregion READ_CONF_PARAM

  
#pragma region READ_POINT_CLOUD

  // LOADING INPUT POINT CLOUD - "/home/ubuntu/Scaricati/cloud_0250.pcd"
  pcl::PointCloud<MyPoint>::Ptr cloudIn(new pcl::PointCloud<MyPoint>);
  if (pcl::io::loadPCDFile(args.pointCloud, *cloudIn) < 0) {
    std::cerr << "Cannot load point cloud from \"" << args.pointCloud << "\"" << std::endl;
    return -1;
  }
  std::cout << "Reading completed successfully. Number of points: " << cloudIn->size()  << "\n" << std::endl;

#pragma endregion READ_POINT_CLOUD

  pcl::PointCloud<MyPoint>::Ptr cloudPlane(new pcl::PointCloud<MyPoint>);
  pcl::PointCloud<MyPoint>::Ptr cloud1(new pcl::PointCloud<MyPoint>);
  pcl::PointCloud<MyPoint>::Ptr cloud2(new pcl::PointCloud<MyPoint>);
  pcl::PointCloud<MyPoint>::Ptr cloudNoGround(new pcl::PointCloud<MyPoint>);

  //Plane coefficients extracted from the point cloud with RANSAC
  Eigen::VectorXf planeCoeffs;


  std::cout << "Start top plane extraction" << std::endl;
  ransacTopPlaneExtraction(cloudIn, cloud1, cloud2, cloudNoGround, cloudPlane, planeCoeffs);

  onPlaneProjection(*cloudPlane, planeCoeffs);

  pcl::PointCloud<MyPoint>::Ptr cloudAligned(new pcl::PointCloud<MyPoint>);

  maf::PointCloudPlaneAligner<MyPoint> aligner;
  aligner.setInputCloud(cloudIn);
  aligner.setPlaneCoeffs(planeCoeffs);
  aligner.compute();
  aligner.transform(*cloudAligned);

  cv::Mat imageIntensities(cloudIn->height, cloudIn->width, CV_32FC1);
  cv::Mat imageDepth(cloudIn->height, cloudIn->width, CV_32FC1);
  cv::Mat imageDepthPlane(cloudIn->height, cloudIn->width, CV_32FC1);
  cv::Mat imageIntensitiesPlane(cloudIn->height, cloudIn->width, CV_32FC1);

  std::cout << "Building image intensities and depth" << std::endl;
  cloudToImgDepth(*cloudIn, imageDepth);
  cloudToImgIntensities(*cloudIn, imageIntensities);
  cloudToImgIntensities(*cloudPlane, imageIntensitiesPlane);
  cloudToImgDepth(*cloudPlane, imageDepthPlane);

  cv::Mat imageLambert;
  if (params.doLambert) {
    std::cout << "Lambert compensation" << std::endl;
    cloudToImgIntensitiesLambertCompensation(*cloudPlane, planeCoeffs, imageLambert);
    //     cloudToImgIntensitiesLambertCompensation(*cloudIn, planeCoeffs, imageLambert);
  }

  std::cout << "Image histogram normalization" << std::endl;
  cv::Mat imageNormalized;
  if (params.doLambert)
    histogramNormalization(imageLambert, imageNormalized, 0.01, 0.01);
  else
    histogramNormalization(imageIntensitiesPlane, imageNormalized, 0.01, 0.01);
  //   else histogramNormalization(imageIntensities, imageNormalized, 0.01, 0.01);

  std::cout << "Image blurring" << std::endl;
  cv::Mat imageBlurred;
  // Blurring for noise removeAllShapes
  cv::bilateralFilter(imageNormalized, imageBlurred, 5, 100, 100);
  //   cv::GaussianBlur(imageNormalized, imageBlurred,cv::Size(5,5),0.7);

  // Blurring re-introduces some invalid (nan) pixels: remove them
  for (int r = 0; r < imageBlurred.rows; ++r) {
    for (int c = 0; c < imageBlurred.cols; ++c) {
      if (imageNormalized.at<float>(r, c) < 1) {
        imageBlurred.at<float>(r, c) = 0;
      }
    }
  }

  std::cout << "Compute image gradient and orientation" << std::endl;
  // Gradient and orientation
  cv::Mat imageSobelX;
  cv::Mat imageSobelY;
  cv::Sobel(imageBlurred, imageSobelX, CV_32FC1, 1, 0, 3, 1.0 / 256, 0);
  cv::Sobel(imageBlurred, imageSobelY, CV_32FC1, 0, 1, 3, 1.0 / 256, 0);
  cv::Mat imageGradient = cv::Mat(imageBlurred.rows, imageBlurred.cols, CV_32FC1);

  for (int r = 0; r < imageGradient.rows; ++r) {
    for (int c = 0; c < imageGradient.cols; ++c) {
      if (imageNormalized.at<float>(r, c) < 1) {
        imageGradient.at<float>(r, c) = 0;
      } else {
        imageGradient.at<float>(r, c) = std::sqrt(std::pow(imageSobelX.at<float>(r, c), 2) +
          std::pow(imageSobelY.at<float>(r, c), 2));
      }
    }
  }

  cv::Mat orientation = cv::Mat(imageGradient.rows, imageGradient.cols, CV_8UC1);
  for (int r = 0; r < imageGradient.rows; ++r) {
    for (int c = 0; c < imageGradient.cols; ++c) {
      orientation.at<unsigned char>(r, c) = (std::atan2(imageSobelY.at<float>(r, c),
        imageSobelX.at<float>(r, c)) *
        0.5 / M_PI +
        0.5) *
        256;
    }
  }
  cv::Mat orientationColored;
  cv::applyColorMap(orientation, orientationColored, cv::COLORMAP_HSV);
  for (int r = 0; r < imageGradient.rows; ++r) {
    for (int c = 0; c < imageGradient.cols; ++c) {
      if (imageNormalized.at<float>(r, c) <= 0.01) {
        orientationColored.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
      }
    }
  }

  //   std::cout<<"Gradient thresholding"<<std::endl;
  //   cv::Mat imageGradientThresholded;
  //   cv::threshold(imageGradient,imageGradientThresholded,10*4.0/256,0,cv::THRESH_TOZERO);

  std::cout << "Non Maxima Suppression" << std::endl;
  cv::Mat imageGradientSuppressed;
  nonMaximumSuppression(imageGradient, imageGradientSuppressed, 10 * 4.0 / 256, 1, 3);

  cv::Mat opposite;
  cv::Mat oppositeTh;
  if (params.doLineFilter) {
    std::cout << "Line Gradient Filter" << std::endl;
    //   lineGradientFilter(*cloudPlaneOrgProj, imageBlurred, imageGradientSuppressed,
    //                      imageSobelX, imageSobelY, imageGradient, opposite, 0.1);
    lineGradientFilter(*cloudPlane, imageBlurred, imageGradientSuppressed,
      imageSobelX, imageSobelY, imageGradient, opposite, 0.1);
    cv::threshold(opposite, oppositeTh, 210, 255, cv::THRESH_BINARY);
    std::cout << "lgf: " << oppositeTh.type() << std::endl;
  } else {
    //     imageGradientSuppressed.convertTo(opposite, CV_8UC1);
    cv::normalize(imageGradientSuppressed, opposite, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::threshold(opposite, oppositeTh, 30, 255, cv::THRESH_BINARY);
  }

  // Build visualizer scene
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Box Detection"));

  viewer->addCoordinateSystem(0.3);
  viewer->addPointCloud<MyPoint>(cloudIn, "cloudIn");
  viewer->addPointCloud<MyPoint>(cloudPlane, "cloudPlane");
  //   viewer->addPointCloud<MyPoint>(cloudPlaneOrgProj, "cloudPlaneProj");
  //   viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0,0,1, "cloudPlaneProj");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "cloudPlane");
  viewer->addPointCloud<MyPoint>(cloudAligned, "cloudAligned");

  //   viewer->addPointCloud<MyPoint>(cloudOut, "cloudOut");
  //   viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1,0,1, "cloudOut");

  //   for (int i = 0; i < planes.size(); ++i) {
  //     pcl::ModelCoefficients coefficients;
  //     float x, y, z;
  //     int n;
  //     std::stringstream ss;
  //
  //     plotPlane(planes[i], *cloudIn, 2.0 * rhoStep, coefficients, x, y, z, n);
  //
  //     ss << "plane_" << i;
  //     std::cout << "  " << ss.str() << " [" << planes[i].transpose() << "]  centered in [" << x << "," << y << "," << z << "]  inliers " << n << std::endl;
  //
  //     if (n > 1000) {
  //       viewer->addPlane(coefficients, x, y, z, ss.str());
  // //       PointIn textPos;
  // //       textPos.x = x + 0.10 * planes[i](0);
  // //       textPos.y = y + 0.10 * planes[i](1);
  // //       textPos.z = z + 0.10 * planes[i](2);
  // //       ss.str("");
  // //       ss << "pl_" << i;
  // //       viewer->addText3D(ss.str(), textPos, 0.05, 0.0, 0.0, 0.0);
  //     }
  //   }

  MyPoint normalP1, normalP2;
  //  if ((!hsMaxima.empty())
  //  {
  //    normalP1.x = normalP1.y = normalP1.z = 0.0f;
  //    normalP2.x = hpd.getNormal(hsMaxima[0])(0);
  //    normalP2.y = hpd.getNormal(hsMaxima[0])(1);
  //    normalP2.z = hpd.getNormal(hsMaxima[0])(2);
  //    viewer->addArrow(normalP2, normalP1, 1.0f, 1.0f, 0.0, true, "normalMax");
  //    normalP2.x = -0.694553;
  //    normalP2.y = 0.0121235;
  //    normalP2.z = 0.71934;
  //    viewer->addArrow(normalP2, normalP1, 1.0f, 0.0f, 0.0, true, "max1");
  //    normalP2.x = -0.920505;
  //    normalP2.y = 0;
  //    normalP2.z = 0.390731;
  //    viewer->addArrow(normalP2, normalP1, 0.0f, 1.0f, 0.0, true, "max2");
  //    normalP2.x = -0.990268;
  //    normalP2.y = 0;
  //    normalP2.z = 0.139173;
  //    viewer->addArrow(normalP2, normalP1, 0.0f, 0.0f, 1.0, true, "max3");
  //  }

  //  for (int i = 0; i < planes.size(); ++i)
  //  {
  //    pcl::ModelCoefficients coefficients;
  //    float x, y, z;
  //    int n;
  //    std::stringstream ss;

  //    plotPlane(planes[i], *cloudIn, 2.0 * std::abs(houghSteps(2)), coefficients, x, y, z, n);

  //    ss << "plane_" << i;
  //    std::cout << "  " << ss.str() << " [" << planes[i].transpose() << "]  centered in [" << x << "," << y << "," << z << "]  inliers " << n << std::endl;

  //    if (n > 100)
  //    {
  //      viewer->addPlane(coefficients, x, y, z, ss.str());
  //      MyPoint textPos;
  //      textPos.x = x + 0.10 * planes[i](0);
  //      textPos.y = y + 0.10 * planes[i](1);
  //      textPos.z = z + 0.10 * planes[i](2);
  //      ss.str("");
  //      ss << "pl_" << i;
  //      viewer->addText3D(ss.str(), textPos, 0.05, 0.0, 0.0, 0.0);
  //    }
  //  }

  pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Box hull"));
  viewer2->addCoordinateSystem(0.3);

  // Set up opencv windows
  setupWindows();

  std::cout << "Start loop" << std::endl;
  bool exit_loop = false;
  while (!exit_loop) {
    viewer2->removeAllPointClouds();

    // Compute Probabilistic Hough
    cv::Mat cdst;
    cvtColor(oppositeTh, cdst, cv::COLOR_GRAY2BGR);
    cv::Mat cdstP;
    cdstP = cdst.clone();

    std::vector<cv::Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(oppositeTh, linesP, 1, CV_PI / 180,
      houghP_threshold_value, houghP_minLL_value, houghP_maxLG_value); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++) {
      cv::Vec4i l = linesP[i];
      cv::line(cdstP, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
        cv::Scalar(0, 0, 255), 4, cv::LINE_AA);
    }

    // Find contours and draw
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat drawing;
    cv::RNG rng;
    cv::Mat drawingHull;

    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> faceCentroids;
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> maf_centroids_preReg;
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> maf_centroids_postReg;
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> maf_faceCentroids_postReg;

    boxDetector.findContoursAndDraw(cdstP, imageIntensities, contours, hierarchy, rng, drawing,
      drawingHull, viewer2, cloudAligned, faceCentroids, maf_centroids_preReg,
      maf_centroids_postReg, maf_faceCentroids_postReg, params.longEdge, params.shortEdge);



    if (params.estimateError) {
      // LOAD GROUND TRUTH
      std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> centroidsGt;
      centroidsGt = boxDetector.loadGroundTruth(args.boxCenter);

      std::cout << "Risultati:" << std::endl;
      float faceResults = boxDetector.computeError(faceCentroids, centroidsGt);
      float preRegResults = boxDetector.computeError(maf_centroids_preReg, centroidsGt);
      float postRegResults = boxDetector.computeError(maf_centroids_postReg, centroidsGt);
      float facePostRegResults = boxDetector.computeError(maf_faceCentroids_postReg, centroidsGt);
      // //     float aDist = evaluateAssociationDistance(maf_centroids_preReg, centroidsGt);
      //
      std::cout << "face centroids error: " << faceResults << std::endl;
      std::cout << "contour centroids error pre reg: " << preRegResults << std::endl;
      std::cout << "contour centroids error post reg: " << postRegResults << std::endl;
      std::cout << "face centroids error post reg: " << facePostRegResults << std::endl;
      // //     std::cout<<"association error: "<<aDist<<std::endl;
    }

    // Approximation
    for (size_t k = 0; k < contours.size(); k++)
      cv::approxPolyDP(cv::Mat(contours[k]), contours[k], 10, true);
    //       cv::approxPolyDP(cv::Mat(contours[k]), contours[k], 6, true);

    cv::Mat drawingApprox = cv::Mat::zeros(imageIntensities.size(), CV_8UC3);
    cvtColor(imageIntensities, drawingApprox, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < contours.size(); i++) {
      cv::Scalar color = cv::Scalar(rng.uniform(0, 256),
        rng.uniform(0, 256),
        rng.uniform(0, 256));
      cv::drawContours(drawingApprox, contours, (int)i, color, 1, cv::LINE_8, hierarchy, 0);
    }

#pragma region CONTOURS_DEBUG
    //    //Uncomment for debugging
    //     std::cout<<"contours size: "<<contours.size()<<std::endl;
    //     for(size_t i = 0; i < contours.size(); ++i) {
    // //     for(size_t i = 0; i < 1; ++i) {
    //       std::cout<<"c["<<i<<"]: ";
    //       for(size_t k=0; k<contours[i].size(); ++k) {
    //         std::cout<<contours[i][k]<<" ";
    //       }
    //       std::cout<<std::endl;
    //     }
    //
    //     std::cout<<"[Next, Previous, Child, Parent]"<<std::endl;
    //     for(size_t i = 0; i < hierarchy.size(); ++i) {
    //       std::cout<<"h["<<i<<"]: "<<hierarchy[i]<<std::endl;
    //     }
#pragma endregion CONTOURS_DEBUG
    
    // Normalization before viewing
    cv::normalize(imageDepth, imageDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(imageDepthPlane, imageDepthPlane, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(imageIntensities, imageIntensities, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(imageIntensitiesPlane, imageIntensitiesPlane, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    if (params.doLambert)
      cv::normalize(imageLambert, imageLambert, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(imageNormalized, imageNormalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(imageBlurred, imageBlurred, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(imageGradient, imageGradient, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(imageGradientSuppressed, imageGradientSuppressed, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Display opencv windows
    cv::imshow("imageDepth", imageDepth);
    cv::imshow("imageDepthPlane", imageDepthPlane);
    cv::imshow("imageIntensities", imageIntensities);
    cv::imshow("imageIntensitiesPlane", imageIntensitiesPlane);

    if (params.doLambert)
    cv::imshow("imageLambert", imageLambert);
    cv::imshow("imageNormalized", imageNormalized);
    cv::imshow("imageBlurred", imageBlurred);

    cv::imshow("imageGradient", imageGradient);
    cv::imshow("imageGradientSuppressed", imageGradientSuppressed);

    cv::imshow("imageOrientation", orientationColored);

    if (params.doLineFilter)
      cv::imshow("opposite", opposite);
    cv::imshow("oppositeTh", oppositeTh);

    cv::imshow("Probabilistic Line Transform", cdstP);

    cv::imshow("findContours", drawing);
    cv::imshow("approximation", drawingApprox);
    cv::imshow("hull", drawingHull);

    viewer->spinOnce(100);
    viewer2->spinOnce(100);

    char key = cv::waitKey(1);
    if (key == 'q')
      exit_loop = true;
  }

  // Reset pcl viewer
  viewer->resetStoppedFlag();
  viewer->removeAllPointClouds();
  viewer->removeAllShapes();

  viewer2->resetStoppedFlag();
  viewer2->removeAllPointClouds();
  viewer2->removeAllShapes();

  // Destroy opencv windows
  cv::destroyAllWindows();

  return 0;
}

bool ParseInputs(ArgumentList& args, int argc, char** argv) {
  int c;

  while ((c = getopt(argc, argv, "hc:p:b:")) != -1)
    switch (c) {
    case 'c':
      args.config = optarg;
      break;
    case 'p':
      args.pointCloud = optarg;
      break;
    case 'b':
      args.boxCenter = optarg;
      break;
    case 'h':
    default:
      std::cout << "usage: " << argv[0] << " -c <config> -p <point cloud> -b <box_center>" << std::endl;
      std::cout << "Allowed options:" << std::endl <<
        "   -h                       produce help message" << std::endl <<
        "   -c 'path'                path to the configuration file" << std::endl <<
        "   -p 'path'                path to the pointcloud" << std::endl <<
        "   -b 'path'                path to the box center file" << std::endl << std::endl;
      return false;
    }
  return true;
}