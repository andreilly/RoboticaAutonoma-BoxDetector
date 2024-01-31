#include "BoxDetector.h"

float BoxDetector::computeError(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &estimated, const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &groundTruth)
{
  if (estimated.size() != groundTruth.size())
  {
    std::cout << "estimatedSize: " << estimated.size() << " gtSize: " << groundTruth.size() << std::endl;
    throw std::runtime_error("Error! Estimated centroids size different from ground truth size!");
  }

  std::cout << "Estimated:" << std::endl;
  std::vector<float> estDistances = BoxDetector::computePaiwiseDistanceSet(estimated);
  std::cout << "Gt: " << std::endl;
  std::vector<float> gtDistances = BoxDetector::computePaiwiseDistanceSet(groundTruth);

  std::sort(estDistances.begin(), estDistances.end());
  std::sort(gtDistances.begin(), gtDistances.end());

  float acc = 0;
  for (int i = 0; i < gtDistances.size(); ++i)
  {
    acc += std::abs(gtDistances[i] - estDistances[i]);
  }

  return acc / gtDistances.size();
}

std::vector<float> BoxDetector::computePaiwiseDistanceSet(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &points)
{//prende vettore punti in input e calcola la norma tra ogni punto e tutti gli altri
  std::vector<float> result;
  result.reserve(points.size() * (points.size() - 1) / 2);

  for (int i = 0; i < points.size(); ++i)
  {
    for (int j = i + 1; j < points.size(); ++j)
    {
      result.push_back((points[i] - points[j]).norm());
      std::cout << result.back() << std::endl;
    }
  }

  return result;
}

void BoxDetector::findContoursAndDraw(cv::Mat &cdstP,cv::Mat &imageIntensities, std::vector<std::vector<cv::Point>> &contours, 
    std::vector<cv::Vec4i> &hierarchy,cv::RNG &rng, cv::Mat &drawing, cv::Mat &drawingHull,
    pcl::visualization::PCLVisualizer::Ptr &viewer2,pcl::PointCloud<MyPoint>::Ptr cloudAligned, 
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &faceCentroids,
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &maf_centroids_preReg,
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &maf_centroids_postReg,
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> &maf_faceCentroids_postReg,
    float longEdge, float shortEdge)
{
  rng=cv::RNG(12345);
  
  cv::Mat fcSrc;
  cvtColor(cdstP, fcSrc, cv::COLOR_BGR2GRAY);
  cv::findContours(fcSrc, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  drawing = cv::Mat::zeros(imageIntensities.size(), CV_8UC3);
  cvtColor(imageIntensities, drawing, cv::COLOR_GRAY2BGR);

  for (size_t i = 0; i < contours.size(); i++)
  {
    cv::Scalar color = cv::Scalar(rng.uniform(0, 256),
                                  rng.uniform(0, 256),
                                  rng.uniform(0, 256));
    cv::drawContours(drawing, contours, (int)i, color, 1, cv::LINE_8, hierarchy, 0);
  }

  cvtColor(imageIntensities, drawingHull, cv::COLOR_GRAY2BGR);

  for (int i = 0; i < contours.size(); ++i)
    {
      //       if(hierarchy[i][3] != 0) continue;
      // Filtering non boxes contours (eg. connected components)
      double area0 = cv::contourArea(contours[i]);
      std::cout << "contours[" << i << "] - area0 =" << area0 << std::endl;
      // Filter by size and by hierarchy (they should have a parent)
      // dataset coorsa 6000 pixel per scatola. invece scatole bianche 9000
      //       if( (area0 < 1000 || area0 > 6000) || hierarchy[i][3] == -1 ) {
      if ((area0 < 1000 || area0 > 9000) || hierarchy[i][3] == -1)
      {
        std::cout << "not a box" << std::endl;
        continue;
      }

      pcl::PointCloud<MyPoint>::Ptr face(new pcl::PointCloud<MyPoint>);

      // calcolo hull
      std::vector<cv::Point> hull;
      cv::convexHull(contours[i], hull, false, true);
      cv::Scalar color = cv::Scalar(rng.uniform(0, 256),
                                    rng.uniform(0, 256),
                                    rng.uniform(0, 256));
      cv::polylines(drawingHull, hull, true, color, 1, cv::LINE_AA);

      Eigen::Vector2f hullCentroid = Eigen::Vector2f::Zero();

      maf::BoxRegistration2f::VectorPoint maf_face;

      for (const cv::Point &pt : hull)
      {
        hullCentroid += Eigen::Vector2f(pt.x, pt.y);
      }
      hullCentroid /= hull.size();

      Eigen::Vector2f faceCentroid = Eigen::Vector2f::Zero();
      // Compute point cloud clusters of boxes (contours and points inside the contour)
      for (int r = 0; r < imageIntensities.rows; ++r)
      {
        for (int c = 0; c < imageIntensities.cols; ++c)
        {
          if (cv::pointPolygonTest(hull, cv::Point2f(c, r), false) >= 0)
          {
            // FIXME: cloudIn o cloudAligned
            //             MyPoint pt = cloudIn->at(r*imageIntensities.cols+c);
            MyPoint pt = cloudAligned->at(r * imageIntensities.cols + c);
            face->push_back(pt);

            maf_face.emplace_back(pt.x, pt.y);

            //             faceCentroid+=Eigen::Vector2f(c,r);
            faceCentroid += Eigen::Vector2f(pt.x, pt.y);
          }
        }
      }
      faceCentroid /= face->size();

      faceCentroids.push_back(faceCentroid);

      // BoxRegistration
      maf::BoxRegistration2f::VectorPoint maf_hull;
      for (int j = 0; j < hull.size(); ++j)
      {
        // FIXME: cloudIn o cloudAligned
        //         MyPoint pt =  cloudIn->at(hull[j].y*imageIntensities.cols+hull[j].x);
        MyPoint pt = cloudAligned->at(hull[j].y * imageIntensities.cols + hull[j].x);
        maf_hull.emplace_back(pt.x, pt.y);
      }
      maf::BoxRegistration2f boxreg;
      maf::Transform2 transf;
      boxreg.setDimension(maf::Point2(longEdge, shortEdge));
      // Registrazione hull
      boxreg.estimateInitTransform(maf_hull, transf);

      maf_centroids_preReg.emplace_back(transf.translation());

      //       std::cout<<i<<"-centroide: "<<faceCentroid.transpose()<<std::endl;
      //       std::cout<<"transf: \n"<<transf.matrix()<<std::endl;
      // //     transfEstim2 = maf::Transform2::Identity();
      // //     transfEstim2.pretranslate(center);
      //
      boxreg.computeTransform(maf_hull, transf);
      maf_centroids_postReg.emplace_back(transf.translation());

      boxreg.estimateInitTransform(maf_face, transf);
      boxreg.computeTransform(maf_face, transf);
      maf_faceCentroids_postReg.emplace_back(transf.translation());

      viewer2->addPointCloud<MyPoint>(face, "face" + std::to_string(i));
      viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[2] / 255, color[1] / 255, color[0] / 255, "face" + std::to_string(i));
      // color bgr in accordance to opencv image hull

      //       viewer2->addPolygon<MyPoint>(box, 1.0,0.0,0.0, "polygon"+std::to_string(i));
      //       std::cout<<"point cloud added"<<std::endl;

      //       viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SIZE, 5,
    }
}

std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> BoxDetector::loadGroundTruth(const std::string& filenameIn) 
{
    std::string gtIn = filenameIn;
    std::cout << "gtIn: " << gtIn << std::endl;
    bool done = false;
    while (!done) {
        if (gtIn[gtIn.length() - 1] == '/') {
            done = true;
        } else {
            gtIn.pop_back();
        }
    }
    gtIn += "gt/box_centers.txt";

    std::cout << "Loading ground truth from \"" << gtIn << "\"" << std::endl;

    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> centroidsGt;
    std::ifstream gt;
    std::string boxName;
    float cx, cy, cz;
    gt.open(gtIn, std::ifstream::in);
    if (gt.is_open()) {
        while (true) {
            gt >> boxName >> cx >> cy >> cz;
            if (gt.eof()) {
                break;
            }
            Eigen::Vector2f centroid;
            centroid(0) = cx;
            centroid(1) = cy;
            centroidsGt.push_back(centroid);
        }
    }
    gt.close();

    return centroidsGt;
}

/* void roflTopPlaneExtraction(pcl::PointCloud<MyPoint>::Ptr cloudIn,
                            pcl::PointCloud<MyPoint>::Ptr &cloudPlane, 
                            Eigen::VectorXf &planeCoeffs,
                            float distancePlaneThreshold,
                            rofl::HoughPlaneDetector::Indices3 houghNums,
                            Eigen::Vector3f houghMins,
                            Eigen::Vector3f houghSteps,
                            rofl::Scalar thetaWin,
                            rofl::Scalar phiWin,
                            rofl::Scalar rhoWin,
                            float rhoStep,
                            float thetaNum,
                            float phiNum,
                            float rhoNum)
{
  
} */
 