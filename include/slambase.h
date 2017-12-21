#pragma once

//c++
#include <fstream>
#include <vector>
#include <string>
#include <map>
using namespace std;

//eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
//opencv feature detection
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//opencv for eigen
#include <opencv2/core/eigen.hpp>

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

//g2o
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>


//typedef
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
typedef g2o::LinearSolverEigen< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 


/*
intrinsic camera parameters*/
struct CAMERA_INTRINSIC_PARAMS{
  double cx, cy, fx, fy, scale;
};


/*
Frame struct*/
struct FRAME{
  int frameID;
  cv::Mat rgb, depth;
  //keypoint
  vector<cv::KeyPoint> kp;
  //descriptor
  cv::Mat desp;
};


/*
PnP result*/
struct RESULT_OF_PNP{
  cv::Mat rvec, tvec;
  int inliers;
};


/*
the defination for the relationships between two frames */
enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME}; 


/*
class for read in parameters,
use # for comment*/
class ParameterReader{
public:
  //find parameter name
  ParameterReader(string filename="./parameters.txt"){

    ifstream fin(filename.c_str()) ;
      if (!fin){
        cerr<<"parameter file does not exist."<<endl;
        return;
      }
      
      while(!fin.eof()){
        string str;
        getline( fin, str );
      
        if (str[0] == '#')
          continue;

        int pos = str.find("=");
        
        if (pos == -1)
          continue;
        
        string key = str.substr( 0, pos );
        string value = str.substr( pos+1, str.length() );
        data[key] = value;

        if (!fin.good())
          break;
     }
  }

  //read in the data 
  string getData(string key){
    map<string, string>::iterator iter = data.find(key);
    
    if (iter == data.end()){
        cerr<<"Parameter name "<<key<<" not found!"<<endl;
        return string("NOT_FOUND");
    }
        return iter->second;
  }

public:
  map<string, string> data;

};



/*
FUNCTIONS
*/

//get camera params
inline static CAMERA_INTRINSIC_PARAMS getDefaultCamera(){
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
    return camera;
}

//read in frame
FRAME readFrame(int index, ParameterReader& pd);

//translate image to pointloud
PointCloud::Ptr img2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMS& camera);

//translate a single point from image location (u,v,d) to 3D-space location (x,y,z)
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMS& camera);

//compute keypoints and extract descriptors
void computeKpAndDesp(FRAME& frame, string detector);

//estimate the motion between 2 frames
RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMS& camera);

//translate cv::mat into eigen
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec );

//project the new frame into the pointcloud and add the new pointcloud into the previous one
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMS& camera);

//normalize the transform matrix to calculate whether the motion between 2 frames are so large
double normofTransform(cv::Mat rvec, cv::Mat tvec);

//check whether is keyframes
CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false);

//nearby loops
void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );

//random loops
void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );
