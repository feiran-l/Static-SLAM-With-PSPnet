#include <slambase.h>


/**
* read in frame*/
FRAME readFrame(int index, ParameterReader& pd){

  FRAME f;
  string rgbDir = pd.getData("rgb_dir");
  string depthDir = pd.getData("depth_dir");
    
  string rgbExt = pd.getData("rgb_extension");
  string depthExt = pd.getData("depth_extension");

  stringstream ss;
  ss<<rgbDir<<index<<rgbExt;
  string filename;
  ss>>filename;
  f.rgb = cv::imread(filename);

  ss.clear();
  filename.clear();
  ss<<depthDir<<index<<depthExt;
  ss>>filename;

  f.depth = cv::imread(filename, -1);
  f.frameID = index;
  return f;
}


/**
* translate image to pointloud*/
PointCloud::Ptr img2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMS& camera){
  
  //PointCloud::Ptr is an auto ptr, new a PointCloud-type 
  PointCloud::Ptr cloud (new PointCloud);
  
  for (int m = 0; m < depth.rows; m+=2)
     for (int n = 0; n < depth.cols; n+=2){
        //get the depth_value of the (m,n) point
	ushort d = depth.ptr<ushort>(m)[n];
	//if the depth cannot be detected, jump this point
	if (d == 0)
	  continue;
	//if the depth exists, add a point to the point cloud
	PointT p;
	
	//calculate the 3d location of this point
	p.z = double(d)/camera.scale;
	p.x = (n - camera.cx) * p.z / camera.fx;
        p.y = (m - camera.cy) * p.z / camera.fy; 
	
	//get the rbg data from the rgb camera
	p.b = rgb.ptr<uchar>(m)[n*3];
  p.g = rgb.ptr<uchar>(m)[n*3+1];
  p.r = rgb.ptr<uchar>(m)[n*3+2];
	
	//add p to the PointCloud
	cloud->points.push_back(p);
      }

  //set and save point cloud
  cloud->height = 1;
  cloud->width = cloud->points.size();
  cloud->is_dense = false;
  
  return cloud;
}


/**
* translate a single point from image location (u,v,d) to 3D-space location (x,y,z)*/
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMS& camera){
  
  cv::Point3f p;
  p.z = double(point.z)/camera.scale;
  p.x = (point.x - camera.cx) * p.z / camera.fx;
  p.y = (point.y - camera.cy) * p.z / camera.fy; 
  
  return p;	
}


/**
* compute keypoints and extract descriptors
* NOTE: in opencv2.4, the descriptor type should be defined, in opencv3.3, it is not asked*/
void computeKpAndDesp(FRAME& frame, string detector){
  
  cv::Ptr<cv::Feature2D> _detector;

	if (detector.compare("SIFT") == 0 || detector.compare("sift") == 0 )
		_detector = cv::xfeatures2d::SIFT::create();
	else if (detector.compare("SURF") == 0 || detector.compare("surf") == 0 )
		_detector = cv::xfeatures2d::SURF::create();
	else if (detector.compare("ORB") == 0 || detector.compare("orb") == 0 )
		_detector = cv::ORB::create(1000);
	else {
		_detector = cv::xfeatures2d::SIFT::create();
	}

 	_detector->detect( frame.rgb, frame.kp );
  _detector->compute( frame.rgb, frame.kp, frame.desp );
}


/*
* estimate the motion between 2 frames
* PARAMETER:"good_match_threshold" is used to filters some matches which has a long distance, usually
*           set as 4, which means that the matches who distance is larger than 4 times of the minimum
*           matching distance is discarded*/
RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMS& camera){
  
  static ParameterReader pd;
  RESULT_OF_PNP result;

  //use "Fast Library for Approximate Nearest Neighbour, FLANN" for descriptor matching
  vector< cv::DMatch > matches;
  cv::BFMatcher matcher;
  matcher.match( frame1.desp, frame2.desp, matches );
  cout<<"find total "<<matches.size()<<" matches."<<endl;

  //find out good matches
  vector< cv::DMatch > goodMatches;
  double minDis = 9999;
  double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    for ( size_t i=0; i<matches.size(); i++ ){
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }
    
    if ( minDis < 10 ) 
      minDis = 10;

    for ( size_t i=0; i<matches.size(); i++ ){
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] );
    }
    cout<<"good matches: "<<goodMatches.size()<<endl;

    if (goodMatches.size() <= 5){
        result.inliers = -1;
        return result;
    }
  
  //the 3d location(x,y,z) of the keypoints in first frame
  vector<cv::Point3f> pts_obj;
  //the 2d location(u,v) of the keypoints in the second frame
  vector<cv::Point2f> pts_img;
  
  //process the pts_obj and pts_img and push them into stack
  for (size_t i=0; i<goodMatches.size(); i++){
    //first deal with the pts_obj
      //query is the first frame, train is the second frame
      cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
      //NOTICE！ depend on the opencv coordinate of an image, y is row and x is col
      ushort d = frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
      if (d == 0)
        continue;
      //tranlate (u,v,d) into (x,y,z)
      cv::Point3f pt (p.x, p.y, d);
      cv::Point3f pt_After_trans = point2dTo3d(pt, camera);
      pts_obj.push_back(pt_After_trans);

    //deal with the pts_img
      pts_img.push_back(cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt));
  }

    if (pts_obj.size() ==0 || pts_img.size()==0){
        result.inliers = -1;
        return result;
    }

  //build camera intrinsic matrix, which is needed for the solve-pnp-ransac func
  double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };
  cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );

  //solving pnp
  cout<<"solving pnp"<<endl;
  cv::Mat rvec, tvec, inliers;
  try{
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;
  }
  catch (...){
	  cout<<"PnPSlover Error, Bad matching, jump to next frame!"<<endl;
	}
  
  return result;
}


/**
* translate cv::mat into eigen*/
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec){
  
    //translate the rvec from vector to matrix
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    //cv::cv2eigen(R, r);
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ ) 
            r(i,j) = R.at<double>(i,j);

    //compose the t-vector and r-matrix into the transformation matrix
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
   // Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);
    return T;
}


/**
* project the new frame into the pointcloud and add the new pointcloud into the previous one*/
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMS& camera){

   //project the new frame into a new pointcloud
   PointCloud::Ptr newCloud = img2PointCloud( newFrame.rgb, newFrame.depth, camera );
   
   //joint the new pointcloud with the previous one
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud(*original, *output, T.matrix());
    *newCloud += *output;

    //voxel filter for down-sampling
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
}


/*
normalize the transform matrix to calculate whether the motion between 2 frames are so large*/
double normofTransform(cv::Mat rvec, cv::Mat tvec){

  return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}



/**
* check whether is keyframe,
* Notice: in the g2o solver, only the edges are optimized*/
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops){

  static ParameterReader pd;
  static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
  static double max_norm = atof( pd.getData("max_norm").c_str() );
  static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
  static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
  static CAMERA_INTRINSIC_PARAMS camera = getDefaultCamera();
  //static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct( "Cauchy" );
    
  //compare f1 and f2
  RESULT_OF_PNP result = estimateMotion( f1, f2, camera );

  //a frame is discarded if inliers is to few
  if (result.inliers < min_inliers) 
    return NOT_MATCHED;
  
  //calculate if the motion is too large
  double norm = normofTransform(result.rvec, result.tvec);
  
  if (is_loops == false){
     if (norm >= max_norm)
        return TOO_FAR_AWAY;   // too far away, may be error
  }
  else{
     if (norm >= max_norm_lp)
        return TOO_FAR_AWAY;
  }

  if (norm <= keyframe_threshold)
    return TOO_CLOSE;   // too adjacent frame
   
  //add vertex and set the related-edge of this frame to g2o-solver
    //add vertex, done by just set id 
    if (is_loops == false){
      
      cout<<"is_loop is "<<is_loops<<endl;
      g2o::VertexSE3 *v = new g2o::VertexSE3();
      v->setId( f2.frameID );
      v->setEstimate(Eigen::Isometry3d::Identity());
      opti.addVertex(v);
    }

    //set edge
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();

    //id of the two vertices of this edge
    /*edge->vertices() [0] = opti.vertex( f1.frameID );
    edge->vertices() [1] = opti.vertex( f2.frameID );
    edge->setRobustKernel( robustKernel );*/

    edge->setVertex( 0, opti.vertex(f1.frameID ));
    edge->setVertex( 1, opti.vertex(f2.frameID ));
    edge->setRobustKernel( new g2o::RobustKernelHuber() );

    //information matrix
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
  
    /*the information matrix is the inverse matrix of the covariance matrix, which
      presents the estimation of the edges' accuracy, since the pose is 6D(x,y,z,3-rotation),
      so the information matrix is 6*6 */ 
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;

    edge->setInformation( information );
    
    /*the estimation of the edges are the result of the PnP, 
      notice the information means accuracy and the measurement means the transform matrix itself */
    Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
    edge->setMeasurement( T.inverse() );
    
    //add the edge into solver
    opti.addEdge(edge);

  return KEYFRAME;
}


/**
* nearby frames for loop closure, 
* if can be matched, add a new edge into the graph*/
void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti){
   
  static ParameterReader pd;
  static int nearby_loops = atoi( pd.getData("nearby_loops").c_str() );
    
  if (frames.size() <= nearby_loops){
    // no enough keyframes, check everyone
    for (size_t i=0; i<frames.size(); i++){
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  }
  else{
    // check the nearest ones
    for (size_t i = frames.size()-nearby_loops; i<frames.size(); i++){
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  }
}


/**
* random frames for loop closure,
* if can be matched, add a new edge into the graph*/
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti){
 
  static ParameterReader pd;
  static int random_loops = atoi( pd.getData("random_loops").c_str() );
  srand((unsigned int) time(NULL));
   
  if (frames.size() <= random_loops){
    // no enough keyframes, check everyone
    for (size_t i=0; i<frames.size(); i++){
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  }
  else{
    // randomly check loops
    for (int i=0; i<random_loops; i++){
      int index = rand()%frames.size();
      checkKeyframes(frames[index], currFrame, opti, true);
    }
  }
}
















































