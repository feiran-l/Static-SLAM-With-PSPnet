#include "ICnet.h"

cv::Mat imgSegAndDynamicRemove(cv::Mat& img, caffe::Net<float>* _net);

int main(int argc, char *argv[])
{
    int keyframe_count = 0;
    int new_pcl_count = 0;
    ParameterReader pd;
    int startIndex = atoi(pd.getData("start_index").c_str());
    int endIndex = atoi(pd.getData("end_index").c_str());

    vector<FRAME> keyframes;

    //initialize
    cout<<"Initializing ..."<<endl;

    int currIndex = startIndex; 
    FRAME currFrame = readFrame(currIndex, pd);
    string detector = pd.getData("detector");


    CAMERA_INTRINSIC_PARAMS camera = getDefaultCamera();
    computeKpAndDesp(currFrame, detector);

    PointCloud::Ptr cloud = img2PointCloud(currFrame.rgb, currFrame.depth, camera);
   
      //initialize the solver
      SlamLinearSolver* linearSolver = new SlamLinearSolver();
      linearSolver->setBlockOrdering(false);
      SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
      g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

      //the last solver
      g2o::SparseOptimizer globalOptimizer;
      globalOptimizer.setAlgorithm(solver); 

      //do not display the processing information
      globalOptimizer.setVerbose(false);

      //add the first vertex into solver
      g2o::VertexSE3* v = new g2o::VertexSE3();
      v->setId(currIndex);
      //set the estimation as identity matrix
      v->setEstimate( Eigen::Isometry3d::Identity() ); 
      //the first vertex is stricted and do not need to be optimized
      v->setFixed(true); 
      globalOptimizer.addVertex(v);
      
      //add the first frame into keyframe
      keyframes.push_back(currFrame);

      //read in parameters
      double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
      bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");

      for (currIndex=startIndex+1; currIndex<endIndex; currIndex++){
        cout<<"\n"<<endl;
        cout<<"Reading files "<<currIndex<<endl;
        //read in current frame
        FRAME currFrame = readFrame(currIndex,pd); 
        //extract detector and drescriptor
        computeKpAndDesp(currFrame, detector); 
        //fit current-frame and the last keyframe
        CHECK_RESULT result = checkKeyframes(keyframes.back(), currFrame, globalOptimizer); 
        switch (result) 
        {
        case NOT_MATCHED:
            //cannot matched
            cout<<"Not enough inliers."<<endl;
            break;
        case TOO_FAR_AWAY:
            //motion is too small
            cout<<"Too far away, may be an error."<<endl;
            break;
        case TOO_CLOSE:
            //motion is to large
            cout<<"Too close, not a keyframe"<<endl;
            break;
        case KEYFRAME:
            cout<<"This is a new keyframe"<<endl;
            keyframe_count++;
            cout<<"keyframe_count is "<<keyframe_count<<endl;
           
            // check loop closure
            if (check_loop_closure){
                checkNearbyLoops(keyframes, currFrame, globalOptimizer);
                checkRandomLoops(keyframes, currFrame, globalOptimizer);
            }
            keyframes.push_back(currFrame);
            break;
        default:
            break;
        }   
      }
      
      //global optimization
      cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
      globalOptimizer.initializeOptimization();
      //the optimization steps could be set
      globalOptimizer.optimize(100); 
      cout<<"Optimization done."<<endl;

      //joint the pointcloud together
      cout<<"saving the point cloud map..."<<endl;
      //global pointcloud
      PointCloud::Ptr output (new PointCloud()); 
      PointCloud::Ptr tmp (new PointCloud());

      //voxel filter to adjust the pointcloud's resolution
      pcl::VoxelGrid<PointT> voxel;
      //the filter in the z director, too throw away the large depth which means in a pretty far away place
      pcl::PassThrough<PointT> pass; 
      pass.setFilterFieldName("z");
      //throw away the depth which means a distance larger than 4 meters
      pass.setFilterLimits(0.0, 4.0);
  
      double gridsize = atof(pd.getData("voxel_grid").c_str()); 
      voxel.setLeafSize(gridsize, gridsize, gridsize);

      //load net
      string prototxt = "./model/pspnet50_ADE20K_473.prototxt";
      string caffemodel = "./model/pspnet50_ADE20K.caffemodel";
      caffe::Net<float>* net = loadNet<float>(prototxt, caffemodel, caffe::TEST);
      
      for (size_t i=0; i<keyframes.size(); i++){
        //get one optimized frame out from the g2o-solver
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID));
        //the optimized pose
        Eigen::Isometry3d pose = vertex->estimate(); 

       //Semantic segmentation and dynamic object remove
        cv::Mat refer_img = imgSegAndDynamicRemove(keyframes[i].rgb, net);
        PointCloud::Ptr newCloud = img2PointCloudIC(keyframes[i].rgb, keyframes[i].depth, camera, refer_img);
       
        //filtering
        voxel.setInputCloud(newCloud);
        voxel.filter(*tmp);
        pass.setInputCloud(tmp);
        pass.filter(*newCloud);
        //add this new frame into global pointcloud
        pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
        *output += *tmp;
        tmp->clear();
        newCloud->clear();
        new_pcl_count++;
        cout<<"new_pcl_count is "<<new_pcl_count<<endl;
      }

      voxel.setInputCloud(output);
      voxel.filter(*tmp);

      //employ the statistical_outlier_filter of pcl
      tmp = pclSorFilter(tmp);

      //save the pointcloud
      pcl::io::savePCDFile("./result/result_ICslam.pcd", *tmp);
    
      cout<<"Final map is saved."<<endl;

   return 0;
}



/**
* Use the loaded net form semantic segmentation and then remove the dynamic objects
* On the returned img, the vehicles are set as float 100 and other areas are as float 0 */
cv::Mat imgSegAndDynamicRemove(cv::Mat& img, caffe::Net<float>* _net){
    
   //resize the img as the input size
      vector<int> compression_params;
      compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
      compression_params.push_back(3);
      
      //get the original size for upsample
      int Orig_H = img.rows, Orig_W = img.cols;
      //resize the image 
      pair<cv::Mat,vector<int>> processed_Img = imgPreProcess(img);  
      int aft_H = processed_Img.second[1], aft_W = processed_Img.second[0];

   //process the img with caffe net
      std::vector<cv::Mat> img2caffe = {processed_Img.first};
      std::vector<int> label2caffe = {0};

      caffe::MemoryDataLayer<float>* m_layer_ = (caffe::MemoryDataLayer<float>*) _net->layers()[0].get();
      
      if (!m_layer_){
        cout<<"The first layer is not a MemoryDataLayer!"<<endl;
        exit(EXIT_FAILURE);  
      }

      m_layer_->AddMatVector(img2caffe, label2caffe);

      //do a forward process
      std::vector<caffe::Blob<float>*> input_vec;
      _net->Forward(input_vec);
      cout<<"forward is done"<<endl;
      
      //get the output of a certain layer
      boost::shared_ptr<caffe::Blob<float>> layerData = _net->blob_by_name("interp_argmax");
      
      const float* pstart = layerData->cpu_data();

   //make an temp cv::mat to represent the segments
      cv::Mat temp_gray = cv::Mat::zeros(cv::Size(473,473),CV_32F);
      float tmp_ptr;
      for(int r_count = 0; r_count < 473 ; r_count++){
           for(int c_count = 0; c_count< 473; c_count++){

              tmp_ptr = *pstart; 

              //12 means vehicles in the certain model
              if(fabs(tmp_ptr-12) <= 1E-6 )
                temp_gray.at<float>(r_count, c_count) = 100;
              else
                temp_gray.at<float>(r_count, c_count) = 0;

              pstart++;
           }
      }

   //resize the image back into the original size
      temp_gray = temp_gray(cv::Rect(0,0, aft_W, aft_H));
      cv::resize(temp_gray, temp_gray, cv::Size(Orig_W,Orig_H));
    
   //dilation to deal with the edge
      for(int i = 0; i < 2; i++)
        temp_gray = imgDilation(temp_gray);
    
    return temp_gray;
}

