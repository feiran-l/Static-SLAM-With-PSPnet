#include "slambase.h"


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

      for (size_t i=0; i<keyframes.size(); i++){
        //get one optimized frame out from the g2o-solver
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID));
        //the optimized pose
        Eigen::Isometry3d pose = vertex->estimate(); 
        PointCloud::Ptr newCloud = img2PointCloud(keyframes[i].rgb, keyframes[i].depth, camera);
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
      //save the pointcloud
      pcl::io::savePCDFile("./result/result_slam.pcd", *tmp);
    
      cout<<"Final map is saved."<<endl;

   return 0;
}