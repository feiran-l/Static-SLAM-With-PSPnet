#include "ICnet.h"


/**
* Pre-process the image into the network required size */
pair<cv::Mat,vector<int>>  imgPreProcess(cv::Mat img){
    
    // resize the input image to size
    cv::Mat tmp_size;
    vector<cv::Mat> channels;
    int height = int(img.rows), width = int(img.cols), newW, newH;
    cout<<"Image sizes are "<<width<<","<<height<<endl;

    if (width > height){
      newW = 473;
      newH = 473;
      cv::resize(img,tmp_size,cv::Size(newW, newH));
    
    }
    else{
      newW = 473;
      newH = 473;
      cv::resize(img,tmp_size,cv::Size(newW,newH));
    }

    return pair<cv::Mat, vector<int> > (tmp_size, vector<int> {newW, newH});
}


/**
* Test func for testing the output of classification */
void CalculateCls(cv::Mat img){
  
    vector<float> element;
    element.push_back(img.at<float>(0, 0));

    for(int row = 0; row < img.rows; row++)
      for(int col = 0; col < img.cols; col++){

        vector<float>::iterator it;
        it = find(element.begin(), element.end(), img.at<float>(row, col));
        if(it == element.end())
           element.push_back(img.at<float>(row, col));  
      }

    //output
    cout<<"*******************************\n"<<endl;
    for(auto i = element.begin(); i < element.end(); i++)
      cout<<*i<<" "<<endl;
    cout<<"\n********************************\n"<<endl;
}


/**
* Translate the segmented image to pointloud, refer_img is the gray_temp img generated in the func above */
PointCloud::Ptr img2PointCloudIC(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMS& camera, cv::Mat& refer_img){

    //PointCloud::Ptr is an auto ptr, new a PointCloud-type 
    PointCloud::Ptr cloud (new PointCloud);
  
    for(int m = 0; m < depth.rows; m+=2)
      for(int n = 0; n < depth.cols; n+=2){
        if(refer_img.at<float>(m, n) == 0){
            //get the depth_value of the (m,n) point
	        ushort d = depth.ptr<ushort>(m)[n];
	        //if the depth cannot be detected, jump this point
	        if(d == 0)
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
        else 
            continue;
    }

    //set and save point cloud
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
  
    return cloud;
}


/**
* Dilation to deal with the edge*/
cv::Mat imgDilation(cv::Mat& img){
  
  cv::Mat dilated;  
  cv::Mat element = cv::getStructuringElement(0, cv::Size(7,7));  
  cv::dilate(img, dilated, element);  
 
  return dilated;
}


/**
* pcl_StatisticalOutlierRemoval filter to deal with the outliers after 2d-dilation*/
PointCloud::Ptr pclSorFilter(PointCloud::Ptr cloud){

  //initialize the filter
  pcl::StatisticalOutlierRemoval<PointT> sor_filter;
  sor_filter.setInputCloud (cloud);
  sor_filter.setMeanK (50);
  sor_filter.setStddevMulThresh (1.0);

  //filtering
  sor_filter.filter(*cloud);

  return cloud;
}



