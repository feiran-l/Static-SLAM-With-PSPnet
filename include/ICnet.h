#pragma once
//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

//Caffe
#include <caffe/caffe.hpp>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/net.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>

//Boost
#include <boost/shared_ptr.hpp>
#include <boost/pointer_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <google/protobuf/text_format.h>

//c++
#include <iostream>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iterator>
using namespace std;

#include "slambase.h"


/*
FUNCTION DECLARATION
*/

//load net
//template <typename Dtype>
//caffe::Net<Dtype>* loadNet(string param_file, string pretrained_param_file, caffe::Phase phase);

template <typename Dtype>
caffe::Net<Dtype>* loadNet(string param_file, string pretrained_param_file, caffe::Phase phase){
    
    caffe::Net<Dtype>* net(new caffe::Net<Dtype>(param_file, phase));
    net->CopyTrainedLayersFrom(pretrained_param_file);

    return net;
}
    
//image preprocess; cut the image into 512*713
pair<cv::Mat,vector<int>>  imgPreProcess (cv::Mat img);

//translate the segmented image to pointloud
PointCloud::Ptr img2PointCloudIC(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMS& camera, cv::Mat& refer_img);

//test func for classify the output of segmentation, not useful in the main code
void CalculateCls(cv::Mat img);

//dilation to deal with the edge
cv::Mat imgDilation(cv::Mat& img);

//pcl_StatisticalOutlierRemoval filter to deal with the outliers after 2d-dilation
PointCloud::Ptr pclSorFilter(PointCloud::Ptr cloud);




