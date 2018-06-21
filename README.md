## Static_SLAM_based_on_PSPNet/ICNet
This is a refined version of [Yilei's work](https://github.com/yilei0620/RGBD-Slam-Semantic-Seg-DeepLab). 
We seperately test [PSPNet](https://github.com/hszhao/PSPNet) and [ICNet](https://github.com/hszhao/ICNet) to recognize humans from input images and then mask them out.
For 3D map construction, only the unmasked areas are projected into pointcloud.  

Also since the segmentation's qualities can be pretty poor on blurried images, we first used a dilation filter to slightly
amplify the mask. Some stubborn pixels may still exist after the dilation, a points filter is also employed.

## Installation
The package depends on 
[Boost](https://www.boost.org/),
[Opencv3](https://opencv.org/opencv-3-3.html), 
[PCL1.7](http://mobile.pointclouds.org/http://mobile.pointclouds.org/news/2013/07/23/pcl-1.7/) 
and [G2O](https://github.com/RainerKuemmerle/g2o)

For compiling, please follow the <br>
`mkdir build` <br>
`cd build` <br>
`cmake..` <br>
`make`<br>
process. Also please notice that the package only provides CPU version.

## Test
For testing, please `mkdir model` under the generated `bin` folder.
Then copy the `parameters.txt` into `bin` and the `.caffemodel` and `.prototxt` into `bin/model`.
Since the model files are too large, you need to download it from the original PSPNet/ICNet repositories. <br>
Then run `./ICSlam` for the refined result and `./Slam` for the original one.

## Result
#### Origin
![image](https://github.com/SILI1994/static-SLAM-based-on-PSPnet/blob/master/1.png)

#### Refined
![image](https://github.com/SILI1994/static-SLAM-based-on-PSPnet/blob/master/2.png)
