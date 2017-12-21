#LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

#PCL
FIND_PACKAGE( PCL REQUIRED )
LIST(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this for ubuntu 16.04
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

#Opencv
FIND_PACKAGE( OpenCV 3.3.0 REQUIRED )

#G2O
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED )

#CSparse
FIND_PACKAGE( CSparse REQUIRED )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

#Caffe
FIND_PACKAGE(Caffe REQUIRED)
INCLUDE_DIRECTORIES(${Caffe_INCLUDE_DIRS})
ADD_DEFINITIONS(${Caffe_DEFINITIONS})

#Boost
set(Boost_INCLUDE_DIR /usr/include)
set(Boost_LIBRARY_DIR /usr/lib)


ADD_LIBRARY( slambase slambase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension       
    ${CSPARSE_LIBRARY} )

ADD_LIBRARY( ICnet ICnet.cpp )
TARGET_LINK_LIBRARIES( ICnet
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES}
    ${Caffe_LIBS} 
    ${Boost_LIBRARY_DIR}
    boost_system boost_filesystem )

ADD_EXECUTABLE( slam slam.cpp )
TARGET_LINK_LIBRARIES( slam
    slambase
    ${OpenCV_LIBS}  
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension     
    ${CSPARSE_LIBRARY} )

ADD_EXECUTABLE( ICslam ICslam.cpp )
TARGET_LINK_LIBRARIES( ICslam
    ICnet
    slambase
    ${OpenCV_LIBS}  
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension     
    ${CSPARSE_LIBRARY}
    ${Caffe_LIBS} 
    ${Boost_LIBRARY_DIR}
    boost_system boost_filesystem )



