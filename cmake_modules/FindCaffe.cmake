# Caffe package for CNN Triplet training
unset(Caffe_FOUND)

find_path(Caffe_INCLUDE_DIRS NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/proto/caffe.pb.h caffe/util/io.hpp caffe/layers/memory_data_layer.hpp caffe/blob.hpp
  HINTS
  ~/PSPNet/include)

find_library(Caffe_LIBS NAMES libcaffe.so
  HINTS
  ~/PSPNet/build/lib)

if(Caffe_LIBS AND Caffe_INCLUDE_DIRS)
    set(Caffe_FOUND 1)
endif()
