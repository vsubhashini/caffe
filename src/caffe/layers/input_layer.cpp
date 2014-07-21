// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InputLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  const int num_top = top->size();
  const InputParameter& param = this->layer_param_.input_param();
  CHECK(param.num_size() == 1 || param.num_size() == num_top)
      << "Must specify either a single (1) 'num' or one for each top blob "
      << "(" << num_top << "); you specified " << param.num_size() << ".";
  CHECK(param.channels_size() == 1 || param.channels_size() == num_top)
      << "Must specify either a single (1) 'channels' or one for each top blob "
      << "(" << num_top << "); you specified " << param.channels_size() << ".";
  CHECK(param.height_size() == 1 || param.height_size() == num_top)
      << "Must specify either a single (1) 'height' or one for each top blob "
      << "(" << num_top << "); you specified " << param.height_size() << ".";
  CHECK(param.width_size() == 1 || param.width_size() == num_top)
      << "Must specify either a single (1) 'width' or one for each top blob "
      << "(" << num_top << "); you specified " << param.width_size() << ".";
  for (int i = 0; i < num_top; ++i) {
    const int num = param.num_size() > 1 ? param.num(i) : param.num(0);
    const int channels =
        param.channels_size() > 1 ? param.channels(i) : param.channels(0);
    const int height =
        param.height_size() > 1 ? param.height(i) : param.height(0);
    const int width = param.width_size() > 1 ? param.width(i) : param.width(0);
    (*top)[i]->Reshape(num, channels, height, width);
  }
}

INSTANTIATE_CLASS(InputLayer);

}  // namespace caffe
