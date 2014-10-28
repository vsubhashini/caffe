#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/python_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PythonLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PythonLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PythonLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PythonLayerTest, TestDtypesAndDevices);

TYPED_TEST(PythonLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_python_param()->set_module("test_python_layer");
  layer_param.mutable_python_param()->set_layer("MyLayer");
  shared_ptr<PythonLayer<Dtype> > layer(
      new PythonLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(PythonLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_python_param()->set_module("test_python_layer");
  layer_param.mutable_python_param()->set_layer("MyLayer");
  shared_ptr<PythonLayer<Dtype> > layer(
      new PythonLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int count = this->blob_top_->count();
  const Dtype* in_data = this->blob_bottom_->cpu_data();
  const Dtype* out_data = this->blob_top_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(out_data[i], 10 * in_data[i]);
  }
}

TYPED_TEST(PythonLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_python_param()->set_module("test_python_layer");
  layer_param.mutable_python_param()->set_layer("MyLayer");
  PythonLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
