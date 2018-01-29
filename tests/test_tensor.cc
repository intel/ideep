#include <ideep.hpp>

using namespace ideep;

tensor::descriptor test_tensor_construction(const tensor& t_) {
  auto a = t_.get_descriptor();

  return a;
}

INIT_GLOBAL_ENGINE;

int main() {
  tensor::dims dim1 = {5};
  tensor::dims dim2 = {2, 4};
  tensor::dims dim3 = {2, 2, 3};
  tensor::dims dim4 = { 3, 16, 8, 8 };
  tensor::dims view_dim4 = {1, 16, 8, 8};
  tensor::dims off_dim4 = {1, 0, 0, 0};
  // Error
  tensor::dims dim0 = {0, 16, 8, 8};

  tensor::data_type type = tensor::data_type::f32;

  tensor::descriptor empty;
  tensor empty_t;
  tensor::descriptor adesc(dim0, type);
  tensor::descriptor adesc_chwn(dim0, type,
      static_cast<ideep::format>(mkldnn::memory::format::chwn));
  tensor::descriptor adesc_nhwc(dim0, type,
      static_cast<ideep::format>(mkldnn::memory::format::nhwc));
  tensor atensor(adesc);
  tensor another(std::move(adesc));

  auto ret = test_tensor_construction(atensor);
  auto ret_another = test_tensor_construction(another);
  (void)ret;
  (void)ret_another;
  assert(empty_t.get() != nullptr);
  tensor::dims d = empty_t.get_dims();
  assert(d.size() == 0);

  static const float data[32 * 64 * 8 * 8] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13};

  tensor _3d ({dim3, type});
  tensor _3d_dst({dim3, type});
  tensor src (adesc_nhwc);
  tensor dst (adesc_chwn);
  tensor _4d ({dim4, type});
  tensor _4d_src ({view_dim4, type}, const_cast<float *>(data));
  auto _4d_view = _4d.create_view(view_dim4, off_dim4);
  
  reorder filler;

  filler.reinit(_4d_src.get_descriptor(), _4d_view, _4d.get_descriptor());
  filler(_4d_src, _4d);

  _3d.reorder_from(dim3, type, static_cast<const void *>(data));
  _3d.reorder_to(_3d_dst.get_data_handle());

  src.reorder_from(dim0, type, static_cast<const void *>(data));
  src.reorder_to(dst);

  // auto result = adesc.reshape({2, 4*6*8}).format_to(ideep::format::oi);
  // std::cout<<result.get_size()<<std::endl;
}
