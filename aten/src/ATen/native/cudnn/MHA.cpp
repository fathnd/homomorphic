#include <ATen/ATen.h>
//#ifndef AT_PER_OPERATOR_HEADERS
//#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

}} // namespace at::native

#else // AT_CUDNN_ENABLED
#include <ATen/native/cudnn/MHA.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

#include <ATen/cuda/Exceptions.h>
#include <cudnn_frontend.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <cudnn.h>

#include <iostream>

namespace at { namespace native {

#if (CUDNN_VERSION >= 8900)
#include <cudnn_frontend.h>
void
run_cudnn_LLM_fprop(int64_t b,
                    int64_t h,
                    int64_t s_q,
                    int64_t s_kv,
                    int64_t d,
                    float scaling_factor,
                    bool return_softmaxstats,
		    bool is_causal,
                    double dropout_probability,
                    const Tensor& q,
                    const Tensor& k,
                    const Tensor& v,
                    Tensor& softmaxstats,
                    Tensor& o,
                    Tensor& dropoutseed,
                    Tensor& dropoutoffset) {
    std::cout << "running cuDNN" << std::endl;
    cudnnHandle_t handle = getCudnnHandle();
    namespace fe = cudnn_frontend;
    auto dtype = fe::DataType_t::HALF;
    if (q.scalar_type() == kBFloat16) {
      dtype = fe::DataType_t::HALF;
    }
    o = at::empty_strided({b, h, s_q, d}, {h * d, d, b * h * d, 1}, q.options());
    if (return_softmaxstats) {
      softmaxstats = at::zeros({b, h, s_q}, q.options());
    }
    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    std::vector<int64_t> q_dim;
    std::vector<int64_t> q_stride;
    std::vector<int64_t> k_dim;
    std::vector<int64_t> k_stride;
    std::vector<int64_t> v_dim;
    std::vector<int64_t> v_stride;
    q_dim.assign(q.sizes().data(), q.sizes().data() + q.sizes().size());
    q_stride.assign(q.strides().data(), q.strides().data() + q.strides().size());
    k_dim.assign(k.sizes().data(), k.sizes().data() + k.sizes().size());
    k_stride.assign(k.strides().data(), k.strides().data() + k.strides().size());
    v_dim.assign(v.sizes().data(), v.sizes().data() + v.sizes().size());
    v_stride.assign(v.strides().data(), v.strides().data() + v.strides().size());
    std::cout << q.sizes() << q.strides() << k.sizes() << k.strides() << v.sizes() << v.strides() << std::endl;
    auto Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(q_dim)
                                  .set_stride(q_stride));
    std::cout << "q stride: " << q.strides() << std::endl;
    for (auto it = q_stride.begin(); it != q_stride.end(); it++) std::cout << *it << std::endl;
    std::cout << "k stride: " << k.strides() << std::endl;
    for (auto it = k_stride.begin(); it != k_stride.end(); it++) std::cout << *it << std::endl;
    std::cout << "v stride: " << v.strides() << std::endl;
    for (auto it = v_stride.begin(); it != v_stride.end(); it++) std::cout << *it << std::endl;

    auto K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(k_dim)
                                  .set_stride(k_stride));
    auto V = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim(v_dim)
                                  .set_stride(v_stride));
    auto attn_scale = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("attn_scale")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_is_pass_by_value(true)
                                       .set_data_type(fe::DataType_t::FLOAT));
    //auto bias = mha_graph.tensor(fe::graph::Tensor_attributes()
    //                         .set_name("bias")
    //                         .set_dim({b, 1, s_q, s_kv})
    //                         .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    auto seed = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("Seed")
                                     .set_dim({1, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::INT32));
    auto offset = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("Offset")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
    auto scaled_dot_product_flash_attention_options = fe::graph::Scaled_dot_product_flash_attention_attributes()
                                                          .set_name("flash_attention")
                                                          .set_is_inference(return_softmaxstats == false)
                                                          .set_causal_mask(is_causal)
                                                          .set_attn_scale(attn_scale)
                                                          .set_dropout(dropout_probability, seed, offset);
    // Optional bias in flash attention is only supported 8.9.3 onwards
    if (cudnnGetVersion() >= 8904) {
        //scaled_dot_product_flash_attention_options.set_alibi_mask(true);
    }

    auto seq_q  = mha_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("seq_q")
                                    .set_dim({b, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
    auto seq_kv = mha_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("seq_kv")
                                    .set_dim({b, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
    //if (cudnnGetVersion() >= 8903) {
    //    scaled_dot_product_flash_attention_options.set_bias(bias)
    //        .set_padding_mask(true)
    //        .set_seq_len_q(seq_q)
    //        .set_seq_len_kv(seq_kv);
    //}


    auto [O, Stats] = mha_graph.scaled_dot_product_flash_attention(Q, K, V, scaled_dot_product_flash_attention_options);

    //O->set_output(true).set_stride({h * d, d, b * h * d, 1});
    std::vector<int64_t> o_stride;
    o_stride.assign(o.strides().data(), o.strides().data() + o.strides().size());
    std::cout << "out stride set: " << h*d << " " << d << " " << b * h * d << " " << 1 << std::endl;
    std::cout << "tensor stride: " << o.strides() << std::endl;
    O->set_output(true).set_stride(o_stride);

    // Check that Stats tensor is real, which is only when its training step
    if (Stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

    TORCH_INTERNAL_ASSERT(mha_graph.validate().is_good());

    TORCH_INTERNAL_ASSERT(mha_graph.build_operation_graph(handle).is_good());

    auto plans = mha_graph.get_execution_plan_list({fe::HeurMode_t::A});


    TORCH_INTERNAL_ASSERT(plans.check_support(handle).is_good());

    TORCH_INTERNAL_ASSERT(mha_graph.set_execution_plans(plans).is_good());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = { {Q, q.data_ptr()},
         {K, k.data_ptr()},
         {V, v.data_ptr()},
         {attn_scale, &scaling_factor},
         //{bias, bias.data_ptr()},
         {seed, dropoutseed.data_ptr()},
         {offset, dropoutoffset.data_ptr()},
         {O, o.data_ptr()}};

    if (return_softmaxstats) {
        variant_pack[Stats] = softmaxstats.data_ptr();
    }

    auto workspace_size = mha_graph.get_workspace_size();
    auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
    TORCH_INTERNAL_ASSERT(mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
}


}} // namespace at::native

#endif
#endif
