#include "caffe2/operators/tensor_protos_db_input.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(TensorProtosDBInput, TensorProtosDBInput<CPUContext>);
}  // namespace
}  // namespace caffe2
