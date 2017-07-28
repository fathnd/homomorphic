#pragma once

#include <memory>
#include <limits>

#include "ATen/ArrayRef.h"
#include "ATen/Half.h"
#include "ATen/SparseTensorRef.h"

namespace at {

class Context;
struct Storage;
struct Tensor;
class Scalar;
struct Generator;

#define AT_FORALL_SCALAR_TYPES(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(double,Double,d) \
_(float,Float,d) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(int16_t,Short,i) \
_(Half,Half,d)

enum class ScalarType {
#define DEFINE_ENUM(_1,n,_2) \
  n,
  AT_FORALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
  NumOptions
};

enum class Backend {
  CPU,
  CUDA,
  SparseCPU,
  SparseCUDA,
  NumOptions
};


constexpr Backend kCPU = Backend::CPU;
constexpr Backend kCUDA = Backend::CUDA;
constexpr Backend kSparseCPU = Backend::SparseCPU;
constexpr Backend kSparseCUDA = Backend::SparseCUDA;

// Note [Undefined-dim versus 0-dim]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Unlike Torch, ATen treats zero-dimension tensors as having ONE
// element (that is to say, a zero-dimensional tensor is a scalar!)
// This is in contrast to Torch, where a zero-dimension tensor has
// zero elements.
//
// Because we are backed by Torch tensors, we need to be able to
// represent this state (of numel==0).  kUndefinedDimensions represents this
// situation.
constexpr int64_t kUndefinedDimensions = std::numeric_limits<int64_t>::min();

static inline const char * toString(Backend b) {
  switch(b) {
    case Backend::CPU: return "CPU";
    case Backend::CUDA: return "CUDA";
    case Backend::SparseCPU: return "SparseCPU";
    case Backend::SparseCUDA: return "SparseCUDA";
    default: return "UNKNOWN_BACKEND";
  }
}

#define DEFINE_CONSTANT(_,name,_2) \
constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

static inline const char * toString(ScalarType t) {
#define DEFINE_CASE(_,name,_2) \
  case ScalarType:: name : return #name;

  switch(t) {
    AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR_TYPE";
  }
#undef DEFINE_CASE
}

enum class TypeID {
  CPUByte,
  CPUChar,
  CPUDouble,
  CPUFloat,
  CPUInt,
  CPULong,
  CPUShort,
  CPUHalf,
  SparseCPUByte,
  SparseCPUChar,
  SparseCPUDouble,
  SparseCPUFloat,
  SparseCPUInt,
  SparseCPULong,
  SparseCPUShort,
  CUDAByte,
  CUDAChar,
  CUDADouble,
  CUDAFloat,
  CUDAInt,
  CUDALong,
  CUDAShort,
  CUDAHalf,
  SparseCUDAByte,
  SparseCUDAChar,
  SparseCUDADouble,
  SparseCUDAFloat,
  SparseCUDAInt,
  SparseCUDALong,
  SparseCUDAShort,
  NumOptions
};


typedef ArrayRef<int64_t> IntList;
typedef ArrayRef<Tensor> TensorList;

struct Type {
  explicit Type(Context * context)
  : context(context) {}
  virtual ~Type() {}
  virtual ScalarType scalarType() = 0;
  virtual Backend backend() = 0;
  virtual bool isCuda() = 0;
  virtual bool isSparse() = 0;
  virtual bool isDistributed() = 0;
  static void registerAll(Context * context);
  virtual std::unique_ptr<Storage> storage() = 0;
  virtual std::unique_ptr<Storage> storage(size_t size) = 0;
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size) = 0;
  virtual std::unique_ptr<Generator> generator() = 0;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) = 0;
  virtual const char * toString() const = 0;
  Type & toBackend(Backend b);
  Type & toScalarType(ScalarType s);

  // contingious IDs for all types in the system
  // for external dispatch
  virtual TypeID ID() const = 0;

  virtual void copy(const Tensor & src, Tensor & dst) = 0;
  Tensor copy(const Tensor & src);

  Tensor tensorFromBlob(void * data, IntList sizes);
  Tensor tensorFromBlob(void * data, IntList sizes, IntList strides);
  Tensor scalarTensor(Scalar s);

  bool operator==(const Type& other) const;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  virtual int64_t m_storage_offset(const Tensor & self) ;
  virtual Tensor & m_resize_(Tensor & self, IntList size) ;
  virtual Tensor & zeros_out(IntList size, Tensor & result) ;
  virtual Tensor zeros(IntList size) ;
  virtual Tensor & ones_out(IntList size, Tensor & result) ;
  virtual Tensor ones(IntList size) ;
  virtual int64_t numel(const Tensor & self) ;
  virtual Tensor & m_set_(Tensor & self, Storage & storage) ;
  virtual Tensor & m_set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) ;
  virtual Tensor & m_set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size) ;
  virtual Tensor & m_set_(Tensor & self, const Tensor & source) ;
  virtual Tensor & m_set_(Tensor & self) ;
  virtual Tensor & m_fill_(Tensor & self, Scalar value) ;
  virtual bool m_is_same_size(const Tensor & self, const Tensor & other) ;
  virtual bool m_is_contiguous(const Tensor & self) ;
  virtual bool m_is_set_to(const Tensor & self, const Tensor & tensor) ;
  virtual Tensor & m_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) ;
  virtual Tensor & m_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) ;
  virtual Tensor & masked_select_out(const Tensor & self, const Tensor & mask, Tensor & result) ;
  virtual Tensor masked_select(const Tensor & self, const Tensor & mask) ;
  virtual Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) ;
  virtual Tensor & m_transpose_(Tensor & self, int64_t dim0, int64_t dim1) ;
  virtual Tensor t(const Tensor & self) ;
  virtual Tensor & m_t_(Tensor & self) ;
  virtual Tensor & squeeze_out(const Tensor & self, int64_t dim, Tensor & result) ;
  virtual Tensor squeeze(const Tensor & self, int64_t dim) ;
  virtual Tensor & squeeze_out(const Tensor & self, Tensor & result) ;
  virtual Tensor squeeze(const Tensor & self) ;
  virtual Tensor & m_squeeze_(Tensor & self, int64_t dim) ;
  virtual Tensor & m_squeeze_(Tensor & self) ;
  virtual Tensor & unsqueeze_out(const Tensor & self, int64_t dim, Tensor & result) ;
  virtual Tensor unsqueeze(const Tensor & self, int64_t dim) ;
  virtual Tensor & m_unsqueeze_(Tensor & self, int64_t dim) ;
  virtual Tensor & nonzero_out(const Tensor & self, Tensor & result) ;
  virtual Tensor nonzero(const Tensor & self) ;
  virtual Tensor m_contiguous(const Tensor & self) ;
  virtual Tensor m_clone(const Tensor & self) ;
  virtual Tensor m_view(const Tensor & self, IntList size) ;
  virtual Tensor m_expand(const Tensor & self, IntList size) ;
  virtual Tensor & m_resize_as_(Tensor & self, const Tensor & the_template) ;
  virtual Tensor & index_select_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) ;
  virtual Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) ;
  virtual Tensor & m_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) ;
  virtual Tensor & m_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) ;
  virtual Tensor & m_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) ;
  virtual Tensor m_narrow(const Tensor & self, int64_t dimension, int64_t start, int64_t length) ;
  virtual Tensor m_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) ;
  virtual Tensor & range_out(Scalar start, Scalar end, Scalar step, Tensor & result) ;
  virtual Tensor range(Scalar start, Scalar end, Scalar step) ;
  virtual Tensor & range_out(Scalar start, Scalar end, Tensor & result) ;
  virtual Tensor range(Scalar start, Scalar end) ;
  virtual Tensor & m_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) ;
  virtual Tensor & m_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) ;
  virtual Tensor & m_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) ;
  virtual Tensor & gather_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) ;
  virtual Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) ;
  virtual void* m_data_ptr(const Tensor & self) ;
  virtual bool equal(const Tensor & self, const Tensor & other) ;
  virtual Tensor & __and___out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor __and__(const Tensor & self, Scalar value) ;
  virtual Tensor & __and___out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor __and__(const Tensor & self, const Tensor & other) ;
  virtual Tensor & __iand__(Tensor & self, Scalar value) ;
  virtual Tensor & __iand__(Tensor & self, const Tensor & other) ;
  virtual Tensor & __or___out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor __or__(const Tensor & self, Scalar value) ;
  virtual Tensor & __or___out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor __or__(const Tensor & self, const Tensor & other) ;
  virtual Tensor & __ior__(Tensor & self, Scalar value) ;
  virtual Tensor & __ior__(Tensor & self, const Tensor & other) ;
  virtual Tensor & __xor___out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor __xor__(const Tensor & self, Scalar value) ;
  virtual Tensor & __xor___out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor __xor__(const Tensor & self, const Tensor & other) ;
  virtual Tensor & __ixor__(Tensor & self, Scalar value) ;
  virtual Tensor & __ixor__(Tensor & self, const Tensor & other) ;
  virtual Tensor & __lshift___out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor __lshift__(const Tensor & self, Scalar value) ;
  virtual Tensor & __lshift___out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor __lshift__(const Tensor & self, const Tensor & other) ;
  virtual Tensor & __ilshift__(Tensor & self, Scalar value) ;
  virtual Tensor & __ilshift__(Tensor & self, const Tensor & other) ;
  virtual Tensor & __rshift___out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor __rshift__(const Tensor & self, Scalar value) ;
  virtual Tensor & __rshift___out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor __rshift__(const Tensor & self, const Tensor & other) ;
  virtual Tensor & __irshift__(Tensor & self, Scalar value) ;
  virtual Tensor & __irshift__(Tensor & self, const Tensor & other) ;
  virtual Tensor m_lt(const Tensor & self, Scalar value) ;
  virtual Tensor m_lt(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_lt_(Tensor & self, Scalar value) ;
  virtual Tensor & m_lt_(Tensor & self, const Tensor & other) ;
  virtual Tensor & lt_out(const Tensor & tensor, Scalar value, Tensor & result) ;
  virtual Tensor lt(const Tensor & tensor, Scalar value) ;
  virtual Tensor & lt_out(const Tensor & tensor, const Tensor & other, Tensor & result) ;
  virtual Tensor lt(const Tensor & tensor, const Tensor & other) ;
  virtual Tensor m_gt(const Tensor & self, Scalar value) ;
  virtual Tensor m_gt(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_gt_(Tensor & self, Scalar value) ;
  virtual Tensor & m_gt_(Tensor & self, const Tensor & other) ;
  virtual Tensor & gt_out(const Tensor & tensor, Scalar value, Tensor & result) ;
  virtual Tensor gt(const Tensor & tensor, Scalar value) ;
  virtual Tensor & gt_out(const Tensor & tensor, const Tensor & other, Tensor & result) ;
  virtual Tensor gt(const Tensor & tensor, const Tensor & other) ;
  virtual Tensor m_le(const Tensor & self, Scalar value) ;
  virtual Tensor m_le(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_le_(Tensor & self, Scalar value) ;
  virtual Tensor & m_le_(Tensor & self, const Tensor & other) ;
  virtual Tensor & le_out(const Tensor & tensor, Scalar value, Tensor & result) ;
  virtual Tensor le(const Tensor & tensor, Scalar value) ;
  virtual Tensor & le_out(const Tensor & tensor, const Tensor & other, Tensor & result) ;
  virtual Tensor le(const Tensor & tensor, const Tensor & other) ;
  virtual Tensor m_ge(const Tensor & self, Scalar value) ;
  virtual Tensor m_ge(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_ge_(Tensor & self, Scalar value) ;
  virtual Tensor & m_ge_(Tensor & self, const Tensor & other) ;
  virtual Tensor & ge_out(const Tensor & tensor, Scalar value, Tensor & result) ;
  virtual Tensor ge(const Tensor & tensor, Scalar value) ;
  virtual Tensor & ge_out(const Tensor & tensor, const Tensor & other, Tensor & result) ;
  virtual Tensor ge(const Tensor & tensor, const Tensor & other) ;
  virtual Tensor m_eq(const Tensor & self, Scalar value) ;
  virtual Tensor m_eq(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_eq_(Tensor & self, Scalar value) ;
  virtual Tensor & m_eq_(Tensor & self, const Tensor & other) ;
  virtual Tensor & eq_out(const Tensor & tensor, Scalar value, Tensor & result) ;
  virtual Tensor eq(const Tensor & tensor, Scalar value) ;
  virtual Tensor & eq_out(const Tensor & tensor, const Tensor & other, Tensor & result) ;
  virtual Tensor eq(const Tensor & tensor, const Tensor & other) ;
  virtual Tensor m_ne(const Tensor & self, Scalar value) ;
  virtual Tensor m_ne(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_ne_(Tensor & self, Scalar value) ;
  virtual Tensor & m_ne_(Tensor & self, const Tensor & other) ;
  virtual Tensor & ne_out(const Tensor & tensor, Scalar value, Tensor & result) ;
  virtual Tensor ne(const Tensor & tensor, Scalar value) ;
  virtual Tensor & ne_out(const Tensor & tensor, const Tensor & other, Tensor & result) ;
  virtual Tensor ne(const Tensor & tensor, const Tensor & other) ;
  virtual std::tuple<Tensor &,Tensor &> min_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices) ;
  virtual std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual std::tuple<Tensor &,Tensor &> min_out(const Tensor & self, int64_t dim, Tensor & min, Tensor & min_indices) ;
  virtual std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim) ;
  virtual Tensor & min_out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor min(const Tensor & self, const Tensor & other) ;
  virtual Scalar min(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> max_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & max, Tensor & max_indices) ;
  virtual std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual std::tuple<Tensor &,Tensor &> max_out(const Tensor & self, int64_t dim, Tensor & max, Tensor & max_indices) ;
  virtual std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim) ;
  virtual Tensor & max_out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor max(const Tensor & self, const Tensor & other) ;
  virtual Scalar max(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, bool keepdim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, bool keepdim) ;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k) ;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) ;
  virtual std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim) ;
  virtual std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, bool keepdim) ;
  virtual std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim) ;
  virtual std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> median(const Tensor & self, bool keepdim) ;
  virtual std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim) ;
  virtual std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual Scalar median(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim) ;
  virtual std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) ;
  virtual std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k) ;
  virtual std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) ;
  virtual std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest) ;
  virtual std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices) ;
  virtual std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim) ;
  virtual bool m_all(const Tensor & self) ;
  virtual bool m_any(const Tensor & self) ;
  virtual int64_t m_get_device(const Tensor & self) ;
  virtual Tensor & abs_out(const Tensor & self, Tensor & destination) ;
  virtual Tensor abs(const Tensor & self) ;
  virtual Tensor & m_abs_(Tensor & self) ;
  virtual Tensor & m_sigmoid_(Tensor & self) ;
  virtual Tensor & sigmoid_out(const Tensor & self, Tensor & result) ;
  virtual Tensor sigmoid(const Tensor & self) ;
  virtual Tensor & m_log_(Tensor & self) ;
  virtual Tensor & log_out(const Tensor & self, Tensor & result) ;
  virtual Tensor log(const Tensor & self) ;
  virtual Tensor & m_log1p_(Tensor & self) ;
  virtual Tensor & log1p_out(const Tensor & self, Tensor & result) ;
  virtual Tensor log1p(const Tensor & self) ;
  virtual Tensor & lgamma_out(const Tensor & self, Tensor & result) ;
  virtual Tensor lgamma(const Tensor & self) ;
  virtual Tensor & m_lgamma_(Tensor & self) ;
  virtual Tensor & m_exp_(Tensor & self) ;
  virtual Tensor & exp_out(const Tensor & self, Tensor & result) ;
  virtual Tensor exp(const Tensor & self) ;
  virtual Tensor & m_cos_(Tensor & self) ;
  virtual Tensor & cos_out(const Tensor & self, Tensor & result) ;
  virtual Tensor cos(const Tensor & self) ;
  virtual Tensor & m_acos_(Tensor & self) ;
  virtual Tensor & acos_out(const Tensor & self, Tensor & result) ;
  virtual Tensor acos(const Tensor & self) ;
  virtual Tensor & m_cosh_(Tensor & self) ;
  virtual Tensor & cosh_out(const Tensor & self, Tensor & result) ;
  virtual Tensor cosh(const Tensor & self) ;
  virtual Tensor & m_sin_(Tensor & self) ;
  virtual Tensor & sin_out(const Tensor & self, Tensor & result) ;
  virtual Tensor sin(const Tensor & self) ;
  virtual Tensor & m_asin_(Tensor & self) ;
  virtual Tensor & asin_out(const Tensor & self, Tensor & result) ;
  virtual Tensor asin(const Tensor & self) ;
  virtual Tensor & m_sinh_(Tensor & self) ;
  virtual Tensor & sinh_out(const Tensor & self, Tensor & result) ;
  virtual Tensor sinh(const Tensor & self) ;
  virtual Tensor & m_tan_(Tensor & self) ;
  virtual Tensor & tan_out(const Tensor & self, Tensor & result) ;
  virtual Tensor tan(const Tensor & self) ;
  virtual Tensor & m_atan_(Tensor & self) ;
  virtual Tensor & atan_out(const Tensor & self, Tensor & result) ;
  virtual Tensor atan(const Tensor & self) ;
  virtual Tensor & m_tanh_(Tensor & self) ;
  virtual Tensor & tanh_out(const Tensor & self, Tensor & result) ;
  virtual Tensor tanh(const Tensor & self) ;
  virtual Tensor & m_sqrt_(Tensor & self) ;
  virtual Tensor & sqrt_out(const Tensor & self, Tensor & result) ;
  virtual Tensor sqrt(const Tensor & self) ;
  virtual Tensor & m_rsqrt_(Tensor & self) ;
  virtual Tensor & rsqrt_out(const Tensor & self, Tensor & result) ;
  virtual Tensor rsqrt(const Tensor & self) ;
  virtual Tensor & m_ceil_(Tensor & self) ;
  virtual Tensor & ceil_out(const Tensor & self, Tensor & result) ;
  virtual Tensor ceil(const Tensor & self) ;
  virtual Tensor & m_floor_(Tensor & self) ;
  virtual Tensor & floor_out(const Tensor & self, Tensor & result) ;
  virtual Tensor floor(const Tensor & self) ;
  virtual Tensor & m_round_(Tensor & self) ;
  virtual Tensor & round_out(const Tensor & self, Tensor & result) ;
  virtual Tensor round(const Tensor & self) ;
  virtual Tensor & m_trunc_(Tensor & self) ;
  virtual Tensor & trunc_out(const Tensor & self, Tensor & result) ;
  virtual Tensor trunc(const Tensor & self) ;
  virtual Tensor & m_frac_(Tensor & self) ;
  virtual Tensor & frac_out(const Tensor & self, Tensor & result) ;
  virtual Tensor frac(const Tensor & self) ;
  virtual Tensor & mean_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) ;
  virtual Tensor mean(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual Tensor & mean_out(const Tensor & self, int64_t dim, Tensor & destination) ;
  virtual Tensor mean(const Tensor & self, int64_t dim) ;
  virtual Scalar mean(const Tensor & self) ;
  virtual Tensor & var_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination) ;
  virtual Tensor var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) ;
  virtual Tensor & var_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) ;
  virtual Tensor var(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual Tensor & var_out(const Tensor & self, int64_t dim, Tensor & destination) ;
  virtual Tensor var(const Tensor & self, int64_t dim) ;
  virtual Scalar var(const Tensor & self, bool unbiased) ;
  virtual Scalar var(const Tensor & self) ;
  virtual Tensor & std_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination) ;
  virtual Tensor std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) ;
  virtual Tensor & std_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) ;
  virtual Tensor std(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual Tensor & std_out(const Tensor & self, int64_t dim, Tensor & destination) ;
  virtual Tensor std(const Tensor & self, int64_t dim) ;
  virtual Scalar std(const Tensor & self, bool unbiased) ;
  virtual Scalar std(const Tensor & self) ;
  virtual Tensor & norm_out(const Tensor & self, Scalar p, int64_t dim, bool keepdim, Tensor & destination) ;
  virtual Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) ;
  virtual Tensor & norm_out(const Tensor & self, Scalar p, int64_t dim, Tensor & destination) ;
  virtual Tensor norm(const Tensor & self, Scalar p, int64_t dim) ;
  virtual Scalar norm(const Tensor & self, Scalar p) ;
  virtual Scalar norm(const Tensor & self) ;
  virtual Tensor & renorm_out(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm, Tensor & destination) ;
  virtual Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) ;
  virtual Tensor & m_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) ;
  virtual Scalar dist(const Tensor & self, const Tensor & other, Scalar p) ;
  virtual Scalar dist(const Tensor & self, const Tensor & other) ;
  virtual Tensor & reciprocal_out(const Tensor & self, Tensor & destination) ;
  virtual Tensor reciprocal(const Tensor & self) ;
  virtual Tensor & m_reciprocal_(Tensor & self) ;
  virtual Tensor & neg_out(const Tensor & self, Tensor & destination) ;
  virtual Tensor neg(const Tensor & self) ;
  virtual Tensor & m_neg_(Tensor & self) ;
  virtual Tensor & atan2_out(const Tensor & self, const Tensor & other, Tensor & destination) ;
  virtual Tensor atan2(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_atan2_(Tensor & self, const Tensor & other) ;
  virtual Tensor & pow_out(const Tensor & self, Scalar exponent, Tensor & destination) ;
  virtual Tensor pow(const Tensor & self, Scalar exponent) ;
  virtual Tensor & pow_out(const Tensor & self, const Tensor & exponent, Tensor & destination) ;
  virtual Tensor pow(const Tensor & self, const Tensor & exponent) ;
  virtual Tensor & m_pow_(Tensor & self, Scalar exponent) ;
  virtual Tensor & m_pow_(Tensor & self, const Tensor & exponent) ;
  virtual Tensor & lerp_out(const Tensor & self, const Tensor & end, Scalar weight, Tensor & destination) ;
  virtual Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight) ;
  virtual Tensor & m_lerp_(Tensor & self, const Tensor & end, Scalar weight) ;
  virtual Tensor & linspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result) ;
  virtual Tensor linspace(Scalar start, Scalar end, int64_t steps) ;
  virtual Tensor & linspace_out(Scalar start, Scalar end, Tensor & result) ;
  virtual Tensor linspace(Scalar start, Scalar end) ;
  virtual Tensor & logspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result) ;
  virtual Tensor logspace(Scalar start, Scalar end, int64_t steps) ;
  virtual Tensor & logspace_out(Scalar start, Scalar end, Tensor & result) ;
  virtual Tensor logspace(Scalar start, Scalar end) ;
  virtual Tensor & histc_out(const Tensor & self, Tensor & destination) ;
  virtual Tensor histc(const Tensor & self) ;
  virtual Tensor & histc_out(const Tensor & self, int64_t bins, Tensor & destination) ;
  virtual Tensor histc(const Tensor & self, int64_t bins) ;
  virtual Tensor & histc_out(const Tensor & self, int64_t bins, Scalar min, Tensor & destination) ;
  virtual Tensor histc(const Tensor & self, int64_t bins, Scalar min) ;
  virtual Tensor & histc_out(const Tensor & self, int64_t bins, Scalar min, Scalar max, Tensor & destination) ;
  virtual Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) ;
  virtual Tensor & m_zero_(Tensor & self) ;
  virtual Tensor & sum_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result) ;
  virtual Tensor sum(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual Tensor & sum_out(const Tensor & self, int64_t dim, Tensor & result) ;
  virtual Tensor sum(const Tensor & self, int64_t dim) ;
  virtual Scalar sum(const Tensor & self) ;
  virtual Tensor & prod_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result) ;
  virtual Tensor prod(const Tensor & self, int64_t dim, bool keepdim) ;
  virtual Tensor & prod_out(const Tensor & self, int64_t dim, Tensor & result) ;
  virtual Tensor prod(const Tensor & self, int64_t dim) ;
  virtual Scalar prod(const Tensor & self) ;
  virtual Tensor & cumsum_out(const Tensor & self, int64_t dim, Tensor & result) ;
  virtual Tensor cumsum(const Tensor & self, int64_t dim) ;
  virtual Tensor & cumprod_out(const Tensor & self, int64_t dim, Tensor & result) ;
  virtual Tensor cumprod(const Tensor & self, int64_t dim) ;
  virtual Tensor & sign_out(const Tensor & self, Tensor & result) ;
  virtual Tensor sign(const Tensor & self) ;
  virtual Tensor & m_sign_(Tensor & self) ;
  virtual Scalar trace(const Tensor & self) ;
  virtual Tensor & add_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) ;
  virtual Tensor add(const Tensor & self, Scalar value, const Tensor & other) ;
  virtual Tensor & add_out(const Tensor & self, Scalar value, SparseTensor other, Tensor & result) ;
  virtual Tensor add(const Tensor & self, Scalar value, SparseTensor other) ;
  virtual Tensor & add_out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor add(const Tensor & self, Scalar value) ;
  virtual Tensor & add_out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor add(const Tensor & self, const Tensor & other) ;
  virtual Tensor & add_out(const Tensor & self, SparseTensor other, Tensor & result) ;
  virtual Tensor add(const Tensor & self, SparseTensor other) ;
  virtual Tensor & m_add_(Tensor & self, Scalar value, const Tensor & other) ;
  virtual Tensor & m_add_(Tensor & self, Scalar value, SparseTensor other) ;
  virtual Tensor & m_add_(Tensor & self, Scalar value) ;
  virtual Tensor & m_add_(Tensor & self, const Tensor & other) ;
  virtual Tensor & m_add_(Tensor & self, SparseTensor other) ;
  virtual Tensor & sub_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) ;
  virtual Tensor sub(const Tensor & self, Scalar value, const Tensor & other) ;
  virtual Tensor & sub_out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor sub(const Tensor & self, Scalar value) ;
  virtual Tensor & sub_out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor sub(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_sub_(Tensor & self, Scalar value, const Tensor & other) ;
  virtual Tensor & m_sub_(Tensor & self, Scalar value) ;
  virtual Tensor & m_sub_(Tensor & self, const Tensor & other) ;
  virtual Tensor & mul_out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor mul(const Tensor & self, Scalar value) ;
  virtual Tensor & mul_out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor mul(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_mul_(Tensor & self, Scalar value) ;
  virtual Tensor & m_mul_(Tensor & self, const Tensor & other) ;
  virtual Tensor & div_out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor div(const Tensor & self, Scalar value) ;
  virtual Tensor & div_out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor div(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_div_(Tensor & self, Scalar value) ;
  virtual Tensor & m_div_(Tensor & self, const Tensor & other) ;
  virtual Tensor & fmod_out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor fmod(const Tensor & self, Scalar value) ;
  virtual Tensor & fmod_out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor fmod(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_fmod_(Tensor & self, Scalar value) ;
  virtual Tensor & m_fmod_(Tensor & self, const Tensor & other) ;
  virtual Tensor & remainder_out(const Tensor & self, Scalar value, Tensor & result) ;
  virtual Tensor remainder(const Tensor & self, Scalar value) ;
  virtual Tensor & remainder_out(const Tensor & self, const Tensor & other, Tensor & result) ;
  virtual Tensor remainder(const Tensor & self, const Tensor & other) ;
  virtual Tensor & m_remainder_(Tensor & self, Scalar value) ;
  virtual Tensor & m_remainder_(Tensor & self, const Tensor & other) ;
  virtual Tensor & clamp_out(const Tensor & self, Scalar min, Scalar max, Tensor & destination) ;
  virtual Tensor clamp(const Tensor & self, Scalar min, Scalar max) ;
  virtual Tensor & clamp_out(const Tensor & self, Scalar min, Tensor & result) ;
  virtual Tensor clamp(const Tensor & self, Scalar min) ;
  virtual Tensor & m_clamp_(Tensor & self, Scalar min, Scalar max) ;
  virtual Tensor & m_clamp_(Tensor & self, Scalar min) ;
  virtual Scalar dot(const Tensor & self, const Tensor & tensor) ;
  virtual Tensor & tril_out(const Tensor & self, int64_t diagonal, Tensor & destination) ;
  virtual Tensor tril(const Tensor & self, int64_t diagonal) ;
  virtual Tensor & tril_out(const Tensor & self, Tensor & destination) ;
  virtual Tensor tril(const Tensor & self) ;
  virtual Tensor & m_tril_(Tensor & self, int64_t diagonal) ;
  virtual Tensor & m_tril_(Tensor & self) ;
  virtual Tensor & triu_out(const Tensor & self, int64_t diagonal, Tensor & destination) ;
  virtual Tensor triu(const Tensor & self, int64_t diagonal) ;
  virtual Tensor & triu_out(const Tensor & self, Tensor & destination) ;
  virtual Tensor triu(const Tensor & self) ;
  virtual Tensor & m_triu_(Tensor & self, int64_t diagonal) ;
  virtual Tensor & m_triu_(Tensor & self) ;
  virtual Tensor & cross_out(const Tensor & self, const Tensor & other, int64_t dim, Tensor & destination) ;
  virtual Tensor cross(const Tensor & self, const Tensor & other, int64_t dim) ;
  virtual Tensor & cross_out(const Tensor & self, const Tensor & other, Tensor & destination) ;
  virtual Tensor cross(const Tensor & self, const Tensor & other) ;
  virtual Tensor & eye_out(int64_t n, Tensor & result) ;
  virtual Tensor eye(int64_t n) ;
  virtual Tensor & eye_out(int64_t n, int64_t m, Tensor & result) ;
  virtual Tensor eye(int64_t n, int64_t m) ;
  virtual Tensor & diag_out(const Tensor & self, int64_t diagonal, Tensor & result) ;
  virtual Tensor diag(const Tensor & self, int64_t diagonal) ;
  virtual Tensor & diag_out(const Tensor & self, Tensor & result) ;
  virtual Tensor diag(const Tensor & self) ;
  virtual Tensor & addmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor & result) ;
  virtual Tensor addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) ;
  virtual Tensor & addmm_out(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result) ;
  virtual Tensor addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) ;
  virtual Tensor & addmm_out(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result) ;
  virtual Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2) ;
  virtual Tensor & m_addmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & mat1, const Tensor & mat2) ;
  virtual Tensor & m_addmm_(Tensor & self, Scalar beta, const Tensor & mat1, const Tensor & mat2) ;
  virtual Tensor & m_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2) ;
  virtual Tensor & addmv_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor & result) ;
  virtual Tensor addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) ;
  virtual Tensor & addmv_out(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result) ;
  virtual Tensor addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec) ;
  virtual Tensor & addmv_out(const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result) ;
  virtual Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec) ;
  virtual Tensor & m_addmv_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & mat, const Tensor & vec) ;
  virtual Tensor & m_addmv_(Tensor & self, Scalar beta, const Tensor & mat, const Tensor & vec) ;
  virtual Tensor & m_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec) ;
  virtual Tensor & addr_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor & result) ;
  virtual Tensor addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) ;
  virtual Tensor & addr_out(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result) ;
  virtual Tensor addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2) ;
  virtual Tensor & addr_out(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result) ;
  virtual Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2) ;
  virtual Tensor & m_addr_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & vec1, const Tensor & vec2) ;
  virtual Tensor & m_addr_(Tensor & self, Scalar beta, const Tensor & vec1, const Tensor & vec2) ;
  virtual Tensor & m_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2) ;
  virtual Tensor & ger_out(const Tensor & self, const Tensor & vec2, Tensor & result) ;
  virtual Tensor ger(const Tensor & self, const Tensor & vec2) ;
  virtual Tensor & mv_out(const Tensor & self, const Tensor & vec, Tensor & result) ;
  virtual Tensor mv(const Tensor & self, const Tensor & vec) ;
  virtual Tensor & mm_out(const Tensor & self, const Tensor & mat2, Tensor & result) ;
  virtual Tensor mm(const Tensor & self, const Tensor & mat2) ;
  virtual Tensor & bmm_out(const Tensor & self, const Tensor & mat2, Tensor & result) ;
  virtual Tensor bmm(const Tensor & self, const Tensor & mat2) ;
  virtual Tensor & addbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) ;
  virtual Tensor addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & addbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) ;
  virtual Tensor addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & addbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) ;
  virtual Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & m_addbmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & m_addbmm_(Tensor & self, Scalar beta, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & m_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & baddbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) ;
  virtual Tensor baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & baddbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) ;
  virtual Tensor baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & baddbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) ;
  virtual Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & m_baddbmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & m_baddbmm_(Tensor & self, Scalar beta, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & m_baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2) ;
  virtual Tensor & addcmul_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) ;
  virtual Tensor addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) ;
  virtual Tensor & addcmul_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) ;
  virtual Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2) ;
  virtual Tensor & m_addcmul_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) ;
  virtual Tensor & m_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2) ;
  virtual Tensor & addcdiv_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) ;
  virtual Tensor addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) ;
  virtual Tensor & addcdiv_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) ;
  virtual Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2) ;
  virtual Tensor & m_addcdiv_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) ;
  virtual Tensor & m_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2) ;
  virtual std::tuple<Tensor &,Tensor &> gesv_out(const Tensor & self, const Tensor & A, Tensor & solution, Tensor & lu) ;
  virtual std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A) ;
  virtual std::tuple<Tensor &,Tensor &> gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A) ;
  virtual std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) ;
  virtual std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose) ;
  virtual std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper) ;
  virtual std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A) ;
  virtual std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, bool eigenvectors, bool upper, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) ;
  virtual std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors) ;
  virtual std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> symeig(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> eig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) ;
  virtual std::tuple<Tensor &,Tensor &> eig_out(const Tensor & self, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> eig(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> svd_out(const Tensor & self, bool some, Tensor & res1, Tensor & res2, Tensor & res3) ;
  virtual std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some) ;
  virtual std::tuple<Tensor &,Tensor &,Tensor &> svd_out(const Tensor & self, Tensor & res1, Tensor & res2, Tensor & res3) ;
  virtual std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self) ;
  virtual Tensor & inverse_out(const Tensor & self, Tensor & output) ;
  virtual Tensor inverse(const Tensor & self) ;
  virtual Tensor & potrf_out(const Tensor & self, bool upper, Tensor & output) ;
  virtual Tensor potrf(const Tensor & self, bool upper) ;
  virtual Tensor & potrf_out(const Tensor & self, Tensor & output) ;
  virtual Tensor potrf(const Tensor & self) ;
  virtual Tensor & potrs_out(const Tensor & self, const Tensor & input2, bool upper, Tensor & result) ;
  virtual Tensor potrs(const Tensor & self, const Tensor & input2, bool upper) ;
  virtual Tensor & potrs_out(const Tensor & self, const Tensor & input2, Tensor & result) ;
  virtual Tensor potrs(const Tensor & self, const Tensor & input2) ;
  virtual Tensor & potri_out(const Tensor & self, bool upper, Tensor & output) ;
  virtual Tensor potri(const Tensor & self, bool upper) ;
  virtual Tensor & potri_out(const Tensor & self, Tensor & output) ;
  virtual Tensor potri(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, bool upper, Scalar tol, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper, Scalar tol) ;
  virtual std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, bool upper, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper) ;
  virtual std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, Scalar tol, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> pstrf(const Tensor & self, Scalar tol) ;
  virtual std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> pstrf(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> qr_out(const Tensor & self, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> qr(const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> geqrf_out(const Tensor & self, Tensor & res1, Tensor & res2) ;
  virtual std::tuple<Tensor,Tensor> geqrf(const Tensor & self) ;
  virtual std::tuple<Tensor &,const Tensor &> orgqr_out(const Tensor & self, const Tensor & input2, Tensor & result) ;
  virtual std::tuple<Tensor,const Tensor &> orgqr(const Tensor & self, const Tensor & input2) ;
  virtual std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose, Tensor & result) ;
  virtual std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) ;
  virtual std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, Tensor & result) ;
  virtual std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left) ;
  virtual std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, Tensor & result) ;
  virtual std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3) ;
  virtual std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & info, bool pivot, const Tensor & self, Tensor & result, Tensor & pivots) ;
  virtual std::tuple<Tensor,Tensor> btrifact(const Tensor & info, bool pivot, const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & info, const Tensor & self, Tensor & result, Tensor & pivots) ;
  virtual std::tuple<Tensor,Tensor> btrifact(const Tensor & info, const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> btrifact_out(bool pivot, const Tensor & self, Tensor & result, Tensor & pivots) ;
  virtual std::tuple<Tensor,Tensor> btrifact(bool pivot, const Tensor & self) ;
  virtual std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & self, Tensor & result, Tensor & pivots) ;
  virtual std::tuple<Tensor,Tensor> btrifact(const Tensor & self) ;
  virtual Tensor & btrisolve_out(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots, Tensor & result) ;
  virtual Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) ;
  virtual Tensor & randperm_out(Generator & generator, int64_t n, Tensor & result) ;
  virtual Tensor randperm(Generator & generator, int64_t n) ;
  virtual Tensor & randperm_out(int64_t n, Tensor & result) ;
  virtual Tensor randperm(int64_t n) ;
  virtual Tensor & multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement, Tensor & result) ;
  virtual Tensor multinomial(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement) ;
  virtual Tensor & multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, Tensor & result) ;
  virtual Tensor multinomial(Generator & generator, const Tensor & self, int64_t num_samples) ;
  virtual Tensor & multinomial_out(const Tensor & self, int64_t num_samples, bool replacement, Tensor & result) ;
  virtual Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement) ;
  virtual Tensor & multinomial_out(const Tensor & self, int64_t num_samples, Tensor & result) ;
  virtual Tensor multinomial(const Tensor & self, int64_t num_samples) ;
  virtual Tensor & m_uniform_(Tensor & self, Generator & generator, double from, double to) ;
  virtual Tensor & m_uniform_(Tensor & self, Generator & generator, double from) ;
  virtual Tensor & m_uniform_(Tensor & self, double from, double to) ;
  virtual Tensor & m_uniform_(Tensor & self, Generator & generator) ;
  virtual Tensor & m_uniform_(Tensor & self, double from) ;
  virtual Tensor & m_uniform_(Tensor & self) ;
  virtual Tensor & m_cauchy_(Tensor & self, Generator & generator, double median, double sigma) ;
  virtual Tensor & m_cauchy_(Tensor & self, Generator & generator, double median) ;
  virtual Tensor & m_cauchy_(Tensor & self, double median, double sigma) ;
  virtual Tensor & m_cauchy_(Tensor & self, Generator & generator) ;
  virtual Tensor & m_cauchy_(Tensor & self, double median) ;
  virtual Tensor & m_cauchy_(Tensor & self) ;
  virtual Tensor & m_log_normal_(Tensor & self, Generator & generator, double mean, double std) ;
  virtual Tensor & m_log_normal_(Tensor & self, Generator & generator, double mean) ;
  virtual Tensor & m_log_normal_(Tensor & self, double mean, double std) ;
  virtual Tensor & m_log_normal_(Tensor & self, Generator & generator) ;
  virtual Tensor & m_log_normal_(Tensor & self, double mean) ;
  virtual Tensor & m_log_normal_(Tensor & self) ;
  virtual Tensor & rand_out(Generator & generator, IntList size, Tensor & result) ;
  virtual Tensor rand(Generator & generator, IntList size) ;
  virtual Tensor & rand_out(IntList size, Tensor & result) ;
  virtual Tensor rand(IntList size) ;
  virtual Tensor & randn_out(Generator & generator, IntList size, Tensor & result) ;
  virtual Tensor randn(Generator & generator, IntList size) ;
  virtual Tensor & randn_out(IntList size, Tensor & result) ;
  virtual Tensor randn(IntList size) ;
  virtual Tensor & m_geometric_(Tensor & self, Generator & generator, double p) ;
  virtual Tensor & m_geometric_(Tensor & self, double p) ;
  virtual int64_t m_size(const Tensor & self, int64_t dim) ;
  virtual int64_t m_stride(const Tensor & self, int64_t dim) ;
  virtual Tensor tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) ;
  virtual Tensor tensor(Storage & storage, int64_t storageOffset, IntList size) ;
  virtual Tensor tensor(IntList size, IntList stride) ;
  virtual Tensor tensor(IntList size) ;
  virtual Tensor tensor() ;
  virtual Tensor & select_out(const Tensor & self, int dim, int64_t sliceIndex, Tensor & result) ;
  virtual Tensor select(const Tensor & self, int dim, int64_t sliceIndex) ;
  virtual Tensor & m_assign_(Tensor & self, const Tensor & src) ;
  virtual Tensor & cat_out(TensorList tensors, int dim, Tensor & self) ;
  virtual Tensor cat(TensorList tensors, int dim) ;
  virtual void Abs_updateOutput(const Tensor & input, const Tensor & output) ;
  virtual void Abs_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) ;
  virtual void AbsCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) ;
  virtual void AbsCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) ;
  virtual void BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights) ;
  virtual void BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) ;
  virtual void BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights) ;
  virtual void BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) ;
  virtual void ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) ;
  virtual void ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) ;
  virtual void ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) ;
  virtual void ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) ;
  virtual void SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) ;
  virtual void SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) ;
  virtual void SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) ;
  virtual void SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) ;
  virtual void ELU_updateOutput(const Tensor & input, const Tensor & output, Scalar alpha, bool inplace) ;
  virtual void ELU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar alpha, bool inplace) ;
  virtual void DistKLDivCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) ;
  virtual void DistKLDivCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) ;
  virtual void GatedLinear_updateOutput(const Tensor & input, const Tensor & output, int dim) ;
  virtual void GatedLinear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int dim) ;
  virtual void HardShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda) ;
  virtual void HardShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda) ;
  virtual void HardTanh_updateOutput(const Tensor & input, const Tensor & output, Scalar min_val, Scalar max_val, bool inplace) ;
  virtual void HardTanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar min_val, Scalar max_val, bool inplace) ;
  virtual void L1Cost_updateOutput(const Tensor & input, const Tensor & output) ;
  virtual void L1Cost_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) ;
  virtual void L1Cost_updateGradInput(const Tensor & input, const Tensor & gradInput) ;
  virtual void LeakyReLU_updateOutput(const Tensor & input, const Tensor & output, Scalar negval, bool inplace) ;
  virtual void LeakyReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar negval, bool inplace) ;
  virtual void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & hx, const Tensor & output, const Tensor & storage) ;
  virtual void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & hx, const Tensor & output, const Tensor & storage) ;
  virtual void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & hx, const Tensor & output, const Tensor & storage) ;
  virtual void GRUFused_updateGradInput(const Tensor & gradInInput, const Tensor & gradInHidden, const Tensor & gradOutput, const Tensor & gradInputHx, const Tensor & storage) ;
  virtual void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & cell, const Tensor & output, const Tensor & outputCell) ;
  virtual void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & cell, const Tensor & output, const Tensor & outputCell) ;
  virtual void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & cell, const Tensor & output, const Tensor & outputCell) ;
  virtual void LSTMFused_updateGradInput(const Tensor & storage, const Tensor & gradInGates, const Tensor & cx, const Tensor & cy, const Tensor & gradOutput, const Tensor & gradOutputCell, const Tensor & gradInputCx) ;
  virtual void LogSigmoid_updateOutput(const Tensor & input, const Tensor & output, const Tensor & buffer) ;
  virtual void LogSigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & buffer) ;
  virtual void LogSoftMax_updateOutput(const Tensor & input, const Tensor & output) ;
  virtual void LogSoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) ;
  virtual void MarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, Scalar margin) ;
  virtual void MarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, Scalar margin) ;
  virtual void SoftMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) ;
  virtual void SoftMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) ;
  virtual void MSECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) ;
  virtual void MSECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) ;
  virtual void MultiLabelMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, const Tensor & isTarget, bool sizeAverage) ;
  virtual void MultiLabelMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, const Tensor & isTarget, bool sizeAverage) ;
  virtual void MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, const Tensor & weights, Scalar margin) ;
  virtual void MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, Scalar margin) ;
  virtual void MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, const Tensor & weights, Scalar margin) ;
  virtual void MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, Scalar margin) ;
  virtual void PReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, int64_t nOutputPlane) ;
  virtual void PReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int64_t nOutputPlane) ;
  virtual void PReLU_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradWeight, const Tensor & gradWeightBuf, const Tensor & gradWeightBuf2, int64_t nOutputPlane, Scalar scale) ;
  virtual void Linear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & addBuffer) ;
  virtual void Linear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight) ;
  virtual void Linear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & addBuffer, Scalar scale) ;
  virtual void RReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace, Generator & generator) ;
  virtual void RReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace) ;
  virtual void Sigmoid_updateOutput(const Tensor & input, const Tensor & output) ;
  virtual void Sigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) ;
  virtual void Sigmoid_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) ;
  virtual void SmoothL1Criterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) ;
  virtual void SmoothL1Criterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) ;
  virtual void SoftMax_updateOutput(const Tensor & input, const Tensor & output) ;
  virtual void SoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) ;
  virtual void SoftPlus_updateOutput(const Tensor & input, const Tensor & output, Scalar beta, Scalar threshold) ;
  virtual void SoftPlus_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar beta, Scalar threshold) ;
  virtual void SoftShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda) ;
  virtual void SoftShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda) ;
  virtual void IndexLinear_updateOutput(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & normalizedValues, int train) ;
  virtual void IndexLinear_accGradParameters(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, const Tensor & valuesBuffer, Scalar weightDecay, Scalar scale) ;
  virtual void SparseLinear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias) ;
  virtual void SparseLinear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, Scalar weightDecay, Scalar scale) ;
  virtual void Sqrt_updateOutput(const Tensor & input, const Tensor & output, Scalar eps) ;
  virtual void Sqrt_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) ;
  virtual void Square_updateOutput(const Tensor & input, const Tensor & output) ;
  virtual void Square_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) ;
  virtual void Tanh_updateOutput(const Tensor & input, const Tensor & output) ;
  virtual void Tanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) ;
  virtual void Tanh_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) ;
  virtual void Threshold_updateOutput(const Tensor & input, const Tensor & output, Scalar threshold, Scalar val, bool inplace) ;
  virtual void Threshold_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar threshold, Scalar val, bool inplace) ;
  virtual void TemporalConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize, int outputFrameSize) ;
  virtual void TemporalConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW) ;
  virtual void TemporalConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale) ;
  virtual void TemporalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int dW) ;
  virtual void TemporalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int dW) ;
  virtual void TemporalSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize) ;
  virtual void TemporalSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW) ;
  virtual void TemporalSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale) ;
  virtual void TemporalRowConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst) ;
  virtual void TemporalRowConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst) ;
  virtual void TemporalRowConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst, Scalar scale) ;
  virtual void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) ;
  virtual void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) ;
  virtual void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) ;
  virtual void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) ;
  virtual void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) ;
  virtual void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) ;
  virtual void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) ;
  virtual void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) ;
  virtual void SpatialConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) ;
  virtual void SpatialConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) ;
  virtual void SpatialConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale) ;
  virtual void SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) ;
  virtual void SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) ;
  virtual void SpatialConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) ;
  virtual void SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) ;
  virtual void SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) ;
  virtual void SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) ;
  virtual void SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) ;
  virtual void SpatialDepthWiseConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) ;
  virtual void SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) ;
  virtual void SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) ;
  virtual void SpatialConvolutionLocal_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight) ;
  virtual void SpatialConvolutionLocal_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight) ;
  virtual void SpatialConvolutionLocal_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight, Scalar scale) ;
  virtual void SpatialAdaptiveMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight) ;
  virtual void SpatialAdaptiveMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices) ;
  virtual void SpatialAdaptiveAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int owidth, int oheight) ;
  virtual void SpatialAdaptiveAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) ;
  virtual void SpatialAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad) ;
  virtual void SpatialAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad) ;
  virtual void SpatialFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples) ;
  virtual void SpatialFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices) ;
  virtual void SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) ;
  virtual void SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) ;
  virtual void SpatialFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) ;
  virtual void SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale) ;
  virtual void SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale) ;
  virtual void SpatialFullConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) ;
  virtual void SpatialFullConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) ;
  virtual void SpatialFullConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale) ;
  virtual void SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) ;
  virtual void SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) ;
  virtual void SpatialDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) ;
  virtual void SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale) ;
  virtual void SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale) ;
  virtual void SpatialMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode) ;
  virtual void SpatialMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode) ;
  virtual void SpatialDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode) ;
  virtual void SpatialDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode) ;
  virtual void SpatialMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight) ;
  virtual void SpatialMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int owidth, int oheight) ;
  virtual void SpatialSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int kH, int dW, int dH) ;
  virtual void SpatialSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int kH, int dW, int dH) ;
  virtual void SpatialSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int kH, int dW, int dH, Scalar scale) ;
  virtual void SpatialUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor) ;
  virtual void SpatialUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor) ;
  virtual void SpatialUpSamplingBilinear_updateOutput(const Tensor & input, const Tensor & output, int outputHeight, int outputWidth) ;
  virtual void SpatialUpSamplingBilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth) ;
  virtual void SpatialGridSamplerBilinear_updateOutput(const Tensor & input, const Tensor & grid, const Tensor & output) ;
  virtual void SpatialGridSamplerBilinear_updateGradInput(const Tensor & input, const Tensor & gradInput, const Tensor & grid, const Tensor & gradGrid, const Tensor & gradOutput) ;
  virtual void VolumetricAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kT, int kW, int kH, int dT, int dW, int dH) ;
  virtual void VolumetricAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kT, int kW, int kH, int dT, int dW, int dH) ;
  virtual void VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH) ;
  virtual void VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH) ;
  virtual void VolumetricConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, int dT, int dW, int dH, int pT, int pW, int pH) ;
  virtual void VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) ;
  virtual void VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) ;
  virtual void VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) ;
  virtual void VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) ;
  virtual void VolumetricConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) ;
  virtual void VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) ;
  virtual void VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) ;
  virtual void VolumetricFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples) ;
  virtual void VolumetricFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices) ;
  virtual void VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) ;
  virtual void VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) ;
  virtual void VolumetricFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) ;
  virtual void VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale) ;
  virtual void VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale) ;
  virtual void VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) ;
  virtual void VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) ;
  virtual void VolumetricDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) ;
  virtual void VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale) ;
  virtual void VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale) ;
  virtual void VolumetricMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode) ;
  virtual void VolumetricMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode) ;
  virtual void VolumetricDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode) ;
  virtual void VolumetricDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode) ;
  virtual void VolumetricMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH) ;
  virtual void VolumetricMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH) ;
  virtual void SpatialReflectionPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b) ;
  virtual void SpatialReflectionPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b) ;
  virtual void SpatialReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b) ;
  virtual void SpatialReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b) ;
  virtual void VolumetricReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pleft, int pright, int ptop, int pbottom, int pfront, int pback) ;
  virtual void VolumetricReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pleft, int pright, int ptop, int pbottom, int pfront, int pback) ;
  virtual void VolumetricUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor) ;
  virtual void VolumetricUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor) ;
  virtual void VolumetricUpSamplingTrilinear_updateOutput(const Tensor & input, const Tensor & output, int outputDepth, int outputHeight, int outputWidth) ;
  virtual void VolumetricUpSamplingTrilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputDepth, int inputHeight, int inputWidth, int outputDepth, int outputHeight, int outputWidth) ;
  virtual void SpatialCrossMapLRN_updateOutput(const Tensor & input, const Tensor & output, const Tensor & scale, int size, Scalar alpha, Scalar beta, Scalar k) ;
  virtual void SpatialCrossMapLRN_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & scale, const Tensor & output, int size, Scalar alpha, Scalar beta, Scalar k) ;
protected:
  Context* context;
};


}
