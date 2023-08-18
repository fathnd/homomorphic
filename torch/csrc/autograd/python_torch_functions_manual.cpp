#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/python_torch_functions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/out_types.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/structseq.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>

#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>

#include <Python.h>
#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <utility>
#include <vector>

using at::DeviceGuard;
using at::DimnameList;
using at::IntArrayRef;
using at::OptionalDeviceGuard;
using at::Scalar;
using at::Tensor;
using at::TensorList;
using at::TensorOptions;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch {
namespace autograd {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyObject* THPVariableFunctionsModule = nullptr;

inline Tensor dispatch_range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor result) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(result));
  return at::range_out(result, start, end, step);
}

inline Tensor dispatch_range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    const TensorOptions& options) {
  torch::utils::maybe_initialize_cuda(options);
  pybind11::gil_scoped_release no_gil;
  DeviceGuard device_guard(options.device());
  return torch::range(start, end, step, options);
}

static PyObject* THPVariable_range(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "range(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto ret = PyErr_WarnEx(
        PyExc_UserWarning,
        "torch.range is deprecated and will be removed in a future release "
        "because its behavior is inconsistent with Python's range builtin. "
        "Instead, use torch.arange, which produces values in [start, end).",
        1);
    if (ret != 0)
      throw python_error();
    if (r.isNone(3)) {
      const auto options = TensorOptions()
                               .dtype(r.scalartype(4))
                               .device(r.device(6))
                               .layout(r.layout(5))
                               .requires_grad(r.toBool(7));
      return wrap(
          dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), options));
    } else {
      check_out_type_matches(
          r.tensor(3),
          r.scalartype(4),
          r.isNone(4),
          r.layout(5),
          r.device(6),
          r.isNone(6));
      return wrap(
          dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3))
              .set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow torch.as_tensor to be constructed with
// arbitrarily nested python objects - list, tuple, np array, scalar, etc.
static PyObject* THPVariable_as_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "as_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  jit::tracer::warn("torch.as_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::as_tensor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// implemented on python object here because PyObject currently not natively
// declarable See: ATen/native/README.md for more context
static PyObject* THPVariable_from_numpy(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.from_numpy", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_from_numpy(arg));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor& self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static Tensor dispatch_nonzero(const Tensor& self, Tensor out) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return at::nonzero_out(out, self);
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor& self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

static PyObject* THPVariable_nonzero(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs);

#define THPVARIABLE_SPARSE_COMPRESSED_CTOR(NAME, NARGS, SIGNATURES)       \
  static PyObject* THPVariable_##NAME(                                    \
      PyObject* self, PyObject* args, PyObject* kwargs) {                 \
    HANDLE_TH_ERRORS                                                      \
    static PythonArgParser parser SIGNATURES;                             \
    ParsedArgs<NARGS> parsed_args;                                        \
    auto r = parser.parse(args, kwargs, parsed_args);                     \
    if (r.has_torch_function()) {                                         \
      return handle_torch_function(                                       \
          r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch"); \
    }                                                                     \
    jit::tracer::warn("torch." #NAME, jit::tracer::WARN_CONSTRUCTOR);     \
    return THPVariable_Wrap(torch::utils::NAME##_ctor(                    \
        torch::tensors::get_default_dispatch_key(),                       \
        torch::tensors::get_default_scalar_type(),                        \
        r));                                                              \
    END_HANDLE_TH_ERRORS                                                  \
  }

THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_compressed_tensor,
    10,
    ({"sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_csr_tensor,
    10,
    ({"sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_csc_tensor,
    10,
    ({"sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_bsr_tensor,
    10,
    ({"sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_bsc_tensor,
    10,
    ({"sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
      "sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"}))

static PyObject* THPVariable_sparse_coo_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False, bool check_invariants=None)",
      "sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False, bool check_invariants=None)",
      "sparse_coo_tensor(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False, bool check_invariants=None)",
  });

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  jit::tracer::warn("torch.sparse_coo_tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::sparse_coo_tensor_ctor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// implemented on python object to allow torch.tensor to be constructed with
// arbitrarily nested python objects - list, tuple, np array, scalar, etc.
static PyObject* THPVariable_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
  });

  constexpr int ctor_num_args = 6;
  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  jit::tracer::warn("torch.tensor", jit::tracer::WARN_CONSTRUCTOR);
  return THPVariable_Wrap(torch::utils::tensor_ctor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_device(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {
          "get_device(Tensor input)",
      },
      /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(r.tensor(0).get_device());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_frombuffer(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {
          "frombuffer(PyObject* buffer, *, ScalarType dtype, int64_t count=-1, int64_t offset=0, bool requires_grad=False)",
      },
      /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto buffer = r.pyobject(0);
    auto dtype = r.scalartype(1);
    auto count = r.toInt64(2);
    auto offset = r.toInt64(3);
    auto requires_grad = r.toBool(4);

    TORCH_CHECK_VALUE(
        PyObject_CheckBuffer(buffer) != 0,
        "object does not implement Python buffer protocol.");
    return wrap(torch::utils::tensor_frombuffer(
        buffer, dtype, count, offset, requires_grad));
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_asarray(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {
          "asarray(PyObject* obj, *, ScalarType? dtype=None, Device? device=None, bool? copy=None, bool requires_grad=False)",
      },
      /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  if (r.idx == 0) {
    auto obj = r.pyobject(0);
    auto dtype = r.scalartypeOptional(1);
    auto device = r.deviceOptional(2);
    auto copy = r.toBoolOptional(3);
    auto requires_grad = r.toBool(4);
    return wrap(torch::utils::asarray(obj, dtype, device, copy, requires_grad));
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_numel(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs);

static PyObject* THPVariable__to_functional_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {"_to_functional_tensor(Tensor t, *, bool mirror_autograd_meta=False)"},
      /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto mirror_autograd_meta = r.toBool(1);
  auto wrapped = at::functionalization::impl::to_functional_tensor(self_);
  if (mirror_autograd_meta) {
    // Here, we unsafely set the grad function on the wrapper to be the same as
    // the inner. We expect this grad_fn to NEVER be used. It's needed so that
    // .is_leaf metadata is accurate on the wrapper
    auto inner_autograd_meta = impl::get_autograd_meta(self_);
    if (inner_autograd_meta) {
      wrapped.set_requires_grad(self_.requires_grad());
      if (wrapped.requires_grad()) {
        auto new_grad_fn = std::shared_ptr<torch::autograd::Error>(
            new torch::autograd::Error(
                "Cannot backprop through mirrored meta, file a bug in PyTorch"),
            torch::autograd::deleteNode);
        torch::autograd::set_history(wrapped, new_grad_fn);
      }
    }
  }
  return wrap(std::move(wrapped));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__from_functional_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {"_from_functional_tensor(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto unwrapped = at::functionalization::impl::from_functional_tensor(self_);
  return wrap(std::move(unwrapped));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__freeze_functional_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {"_freeze_functional_tensor(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  at::functionalization::impl::freeze_functional_tensor(self_);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__is_functional_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {"_is_functional_tensor(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (at::functionalization::impl::isFunctionalTensor(self_)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__functionalize_has_metadata_mutation(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {"_functionalize_has_metadata_mutation(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(self_);
  if (wrapper->has_metadata_mutation()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__enable_functionalization(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {"_enable_functionalization(*, bool reapply_views=False)"},
      /*traceable=*/true);
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  const auto reapply_views = r.toBool(0);

  if (c10::impl::tls_is_dispatch_key_included(at::DispatchKey::Functionalize)) {
    TORCH_INTERNAL_ASSERT(
        false,
        "multiple layers of mode-style functionalization nesting is not"
        " currently supported, outside of the functionalize() transform");
  }
  c10::impl::tls_set_dispatch_key_included(
      at::DispatchKey::Functionalize, true);
  if (reapply_views) {
    at::functionalization::impl::setFunctionalizationReapplyViewsTLS(true);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__disable_functionalization(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  c10::impl::tls_set_dispatch_key_included(
      at::DispatchKey::Functionalize, false);
  at::functionalization::impl::setFunctionalizationReapplyViewsTLS(false);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__sync(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"_sync(Tensor t)"}, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  at::functionalization::impl::sync(self_);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static PyMethodDef torch_functions_manual[] = {
    {"asarray",
     castPyCFunctionWithKeywords(THPVariable_asarray),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"as_tensor",
     castPyCFunctionWithKeywords(THPVariable_as_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"from_numpy", THPVariable_from_numpy, METH_STATIC | METH_O, nullptr},
    {"frombuffer",
     castPyCFunctionWithKeywords(THPVariable_frombuffer),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_is_functional_tensor",
     castPyCFunctionWithKeywords(THPVariable__is_functional_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_to_functional_tensor",
     castPyCFunctionWithKeywords(THPVariable__to_functional_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_from_functional_tensor",
     castPyCFunctionWithKeywords(THPVariable__from_functional_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_freeze_functional_tensor",
     castPyCFunctionWithKeywords(THPVariable__freeze_functional_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_sync",
     castPyCFunctionWithKeywords(THPVariable__sync),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_enable_functionalization",
     castPyCFunctionWithKeywords(THPVariable__enable_functionalization),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_disable_functionalization",
     castPyCFunctionWithKeywords(THPVariable__disable_functionalization),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_functionalize_has_metadata_mutation",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_has_metadata_mutation),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"nonzero",
     castPyCFunctionWithKeywords(THPVariable_nonzero),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"range",
     castPyCFunctionWithKeywords(THPVariable_range),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_coo_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_coo_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_compressed_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_compressed_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_csr_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_csr_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_csc_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_csc_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_bsr_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_bsr_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"sparse_bsc_tensor",
     castPyCFunctionWithKeywords(THPVariable_sparse_bsc_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"tensor",
     castPyCFunctionWithKeywords(THPVariable_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"get_device",
     castPyCFunctionWithKeywords(THPVariable_get_device),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"numel",
     castPyCFunctionWithKeywords(THPVariable_numel),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
};

static PyObject* THPVariable_nonzero(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "nonzero(Tensor input, *, bool as_tuple=False, Tensor out=None)",
  });
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(
        r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  const auto as_tuple = r.toBool(1);
  const auto has_out = !r.isNone(2);

  if (as_tuple) {
    TORCH_CHECK(
        !has_out,
        "nonzero does not support the out kwarg when as_tuple is True");
    return wrap(dispatch_nonzero_numpy(r.tensor(0)));
  }

  if (has_out) {
    return wrap(dispatch_nonzero(r.tensor(0), r.tensor(2)));
  }

  return wrap(dispatch_nonzero(r.tensor(0)));

  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_numel(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser(
      {
          "numel(Tensor input)",
      },
      /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(
        r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  if (r.idx == 0) {
    return py::cast(r.tensor(0).sym_numel()).release().ptr();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Sharded function definitions
void gatherTorchFunctions_0(std::vector<PyMethodDef>& torch_functions);
void gatherTorchFunctions_1(std::vector<PyMethodDef>& torch_functions);
void gatherTorchFunctions_2(std::vector<PyMethodDef>& torch_functions);

void gatherTorchFunctions(std::vector<PyMethodDef>& torch_functions) {
  constexpr size_t num_functions =
      sizeof(torch_functions_manual) / sizeof(torch_functions_manual[0]);
  torch_functions.assign(
      torch_functions_manual, torch_functions_manual + num_functions);
  // NOTE: Must be synced with num_shards in
  // tools/autograd/gen_python_functions.py
  gatherTorchFunctions_0(torch_functions);
  gatherTorchFunctions_1(torch_functions);
  gatherTorchFunctions_2(torch_functions);

  static std::array<std::pair<const char*, const char*>, 4> aliases{
      {// Canonical function, alias name
       {"sspaddmm", "saddmm"},
       {"mm", "spmm"},
       {"mm", "dsmm"},
       {"hspmm", "hsmm"}}};

  for (const auto& alias : aliases) {
    auto it = std::find_if(
        torch_functions.begin(),
        torch_functions.end(),
        [&](const PyMethodDef& def) {
          return strcmp(def.ml_name, alias.first) == 0;
        });
    TORCH_INTERNAL_ASSERT(
        it != torch_functions.end(),
        "Failed to create function alias from ",
        alias.first,
        " to ",
        alias.second);
    PyMethodDef alias_def = *it;
    alias_def.ml_name = alias.second;

    torch_functions.push_back(alias_def);
  }

  torch_functions.push_back({nullptr});
  torch_functions.shrink_to_fit();
}

static PyTypeObject THPVariableFunctions = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch._C._VariableFunctionsClass", /* tp_name */
    0, /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr /* tp_new */
};

void initTorchFunctions(PyObject* module) {
  static std::vector<PyMethodDef> torch_functions;
  gatherTorchFunctions(torch_functions);
  THPVariableFunctions.tp_methods = torch_functions.data();

  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);

  // Steals
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(
          module,
          "_VariableFunctionsClass",
          reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THPVariableFunctionsModule =
      PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(
          module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }
}

} // namespace autograd
} // namespace torch
