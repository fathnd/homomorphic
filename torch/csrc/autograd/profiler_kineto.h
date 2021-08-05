#pragma once

#include <torch/csrc/autograd/profiler_legacy.h>
#include <vector>

#ifdef USE_KINETO
// skip Kineto dependency on mobile
#ifdef C10_MOBILE
#undef USE_KINETO
#endif
#endif

#ifdef USE_KINETO
namespace libkineto {
struct TraceActivity;
class ActivityTraceInterface;
}
#endif

namespace torch {
namespace autograd {
namespace profiler {

enum class C10_API_ENUM ActivityType {
  CPU = 0,
  CUDA, // CUDA kernels, runtime
  NUM_KINETO_ACTIVITIES, // must be the last one
};

#ifdef USE_KINETO

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct KinetoObserverContext : public at::ObserverContext {
  int64_t startUs;
  uint64_t correlationId;
  uint64_t startThreadId;
  uint64_t endThreadId;
  c10::optional<std::vector<std::vector<int64_t>>> shapes;
  c10::optional<std::vector<std::string>> dtypes;
  int64_t sequenceNr;
  uint64_t fwdThreadId;
  uint8_t recFunScope;
  c10::optional<std::vector<std::string>> stack;
  c10::optional<std::vector<std::string>> module_hierarchy;
  // Extra arguments for computing op flops
  c10::optional<std::unordered_map<std::string, c10::IValue>> extraArgs;
  CUDAEventStub cuda_event_start_ = nullptr;
  CUDAEventStub cuda_event_end_ = nullptr;
};

struct TORCH_API KinetoEvent {
  uint64_t startThreadId() const {
    return start_thread_id_;
  }

  uint64_t endThreadId() const {
    return end_thread_id_;
  }

  uint8_t activityType() const {
    return activity_type_;
  }

  uint64_t fwdThreadId() const {
    return fwd_thread_id_;
  }

  bool hasShapes() const {
    return shapes_ != c10::nullopt;
  }

  const std::vector<std::vector<int64_t>>& shapes() const {
    return *shapes_;
  }

  bool hasTypes() const {
    return dtypes_ != c10::nullopt;
  }

  const std::vector<std::string>& dtypes() const {
    return *dtypes_;
  }

  uint64_t flops() const {
    return flops_;
  }

  int64_t sequenceNr() const {
    return sequence_nr_;
  }

  bool hasStack() const {
    return stack_ != c10::nullopt;
  }

  const std::vector<std::string>& stack() const {
    return *stack_;
  }

  bool hasModuleHierarchy() const {
    return module_hierarchy_ != c10::nullopt;
  }

  const std::vector<std::string>& moduleHierarchy() const {
    return *module_hierarchy_;
  }

  uint8_t scope() const {
    return scope_;
  }

  KinetoEvent& startThreadId(uint64_t start_thread_id) {
    start_thread_id_ = start_thread_id;
    return *this;
  }

  KinetoEvent& endThreadId(uint64_t end_thread_id) {
    end_thread_id_ = end_thread_id;
    return *this;
  }

  KinetoEvent& fwdThreadId(uint64_t fwd_thread_id) {
    fwd_thread_id_ = fwd_thread_id;
    return *this;
  }

  KinetoEvent& shapes(const std::vector<std::vector<int64_t>>& shapes) {
    shapes_ = shapes;
    return *this;
  }

  KinetoEvent& dtypes(const std::vector<std::string>& dtypes) {
    dtypes_ = dtypes;
    return *this;
  }

  KinetoEvent& flops(uint64_t flops) {
    flops_ = flops;
    return *this;
  }

  KinetoEvent& sequenceNr(int64_t sequence_nr) {
    sequence_nr_ = sequence_nr;
    return *this;
  }

  KinetoEvent& stack(const std::vector<std::string>& st) {
    stack_ = st;
    return *this;
  }

  KinetoEvent& moduleHierarchy(const std::vector<std::string>& module_hierarchy) {
    module_hierarchy_ = module_hierarchy;
    return *this;
  }

  KinetoEvent& scope(uint8_t scope) {
    scope_ = scope;
    return *this;
  }

  KinetoEvent& setAsync(bool is_async) {
    is_async_ = is_async;
    return *this;
  }

  KinetoEvent& deviceType(c10::DeviceType device_type) {
    device_type_ = (int8_t)device_type;
    return *this;
  }

  KinetoEvent& deviceIndex(uint8_t device_index) {
    device_index_ = device_index;
    return *this;
  }

  KinetoEvent& nBytes(int64_t nbytes) {
    nbytes_ = nbytes;
    return *this;
  }

  // Kineto fields

  KinetoEvent& activity(const libkineto::TraceActivity& activity);

  std::string name() const {
    return name_;
  }

  bool isAsync() const {
    return is_async_;
  }

  uint8_t deviceIndex() const {
    return device_index_;
  }

  uint64_t startUs() const {
    return start_us_;
  }

  uint64_t durationUs() const {
    return duration_us_;
  }

  uint64_t correlationId() const {
    return correlation_id_;
  }

  KinetoEvent& correlationId(uint64_t correlation_id)  {
    correlation_id_ = correlation_id;
    return *this;
  }

  uint64_t linkedCorrelationId() const {
    return linked_correlation_id_;
  }

  int64_t deviceResourceId() const {
    return device_resource_id_;
  }

  int64_t nBytes() const {
    return nbytes_;
  }

  c10::DeviceType deviceType() const;

  int64_t cudaElapsedUs() const;

  uint64_t start_thread_id_ = 0;
  uint64_t end_thread_id_ = 0;
  uint64_t fwd_thread_id_ = 0;
  int64_t sequence_nr_ = -1;
  uint8_t scope_ = 0;

  uint8_t activity_type_ = 0;
  c10::optional<std::vector<std::vector<int64_t>>> shapes_;
  c10::optional<std::vector<std::string>> stack_;
  c10::optional<std::vector<std::string>> module_hierarchy_;
  c10::optional<std::vector<std::string>> dtypes_;
  uint64_t flops_ = 0;

  std::string name_;
  uint8_t device_index_ = 0;
  int8_t device_type_ = -1;
  uint64_t start_us_ = 0;
  uint64_t duration_us_ = 0;
  uint64_t correlation_id_ = 0;
  uint64_t linked_correlation_id_ = 0;
  int64_t device_resource_id_ = 0;
  int64_t nbytes_ = 0;
  bool is_async_{false};

  CUDAEventStub cuda_event_start_ = nullptr;
  CUDAEventStub cuda_event_end_ = nullptr;
};

// Consolidating events returned directly from Kineto
// with events manually created by us (e.g. start/stop marks,
// memory allocation events)
struct TORCH_API ProfilerResult {
  ProfilerResult();
  ProfilerResult(
      uint64_t start_time,
      std::vector<KinetoEvent> events,
      std::unique_ptr<libkineto::ActivityTraceInterface> trace);
  ~ProfilerResult();

  uint64_t trace_start_us() const {
    return trace_start_us_;
  }

  const std::vector<KinetoEvent>& events() const {
    return events_;
  }

  void save(const std::string& path);

 private:
  bool saved_ = false;
  uint64_t trace_start_us_ = 0;
  std::vector<KinetoEvent> events_;
  std::unique_ptr<libkineto::ActivityTraceInterface> trace_;
};

TORCH_API void enableProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities);

TORCH_API std::unique_ptr<ProfilerResult> disableProfiler();

TORCH_API void prepareProfiler(
    const ProfilerConfig& config,
    const std::set<ActivityType>& activities);

TORCH_API void addMetadataJson(
    const std::string& key, const std::string& value);
#endif // USE_KINETO

} // namespace profiler
}} // namespace torch::autograd
