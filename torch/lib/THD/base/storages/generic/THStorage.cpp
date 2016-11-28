#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "base/storages/generic/THStorage.cpp"
#else

template<>
THStorage<real>::THStorage(): storage(THStorage_(new)()) {}

template<>
THStorage<real>::THStorage(storage_type* storage): storage(storage) {}

template<>
THStorage<real>::THStorage(std::size_t storage_size)
  : storage(THStorage_(newWithSize)(storage_size)) {}

template<>
THStorage<real>::~THStorage() {
  THStorage_(free)(storage);
}

template<>
std::size_t THStorage<real>::elementSize() const {
  return sizeof(real);
}

template<>
std::size_t THStorage<real>::size() const {
  return storage->size;
}

template<>
void* THStorage<real>::data() {
  return storage->data;
}

template<>
const void* THStorage<real>::data() const {
  return storage->data;
}

template<>
auto THStorage<real>::retain() -> THStorage& {
  THStorage_(retain)(storage);
  return *this;
}

template<>
auto THStorage<real>::free() -> THStorage& {
  THStorage_(free)(storage);
  return *this;
}

template<>
auto THStorage<real>::resize(long new_size) -> THStorage& {
  THStorage_(resize)(storage, new_size);
  return *this;
}

template<>
auto THStorage<real>::fill(scalar_type value) -> THStorage& {
  THStorage_(fill)(storage, value);
  return *this;
}

template<>
thd::Type THStorage<real>::type() const {
  return thd::type_traits<real>::type;
}

#endif
