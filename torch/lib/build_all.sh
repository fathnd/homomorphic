set -e

cd "$(dirname "$0")/../.."
BASE_DIR=$(pwd)
cd torch/lib
INSTALL_DIR="$(pwd)/tmp_install"
BASIC_C_FLAGS=" -DTH_INDEX_BASE=0 -DTH_GENERIC_USE_HALF=1 -DCUDA_HAS_FP16=1 -I$INSTALL_DIR/include -I$INSTALL_DIR/include/TH -I$INSTALL_DIR/include/THC "
LDFLAGS="-L$INSTALL_DIR/lib "
LD_POSTFIX=".so.1"
LD_POSTFIX_UNVERSIONED=".so"
if [[ $(uname) == 'Darwin' ]]; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,@loader_path"
    LD_POSTFIX=".1.dylib"
    LD_POSTFIX_UNVERSIONED=".dylib"
else
    LDFLAGS="$LDFLAGS -Wl,-rpath,\$ORIGIN"
fi
C_FLAGS="$BASIC_C_FLAGS $LDFLAGS"
function build() {
  mkdir -p build/$1
  cd build/$1
  cmake ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              -DTorch_FOUND="1" \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$C_FLAGS" \
              -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS" \
              -DCUDA_NVCC_FLAGS="$BASIC_C_FLAGS" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include" \
              -DTH_LIB_PATH="$INSTALL_DIR/lib" \
              -DTH_LIBRARIES="$INSTALL_DIR/lib/libTH$LD_POSTFIX" \
              -DTHC_LIBRARIES="$INSTALL_DIR/lib/libTHC$LD_POSTFIX" \
              -DTH_SO_VERSION=1 \
              -DTHC_SO_VERSION=1 \
              -DTHNN_SO_VERSION=1 \
              -DTHCUNN_SO_VERSION=1 
  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

  rm -rf $INSTALL_DIR/lib/lib$1$LD_POSTFIX_UNVERSIONED

  if [[ $(uname) == 'Darwin' ]]; then
    cd tmp_install/lib
    for lib in *.dylib; do
      echo "Updating install_name for $lib"
      install_name_tool -id @rpath/$lib $lib
    done
    cd ../..
  fi
}
function build_nccl() {
   mkdir -p build/nccl
   cd build/nccl
   cmake ../../nccl -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
               -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
               -DCMAKE_C_FLAGS="$C_FLAGS" \
               -DCMAKE_CXX_FLAGS="$C_FLAGS $CPP_FLAGS"
   make install
   cp "lib/libnccl.so.1" "${INSTALL_DIR}/lib/libnccl.so"
   cd ../..
}

mkdir -p tmp_install
build TH
build THNN

if [[ "$1" == "--with-cuda" ]]; then
    build THC
    build THCUNN
    if [[ $(uname) != 'Darwin' ]]; then
        build_nccl
    fi
fi

CPP_FLAGS=" -std=c++11 "
build libshm


cp $INSTALL_DIR/lib/* .
cp THNN/generic/THNN.h .
cp THCUNN/generic/THCUNN.h .
cp -r tmp_install/include .
cp $INSTALL_DIR/bin/* .
