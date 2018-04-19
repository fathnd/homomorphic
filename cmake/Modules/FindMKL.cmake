# Find the MKL libraries
#
# Options:
#
#   MKL_USE_SINGLE_DYNAMIC_LIBRARY  : use single dynamic library interface
#   MKL_USE_STATIC_LIBS             : use static libraries
#   MKL_MULTI_THREADED              : use multi-threading
#   MKL_USE_IDEEP                   : use IDEEP interface
#   MKL_USE_MKLML                   : use MKLML interface
#
# This module defines the following variables:
#
#   MKL_FOUND            : True mkl is found
#   MKL_INCLUDE_DIR      : unclude directory
#   MKL_LIBRARIES        : the libraries to link against.

# ---[ Options
include(CMakeDependentOption)
option(MKL_USE_IDEEP "Use IDEEP interface" ON)
option(MKL_USE_MKLML "Use MKLML interface" ON)

if(MKL_USE_IDEEP)
  set(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
  set(MKLDNN_ROOT "${IDEEP_ROOT}/mkl-dnn")
  set(__ideep_looked_for IDEEP_ROOT)

  find_path(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
  find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
  if (NOT MKLDNN_INCLUDE_DIR)
    execute_process(COMMAND git submodule update --init mkl-dnn WORKING_DIRECTORY ${IDEEP_ROOT})
    find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
  endif()

  if (MKLDNN_INCLUDE_DIR)
    # to avoid adding conflicting submodels
    set(ORIG_WITH_TEST ${WITH_TEST})
    set(WITH_TEST OFF)
    add_subdirectory(${IDEEP_ROOT})
    set(WITH_TEST ${ORIG_WITH_TEST})

    file(GLOB_RECURSE MKLML_INCLUDE_DIR ${MKLDNN_ROOT}/external/*/mkl_vsl.h)
    if(MKLML_INCLUDE_DIR)
      # if user has multiple version under external/ then guess last
      # one alphabetically is "latest" and warn
      list(LENGTH MKLML_INCLUDE_DIR MKLINCLEN)
      if(MKLINCLEN GREATER 1)
        list(SORT MKLML_INCLUDE_DIR)
        list(REVERSE MKLML_INCLUDE_DIR)
        list(GET MKLML_INCLUDE_DIR 0 MKLINCLST)
        set(MKLML_INCLUDE_DIR "${MKLINCLST}")
      endif()
      get_filename_component(MKLML_INCLUDE_DIR ${MKLML_INCLUDE_DIR} DIRECTORY)
      list(APPEND IDEEP_INCLUDE_DIR ${MKLDNN_INCLUDE_DIR} ${MKLML_INCLUDE_DIR})
      list(APPEND __ideep_looked_for IDEEP_INCLUDE_DIR)

      set(__mklml_libs mklml_intel iomp5)
      set(IDEEP_LIBRARIES "")
      foreach (__mklml_lib ${__mklml_libs})
        string(TOUPPER ${__mklml_lib} __mklml_lib_upper)
        find_library(${__mklml_lib_upper}_LIBRARY
              NAMES ${__mklml_lib}
              PATHS  "${MKLML_INCLUDE_DIR}/../lib"
              DOC "The path to Intel(R) MKLML ${__mklml_lib} library")
        mark_as_advanced(${__mklml_lib_upper}_LIBRARY)
        list(APPEND IDEEP_LIBRARIES ${${__mklml_lib_upper}_LIBRARY})
        list(APPEND __ideep_looked_for ${__mklml_lib_upper}_LIBRARY)
      endforeach()

      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(IDEEP DEFAULT_MSG ${__ideep_looked_for})

      if(IDEEP_FOUND)
        list(APPEND IDEEP_LIBRARIES "${PROJECT_BINARY_DIR}/lib/libmkldnn.so")
        set(CAFFE2_USE_IDEEP 1)
        # Do NOT use MPI if IDEEP is enabled
        set(USE_MPI OFF)
        message(STATUS "Found IDEEP (include: ${IDEEP_INCLUDE_DIR}, lib: ${IDEEP_LIBRARIES})")
      endif()

      caffe_clear_vars(__ideep_looked_for __mklml_libs)
    endif()
  endif()

  if(NOT IDEEP_FOUND)
    message(FATAL_ERROR "Did not find IDEEP files!")
  endif()
endif()

if(MKL_USE_MKLML)

  # ---[ Options
  option(MKL_USE_SINGLE_DYNAMIC_LIBRARY "Use single dynamic library interface" ON)
  cmake_dependent_option(
      MKL_USE_STATIC_LIBS "Use static libraries" OFF
          "NOT MKL_USE_SINGLE_DYNAMIC_LIBRARY" OFF)
  cmake_dependent_option(
      MKL_MULTI_THREADED  "Use multi-threading" ON
      "NOT MKL_USE_SINGLE_DYNAMIC_LIBRARY" OFF)

  # ---[ Root folders
  if(MSVC)
    set(INTEL_ROOT_DEFAULT "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows")
  else()
    set(INTEL_ROOT_DEFAULT "/opt/intel")
  endif()
  set(INTEL_ROOT ${INTEL_ROOT_DEFAULT} CACHE PATH "Folder contains intel libs")
  find_path(MKL_ROOT include/mkl.h PATHS $ENV{MKLROOT} ${INTEL_ROOT}/mkl
                                     DOC "Folder contains MKL")

  # ---[ Find include dir
  find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT} PATH_SUFFIXES include)
  set(__looked_for MKL_INCLUDE_DIR)

  # ---[ Find libraries
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(__path_suffixes lib lib/ia32)
  else()
    set(__path_suffixes lib lib/intel64)
  endif()

  set(__mkl_libs "")
  if(MKL_USE_SINGLE_DYNAMIC_LIBRARY)
    list(APPEND __mkl_libs rt)
  else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
      if(WIN32)
        list(APPEND __mkl_libs intel_c)
      else()
        list(APPEND __mkl_libs intel gf)
      endif()
    else()
      list(APPEND __mkl_libs intel_lp64 gf_lp64)
    endif()

    if(MKL_MULTI_THREADED)
      list(APPEND __mkl_libs intel_thread)
    else()
       list(APPEND __mkl_libs sequential)
    endif()

    list(APPEND __mkl_libs core cdft_core)
  endif()

  foreach (__lib ${__mkl_libs})
    set(__mkl_lib "mkl_${__lib}")
    string(TOUPPER ${__mkl_lib} __mkl_lib_upper)

    if(MKL_USE_STATIC_LIBS)
      set(__mkl_lib "lib${__mkl_lib}.a")
    endif()

    find_library(${__mkl_lib_upper}_LIBRARY
          NAMES ${__mkl_lib}
          PATHS ${MKL_ROOT} "${MKL_INCLUDE_DIR}/.."
          PATH_SUFFIXES ${__path_suffixes}
          DOC "The path to Intel(R) MKL ${__mkl_lib} library")
    mark_as_advanced(${__mkl_lib_upper}_LIBRARY)

    list(APPEND __looked_for ${__mkl_lib_upper}_LIBRARY)
    list(APPEND MKL_LIBRARIES ${${__mkl_lib_upper}_LIBRARY})
  endforeach()

  if(NOT MKL_USE_SINGLE_DYNAMIC_LIBRARY)
    if (MKL_USE_STATIC_LIBS)
      set(__iomp5_libs iomp5 libiomp5mt.lib)
    else()
      set(__iomp5_libs iomp5 libiomp5md.lib)
    endif()

    if(WIN32)
      find_path(INTEL_INCLUDE_DIR omp.h PATHS ${INTEL_ROOT} PATH_SUFFIXES include)
      list(APPEND __looked_for INTEL_INCLUDE_DIR)
    endif()

    find_library(MKL_RTL_LIBRARY ${__iomp5_libs}
       PATHS ${INTEL_RTL_ROOT} ${INTEL_ROOT}/compiler ${MKL_ROOT}/.. ${MKL_ROOT}/../compiler
       PATH_SUFFIXES ${__path_suffixes}
       DOC "Path to OpenMP runtime library")

    list(APPEND __looked_for MKL_RTL_LIBRARY)
    list(APPEND MKL_LIBRARIES ${MKL_RTL_LIBRARY})
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(MKL DEFAULT_MSG ${__looked_for})

  if(MKL_FOUND)
    set(CAFFE2_USE_MKL 1)
    message(STATUS "Found MKL (include: ${MKL_INCLUDE_DIR}, lib: ${MKL_LIBRARIES})")
  endif()

  caffe_clear_vars(__looked_for __mkl_libs __path_suffixes __lib_suffix __iomp5_libs)

endif()

if(IDEEP_FOUND)
  set(MKL_FOUND True)
  list(APPEND MKL_INCLUDE_DIR ${IDEEP_INCLUDE_DIR})
  list(APPEND MKL_LIBRARIES ${IDEEP_LIBRARIES})
endif()

set(USE_MKL OFF)
if(MKL_FOUND)
  set(USE_MKL ON)
endif()
