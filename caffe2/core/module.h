/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * A global dictionary that holds information about what Caffe2 modules have
 * been loaded in the current runtime, and also utility functions to load
 * modules.
 */
#ifndef CAFFE2_CORE_MODULE_H_
#define CAFFE2_CORE_MODULE_H_

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>

#include "caffe2/core/common.h"
#include "caffe2/core/typeid.h"

namespace caffe2 {

/**
 * A module schema that can be used to store specific information about
 * different modules. Currently, we only store the name and a simple
 * description of what this module does.
 */
class ModuleSchema {
 public:
  ModuleSchema(const char* name, const char* description);

 private:
  const char* name_;
  const char* description_;
};


/**
 * @brief Current Modules present in the Caffe2 runtime.
 * Returns:
 *   map: a map of modules and (optionally) their description. The key is the
 *       module name, and the value is the description for that module. The
 *       module name is recommended to be the part that constitutes the trunk
 *       of the dynamic library: for example, a module called
 *       libcaffe2_db_rocksdb.so should have the name "caffe2_db_rocksdb". The
 *       reason we do not use "lib" is because it's somewhat redundant, and
 *       the reason we do not include ".so" is for cross-platform compatibility
 *       on platforms like mac os.
 */
const CaffeMap<string, const ModuleSchema*>& CurrentModules();

/**
 * @brief Checks whether a module is already present in the current binary.
 */
bool HasModule(const string& name);

/**
 * @brief Load a module.
 * Inputs:
 *   name: a module name or a path name.
 *       It is recommended that you use the name of the module, and leave the
 *       full path option to only experimental modules.
 *   filename: (optional) a filename that serves as a hint to load the module.
 */
void LoadModule(const string& name, const string& filename="");


#define CAFFE2_MODULE(name, description)                                    \
  extern "C" {                                                              \
    const bool gCaffe2ModuleSanityCheck##name() { return true; }            \
  }                                                                         \
  namespace {                                                               \
    static ::caffe2::ModuleSchema module_schema_##name(#name, description); \
  }

}  // namespace caffe2
#endif  // CAFFE2_CORE_MODULE_H_
