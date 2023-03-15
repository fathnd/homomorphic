#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# A bunch of custom pip dependencies for ONNX
pip_install \
  beartype==0.10.4 \
  filelock==3.9.0 \
  flatbuffers==2.0 \
  mock==5.0.1 \
  ninja==1.10.2 \
  networkx==2.0 \
  numpy==1.22.4 \
  onnx==1.13.1 \
  onnxruntime==1.14.0 \
  parameterized==0.8.1 \
  pytest-cov==4.0.0 \
  pytest-subtests==0.10.0 \
  tabulate==0.9.0 \
  transformers==4.25.1

# TODO: change this when onnx-script is on testPypi
pip_install "onnx-script@git+https://github.com/microsoft/onnx-script@29241e15f5182be1384f1cf6ba203d7e2e125196"

IMPORT_SCRIPT_FILENAME="/tmp/onnx_import_script.py"
# Cache the transfomers model to be used later by ONNX tests. By default, the model
# is cached at ~/.cache/huggingface/hub/
as_jenkins echo 'import transformers; transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2");' > "${IMPORT_SCRIPT_FILENAME}"

# Very weird quoting behavior here https://github.com/conda/conda/issues/10972,
# so echo the command to a file and run the file instead
conda_run python "${IMPORT_SCRIPT_FILENAME}"

rm "${IMPORT_SCRIPT_FILENAME}" || true
