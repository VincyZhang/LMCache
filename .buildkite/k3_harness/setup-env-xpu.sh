#!/usr/bin/env bash
# Per-job environment setup for XPU multiprocess tests.
# Installs vLLM + LMCache in an Intel XPU-capable environment.
set -euo pipefail

trap 'echo "ERROR: setup-env-xpu.sh failed at line $LINENO (exit code $?)" >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "--- :gear: Enable Intel oneAPI runtime"
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
fi

echo "--- :mag: Verify XPU availability"
python - <<'PY'
import torch

assert hasattr(torch, "xpu"), "torch.xpu is not available"
assert torch.xpu.is_available(), "Intel XPU not available in pod"
print("torch.xpu.is_available() = True")
PY

echo "--- :python: Install/upgrade vLLM and runtime deps for XPU"
# Keep the package list aligned with setup-env.sh's runtime expectations,
# but without CUDA-specific index settings.
uv pip install -U "vllm[runai,tensorizer,flashinfer]>=0.0.0.dev0" \
    --reinstall-package transformers \
    --reinstall-package tokenizers \
    --reinstall-package huggingface-hub \
    --reinstall-package safetensors \
    --reinstall-package vllm

echo "--- :python: Install LMCache from source"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
uv pip install -e . --no-build-isolation

echo "--- :python: Install multiprocess test extras"
uv pip install 'lm-eval[api]' openai pandas matplotlib

echo "--- :white_check_mark: XPU environment ready"
python - <<'PY'
import vllm
import lmcache
import torch

print(f"vLLM={vllm.__version__}, LMCache import OK, XPU available={torch.xpu.is_available()}")
PY
