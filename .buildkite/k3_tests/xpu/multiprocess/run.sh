#!/usr/bin/env bash
set -euo pipefail

# Reuse multiprocess test harness, but switch to the XPU environment/bootstrap.
export TORCH_DEVICE_TYPE="xpu"
export VLLM_TARGET_DEVICE="xpu"
export BK_SETUP_ENV_SCRIPT=".buildkite/k3_harness/setup-lmcache-only-env.sh"

# XPU path for this phase is single-pod only (no baseline server).
export LAUNCH_BASELINE="false"
# Force explicit backend on XPU to avoid auto fallback to Flash Attention.
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
# Keep low-level env override for debugging parity.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
# First-time model pulls on XPU pods can be slow; keep timeout overridable.
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-1800}"
# Turn on verbose vLLM logs by default to debug device-type inference issues.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
# Avoid inheriting CUDA affinity from host/agent env in XPU jobs.
unset CUDA_VISIBLE_DEVICES || true
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

exec bash "$(cd "$(dirname "$0")/../.." && pwd)/multiprocess/run.sh" "$@"
