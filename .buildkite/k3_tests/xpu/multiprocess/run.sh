#!/usr/bin/env bash
set -euo pipefail

# Reuse multiprocess test harness, but switch to the XPU environment/bootstrap.
export TORCH_DEVICE_TYPE="xpu"
export BK_SETUP_ENV_SCRIPT=".buildkite/k3_harness/setup-lmcache-only-env.sh"

# XPU path for this phase is single-pod only (no baseline server).
export LAUNCH_BASELINE="false"
# Let vLLM pick backend on XPU.
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
# Turn on verbose vLLM logs by default to debug device-type inference issues.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
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
