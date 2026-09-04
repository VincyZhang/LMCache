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

echo "--- :mag: Verify prebuilt XPU vLLM stack ABI compatibility"
# IMPORTANT: the XPU CI image ships a matched torch + vllm + vllm_xpu_kernels
# stack. Reinstalling vLLM from PyPI can silently break ABI compatibility and
# fail later with errors like:
#   undefined symbol ... in vllm_xpu_kernels/_C.abi3.so
# Keep the prebuilt stack intact and fail fast here with actionable diagnostics.
python - <<'PY'
import importlib
import traceback

import torch

print(f"torch={torch.__version__}, torch.xpu.is_available={torch.xpu.is_available()}")

for mod in ("vllm", "vllm_xpu_kernels"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}={getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        print(f"ERROR importing {mod}: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise SystemExit(
            "XPU runtime stack mismatch detected. "
            "Use a CI image with matched torch/vllm/vllm_xpu_kernels versions "
            "or pin/install them together as one known-good set."
        )
PY

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
