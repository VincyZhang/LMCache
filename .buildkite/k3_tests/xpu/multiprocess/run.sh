#!/usr/bin/env bash
set -euo pipefail

# Reuse multiprocess test harness, but switch to the XPU environment/bootstrap.
export BK_TEST_BACKEND="xpu"
export BK_SETUP_ENV_SCRIPT=".buildkite/k3_harness/setup-env-xpu.sh"

# XPU path for this phase is single-pod only (no baseline server).
export LAUNCH_BASELINE="false"
# Let vLLM pick backend on XPU.
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"

exec bash "$(cd "$(dirname "$0")/../.." && pwd)/multiprocess/run.sh" "$@"
