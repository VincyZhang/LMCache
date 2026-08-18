#!/usr/bin/env bash
# Resolve the verified Intel XPU vLLM runtime from the unified pin state.
#
# Explicit XPU_IMAGE_REF wins so GitHub Actions can send a candidate runtime to
# Buildkite. Ordinary XPU CI fetches the most recently verified runtime instead.

PINNED_XPU_IMAGE_REF="${PINNED_XPU_IMAGE_REF:-}"
PINNED_XPU_IMAGE_TAG="${PINNED_XPU_IMAGE_TAG:-}"
PINNED_XPU_IMAGE_DIGEST="${PINNED_XPU_IMAGE_DIGEST:-}"
USE_PINNED_XPU="${USE_PINNED_XPU:-true}"
LMCACHE_RUNTIME_PIN_URL="${LMCACHE_RUNTIME_PIN_URL:-https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/verified_runtimes.json}"

if [[ -z "${PINNED_XPU_IMAGE_REF}" && "${USE_PINNED_XPU}" == "true" ]] && command -v curl >/dev/null 2>&1; then
    state="$(curl -fsSL --connect-timeout 5 --max-time 10 "${LMCACHE_RUNTIME_PIN_URL}" 2>/dev/null || true)"
    if [[ -n "${state}" ]]; then
        eval "$(printf '%s' "${state}" | python3 -c '
import json
import shlex
import sys

try:
    record = json.load(sys.stdin)["runtimes"]["xpu"]["linux-intel-xpu"]
except (KeyError, json.JSONDecodeError):
    record = {}
for name, key in (
    ("PINNED_XPU_IMAGE_REF", "image_ref"),
    ("PINNED_XPU_IMAGE_TAG", "image_tag"),
    ("PINNED_XPU_IMAGE_DIGEST", "image_digest"),
):
    print(f"{name}={shlex.quote(record.get(key) or str())}")
')"
    fi
fi

export PINNED_XPU_IMAGE_REF PINNED_XPU_IMAGE_TAG PINNED_XPU_IMAGE_DIGEST

if [[ -n "${PINNED_XPU_IMAGE_REF}" ]]; then
    echo "[resolve-pinned-xpu] Using verified runtime: ${PINNED_XPU_IMAGE_REF}" >&2
else
    echo "[resolve-pinned-xpu] No verified runtime available" >&2
fi