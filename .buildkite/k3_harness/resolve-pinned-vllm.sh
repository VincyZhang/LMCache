#!/usr/bin/env bash
# Resolve the verified CUDA vLLM nightly version that this build should install.
#
# Resolution order (first non-empty wins):
#   1. PINNED_VLLM_VERSION env var  -- explicit per-build override.
#   2. verified_runtimes.json fetched from the tracking branch -- the most
#      recent CUDA runtime that the canary build verified.
#   3. Empty string -- caller falls back to "latest nightly".
#
# The pin file's first non-blank, non-comment line is the bare version
# (kept this way so older `head -n1` consumers still work). Subsequent
# key=value lines carry pre-resolved metadata so consumers can install
# the wheel without making any extra API calls:
#   short_sha=<short-commit-sha>
#   full_sha=<40-char-commit-sha>
#   archive_index_url=https://wheels.vllm.ai/<full-sha>/cu130
#
# Toggles:
#   USE_PINNED_VLLM=false  -- skip step 2 (always probe the latest nightly).
#                             Used by the canary build itself, since pinning
#                             to its own previous result would defeat the
#                             purpose of a freshness check.
#
# Usage:
#   source .buildkite/k3_harness/resolve-pinned-vllm.sh
#   echo "Resolved: ${PINNED_VLLM_VERSION:-<unpinned, using nightly>}"
#   echo "Archive : ${PINNED_VLLM_ARCHIVE_INDEX_URL:-<none>}"
#
# After sourcing the following are set (possibly empty) and exported:
#   PINNED_VLLM_VERSION         -- e.g. 0.23.1rc1.dev508+gc6dd32a81
#   PINNED_VLLM_FULL_SHA        -- 40-char commit SHA, if recorded in pin
#   PINNED_VLLM_ARCHIVE_INDEX_URL
#                               -- permanent PEP 503 simple index, if recorded
#
# The script never fails the build: a missing/unreachable pin file just
# falls through to the unpinned path, mirroring the previous behaviour.

# Allow re-sourcing without "unbound variable" complaints under set -u.
PINNED_VLLM_VERSION="${PINNED_VLLM_VERSION:-}"
PINNED_VLLM_ARCHIVE_INDEX_URL="${PINNED_VLLM_ARCHIVE_INDEX_URL:-}"
PINNED_VLLM_FULL_SHA="${PINNED_VLLM_FULL_SHA:-}"
USE_PINNED_VLLM="${USE_PINNED_VLLM:-true}"

# Override URL if you mirror the pin file elsewhere (e.g. an internal
# raw-file proxy for offline CI).
LMCACHE_VLLM_PIN_URL="${LMCACHE_VLLM_PIN_URL:-https://raw.githubusercontent.com/LMCache/LMCache/buildkite_latest_tested_vllm/verified_runtimes.json}"

if [[ -z "${PINNED_VLLM_VERSION}" && "${USE_PINNED_VLLM}" == "true" ]]; then
    if command -v curl >/dev/null 2>&1; then
        # 5s connect, 10s total -- pin lookup must never dominate setup time.
        fetched="$(curl -fsSL --connect-timeout 5 --max-time 10 \
            "${LMCACHE_VLLM_PIN_URL}" 2>/dev/null || true)"
        eval "$(printf '%s' "${fetched}" | python3 -c '
import json
import shlex
import sys

try:
    record = json.load(sys.stdin)["runtimes"]["cuda"]["cu130"]
except (KeyError, json.JSONDecodeError):
    record = {}
for name, key in (
    ("PINNED_VLLM_VERSION", "vllm_version"),
    ("PINNED_VLLM_ARCHIVE_INDEX_URL", "archive_index_url"),
    ("PINNED_VLLM_FULL_SHA", "vllm_source_commit"),
):
    print(f"{name}={shlex.quote(record.get(key) or str())}")
')"
    fi
fi

export PINNED_VLLM_VERSION
export PINNED_VLLM_ARCHIVE_INDEX_URL
export PINNED_VLLM_FULL_SHA

if [[ -n "${PINNED_VLLM_VERSION}" ]]; then
    echo "[resolve-pinned-vllm] Pinned vLLM version: ${PINNED_VLLM_VERSION}" >&2
    if [[ -n "${PINNED_VLLM_ARCHIVE_INDEX_URL}" ]]; then
        echo "[resolve-pinned-vllm] Archive index:" \
             "${PINNED_VLLM_ARCHIVE_INDEX_URL}" >&2
    fi
else
    echo "[resolve-pinned-vllm] No pinned vLLM; will install latest nightly" >&2
fi
