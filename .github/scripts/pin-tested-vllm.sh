#!/usr/bin/env bash
# Pin a verified vLLM runtime to a dedicated CI tracking branch.
#
# CI_PLATFORM selects the target branch. Callers explicitly supply the backend
# key, runtime ID, and installation form for the pin.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export GIT_TERMINAL_PROMPT=0

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

resolve_ci_context() {
    case "${CI_PLATFORM}" in
        buildkite)
            PIN_VLLM_BRANCH="${PIN_VLLM_BRANCH:-buildkite_latest_tested_vllm}"
            ;;
        github_actions)
            PIN_VLLM_BRANCH="${PIN_VLLM_BRANCH:-github_nightly_tested_vllm}"
            ;;
        *) fail "unknown CI_PLATFORM '${CI_PLATFORM}'" ;;
    esac
}

resolve_wheel_runtime() {
    VLLM_VERSION="${VLLM_VERSION:-$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)}"
    if [[ -z "${VLLM_VERSION}" ]]; then
        [[ "${PIN_VLLM_STATUS}" != tested ]] || fail "could not read vllm.__version__ from the live environment"
        VLLM_VERSION="unknown"
    fi

    VLLM_SHORT_SHA="${VLLM_VERSION##*+g}"
    if [[ "${VLLM_SHORT_SHA}" == "${VLLM_VERSION}" || ! "${VLLM_SHORT_SHA}" =~ ^[0-9a-f]+$ ]]; then
        VLLM_SHORT_SHA=""
    fi

    VLLM_SOURCE_COMMIT="${VLLM_SOURCE_COMMIT:-}"
    if [[ -z "${VLLM_SOURCE_COMMIT}" && -n "${VLLM_SHORT_SHA}" ]]; then
        local auth_args=()
        if [[ -n "${GITHUB_TOKEN:-}" ]]; then
            auth_args=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
        fi
        VLLM_SOURCE_COMMIT="$(curl -fsSL --connect-timeout 5 --max-time 10 \
            -H "Accept: application/vnd.github+json" \
            "${auth_args[@]+"${auth_args[@]}"}" \
            "https://api.github.com/repos/vllm-project/vllm/commits/${VLLM_SHORT_SHA}" \
            2>/dev/null | jq -r '.sha // empty' || true)"
    fi

    VLLM_ARCHIVE_INDEX="${VLLM_ARCHIVE_INDEX:-}"
    if [[ "${VLLM_WHEEL_ARCHIVE_INDEX}" == 1 && -z "${VLLM_ARCHIVE_INDEX}" && -n "${VLLM_SOURCE_COMMIT}" ]]; then
        VLLM_ARCHIVE_INDEX="https://wheels.vllm.ai/${VLLM_SOURCE_COMMIT}/${PIN_RUNTIME_ID}"
    fi

    VLLM_IMAGE_TAG=""
    VLLM_IMAGE_REF=""
    VLLM_IMAGE_DIGEST=""
}

resolve_image_runtime() {
    VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-}"
    VLLM_IMAGE_REF="${VLLM_IMAGE_REF:-}"
    VLLM_IMAGE_DIGEST="${VLLM_IMAGE_DIGEST:-}"
    if [[ "${PIN_VLLM_STATUS}" == tested ]]; then
        [[ -n "${VLLM_IMAGE_REF}" ]] || fail "image pins require VLLM_IMAGE_REF"
        [[ "${VLLM_IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "image pins require VLLM_IMAGE_DIGEST as a sha256 digest"
    fi

    VLLM_VERSION=""
    VLLM_SOURCE_COMMIT=""
    VLLM_ARCHIVE_INDEX=""
}

resolve_runtime() {
    case "${VLLM_INSTALLATION_FORM}" in
        wheel) resolve_wheel_runtime ;;
        image) resolve_image_runtime ;;
        *) fail "unknown VLLM_INSTALLATION_FORM '${VLLM_INSTALLATION_FORM}'" ;;
    esac
}

prepare_work_dir() {
    WORK_DIR="/tmp/pin_vllm_$$"
    trap 'rm -rf "${WORK_DIR}"' EXIT
    if [[ "${PIN_VLLM_DRY_RUN}" == 1 ]]; then
        mkdir -p "${WORK_DIR}"
        return
    fi

    [[ -n "${GITHUB_TOKEN:-}" ]] || fail "GITHUB_TOKEN is required to update the pin state"
    local repo_url="https://x-access-token:${GITHUB_TOKEN}@github.com/LMCache/LMCache.git"
    if git clone --depth=1 --branch "${PIN_VLLM_BRANCH}" "${repo_url}" "${WORK_DIR}" 2>/dev/null; then
        return
    fi

    mkdir -p "${WORK_DIR}"
    git -C "${WORK_DIR}" init -q
    git -C "${WORK_DIR}" remote add origin "${repo_url}"
    git -C "${WORK_DIR}" checkout --orphan "${PIN_VLLM_BRANCH}"
    git -C "${WORK_DIR}" rm -rf --cached . >/dev/null 2>&1 || true
    find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
}

write_runtime_state() {
    local state_file="${WORK_DIR}/verified_runtimes.json"

    PIN_BACKEND="${PIN_BACKEND}" \
    PIN_RUNTIME_ID="${PIN_RUNTIME_ID}" \
    VLLM_INSTALLATION_FORM="${VLLM_INSTALLATION_FORM}" \
    VLLM_VERSION="${VLLM_VERSION}" \
    VLLM_ARCHIVE_INDEX="${VLLM_ARCHIVE_INDEX}" \
    VLLM_IMAGE_REF="${VLLM_IMAGE_REF}" \
    VLLM_IMAGE_DIGEST="${VLLM_IMAGE_DIGEST}" \
    python3 - "${state_file}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    state = json.loads(path.read_text())
except FileNotFoundError:
    state = {"schema_version": 1, "runtimes": {}}

record = {
    "installation_form": os.environ["VLLM_INSTALLATION_FORM"],
}
if record["installation_form"] == "wheel":
    record.update(
        vllm_version=os.environ["VLLM_VERSION"],
        archive_index_url=os.environ["VLLM_ARCHIVE_INDEX"],
    )
else:
    record.update(
        image_ref=os.environ["VLLM_IMAGE_REF"],
        image_digest=os.environ["VLLM_IMAGE_DIGEST"],
    )
state.setdefault("schema_version", 1)
state.setdefault("runtimes", {}).setdefault(os.environ["PIN_BACKEND"], {})[
    os.environ["PIN_RUNTIME_ID"]
] = record
path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
PY
}

publish_runtime_state() {
    if [[ "${PIN_VLLM_DRY_RUN}" == 1 ]]; then
        cat "${WORK_DIR}/verified_runtimes.json"
        return
    fi

    git -C "${WORK_DIR}" add verified_runtimes.json
    git -C "${WORK_DIR}" -c user.email="ci@lmcache.ai" -c user.name="LMCache CI" \
        commit -m "Pin ${PIN_BACKEND} runtime: ${PIN_RUNTIME_ID}"
    git -C "${WORK_DIR}" push origin "HEAD:${PIN_VLLM_BRANCH}"
}

CI_PLATFORM="${CI_PLATFORM:-buildkite}"
PIN_BACKEND="${PIN_BACKEND:-}"
PIN_VLLM_STATUS="${PIN_VLLM_STATUS:-tested}"
PIN_VLLM_REASON="${PIN_VLLM_REASON:-}"
PIN_RUNTIME_ID="${PIN_RUNTIME_ID:-}"
VLLM_INSTALLATION_FORM="${VLLM_INSTALLATION_FORM:-}"
VLLM_WHEEL_ARCHIVE_INDEX="${VLLM_WHEEL_ARCHIVE_INDEX:-0}"
PIN_VLLM_DRY_RUN="${PIN_VLLM_DRY_RUN:-0}"

[[ -n "${PIN_BACKEND}" ]] || fail "PIN_BACKEND is required"
[[ -n "${PIN_RUNTIME_ID}" ]] || fail "PIN_RUNTIME_ID is required"
case "${VLLM_INSTALLATION_FORM}" in
    wheel|image) ;;
    *) fail "VLLM_INSTALLATION_FORM must be wheel or image" ;;
esac
[[ "${VLLM_WHEEL_ARCHIVE_INDEX}" =~ ^[01]$ ]] || fail "VLLM_WHEEL_ARCHIVE_INDEX must be 0 or 1"

resolve_ci_context
resolve_runtime

if [[ "${PIN_VLLM_STATUS}" != tested ]]; then
    echo "[pin-tested-vllm] ${PIN_BACKEND}/${PIN_RUNTIME_ID} failed: ${PIN_VLLM_REASON}" >&2
    exit 0
fi

prepare_work_dir
write_runtime_state
publish_runtime_state
echo "[pin-tested-vllm] pinned ${PIN_BACKEND}/${PIN_RUNTIME_ID}" >&2