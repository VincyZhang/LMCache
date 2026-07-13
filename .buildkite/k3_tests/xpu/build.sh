#!/usr/bin/env sh
# Build and push the XPU runtime image into the host-local registry.
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../../.." && pwd)"

LOCAL_REGISTRY="${LOCAL_REGISTRY:-127.0.0.1:5000}"
IMAGE_TAG="${IMAGE_TAG:-${LOCAL_REGISTRY%/}/lmcache/xpu-ci:latest}"
BUILD_CONTEXT="${BUILD_CONTEXT:-${BUILDKITE_BUILD_CHECKOUT_PATH:-${REPO_ROOT}}}"

echo "[xpu-build] building ${IMAGE_TAG}"
/kaniko/executor \
  --context=dir://${BUILD_CONTEXT} \
  --dockerfile="${BUILD_CONTEXT}/docker/Dockerfile.xpu" \
  --destination="${IMAGE_TAG}" \
  --insecure \
  --insecure-pull

echo "[xpu-build] pushed ${IMAGE_TAG}"