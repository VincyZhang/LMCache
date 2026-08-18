#!/usr/bin/env bash
# Trigger Buildkite's Intel XPU validation pipeline and wait for its verdict.
#
# Required environment:
#   BUILDKITE_API_TOKEN, BUILDKITE_ORGANIZATION, BUILDKITE_XPU_PIPELINE
#   GITHUB_SHA, GITHUB_REF_NAME, XPU_IMAGE_REF, XPU_IMAGE_TAG,
#   XPU_IMAGE_DIGEST, LMCACHE_XPU_ARTIFACT_URL, LMCACHE_XPU_VERSION,
#   LMCACHE_XPU_WHEEL_SHA256
set -euo pipefail

: "${BUILDKITE_API_TOKEN:?BUILDKITE_API_TOKEN is required}"
: "${BUILDKITE_ORGANIZATION:?BUILDKITE_ORGANIZATION is required}"
: "${BUILDKITE_XPU_PIPELINE:?BUILDKITE_XPU_PIPELINE is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"
: "${XPU_IMAGE_REF:?XPU_IMAGE_REF is required}"
: "${XPU_IMAGE_DIGEST:?XPU_IMAGE_DIGEST is required}"
: "${LMCACHE_XPU_ARTIFACT_URL:?LMCACHE_XPU_ARTIFACT_URL is required}"
: "${LMCACHE_XPU_VERSION:?LMCACHE_XPU_VERSION is required}"
: "${LMCACHE_XPU_WHEEL_SHA256:?LMCACHE_XPU_WHEEL_SHA256 is required}"

api_base="https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION}/pipelines/${BUILDKITE_XPU_PIPELINE}"
payload="$(jq -n \
    --arg commit "${GITHUB_SHA}" \
    --arg branch "${GITHUB_REF_NAME}" \
    --arg message "XPU nightly validation for ${GITHUB_SHA}" \
    --arg image_ref "${XPU_IMAGE_REF}" \
    --arg image_tag "${XPU_IMAGE_TAG:-nightly}" \
    --arg image_digest "${XPU_IMAGE_DIGEST}" \
    --arg artifact_url "${LMCACHE_XPU_ARTIFACT_URL}" \
    --arg lmcache_version "${LMCACHE_XPU_VERSION}" \
    --arg wheel_sha256 "${LMCACHE_XPU_WHEEL_SHA256}" \
    '{commit: $commit, branch: $branch, message: $message, env: {
      XPU_IMAGE_REF: $image_ref,
      XPU_IMAGE_TAG: $image_tag,
      XPU_IMAGE_DIGEST: $image_digest,
      LMCACHE_XPU_ARTIFACT_URL: $artifact_url,
      LMCACHE_XPU_VERSION: $lmcache_version,
      LMCACHE_XPU_WHEEL_SHA256: $wheel_sha256,
      VERIFY_AND_PIN_XPU: "true"
    }}')"

build="$(curl -fsSL -X POST \
    -H "Authorization: Bearer ${BUILDKITE_API_TOKEN}" \
    -H 'Content-Type: application/json' \
    --data "${payload}" "${api_base}/builds")"
build_number="$(jq -er '.number' <<<"${build}")"
build_url="$(jq -er '.web_url' <<<"${build}")"
echo "Triggered Buildkite XPU validation: ${build_url}" >&2
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf 'build_url=%s\n' "${build_url}" >> "${GITHUB_OUTPUT}"
fi

deadline=$((SECONDS + ${BUILDKITE_XPU_TIMEOUT_SECONDS:-14400}))
while (( SECONDS < deadline )); do
    build="$(curl -fsSL \
        -H "Authorization: Bearer ${BUILDKITE_API_TOKEN}" \
        "${api_base}/builds/${build_number}")"
    state="$(jq -er '.state' <<<"${build}")"
    case "${state}" in
        passed)
            echo "Buildkite XPU validation passed: ${build_url}" >&2
            exit 0
            ;;
        failed|canceled|blocked|skipped|not_run)
            echo "Buildkite XPU validation ${state}: ${build_url}" >&2
            exit 1
            ;;
        *)
            echo "Buildkite XPU validation is ${state}: ${build_url}" >&2
            sleep 30
            ;;
    esac
done

echo "Timed out waiting for Buildkite XPU validation: ${build_url}" >&2
exit 1