# Buildkite Web UI Setup: XPU Smoke Test

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "xpu"`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

This pipeline is heavier than the regular K8s tests because it has two steps:
1. build the XPU image and push it to the host-local registry
2. run the smoke test in a separate agent-stack-k8s job pod

With this trigger filter, the pipeline only starts when the PR has the `xpu`
label. If the label is missing, the XPU pipeline is not triggered.

## Required host setup

Before creating the pipeline, prepare the machine that will run the `xpu` queue:

1. Run [setup-cluster.sh](../../k3_harness/setup-cluster.sh) to install K3s, the GPU Operator, and the local CI base image.
2. Run [install-agent-stack.sh](../../k3_harness/install-agent-stack.sh) with a Buildkite agent token and a GitHub token.
3. Make sure the host-local registry is reachable as `127.0.0.1:5000` from the K8s node that runs the pods.

## Queue and registry notes

- The pipeline uses the `xpu` queue.
- The build step runs in a K8s pod with `hostNetwork: true` so it can push to the host-local registry.
- The test pod pulls `127.0.0.1:5000/lmcache/xpu-ci:latest` from the node side, so the registry must be present on the same host as the K8s node.
- The XPU test pod requests `deviceclass.resource.kubernetes.io/gpu.intel.com` through DRA.

## Buildkite UI snippet

If you want to create the pipeline manually, paste this into the Steps editor:

```yaml
agents:
  queue: "xpu"

steps:
  - label: ":pipeline: Upload pipeline"
    command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh .buildkite/k3_tests/xpu/pipeline.yml
```

## What this pipeline does

- Builds `lmcache/xpu-ci:latest` from `docker/Dockerfile.xpu`
- Pushes the image to the host-local registry
- Runs the XPU smoke test on the `xpu` queue
- Verifies `torch.xpu.is_available()` inside the job pod
