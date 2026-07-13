#!/usr/bin/env bash
# Shared helpers for the K8s CI harness.
set -euo pipefail

resolve_kubeconfig() {
    if [[ -n "${KUBECONFIG:-}" ]]; then
        IFS=':' read -r first_kubeconfig _ <<<"${KUBECONFIG}"
        if [[ -f "${first_kubeconfig}" ]]; then
            echo "${KUBECONFIG}"
            return 0
        fi
    fi

    local candidate
    for candidate in \
        /etc/rancher/k3s/k3s.yaml \
        /etc/kubernetes/admin.conf \
        "${HOME}/.kube/config"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "Unable to find a kubeconfig. Set KUBECONFIG to your cluster config." >&2
    return 1
}

export KUBECONFIG="$(resolve_kubeconfig)"