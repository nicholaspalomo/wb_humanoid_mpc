#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🚀 Running local CI emulator using official GitHub Actions runner container (ros:jazzy)..."

rm -rf "${SCRIPT_DIR}/.bazel" "${SCRIPT_DIR}"/bazel-* 2>/dev/null || true

docker run --rm \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  ros:jazzy \
  bash -c "
    set -eo pipefail
    echo '=== 1. Installing Base System Dependencies ==='
    apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg2 lsb-release software-properties-common \
      ca-certificates locales git gettext-base sudo wget build-essential

    echo '=== 2. Installing Bazelisk ==='
    curl -sSL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    chmod +x /usr/local/bin/bazel
    bazel --version

    echo '=== 3. Installing Project Dependencies ==='
    grep -v '^\s*#' dependencies.txt | envsubst | xargs apt-get install -y --no-install-recommends

    echo '=== 4. Running CI Build ==='
    source ./setup_env.sh
    make build-all

    echo '=== 5. Running CI Tests ==='
    make test-all

    echo '✅ Local CI Verification Passed Successfully!'
    rm -rf /workspace/.bazel /workspace/bazel-*
"

rm -rf "${SCRIPT_DIR}/.bazel" "${SCRIPT_DIR}"/bazel-* 2>/dev/null || true
