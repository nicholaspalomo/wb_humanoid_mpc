#!/usr/bin/env bash
# Runs .github/workflows/build_test.yml locally: the same container, the same steps, and the resources of the runner
# it runs on. The resources matter. A build that fits a workstation can still fail on the runner, and GitHub's job log
# is the only other place that says so.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# LINT.IfChange(runner_resources)
# A GitHub-hosted ubuntu-latest runner: 4 vCPUs, 16 GB of RAM and a 4 GB swap file.
RUNNER_RESOURCES=(--cpus=4 --memory=16g --memory-swap=20g)
# LINT.ThenChange(//.github/workflows/build_test.yml:runner)

# LINT.IfChange(ros_distro)
echo "🚀 Running local CI emulator using official GitHub Actions runner container (ros:jazzy)..."

rm -rf "${SCRIPT_DIR}/.bazel" "${SCRIPT_DIR}"/bazel-* 2>/dev/null || true

docker run --rm "${RUNNER_RESOURCES[@]}" \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  ros:jazzy \
  bash -c "
# LINT.ThenChange(//docker/Dockerfile:ros_distro, //setup_env.sh:ros_distro, //bazel/ros2.bzl:ros_distro, //bazel/system_libs.bzl:ros_distro)
    set -eo pipefail
    echo '=== 1. Installing Base System Dependencies ==='
    # LINT.IfChange(ci_base_dependencies)
    apt-get update && apt-get install -y --no-install-recommends \
      curl gnupg2 lsb-release software-properties-common \
      locales git gettext-base sudo wget build-essential python3-pip python3-venv
    # LINT.ThenChange(//.github/workflows/build_test.yml:system_dependencies)

    echo '=== 2. Installing Bazelisk ==='
    curl -sSL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    chmod +x /usr/local/bin/bazel
    bazel --version

    # .bazelrc sizes the build by HOST_RAM, and inside a container that is still the host's RAM, not the 16 GB limit.
    # Pin what its factors give for 16 GB (a runner, with a little less usable, may get one job fewer) in the
    # container's own ~/.bazelrc, which Bazel reads after the workspace's and which leaves the checkout's user.bazelrc
    # alone.
    # LINT.IfChange(runner_jobs)
    printf 'build --jobs=4\ntest --local_test_jobs=2\n' > ~/.bazelrc
    # LINT.ThenChange(//.bazelrc:bazel_memory_bound)

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
