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

# LINT.IfChange(ubuntu_release)
# The workflow's container, which is the dev image's base.
CI_IMAGE="ubuntu:24.04"
# LINT.ThenChange(//docker/Dockerfile:ubuntu_release, //.github/workflows/build_test.yml:container_image)

echo "🚀 Running local CI emulator in the workflow's container (${CI_IMAGE})..."

rm -rf "${SCRIPT_DIR}/.bazel" "${SCRIPT_DIR}"/bazel-* 2>/dev/null || true

# LINT.IfChange(noninteractive_apt)
# The workflow's job environment: tzdata, which python3-pip pulls in, would otherwise wait for an answer.
CI_ENVIRONMENT=(-e DEBIAN_FRONTEND=noninteractive -e TZ=Etc/UTC)
# LINT.ThenChange(//.github/workflows/build_test.yml:noninteractive_apt)

docker run --rm "${RUNNER_RESOURCES[@]}" "${CI_ENVIRONMENT[@]}" \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  "${CI_IMAGE}" \
  bash -c "
    set -eo pipefail
    echo '=== 1. Installing Base System Dependencies ==='
    # LINT.IfChange(ci_base_dependencies)
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release \
      locales git gettext-base sudo wget build-essential python3-pip python3-venv
    # LINT.ThenChange(//.github/workflows/build_test.yml:system_dependencies)

    echo '=== 2. Installing Pinocchio (robotpkg) ==='
    sh docker/install_robotpkg.sh
    # The dev image's environment for /opt/openrobots; no LD_LIBRARY_PATH (docker/Dockerfile says why).
    # LINT.IfChange(robotpkg_environment)
    export PATH=/opt/openrobots/bin:\${PATH}
    export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages
    export CMAKE_PREFIX_PATH=/opt/openrobots
    export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig
    # LINT.ThenChange(//docker/Dockerfile:robotpkg_environment, //.github/workflows/build_test.yml:robotpkg_environment)

    echo '=== 3. Installing Bazelisk ==='
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

    echo '=== 4. Installing Project Dependencies ==='
    grep -v '^\s*#' dependencies.txt | envsubst | xargs apt-get install -y --no-install-recommends

    echo '=== 5. Running CI Build ==='
    source ./setup_env.sh
    make build-all

    echo '=== 6. Running CI Tests ==='
    make test-all

    echo '=== 7. Linting C++ (clang-tidy) ==='
    # LINT.IfChange(lint_tidy)
    sh docker/install_llvm_tools.sh tidy
    make lint-tidy
    # LINT.ThenChange(//.github/workflows/build_test.yml:lint_tidy, //.bazelrc:clang_tidy_config)

    echo '✅ Local CI Verification Passed Successfully!'
    rm -rf /workspace/.bazel /workspace/bazel-*
"

rm -rf "${SCRIPT_DIR}/.bazel" "${SCRIPT_DIR}"/bazel-* 2>/dev/null || true
