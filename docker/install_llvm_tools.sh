#!/bin/sh
# Installs the repository's LLVM tools: clang-format, which formats the C++ (`make format`), and clang-tidy, which lints
# it (`make lint-tidy`, tools/clang_tidy/README.md).
#
# The dev image (docker/Dockerfile), CI's build job (.github/workflows/build_test.yml) and the local CI emulator
# (tools/ci_local.sh) all run this one script, so all three lint with the same clang-tidy.
#
# - clang-format comes from Ubuntu's own archive. CI's format job (.github/workflows/format_test.yml) installs that same
#   package on its runner, and a clang-format of another build may format differently, so the hook and CI would
#   disagree. Ubuntu's archive also keeps the package for the life of the release, which apt.llvm.org does not promise
#   for an old major version.
# - clang-tidy comes from apt.llvm.org, which carries newer major versions than Ubuntu. From 19 on, clang reads Abseil's
#   nullability annotations (absl_nonnull, absl_nullable) as _Nonnull / _Nullable, and the checks of .clang-tidy and
#   tools/clang_tidy/sweep.clang-tidy exist. Its packages pull in a clang compiler (clang-N), which nothing may run by
#   hand (AGENTS.md, "Builds share one machine's memory"); this script links no unversioned `clang` or `clang++`.
#
# Usage, as root: sh docker/install_llvm_tools.sh format|tidy|all
# CLANG_FORMAT_VERSION and CLANG_TIDY_VERSION in the environment override the major versions below; the Dockerfile
# passes its build arguments that way. The apt lists are left in place, as docker/install_robotpkg.sh leaves them; the
# image deletes them in the same layer. POSIX sh, so that it runs in any container before anything else is set up.
set -eu

# LINT.IfChange(llvm_versions)
CLANG_FORMAT_VERSION="${CLANG_FORMAT_VERSION:-18}"
CLANG_TIDY_VERSION="${CLANG_TIDY_VERSION:-21}"
# LINT.ThenChange(//docker/Dockerfile:clang_format_version, //docker/Dockerfile:clang_tidy_version, //tools/clang_tidy/run_clang_tidy.sh:clang_tidy_version)

# apt.llvm.org's signing key ("Sylvestre Ledru - Debian LLVM packages"). The key is checked against its fingerprint
# before apt is told to trust it, as docker/install_robotpkg.sh checks robotpkg's.
LLVM_URL="https://apt.llvm.org"
LLVM_KEY_FINGERPRINT="6084F3CF814B57C1CF12EFD515CF4D18AF4F7421"
LLVM_KEYRING="/etc/apt/keyrings/llvm.asc"

usage() {
  echo "usage: sh docker/install_llvm_tools.sh format|tidy|all" >&2
  exit 2
}

install_clang_format() {
  apt-get update
  apt-get install -y --no-install-recommends "clang-format-${CLANG_FORMAT_VERSION}"
  ln -sf "/usr/bin/clang-format-${CLANG_FORMAT_VERSION}" /usr/bin/clang-format
  clang-format --version
}

install_clang_tidy() {
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl gnupg

  install -d -m 0755 /etc/apt/keyrings
  download="$(mktemp)"
  gnupg_home="$(mktemp -d)"
  trap 'rm -rf "${download}" "${gnupg_home}"' EXIT
  curl -fsSL "${LLVM_URL}/llvm-snapshot.gpg.key" -o "${download}"
  fingerprint="$(GNUPGHOME="${gnupg_home}" gpg --batch --show-keys --with-colons "${download}" \
    | awk -F: '$1 == "fpr" { print $10; exit }')"
  if [ "${fingerprint}" != "${LLVM_KEY_FINGERPRINT}" ]; then
    echo "install_llvm_tools.sh: the apt.llvm.org key has fingerprint '${fingerprint}', expected" \
      "${LLVM_KEY_FINGERPRINT}." >&2
    exit 1
  fi
  install -m 0644 "${download}" "${LLVM_KEYRING}"

  # One repository, of the clang-tidy major version only: clang-format stays Ubuntu's.
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
  echo "deb [signed-by=${LLVM_KEYRING}] ${LLVM_URL}/${codename}/ llvm-toolchain-${codename}-${CLANG_TIDY_VERSION} main" \
    > /etc/apt/sources.list.d/llvm.list

  apt-get update
  apt-get install -y --no-install-recommends "clang-tidy-${CLANG_TIDY_VERSION}"
  ln -sf "/usr/bin/clang-tidy-${CLANG_TIDY_VERSION}" /usr/bin/clang-tidy
  clang-tidy --version
}

[ "$#" -eq 1 ] || usage
case "$1" in
  format) install_clang_format ;;
  tidy) install_clang_tidy ;;
  all)
    install_clang_format
    install_clang_tidy
    ;;
  *) usage ;;
esac
