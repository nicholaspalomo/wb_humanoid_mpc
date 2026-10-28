#!/bin/sh
# ------------------------------------------------------------------
# reset_stale_bazel_cache.sh - Delete the dev container's Bazel disk cache when the image's packages changed.
#
# Bazel keys an action by its command line and its inputs, not by the compiler binary or the system libraries that
# the image installs, so a disk cache filled by one image would hand compiled outputs of the old compiler and the old
# headers to the next (.devcontainer/bazel.bazelrc). The container's start command (docker-compose.yaml) runs this
# before any build: it fingerprints the machine and the installed Debian packages, which hold the compiler, the system
# libraries and robotpkg's Pinocchio, and deletes the disk cache when the fingerprint differs from the one the cache
# was filled under, or when the cache has none. The repository cache holds downloads by their checksum and is kept.
#
# Usage:
#   sh .devcontainer/reset_stale_bazel_cache.sh <cache root>
#
# POSIX sh: the container's BASH_ENV makes every bash script source the shell setup first.
# ------------------------------------------------------------------
set -eu

if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "usage: $0 <cache root>" >&2
    exit 2
fi
cache_root="$1"
disk_cache="${cache_root}/bazel-disk"
stamp="${cache_root}/image_fingerprint"

fingerprint="$(
    {
        uname -m
        dpkg-query -W -f='${Package} ${Version} ${Architecture}\n' | LC_ALL=C sort
    } | sha256sum | cut -d ' ' -f 1
)"

if [ -d "${disk_cache}" ] && [ "$(cat "${stamp}" 2>/dev/null || true)" != "${fingerprint}" ]; then
    echo "reset_stale_bazel_cache: the image's packages differ from the ones ${disk_cache} was filled under;" \
        "deleting it" >&2
    rm -rf "${disk_cache}"
fi
mkdir -p "${cache_root}"
printf '%s\n' "${fingerprint}" > "${stamp}"
