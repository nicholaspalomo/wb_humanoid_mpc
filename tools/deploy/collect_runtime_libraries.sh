#!/bin/sh
# Finds the shared libraries the robot bundle's ELF files need, for the robot-runtime image (docker/Dockerfile):
#
#   collect_runtime_libraries.sh collect <bundle> <stage>   # in the builder stage, where every library is installed
#   collect_runtime_libraries.sh check <bundle>             # in the runtime image: fails when one is not found
#
# `collect` writes the Ubuntu packages that hold the libraries to <stage>/packages.txt, one per line, for the runtime
# image to install with apt, and copies to <stage>/root/ (at their paths) the libraries it must not install that way:
# those of /opt/openrobots (Pinocchio and coal, which are not Ubuntu packages) and those of COPIED_PACKAGES below. A
# library in the bundle itself (libmujoco in a binary's runfiles) is left where it is. Both commands fail, naming the
# file, when ldd cannot find a library.
#
# POSIX sh: it runs in the runtime image, which has no bash.
set -eu

# The robot binaries link GLFW, GLEW and libGL's dispatch library because the MuJoCo viewer is compiled in, but
# robot-runtime runs without it (--headless). Installed with apt, these packages pull in Mesa and LLVM (some 180 MB)
# through libglx0's dependency on libglx-mesa0, so their libraries are copied instead; robot-sim installs the packages
# themselves, with Mesa, over the copies.
# LINT.IfChange(copied_packages)
COPIED_PACKAGES="libgl1 libglew2.2 libglfw3 libglvnd0 libglx0"
# LINT.ThenChange(//docker/Dockerfile:robot_images)

usage() {
    echo "usage: collect_runtime_libraries.sh collect <bundle> <stage> | check <bundle>" >&2
    exit 2
}

elf_files() {
    find "$1" -type f \( -perm -u+x -o -name '*.so*' \) | sort | while read -r file; do
        if [ "$(head -c 4 "${file}" | od -An -c | tr -d ' ')" = "177ELF" ]; then
            echo "${file}"
        fi
    done
}

# The `name => path` lines of ldd for every ELF file of the bundle, as "file<TAB>path"; fails on a missing library.
resolved_libraries() {
    missing=0
    for file in $(elf_files "$1"); do
        output="$(ldd "${file}" 2>&1 || true)"
        case "${output}" in
            *"not a dynamic executable"*) continue ;;
        esac
        if echo "${output}" | grep -q "not found"; then
            echo "collect_runtime_libraries: ${file} needs libraries that are not installed:" >&2
            echo "${output}" | grep "not found" >&2
            missing=1
        fi
        echo "${output}" | awk -v file="${file}" '$2 == "=>" && $3 ~ /^\// { print file "\t" $3 }'
    done
    [ "${missing}" -eq 0 ]
}

[ "$#" -ge 2 ] || usage
command="$1"
bundle="$2"
case "${command}" in
    check)
        [ "$#" -eq 2 ] || usage
        resolved_libraries "${bundle}" > /dev/null
        echo "collect_runtime_libraries: every library of ${bundle} is found"
        ;;
    collect)
        [ "$#" -eq 3 ] || usage
        stage="$3"
        mkdir -p "${stage}/root"
        list="$(mktemp)"
        resolved_libraries "${bundle}" > "${list}"
        : > "${stage}/packages.txt"
        : > "${stage}/copied.txt"
        cut -f 2 "${list}" | sort -u | while read -r library; do
            case "${library}" in
                "${bundle}"/*) continue ;;
            esac
            real="$(readlink -f "${library}")"
            package=""
            case "${library}" in
                /opt/openrobots/*) ;;
                *) package="$(dpkg -S "${real}" | head -n 1 | cut -d: -f1)" ;;
            esac
            case " ${COPIED_PACKAGES} " in
                *" ${package} "*) package="" ;;
            esac
            if [ -n "${package}" ]; then
                echo "${package}" >> "${stage}/packages.txt"
                continue
            fi
            # The file and, under the name the binaries ask for, a link to it; in /usr/lib, which /lib links to.
            directory="$(readlink -f "$(dirname "${library}")")"
            mkdir -p "${stage}/root${directory}"
            cp "${real}" "${stage}/root${directory}/$(basename "${real}")"
            if [ "$(basename "${library}")" != "$(basename "${real}")" ]; then
                ln -sf "$(basename "${real}")" "${stage}/root${directory}/$(basename "${library}")"
            fi
            echo "${library}" >> "${stage}/copied.txt"
        done
        sort -u -o "${stage}/packages.txt" "${stage}/packages.txt"
        rm -f "${list}"
        echo "collect_runtime_libraries: $(wc -l < "${stage}/packages.txt") Ubuntu packages to install," \
            "$(wc -l < "${stage}/copied.txt") libraries copied:" $(cat "${stage}/copied.txt")
        ;;
    *) usage ;;
esac
