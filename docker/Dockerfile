# The dev image: Ubuntu 24.04 with the toolchain, the system libraries the Bazel build wraps (bazel/system_libs.bzl),
# Pinocchio from robotpkg, and a VNC desktop for the MuJoCo viewer and the operator GUI. There is no ROS: the processes
# talk over ZeroMQ + Protocol Buffers, which Bazel builds from the BCR, and draw with Rerun
# (humanoid_nmpc/docs/distributed_runtime).

# LINT.IfChange(ubuntu_release)
# The Ubuntu release the image, CI and the local CI emulator build on.
ARG BASE_IMAGE=ubuntu:24.04
# LINT.ThenChange(//tools/ci_local.sh:ubuntu_release, //.github/workflows/build_test.yml:container_image)
FROM ${BASE_IMAGE} AS base

ARG CPP_VERSION=20
ARG WB_HUMANOID_MPC_DIR="/wb_humanoid_mpc_ws"
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive
# The ROS base image used to set these. Without them Python, black and the hooks fall back to ASCII.
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# LINT.IfChange(system_dependencies)
# The base tools, installed first because every later step fetches something.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg lsb-release \
    locales \
    git \
    ssh-client \
    x11-apps \
    mesa-utils \
    gettext-base \
    sudo \
    ccache \
    bash-completion \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*
# LINT.ThenChange(//dependencies.txt:system_dependencies, //.github/workflows/build_test.yml:system_dependencies)

# clang-format (make format), from Ubuntu's archive as in CI's format job, and clang-tidy (make lint-tidy), from
# apt.llvm.org. docker/install_llvm_tools.sh says why each comes from where; CI's build job runs the same script.
# LINT.IfChange(clang_format_version)
ARG CLANG_FORMAT_VERSION=18
# LINT.ThenChange(//.github/workflows/format_test.yml:formatter_versions, //docker/install_llvm_tools.sh:llvm_versions)
# LINT.IfChange(clang_tidy_version)
ARG CLANG_TIDY_VERSION=21
ARG CLANG_TIDY_PACKAGE_VERSION=1:21.1.8~++20251221032922+2078da43e25a-1~exp1~20251221153059.70
# LINT.ThenChange(//tools/clang_tidy/run_clang_tidy.sh:clang_tidy_version, //docker/install_llvm_tools.sh:llvm_versions)

COPY docker/install_llvm_tools.sh /tmp/install_llvm_tools.sh
RUN CLANG_FORMAT_VERSION="${CLANG_FORMAT_VERSION}" CLANG_TIDY_VERSION="${CLANG_TIDY_VERSION}" \
        CLANG_TIDY_PACKAGE_VERSION="${CLANG_TIDY_PACKAGE_VERSION}" sh /tmp/install_llvm_tools.sh all && \
    rm -rf /var/lib/apt/lists/* /tmp/install_llvm_tools.sh

# Pinocchio, coal and eigenpy from robotpkg into /opt/openrobots. The script says why, and why Boost is on the system
# only as their dependency. robotpkg publishes amd64 packages only: on another architecture (an arm64 build machine for
# an ARM robot, tools/deploy/README.md "aarch64") the same releases are built from source, without the Python bindings.
COPY docker/install_robotpkg.sh /tmp/install_robotpkg.sh
COPY docker/build_pinocchio_from_source.sh /tmp/build_pinocchio_from_source.sh
RUN if [ "$(dpkg --print-architecture)" = "amd64" ]; then sh /tmp/install_robotpkg.sh; \
    else sh /tmp/build_pinocchio_from_source.sh; fi && \
    rm -rf /var/lib/apt/lists/* /tmp/install_robotpkg.sh /tmp/build_pinocchio_from_source.sh

# /opt/openrobots/lib is deliberately NOT on LD_LIBRARY_PATH: nothing needs it there. robotpkg's libraries and Python
# modules find each other through their own RUNPATH, and Bazel links them with an rpath. It would also put robotpkg's
# own libblasfeo.so, of another version, in front of every binary started from such a shell (bazel/system_libs.bzl
# links the solver's blasfeo statically for that reason, and .bazelrc keeps the shell's LD_LIBRARY_PATH out of the
# actions and tests).
# LINT.IfChange(robotpkg_environment)
# The base image sets only PATH, so the others start here.
ENV PATH="/opt/openrobots/bin:${PATH}" \
    PYTHONPATH="/opt/openrobots/lib/python3.12/site-packages" \
    CMAKE_PREFIX_PATH="/opt/openrobots" \
    PKG_CONFIG_PATH="/opt/openrobots/lib/pkgconfig"
# LINT.ThenChange(//docker/install_robotpkg.sh:robotpkg_packages, //.github/workflows/build_test.yml:robotpkg_environment, //tools/ci_local.sh:robotpkg_environment, //setup_env.sh:robotpkg_environment, //bazel/system_libs.bzl:robotpkg_prefix)

# Install Bazelisk (Bazel version manager) - reads .bazelversion for the correct version
RUN curl -fSsL -o /usr/local/bin/bazel \
    "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-$(dpkg --print-architecture)" \
    && chmod +x /usr/local/bin/bazel

# The VNC desktop for headless visualization (.devcontainer/start_vnc.sh): Xvfb on display :99 with Mesa's llvmpipe for
# GLX and EGL, x11vnc, noVNC and the openbox window manager. It shows the MuJoCo viewer, the Tk operator GUI and, when
# it runs inside the container, the native Rerun viewer, which draws through Vulkan (Mesa's lavapipe) or GL. psmisc,
# iproute2 and x11-xserver-utils provide the fuser, ip and xsetroot that start_vnc.sh calls.
RUN apt-get update && apt-get install -y --no-install-recommends \
    novnc \
    websockify \
    x11vnc \
    openbox \
    xvfb \
    xterm \
    dbus-x11 \
    x11-xserver-utils \
    psmisc \
    iproute2 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libegl-mesa0 \
    libvulkan1 \
    mesa-vulkan-drivers \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# No Abseil is installed here (nor by dependencies.txt): Bazel builds against its own @abseil-cpp module. A second,
# older copy built from source into /usr/local used to sit on the compiler's default include path, where a target that
# included an Abseil header with angle brackets or without its Bazel dependency silently compiled against it here and
# failed in CI's clean container (tools/hooks/include_style.py, tools/hooks/test_include_style.py). The same holds for
# Protocol Buffers, ZeroMQ and googletest, which Bazel builds from the BCR: none of their development packages is
# installed, so their headers exist only where Bazel puts them.
COPY dependencies.txt /tmp/dependencies.txt

# Install dependencies from dependencies.txt
RUN apt-get update && \
    grep -v '^\s*#' /tmp/dependencies.txt | envsubst | xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# The lint tools of `make lint` (tools/hooks/lint_code.py): cpplint, pylint, mypy and isort at the versions of the hashed
# lock that CI's format job and Bazel's @lint_deps hub install too (tools/hooks/test_lint_tool_versions.py). A venv of
# their own: Ubuntu 24.04 forbids pip on the system Python, and the container's site-packages must not change what
# pylint infers. black stays Ubuntu's (dependencies.txt).
COPY tools/hooks/lint_requirements_lock.txt /tmp/lint_requirements_lock.txt
RUN python3 -m venv /opt/wb-lint && \
    /opt/wb-lint/bin/pip install --no-cache-dir --require-hashes -r /tmp/lint_requirements_lock.txt && \
    ln -sf /opt/wb-lint/bin/cpplint /opt/wb-lint/bin/pylint /opt/wb-lint/bin/mypy /opt/wb-lint/bin/isort /usr/local/bin/ && \
    rm /tmp/lint_requirements_lock.txt

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Set up workspace
WORKDIR ${WB_HUMANOID_MPC_DIR}
RUN mkdir -p ${WB_HUMANOID_MPC_DIR}/src/wb_humanoid_mpc

# Copy the Makefile and other necessary files
COPY . ${WB_HUMANOID_MPC_DIR}/src/wb_humanoid_mpc

# Declare build arguments for user/group ID (passed from docker-compose.yml)
ARG USER_ID=1000
ARG GROUP_ID=1000

# Use the ubuntu user that already exists in the base image
RUN USERNAME=$(getent passwd ${USER_ID} | cut -d: -f1) && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    # Create symbolic link from /home/devuser to ubuntu's home dir for compatibility
    ln -sf /home/${USERNAME} /home/devuser && \
    # Set up directories and permissions
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME} && \
    chown -R ${USERNAME}:${USERNAME} ${WB_HUMANOID_MPC_DIR} && \
    # Ensure volume-mounted directories have correct permissions
    chown -R ${USERNAME}:${USERNAME} /usr/local && \
    # Check if we found the right username
    echo "Found user with UID ${USER_ID}: ${USERNAME}"

# Set up colorful bash prompt for ubuntu user
RUN USERNAME=$(getent passwd ${USER_ID} | cut -d: -f1) && \
    echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /home/${USERNAME}/.bashrc && \
    echo 'alias ls="ls --color=auto"' >> /home/${USERNAME}/.bashrc

# Switch to ubuntu user for subsequent commands
USER ${USER_ID}

# ==============================================================================
# The robot images (tools/deploy/README.md)
# ==============================================================================
# robot-runtime is what the robot's realtime computer runs: the robot bundle (//tools/deploy:robot_bundle, the robot
# binaries with their runfiles, the robot models and the entry scripts), the shared libraries it links, tc for NETEM,
# and nothing to build or view with: no compiler, no Bazel, no Python, no VNC, no Mesa. robot-sim is robot-runtime plus
# the GL of the MuJoCo viewer, for simulation on the laptop, which runs the robot's image
# (humanoid_nmpc/docs/distributed_runtime/README.md, "Deployment"). tools/deploy/deploy_robot.sh builds them, with the
# bundle as the build context `bundle`, a directory holding robot_bundle.tar:
#
#   docker build -f docker/Dockerfile --target robot-runtime --build-context bundle=.deploy/context \
#       -t wb-humanoid-robot:latest .
#
# Only the named stages and what they copy are built: none of the dev image above. The image's architecture is the
# build's (`docker buildx build --platform linux/arm64`): Pinocchio comes from robotpkg on amd64 and from source on
# arm64, and the bundle must have been built for the same architecture.
# LINT.IfChange(robot_images)

# Pinocchio and coal in /opt/openrobots, per architecture. Only the libraries the bundle links reach the runtime image.
FROM ${BASE_IMAGE} AS robot-pinocchio-amd64
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/install_robotpkg.sh /tmp/install_robotpkg.sh
RUN sh /tmp/install_robotpkg.sh

FROM ${BASE_IMAGE} AS robot-pinocchio-arm64
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/build_pinocchio_from_source.sh /tmp/build_pinocchio_from_source.sh
RUN PINOCCHIO_PYTHON=OFF sh /tmp/build_pinocchio_from_source.sh

# The bundle and its libraries, with the dev image's system packages (dependencies.txt) installed, so that every
# library the binaries were linked against resolves here as it did in the build. collect_runtime_libraries.sh copies the
# ones of /opt/openrobots and names the Ubuntu packages of the others.
FROM robot-pinocchio-${TARGETARCH} AS robot-libraries
COPY dependencies.txt /tmp/dependencies.txt
RUN apt-get update && \
    grep -v '^\s*#' /tmp/dependencies.txt | xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*
COPY --from=bundle robot_bundle.tar /tmp/robot_bundle.tar
RUN mkdir -p /opt/wb-humanoid-robot && \
    tar -xf /tmp/robot_bundle.tar -C /opt/wb-humanoid-robot --no-same-owner && \
    sh /opt/wb-humanoid-robot/bin/collect_runtime_libraries.sh collect /opt/wb-humanoid-robot /stage

FROM ${BASE_IMAGE} AS robot-runtime
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
COPY --from=robot-libraries /stage/packages.txt /tmp/robot-packages.txt
# iproute2: tc, for NETEM (tools/deploy/netem.sh).
RUN apt-get update && \
    xargs apt-get install -y --no-install-recommends iproute2 < /tmp/robot-packages.txt && \
    rm -rf /var/lib/apt/lists/* /tmp/robot-packages.txt
# Pinocchio and coal, and the GL client libraries the binaries link but robot-runtime does not install (their packages
# would pull in Mesa: collect_runtime_libraries.sh, COPIED_PACKAGES).
COPY --from=robot-libraries /stage/root/ /
COPY --from=robot-libraries /opt/wb-humanoid-robot/ /opt/wb-humanoid-robot/
# The build fails here when a library of the bundle did not make it into the image.
RUN sh /opt/wb-humanoid-robot/bin/collect_runtime_libraries.sh check /opt/wb-humanoid-robot
# No GL: the MuJoCo backend runs without its viewer unless WB_ROBOT_HEADLESS says otherwise.
ENV WB_ROBOT_DEFAULT_HEADLESS=true
WORKDIR /opt/wb-humanoid-robot
ENTRYPOINT ["/opt/wb-humanoid-robot/bin/robot_entrypoint.sh"]

# The MuJoCo viewer's GL (GLFW over X11 with Mesa) on top of robot-runtime, and nothing else: the GL packages whose
# libraries robot-runtime copied, now with their Mesa, and the X extension libraries GLFW loads at run time, which its
# package does not pull in.
FROM robot-runtime AS robot-sim
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libgl1-mesa-dri \
        libglew2.2 \
        libglfw3 \
        libglvnd0 \
        libglx-mesa0 \
        libglx0 \
        libx11-xcb1 \
        libxcursor1 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxxf86vm1 && \
    rm -rf /var/lib/apt/lists/*
ENV WB_ROBOT_DEFAULT_HEADLESS=false
# LINT.ThenChange(//tools/deploy/test_deploy_files.py:robot_images, //tools/deploy/README.md:robot_images, //tools/deploy/collect_runtime_libraries.sh:copied_packages)
