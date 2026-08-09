ARG ROS_DISTRO=jazzy
ARG BASE_IMAGE=ros:${ROS_DISTRO}-ros-base
FROM ${BASE_IMAGE} AS base

ARG CPP_VERSION=20
ARG WB_HUMANOID_MPC_DIR="/wb_humanoid_mpc_ws"
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=${ROS_DISTRO}

# Install clang-format (for linting).
ARG CLANG_VERSION=18

# Install LLVM GPG key and add repository.
RUN UBUNTU_DISTRO=$(lsb_release -cs) && \
    apt-get update && \
    apt-get install -y wget gnupg && \
    wget https://apt.llvm.org/llvm-snapshot.gpg.key && \
    apt-key add llvm-snapshot.gpg.key && \
    rm llvm-snapshot.gpg.key && \
    echo "deb http://apt.llvm.org/${UBUNTU_DISTRO}/ llvm-toolchain-${UBUNTU_DISTRO}-${CLANG_VERSION} main" | tee /etc/apt/sources.list.d/llvm.list

# Update package list and install clang-format.
RUN apt-get update && \
    apt-get install -y clang-format-${CLANG_VERSION} && \
    ln -s /usr/bin/clang-format-${CLANG_VERSION} /usr/bin/clang-format

# Update package list and install clang-tidy.
RUN apt-get update && \
    apt-get install -y clang-tidy-${CLANG_VERSION} && \
    ln -s /usr/bin/clang-tidy-${CLANG_VERSION} /usr/bin/clang-tidy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install system dependencies, ROS, Python, and development tools in a single step
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release software-properties-common \
    locales \
    git \
    ssh-client \
    gnupg \
    x11-apps \
    mesa-utils \
    gettext-base \
    sudo \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Install VNC, noVNC, and a lightweight window manager for headless remote visualization.
# Uses Xvfb (Mesa GLX) + x11vnc to provide the GLX context that RViz2/OGRE requires.
RUN apt-get update && apt-get install -y --no-install-recommends \
    novnc \
    websockify \
    x11vnc \
    openbox \
    xvfb \
    xterm \
    dbus-x11 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libegl-mesa0 \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-pinocchio \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
    ros-${ROS_DISTRO}-urdfdom-headers \
    && rm -rf /var/lib/apt/lists/*

# Create pkg-config metadata for pinocchio in ROS 2 Jazzy
RUN mkdir -p /usr/lib/x86_64-linux-gnu/pkgconfig /opt/ros/jazzy/lib/pkgconfig && \
    echo "prefix=/opt/ros/jazzy\nexec_prefix=\${prefix}\nlibdir=\${prefix}/lib/x86_64-linux-gnu\nincludedir=\${prefix}/include\n\nName: pinocchio\nDescription: Rigid Body Dynamics algorithms\nVersion: 4.0.0\nCflags: -I\${includedir}\nLibs: -L\${libdir} -lpinocchio_default -lpinocchio_parsers -lpinocchio_collision" > /usr/lib/x86_64-linux-gnu/pkgconfig/pinocchio.pc

# Set up environment variables
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/ros/${ROS_DISTRO}"

# Create directory for the Abseil volume mount and set appropriate permissions
RUN mkdir -p /usr/local/include/absl /usr/local/lib && \
    chmod -R 777 /usr/local

# Install Abseil with better volume persistence handling
RUN if [ ! -f /usr/local/lib/libabsl_base.a ]; then \
      echo "Installing Abseil from source..." && \
      git clone -b 20240722.0 https://github.com/abseil/abseil-cpp.git && \
      cd abseil-cpp && \
      mkdir -p build && \
      cd build && \
      cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release .. && \
      cmake --build . --target install -j$(nproc) && \
      ldconfig && \
      cd / && \
      rm -rf abseil-cpp && \
      echo "Abseil installation completed."; \
    else \
      echo "Abseil already installed in volume, skipping compilation"; \
    fi

COPY dependencies.txt /tmp/dependencies.txt

# Install dependencies from dependencies.txt
RUN apt-get update && \
    envsubst < /tmp/dependencies.txt | xargs apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

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
    echo 'alias ls="ls --color=auto"' >> /home/${USERNAME}/.bashrc && \
    echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /home/${USERNAME}/.bashrc

# Switch to ubuntu user for subsequent commands
USER ${USER_ID}
