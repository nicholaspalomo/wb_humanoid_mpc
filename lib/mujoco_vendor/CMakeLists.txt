cmake_minimum_required(VERSION 3.16)
project(mujoco_vendor)

include(ExternalProject)

# Set the Mujoco version and download details
# LINT.IfChange(mujoco_version)
set(MUJOCO_VERSION "4.0.1")
# LINT.ThenChange(//lib/mujoco_vendor/BUILD.bazel:mujoco_version)
# The prebuilt release of the architecture being built for (the robot computer's may be x86_64 or aarch64), with the
# checksums of https://github.com/google-deepmind/mujoco/releases/tag/4.0.1 (<tarball>.sha256).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(MUJOCO_ARCHITECTURE "x86_64")
  set(MUJOCO_SHA256 "8df1308b2900a4ed6c18717ebbc9d66d53ab3fe7a4da860a7d5b21f94607299d")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(MUJOCO_ARCHITECTURE "aarch64")
  set(MUJOCO_SHA256 "e16dc0ed24d31279bce674bd03b9f8ef75f209d1189befe67b7c3b366978a6b3")
else()
  message(FATAL_ERROR "mujoco_vendor: no prebuilt MuJoCo ${MUJOCO_VERSION} for the processor '${CMAKE_SYSTEM_PROCESSOR}' (x86_64 and aarch64 only)")
endif()
set(MUJOCO_DOWNLOAD_URL "https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-${MUJOCO_ARCHITECTURE}.tar.gz")
set(MUJOCO_TARBALL "${CMAKE_BINARY_DIR}/mujoco_prebuilt/src/mujoco-${MUJOCO_VERSION}-linux-${MUJOCO_ARCHITECTURE}.tar.gz")
set(MUJOCO_INSTALL_DIR ${CMAKE_BINARY_DIR}/mujoco) 

# Add Mujoco as an external project
ExternalProject_Add(mujoco_prebuilt
    URL ${MUJOCO_DOWNLOAD_URL}
    URL_HASH SHA256=${MUJOCO_SHA256}
    PREFIX ${CMAKE_BINARY_DIR}/mujoco_prebuilt
    INSTALL_DIR ${MUJOCO_INSTALL_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND tar --strip-components=1 -xzf ${MUJOCO_TARBALL} -C ${MUJOCO_INSTALL_DIR}
)

# Create an interface library for linking with Mujoco
add_library(mujoco INTERFACE)
add_dependencies(mujoco mujoco_prebuilt)

# Specify include directories and link libraries for Mujoco
target_include_directories(mujoco INTERFACE
    ${MUJOCO_INSTALL_DIR}/include
)
target_link_libraries(mujoco INTERFACE
  ${MUJOCO_INSTALL_DIR}/lib/libmujoco.so.${MUJOCO_VERSION}
)

# Install Mujoco headers and library
install(DIRECTORY ${MUJOCO_INSTALL_DIR}/include/
        DESTINATION include)

install(FILES ${MUJOCO_INSTALL_DIR}/lib/libmujoco.so.${MUJOCO_VERSION}
        DESTINATION lib)

# Create a symlink to the versioned target 
install(CODE "
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink libmujoco.so.${MUJOCO_VERSION} libmujoco.so
        WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib
    )
")
