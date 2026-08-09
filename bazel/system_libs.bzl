"""Repository rules for system-installed libraries.

These rules create Bazel-compatible wrappers around libraries installed
via apt or built from source in the Docker container.
"""

def _eigen_repository(repo_ctx):
    """Wraps system-installed Eigen3."""
    repo_ctx.symlink("/usr/include/eigen3", "include")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "eigen",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""")

eigen_repository = repository_rule(
    implementation = _eigen_repository,
    local = True,
)

def _boost_repository(repo_ctx):
    """Wraps system-installed Boost (system, filesystem, log, log_setup)."""
    repo_ctx.symlink("/usr/include/boost", "include/boost")

    # Find Boost shared libraries
    lib_dir = "/usr/lib/x86_64-linux-gnu"
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
cc_library(
    name = "headers",
    hdrs = glob(["include/boost/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "system_lib",
    shared_library = "lib/libboost_system.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "filesystem_lib",
    shared_library = "lib/libboost_filesystem.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "log_lib",
    shared_library = "lib/libboost_log.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "log_setup_lib",
    shared_library = "lib/libboost_log_setup.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "thread_lib",
    shared_library = "lib/libboost_thread.so",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "boost",
    visibility = ["//visibility:public"],
    deps = [
        ":headers",
        ":system_lib",
        ":filesystem_lib",
        ":log_lib",
        ":log_setup_lib",
        ":thread_lib",
    ],
)
""")

    # Symlink the library directory
    repo_ctx.symlink(lib_dir, "lib_scan")
    # Create lib dir with specific symlinks
    result = repo_ctx.execute(["bash", "-c", """
        mkdir -p lib
        for comp in system filesystem log log_setup thread; do
            src=$(find /usr/lib/x86_64-linux-gnu -name "libboost_${comp}.so*" -not -type d | head -1)
            if [ -n "$src" ]; then
                ln -sf "$src" lib/libboost_${comp}.so
            fi
        done
    """])

boost_repository = repository_rule(
    implementation = _boost_repository,
    local = True,
)

def _pinocchio_repository(repo_ctx):
    """Wraps system-installed Pinocchio (from ROS2 package)."""
    ros_distro = repo_ctx.os.environ.get("ROS_DISTRO", "jazzy")
    ros_prefix = "/opt/ros/" + ros_distro

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "pinocchio",
    hdrs = glob(["include/**"]),
    includes = ["include", "include/pinocchio/deprecated"],
    linkopts = [
        "-L{ros_prefix}/lib/x86_64-linux-gnu",
        "-lpinocchio_parsers",
        "-lpinocchio_default",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@boost",
        "@urdf",
    ],
)
""".format(ros_prefix = ros_prefix))

    # Symlink pinocchio headers from the ROS2 install
    repo_ctx.execute(["bash", "-c", """
        mkdir -p include
        # Link pinocchio headers - they may be in various locations
        for dir in {ros_prefix}/include/pinocchio /usr/include/pinocchio /usr/local/include/pinocchio; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/pinocchio
                break
            fi
        done
        # Also link eigenpy if present
        for dir in {ros_prefix}/include/eigenpy /usr/include/eigenpy; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/eigenpy
                break
            fi
        done
        # Link urdf_parser headers (needed by pinocchio urdf parser)
        for dir in /usr/include/urdf_parser; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/urdf_parser
                break
            fi
        done
        # Link console_bridge headers (needed by urdfdom)
        for dir in /usr/include/console_bridge; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/console_bridge
                break
            fi
        done
        # Link hpp-fcl / coal headers (needed by ocs2_self_collision)
        # In pinocchio 3.x, hpp-fcl was renamed to coal but provides hpp/fcl/ compat headers
        for dir in {ros_prefix}/include/coal/coal /usr/include/coal /usr/local/include/coal; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/coal
                break
            fi
        done
        # hpp/fcl backward-compat headers
        for dir in {ros_prefix}/include/coal/hpp/fcl {ros_prefix}/include/hpp-fcl/hpp/fcl /usr/include/hpp/fcl; do
            if [ -d "$dir" ]; then
                mkdir -p include/hpp
                ln -sf "$dir" include/hpp/fcl
                break
            fi
        done
        # octomap headers (needed by hpp-fcl/coal)
        for dir in /usr/include/octomap; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/octomap
                break
            fi
        done
    """.format(ros_prefix = ros_prefix)])

pinocchio_repository = repository_rule(
    implementation = _pinocchio_repository,
    local = True,
    environ = ["ROS_DISTRO"],
)

def _glfw_repository(repo_ctx):
    """Wraps system-installed GLFW3."""
    repo_ctx.symlink("/usr/include/GLFW", "include/GLFW")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "glfw",
    hdrs = glob(["include/GLFW/**"]),
    includes = ["include"],
    linkopts = ["-lglfw"],
    visibility = ["//visibility:public"],
)
""")

glfw_repository = repository_rule(
    implementation = _glfw_repository,
    local = True,
)

def _glew_repository(repo_ctx):
    """Wraps system-installed GLEW."""
    repo_ctx.symlink("/usr/include/GL", "include/GL")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "glew",
    hdrs = glob(["include/GL/**"]),
    includes = ["include"],
    linkopts = ["-lGLEW", "-lGL"],
    visibility = ["//visibility:public"],
)
""")

glew_repository = repository_rule(
    implementation = _glew_repository,
    local = True,
)

def _abseil_system_repository(repo_ctx):
    """Wraps system-installed Abseil (source-built lts_20240722 in /usr/local)."""
    # Abseil headers - prefer /usr/local/include (matches the lts_20240722 static libs)
    repo_ctx.execute(["bash", "-c", """
        mkdir -p include lib
        for dir in /usr/local/include/absl /usr/include/absl; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/absl
                break
            fi
        done
        # Symlink the static archives from /usr/local/lib
        for f in /usr/local/lib/libabsl_*.a; do
            if [ -f "$f" ]; then
                ln -sf "$f" lib/
            fi
        done
    """])

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_import(
    name = "absl_hash_lib",
    static_library = "lib/libabsl_hash.a",
)
cc_import(
    name = "absl_city_lib",
    static_library = "lib/libabsl_city.a",
)
cc_import(
    name = "absl_raw_hash_set_lib",
    static_library = "lib/libabsl_raw_hash_set.a",
)
cc_import(
    name = "absl_hashtablez_sampler_lib",
    static_library = "lib/libabsl_hashtablez_sampler.a",
)
cc_import(
    name = "absl_low_level_hash_lib",
    static_library = "lib/libabsl_low_level_hash.a",
)
cc_import(
    name = "absl_base_lib",
    static_library = "lib/libabsl_base.a",
)
cc_import(
    name = "absl_spinlock_wait_lib",
    static_library = "lib/libabsl_spinlock_wait.a",
)
cc_import(
    name = "absl_synchronization_lib",
    static_library = "lib/libabsl_synchronization.a",
)
cc_import(
    name = "absl_malloc_internal_lib",
    static_library = "lib/libabsl_malloc_internal.a",
)
cc_import(
    name = "absl_throw_delegate_lib",
    static_library = "lib/libabsl_throw_delegate.a",
)
cc_import(
    name = "absl_raw_logging_internal_lib",
    static_library = "lib/libabsl_raw_logging_internal.a",
)
cc_import(
    name = "absl_log_severity_lib",
    static_library = "lib/libabsl_log_severity.a",
)
cc_import(
    name = "absl_exponential_biased_lib",
    static_library = "lib/libabsl_exponential_biased.a",
)
cc_import(
    name = "absl_int128_lib",
    static_library = "lib/libabsl_int128.a",
)
cc_import(
    name = "absl_strings_lib",
    static_library = "lib/libabsl_strings.a",
)
cc_import(
    name = "absl_strings_internal_lib",
    static_library = "lib/libabsl_strings_internal.a",
)
cc_import(
    name = "absl_time_lib",
    static_library = "lib/libabsl_time.a",
)
cc_import(
    name = "absl_time_zone_lib",
    static_library = "lib/libabsl_time_zone.a",
)
cc_import(
    name = "absl_graphcycles_internal_lib",
    static_library = "lib/libabsl_graphcycles_internal.a",
)
cc_import(
    name = "absl_kernel_timeout_internal_lib",
    static_library = "lib/libabsl_kernel_timeout_internal.a",
)
cc_import(
    name = "absl_stacktrace_lib",
    static_library = "lib/libabsl_stacktrace.a",
)
cc_import(
    name = "absl_symbolize_lib",
    static_library = "lib/libabsl_symbolize.a",
)
cc_import(
    name = "absl_debugging_internal_lib",
    static_library = "lib/libabsl_debugging_internal.a",
)
cc_import(
    name = "absl_demangle_internal_lib",
    static_library = "lib/libabsl_demangle_internal.a",
)
cc_import(
    name = "absl_strerror_lib",
    static_library = "lib/libabsl_strerror.a",
)

cc_library(
    name = "abseil_system",
    hdrs = glob(["include/absl/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":absl_hash_lib",
        ":absl_city_lib",
        ":absl_raw_hash_set_lib",
        ":absl_hashtablez_sampler_lib",
        ":absl_low_level_hash_lib",
        ":absl_base_lib",
        ":absl_spinlock_wait_lib",
        ":absl_synchronization_lib",
        ":absl_malloc_internal_lib",
        ":absl_throw_delegate_lib",
        ":absl_raw_logging_internal_lib",
        ":absl_log_severity_lib",
        ":absl_exponential_biased_lib",
        ":absl_int128_lib",
        ":absl_strings_lib",
        ":absl_strings_internal_lib",
        ":absl_time_lib",
        ":absl_time_zone_lib",
        ":absl_graphcycles_internal_lib",
        ":absl_kernel_timeout_internal_lib",
        ":absl_stacktrace_lib",
        ":absl_symbolize_lib",
        ":absl_debugging_internal_lib",
        ":absl_demangle_internal_lib",
        ":absl_strerror_lib",
    ],
    linkopts = ["-lpthread"],
)
""")

abseil_system_repository = repository_rule(
    implementation = _abseil_system_repository,
    local = True,
)

def _urdf_repository(repo_ctx):
    """Wraps system-installed urdf library (from ROS2)."""
    ros_distro = repo_ctx.os.environ.get("ROS_DISTRO", "jazzy")
    ros_prefix = "/opt/ros/" + ros_distro

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "urdf",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    linkopts = [
        "-L{ros_prefix}/lib",
        "-L/usr/lib/x86_64-linux-gnu",
        "-lurdf",
        "-lurdfdom_model",
        "-lurdfdom_world",
    ],
    visibility = ["//visibility:public"],
)
""".format(ros_prefix = ros_prefix))

    repo_ctx.execute(["bash", "-c", """
        mkdir -p include
        for dir in {ros_prefix}/include/urdf /usr/include/urdf; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/urdf
                break
            fi
        done
        for dir in {ros_prefix}/include/urdf_parser_plugin /usr/include/urdf_parser_plugin; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/urdf_parser_plugin
                break
            fi
        done
        # urdf_parser headers (for urdf_parser/urdf_parser.h)
        # Actual location is /usr/include/urdfdom/urdf_parser/
        for dir in /usr/include/urdfdom/urdf_parser /usr/include/urdf_parser; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/urdf_parser
                break
            fi
        done
        # urdfdom headers - also symlink inner dirs for flat access
        for dir in {ros_prefix}/include/urdfdom /usr/include/urdfdom; do
            if [ -d "$dir" ]; then
                # symlink each subdir inside urdfdom to include/
                for sub in "$dir"/*/; do
                    subname=$(basename "$sub")
                    if [ ! -e "include/$subname" ]; then
                        ln -sf "$sub" "include/$subname"
                    fi
                done
                break
            fi
        done
        # urdfdom_headers
        for dir in {ros_prefix}/include/urdfdom_headers /usr/include/urdfdom_headers; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/urdfdom_headers
                break
            fi
        done
        # console_bridge (required by urdfdom)
        for dir in /usr/include/console_bridge; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/console_bridge
                break
            fi
        done
        # tinyxml2
        if [ -f /usr/include/tinyxml2.h ]; then
            ln -sf /usr/include/tinyxml2.h include/tinyxml2.h
        fi
    """.format(ros_prefix = ros_prefix)])

urdf_repository = repository_rule(
    implementation = _urdf_repository,
    local = True,
    environ = ["ROS_DISTRO"],
)

def _urdfdom_repository(repo_ctx):
    """Wraps system-installed urdfdom."""
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "urdfdom",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    linkopts = ["-lurdfdom_model", "-lurdfdom_world", "-lurdfdom_sensor"],
    visibility = ["//visibility:public"],
)
""")
    repo_ctx.execute(["bash", "-c", """
        mkdir -p include
        for dir in /usr/include/urdfdom /usr/local/include/urdfdom; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/urdfdom
                break
            fi
        done
        for dir in /usr/include/urdfdom_headers /usr/local/include/urdfdom_headers; do
            if [ -d "$dir" ]; then
                ln -sf "$dir" include/urdfdom_headers
                break
            fi
        done
    """])

urdfdom_repository = repository_rule(
    implementation = _urdfdom_repository,
    local = True,
)

# ==============================================================================
# blasfeo (pre-built from colcon install)
# ==============================================================================

def _blasfeo_repository(repo_ctx):
    """Wraps pre-built blasfeo from colcon install."""
    install_prefix = "/wb_humanoid_mpc_ws/install/blasfeo_catkin"
    repo_ctx.symlink(install_prefix + "/include", "include")
    repo_ctx.symlink(install_prefix + "/lib", "lib")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "blasfeo",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    linkopts = ["-L{prefix}/lib", "-lblasfeo"],
    visibility = ["//visibility:public"],
)
""".format(prefix = install_prefix))

blasfeo_repository = repository_rule(
    implementation = _blasfeo_repository,
    local = True,
)

# ==============================================================================
# hpipm (pre-built from colcon install)
# ==============================================================================

def _hpipm_repository(repo_ctx):
    """Wraps pre-built hpipm from colcon install."""
    install_prefix = "/wb_humanoid_mpc_ws/install/hpipm_catkin"
    repo_ctx.symlink(install_prefix + "/include", "include")
    repo_ctx.symlink(install_prefix + "/lib", "lib")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "hpipm",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    linkopts = ["-L{prefix}/lib", "-lhpipm", "-lhpipm_catkin"],
    deps = ["@blasfeo"],
    visibility = ["//visibility:public"],
)
""".format(prefix = install_prefix))

hpipm_repository = repository_rule(
    implementation = _hpipm_repository,
    local = True,
)

# ==============================================================================
# Public function to register all system library repositories
# ==============================================================================

def register_system_libs():
    """Registers all system library repositories."""
    eigen_repository(name = "eigen")
    boost_repository(name = "boost")
    pinocchio_repository(name = "pinocchio")
    glfw_repository(name = "glfw")
    glew_repository(name = "glew")
    abseil_system_repository(name = "abseil_system")
    urdf_repository(name = "urdf")
    urdfdom_repository(name = "urdfdom")
    blasfeo_repository(name = "blasfeo")
    hpipm_repository(name = "hpipm")
