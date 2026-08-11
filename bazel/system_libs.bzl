"""Repository rules for system-installed libraries.

These rules create Bazel-compatible wrappers around libraries installed
via apt or built from source in the Docker container.
"""

# ==============================================================================
# Helper: find first existing directory from a list of candidates
# ==============================================================================
def _find_dir(repo_ctx, candidates):
    """Returns the first existing directory path from candidates, or None."""
    for d in candidates:
        p = repo_ctx.path(d)
        if p.exists:
            return d
    return None

def _symlink_if_exists(repo_ctx, source, target):
    """Symlinks source to target if source exists."""
    if repo_ctx.path(source).exists:
        repo_ctx.symlink(source, target)
        return True
    return False

# ==============================================================================
# Eigen
# ==============================================================================
def _eigen_repository(repo_ctx):
    """Wraps system-installed Eigen3."""
    repo_ctx.symlink("/usr/include/eigen3", "include")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "eigen",
    hdrs = glob(["include/**"], allow_empty = True),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""")

eigen_repository = repository_rule(
    implementation = _eigen_repository,
    local = True,
)

# ==============================================================================
# Boost
# ==============================================================================
def _boost_repository(repo_ctx):
    """Wraps system-installed Boost (system, filesystem, log, log_setup)."""
    repo_ctx.symlink("/usr/include/boost", "include/boost")

    # Find and symlink Boost shared libraries individually
    lib_dir = "/usr/lib/x86_64-linux-gnu"
    for comp in ["system", "filesystem", "log", "log_setup", "thread"]:
        # Find the actual .so file
        result = repo_ctx.execute(["find", lib_dir, "-name", "libboost_%s.so*" % comp, "-not", "-type", "d"])
        if result.return_code == 0 and result.stdout.strip():
            first_match = result.stdout.strip().split("\n")[0]
            repo_ctx.symlink(first_match, "lib/libboost_%s.so" % comp)

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
cc_library(
    name = "headers",
    hdrs = glob(["include/boost/**"], allow_empty = True),
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

boost_repository = repository_rule(
    implementation = _boost_repository,
    local = True,
)

# ==============================================================================
# Pinocchio
# ==============================================================================
def _pinocchio_repository(repo_ctx):
    """Wraps system-installed Pinocchio (from ROS2 package)."""
    ros_distro = repo_ctx.os.environ.get("ROS_DISTRO", "jazzy")
    ros_prefix = "/opt/ros/" + ros_distro

    # Symlink pinocchio headers (doubly-nested in ROS2 Jazzy)
    pinocchio_candidates = [
        ros_prefix + "/include/pinocchio/pinocchio",
        ros_prefix + "/include/pinocchio",
        "/usr/include/pinocchio",
        "/usr/local/include/pinocchio",
    ]
    for d in pinocchio_candidates:
        if _symlink_if_exists(repo_ctx, d, "include/pinocchio"):
            break

    # Pinocchio deprecated compat headers
    _symlink_if_exists(repo_ctx, ros_prefix + "/include/pinocchio/deprecated", "include/pinocchio_deprecated")

    # eigenpy
    for d in [ros_prefix + "/include/eigenpy/eigenpy", ros_prefix + "/include/eigenpy", "/usr/include/eigenpy"]:
        if _symlink_if_exists(repo_ctx, d, "include/eigenpy"):
            break

    # urdf_parser
    _symlink_if_exists(repo_ctx, "/usr/include/urdf_parser", "include/urdf_parser")

    # console_bridge
    _symlink_if_exists(repo_ctx, "/usr/include/console_bridge", "include/console_bridge")

    # coal (hpp-fcl replacement in pinocchio 3.x)
    for d in [ros_prefix + "/include/coal/coal", "/usr/include/coal", "/usr/local/include/coal"]:
        if _symlink_if_exists(repo_ctx, d, "include/coal"):
            break

    # hpp/fcl backward-compat headers
    for d in [ros_prefix + "/include/coal/hpp/fcl", ros_prefix + "/include/hpp-fcl/hpp/fcl", "/usr/include/hpp/fcl"]:
        if repo_ctx.path(d).exists:
            repo_ctx.symlink(d, "include/hpp/fcl")
            break

    # octomap
    _symlink_if_exists(repo_ctx, "/usr/include/octomap", "include/octomap")

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "pinocchio",
    hdrs = glob(["include/**"], allow_empty = True),
    includes = ["include", "include/pinocchio_deprecated"],
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

pinocchio_repository = repository_rule(
    implementation = _pinocchio_repository,
    local = True,
    environ = ["ROS_DISTRO"],
)

# ==============================================================================
# GLFW
# ==============================================================================
def _glfw_repository(repo_ctx):
    """Wraps system-installed GLFW3."""
    repo_ctx.symlink("/usr/include/GLFW", "include/GLFW")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "glfw",
    hdrs = glob(["include/GLFW/**"], allow_empty = True),
    includes = ["include"],
    linkopts = ["-lglfw"],
    visibility = ["//visibility:public"],
)
""")

glfw_repository = repository_rule(
    implementation = _glfw_repository,
    local = True,
)

# ==============================================================================
# GLEW
# ==============================================================================
def _glew_repository(repo_ctx):
    """Wraps system-installed GLEW."""
    repo_ctx.symlink("/usr/include/GL", "include/GL")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "glew",
    hdrs = glob(["include/GL/**"], allow_empty = True),
    includes = ["include"],
    linkopts = ["-lGLEW", "-lGL"],
    visibility = ["//visibility:public"],
)
""")

glew_repository = repository_rule(
    implementation = _glew_repository,
    local = True,
)



# ==============================================================================
# URDF
# ==============================================================================
def _urdf_repository(repo_ctx):
    """Wraps system-installed urdf library (from ROS2)."""
    ros_distro = repo_ctx.os.environ.get("ROS_DISTRO", "jazzy")
    ros_prefix = "/opt/ros/" + ros_distro

    # Symlink urdf headers (ROS2 Jazzy doubly-nested layout)
    for d in [ros_prefix + "/include/urdf/urdf", ros_prefix + "/include/urdf", "/usr/include/urdf"]:
        if _symlink_if_exists(repo_ctx, d, "include/urdf"):
            break

    for d in [ros_prefix + "/include/urdf_parser_plugin/urdf_parser_plugin", ros_prefix + "/include/urdf_parser_plugin", "/usr/include/urdf_parser_plugin"]:
        if _symlink_if_exists(repo_ctx, d, "include/urdf_parser_plugin"):
            break

    # urdf_parser headers
    for d in ["/usr/include/urdfdom/urdf_parser", "/usr/include/urdf_parser"]:
        if _symlink_if_exists(repo_ctx, d, "include/urdf_parser"):
            break

    # urdfdom inner dirs (flat access)
    urdfdom_dir = _find_dir(repo_ctx, [ros_prefix + "/include/urdfdom", "/usr/include/urdfdom"])
    if urdfdom_dir:
        result = repo_ctx.execute(["find", urdfdom_dir, "-mindepth", "1", "-maxdepth", "1", "-type", "d"])
        if result.return_code == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    subname = line.split("/")[-1]
                    if not repo_ctx.path("include/" + subname).exists:
                        repo_ctx.symlink(line, "include/" + subname)

    # urdfdom_headers
    for d in [ros_prefix + "/include/urdfdom_headers/urdfdom_headers", ros_prefix + "/include/urdfdom_headers", "/usr/include/urdfdom_headers"]:
        if _symlink_if_exists(repo_ctx, d, "include/urdfdom_headers"):
            break

    # console_bridge
    _symlink_if_exists(repo_ctx, "/usr/include/console_bridge", "include/console_bridge")

    # tinyxml2
    _symlink_if_exists(repo_ctx, "/usr/include/tinyxml2.h", "include/tinyxml2.h")

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "urdf",
    hdrs = glob(["include/**"], allow_empty = True),
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

urdf_repository = repository_rule(
    implementation = _urdf_repository,
    local = True,
    environ = ["ROS_DISTRO"],
)

# ==============================================================================
# urdfdom (standalone)
# ==============================================================================
def _urdfdom_repository(repo_ctx):
    """Wraps system-installed urdfdom."""
    for d in ["/usr/include/urdfdom", "/usr/local/include/urdfdom"]:
        if _symlink_if_exists(repo_ctx, d, "include/urdfdom"):
            break
    for d in ["/usr/include/urdfdom_headers", "/usr/local/include/urdfdom_headers"]:
        if _symlink_if_exists(repo_ctx, d, "include/urdfdom_headers"):
            break

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "urdfdom",
    hdrs = glob(["include/**"], allow_empty = True),
    includes = ["include"],
    linkopts = ["-lurdfdom_model", "-lurdfdom_world", "-lurdfdom_sensor"],
    visibility = ["//visibility:public"],
)
""")

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
    _symlink_if_exists(repo_ctx, install_prefix + "/include", "include")
    _symlink_if_exists(repo_ctx, install_prefix + "/lib", "lib")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "blasfeo",
    hdrs = glob(["include/**"], allow_empty = True),
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
    _symlink_if_exists(repo_ctx, install_prefix + "/include", "include")
    _symlink_if_exists(repo_ctx, install_prefix + "/lib", "lib")
    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "hpipm",
    hdrs = glob(["include/**"], allow_empty = True),
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

# LINT.IfChange(system_repositories)
def register_system_libs():
    """Registers all system library repositories."""
    eigen_repository(name = "eigen")
    boost_repository(name = "boost")
    pinocchio_repository(name = "pinocchio")
    glfw_repository(name = "glfw")
    glew_repository(name = "glew")
    urdf_repository(name = "urdf")
    urdfdom_repository(name = "urdfdom")
    blasfeo_repository(name = "blasfeo")
    hpipm_repository(name = "hpipm")
# LINT.ThenChange(//MODULE.bazel:system_repositories)
