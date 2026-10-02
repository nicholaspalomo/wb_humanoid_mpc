"""Repository rules for the libraries the build takes from the system, or builds outside Bazel.

Eigen, yaml-cpp, GLFW, GLEW and urdfdom come from Ubuntu (dependencies.txt), Pinocchio from robotpkg in /opt/openrobots
(docker/install_robotpkg.sh), and blasfeo and hpipm are built from pinned sources with CMake. The dev image
(docker/Dockerfile), CI (.github/workflows/build_test.yml) and the local CI emulator (tools/ci_local.sh) install the
same packages, so these rules see the same files everywhere.
"""

def _symlink_if_exists(repo_ctx, source, target):
    """Symlinks source to target if source exists."""
    if repo_ctx.path(source).exists:
        repo_ctx.symlink(source, target)
        return True
    return False

def _run(repo_ctx, arguments, what, timeout = 600):
    """Runs a command of a repository rule and fails the fetch, with the command's output, when it fails."""
    result = repo_ctx.execute(arguments, quiet = False, timeout = timeout)
    if result.return_code != 0:
        fail("{what} failed (exit code {code}): {command}\n{stdout}\n{stderr}".format(
            what = what,
            code = result.return_code,
            command = " ".join([str(argument) for argument in arguments]),
            stdout = result.stdout,
            stderr = result.stderr,
        ))
    return result

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
# Headers only, and only @pinocchio depends on it. The project itself uses no Boost: it remains solely because
# Pinocchio's public headers require it (a joint model is a boost::variant over its joint types, and the URDF parser
# takes boost::optional arguments). Code that uses that part of Pinocchio's API, such as boost::get on a joint's
# variant, receives these headers through @pinocchio, never by listing @boost itself. The headers are Ubuntu's Boost,
# which robotpkg's Pinocchio packages install as their dependency (docker/install_robotpkg.sh).
def _boost_repository(repo_ctx):
    """Wraps the system-installed Boost headers that Pinocchio's headers include."""
    repo_ctx.symlink("/usr/include/boost", "include/boost")

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "boost",
    hdrs = glob(["include/boost/**"], allow_empty = True),
    includes = ["include"],
    # Only Pinocchio may depend on it. The root module does not import @boost (MODULE.bazel), so a project target cannot
    # even name it; this visibility is the second guard. Neither stops an #include of a Boost header in project code,
    # which the compiler would still find in /usr/include; tools/hooks/boost_usage.py, run by the linter and the
    # pre-commit hook, does.
    visibility = ["@pinocchio//:__pkg__"],
)
""")

boost_repository = repository_rule(
    implementation = _boost_repository,
    local = True,
)

# ==============================================================================
# Pinocchio
# ==============================================================================
# LINT.IfChange(robotpkg_prefix)
# Where docker/install_robotpkg.sh installs Pinocchio, coal and eigenpy. Its headers are singly nested
# (include/pinocchio/algorithm/...), and its libraries find their own dependencies through their RUNPATH.
_ROBOTPKG_PREFIX = "/opt/openrobots"
# LINT.ThenChange(//docker/install_robotpkg.sh:robotpkg_packages, //docker/Dockerfile:robotpkg_environment)

def _pinocchio_repository(repo_ctx):
    """Wraps Pinocchio from robotpkg (docker/install_robotpkg.sh)."""
    include_dir = _ROBOTPKG_PREFIX + "/include"
    lib_dir = _ROBOTPKG_PREFIX + "/lib"
    if not repo_ctx.path(include_dir + "/pinocchio/config.hpp").exists:
        fail(("Pinocchio is not installed: {include_dir}/pinocchio/config.hpp does not exist. Install it from robotpkg " +
              "with `sudo sh docker/install_robotpkg.sh` (the dev image, CI and tools/ci_local.sh do).").format(
            include_dir = include_dir,
        ))

    # Only the directories Pinocchio's headers need, never robotpkg's include directory as a whole: it also holds
    # robotpkg's own blasfeo headers, which must not shadow the blasfeo that @blasfeo builds for hpipm.
    repo_ctx.symlink(include_dir + "/pinocchio", "include/pinocchio")
    # The deprecated compatibility headers, which Pinocchio installs under include/pinocchio/deprecated/pinocchio.
    _symlink_if_exists(repo_ctx, include_dir + "/pinocchio/deprecated", "include/pinocchio_deprecated")
    # coal (formerly hpp-fcl) and its hpp/fcl compatibility headers, for Pinocchio's collision headers.
    _symlink_if_exists(repo_ctx, include_dir + "/coal", "include/coal")
    _symlink_if_exists(repo_ctx, include_dir + "/hpp/fcl", "include/hpp/fcl")
    _symlink_if_exists(repo_ctx, "/usr/include/octomap", "include/octomap")

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "pinocchio",
    hdrs = glob(["include/**"], allow_empty = True),
    includes = ["include", "include/pinocchio_deprecated"],
    # Linked with an rpath, not through LD_LIBRARY_PATH: /opt/openrobots/lib also holds robotpkg's libblasfeo.so, which
    # must never stand in for the solver's own blasfeo (see @blasfeo, which is linked statically for that reason).
    linkopts = [
        "-L{lib_dir}",
        "-Wl,-rpath,{lib_dir}",
        "-lpinocchio_parsers",
        "-lpinocchio_default",
    ],
    # Pinocchio's configuration of the Boost and urdfdom headers it includes. As `defines` they reach every target that
    # depends on Pinocchio, so every translation unit that sees its headers agrees on them, and no other target carries
    # Boost flags. The joint model is a boost::variant over an MPL list of more joint types than Boost's default limit
    # of 20, its joint visitors dispatch through boost::fusion::invoke, whose arity limit is raised to match, and the
    # urdfdom it parses hands out std::shared_ptr.
    defines = [
        # NOLINTBEGIN(boost): Boost's configuration for Pinocchio's headers, on this target and on no other.
        "BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS",
        "BOOST_MPL_LIMIT_LIST_SIZE=50",
        "BOOST_VARIANT_LIMIT_TYPES=50",
        "BOOST_FUSION_INVOKE_MAX_ARITY=10",
        # NOLINTEND(boost)
        "PINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR",
        "PINOCCHIO_URDFDOM_USE_STD_SHARED_PTR",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@boost",  # NOLINT(boost): Pinocchio's public headers include Boost; no other target depends on it.
        "@urdf",
    ],
)
""".format(lib_dir = lib_dir))

pinocchio_repository = repository_rule(
    implementation = _pinocchio_repository,
    local = True,
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
# urdfdom (@urdf)
# ==============================================================================
# The URDF parser that Pinocchio, the robot description and the MuJoCo contact utilities read URDF files with
# (urdf::parseURDF, urdf::ModelInterface). Ubuntu's liburdfdom-dev installs the model headers flat in /usr/include
# (urdf_model/, urdf_world/, ...) and the parser under /usr/include/urdfdom/urdf_parser.
_URDFDOM_HEADER_DIRS = [
    "urdf_exception",
    "urdf_model",
    "urdf_model_state",
    "urdf_sensor",
    "urdf_world",
]

def _urdf_repository(repo_ctx):
    """Wraps system-installed urdfdom and the headers it includes."""
    if not repo_ctx.path("/usr/include/urdfdom/urdf_parser/urdf_parser.h").exists:
        fail("urdfdom is not installed: /usr/include/urdfdom/urdf_parser/urdf_parser.h does not exist. Install " +
             "liburdfdom-dev (dependencies.txt).")
    for header_dir in _URDFDOM_HEADER_DIRS:
        _symlink_if_exists(repo_ctx, "/usr/include/" + header_dir, "include/" + header_dir)

    # Both spellings of the parser header: <urdf_parser/urdf_parser.h> and <urdfdom/urdf_parser/urdf_parser.h>.
    repo_ctx.symlink("/usr/include/urdfdom", "include/urdfdom")
    repo_ctx.symlink("/usr/include/urdfdom/urdf_parser", "include/urdf_parser")

    _symlink_if_exists(repo_ctx, "/usr/include/console_bridge", "include/console_bridge")
    _symlink_if_exists(repo_ctx, "/usr/include/tinyxml2.h", "include/tinyxml2.h")

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "urdf",
    hdrs = glob(["include/**"], allow_empty = True),
    includes = ["include"],
    linkopts = [
        "-lurdfdom_model",
        "-lurdfdom_world",
        "-lurdfdom_sensor",
    ],
    visibility = ["//visibility:public"],
)
""")

urdf_repository = repository_rule(
    implementation = _urdf_repository,
    local = True,
)

# ==============================================================================
# blasfeo and hpipm, built from source
# ==============================================================================
# Both are built as static archives (their CMake default, compiled with -fPIC), so that no libblasfeo.so is looked up
# at run time: robotpkg installs a libblasfeo.so of its own, of another version, in /opt/openrobots/lib, which every
# binary that links Pinocchio has on its rpath. A shared blasfeo would be found there or here depending on the order
# of the rpath entries. Their examples and test problems are not built: each example defines blasfeo.h's
# BLASFEO_PROCESSOR_FEATURES itself, which collides with the archive's definition (GCC's default -fno-common), and the
# project includes only hpipm's headers, which do not define it.
_BLASFEO_GIT_REPO = "https://github.com/giaf/blasfeo"
_BLASFEO_GIT_COMMIT = "ae6e2d1dea015862a09990b95905038a756ffc7d"

def _cmake_build_and_install(repo_ctx, name, extra_arguments):
    """Configures, builds and installs the CMake project in src/ into install/, as static libraries."""
    _run(repo_ctx, ["cmake", "-S", "src", "-B", "build",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=" + str(repo_ctx.path("install")),
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DBUILD_SHARED_LIBS=OFF",
        # LINT.IfChange(hpipm_target)
        "-DTARGET=GENERIC",
        # LINT.ThenChange(//lib/ocs2/sqp/sqp/test/testSqpFlatParity.cpp:hpipm_build_sensitive)
    ] + extra_arguments, "Configuring " + name)
    _run(repo_ctx, ["cmake", "--build", "build", "-j4"], "Building " + name)
    _run(repo_ctx, ["cmake", "--install", "build"], "Installing " + name)

def _clone(repo_ctx, url, commit, name):
    _run(repo_ctx, ["git", "clone", "--quiet", url, "src"], "Cloning " + name)
    _run(repo_ctx, ["git", "-C", "src", "checkout", "--quiet", commit], "Checking out " + name + " " + commit)

def _blasfeo_repository(repo_ctx):
    """Builds blasfeo from source as a static library."""
    _clone(repo_ctx, _BLASFEO_GIT_REPO, _BLASFEO_GIT_COMMIT, "blasfeo")
    _cmake_build_and_install(repo_ctx, "blasfeo", ["-DBLASFEO_EXAMPLES=OFF", "-DBLASFEO_BENCHMARKS=OFF"])
    repo_ctx.symlink("install/include", "include")
    repo_ctx.symlink("install/lib", "lib")

    repo_ctx.file("BUILD.bazel", content = """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "blasfeo",
    srcs = ["lib/libblasfeo.a"],
    hdrs = glob(["include/**"], allow_empty = True),
    includes = ["include"],
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
)
""")

blasfeo_repository = repository_rule(
    implementation = _blasfeo_repository,
    local = True,
)

_HPIPM_GIT_REPO = "https://github.com/giaf/hpipm"
_HPIPM_GIT_COMMIT = "255ffdf38d3a5e2c3285b29568ce65ae286e5faf"

def _hpipm_repository(repo_ctx):
    """Builds hpipm from source as a static library, against @blasfeo's headers."""
    blasfeo_prefix = str(repo_ctx.path(Label("@blasfeo//:BUILD.bazel")).dirname) + "/install"
    _clone(repo_ctx, _HPIPM_GIT_REPO, _HPIPM_GIT_COMMIT, "hpipm")
    _cmake_build_and_install(repo_ctx, "hpipm", [
        "-DBLASFEO_PATH=" + blasfeo_prefix,
        "-DBLASFEO_INCLUDE_DIR=" + blasfeo_prefix + "/include",
        "-DHPIPM_BLASFEO_LIB=Static",
        "-DHPIPM_TESTING=OFF",
    ])
    repo_ctx.symlink("install/include", "include")
    repo_ctx.symlink("install/lib", "lib")

    repo_ctx.file("BUILD.bazel", content = """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "hpipm",
    srcs = ["lib/libhpipm.a"],
    hdrs = glob(["include/**"], allow_empty = True),
    includes = ["include"],
    deps = ["@blasfeo"],
    visibility = ["//visibility:public"],
)
""")

hpipm_repository = repository_rule(
    implementation = _hpipm_repository,
    local = True,
)

# ==============================================================================
# yaml-cpp
# ==============================================================================
def _yaml_cpp_repository(repo_ctx):
    """Wraps system-installed yaml-cpp."""
    repo_ctx.symlink("/usr/include/yaml-cpp", "include/yaml-cpp")

    # Find and symlink yaml-cpp shared library, in the multiarch directory of this machine's architecture (an arm64
    # dev container builds the bundle of an ARM robot, tools/deploy/README.md "aarch64").
    lib_dir = "/usr/lib/" + ("aarch64" if repo_ctx.os.arch in ("aarch64", "arm64") else "x86_64") + "-linux-gnu"
    result = repo_ctx.execute(["find", lib_dir, "-name", "libyaml-cpp.so*", "-not", "-type", "d"])
    if result.return_code == 0 and result.stdout.strip():
        first_match = result.stdout.strip().split("\n")[0]
        repo_ctx.symlink(first_match, "lib/libyaml-cpp.so")

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
cc_library(
    name = "headers",
    hdrs = glob(["include/yaml-cpp/**"], allow_empty = True),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "yaml_cpp_lib",
    shared_library = "lib/libyaml-cpp.so",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "yaml_cpp",
    visibility = ["//visibility:public"],
    deps = [
        ":headers",
        ":yaml_cpp_lib",
    ],
)
""")

yaml_cpp_repository = repository_rule(
    implementation = _yaml_cpp_repository,
    local = True,
)

# ==============================================================================
# Public function to register all system library repositories
# ==============================================================================

# LINT.IfChange(system_repositories)
def register_system_libs():
    """Registers all system library repositories."""
    eigen_repository(name = "eigen")

    # Not imported by MODULE.bazel: only @pinocchio, registered here too, sees it.
    boost_repository(name = "boost")
    pinocchio_repository(name = "pinocchio")
    glfw_repository(name = "glfw")
    glew_repository(name = "glew")
    urdf_repository(name = "urdf")
    blasfeo_repository(name = "blasfeo")
    hpipm_repository(name = "hpipm")
    yaml_cpp_repository(name = "yaml_cpp")
# LINT.ThenChange(//MODULE.bazel:system_repositories)
