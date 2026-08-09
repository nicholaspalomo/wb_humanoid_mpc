"""Repository rules for ROS2 system packages.

These rules create Bazel-compatible wrappers around ROS2 packages installed
via apt (e.g., ros-jazzy-rclcpp, ros-jazzy-std-msgs, etc.).
"""

def _ros2_package_repository(repo_ctx):
    """Generic repository rule for a ROS2 system package.

    Creates a cc_library that wraps the headers and shared libraries
    of a ROS2 package installed under /opt/ros/${ROS_DISTRO}.
    """
    ros_distro = repo_ctx.os.environ.get("ROS_DISTRO", "jazzy")
    ros_prefix = "/opt/ros/" + ros_distro
    pkg_name = repo_ctx.attr.pkg_name
    extra_linkopts = repo_ctx.attr.extra_linkopts
    extra_deps = repo_ctx.attr.extra_deps

    # Build the deps string for BUILD file
    deps_str = ""
    if extra_deps:
        deps_entries = ", ".join(['"%s"' % d for d in extra_deps])
        deps_str = "    deps = [%s],\n" % deps_entries

    linkopts_str = ""
    # Always include the ROS2 library path so -l flags can resolve
    all_linkopts = ["-L" + ros_prefix + "/lib"] + extra_linkopts
    if all_linkopts:
        linkopts_entries = ", ".join(['"%s"' % l for l in all_linkopts])
        linkopts_str = "    linkopts = [%s],\n" % linkopts_entries

    # In bzlmod, repo_ctx.attr.name returns the canonical name with module
    # prefix (e.g., '_main~system_libs~ros2_foo'). We need the apparent name
    # (e.g., 'ros2_foo') so that target references like @ros2_foo//:ros2_foo work.
    apparent_name = repo_ctx.attr.name
    # Bazel 9.x uses '+' separator (e.g., '+system_libs+ros2_foo')
    # Older versions use '~' separator (e.g., '_main~system_libs~ros2_foo')
    if "+" in apparent_name:
        apparent_name = apparent_name.split("+")[-1]
    elif "~" in apparent_name:
        apparent_name = apparent_name.split("~")[-1]

    repo_ctx.file("BUILD.bazel", content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "{name}",
    hdrs = glob(["include/**"]),
    includes = ["include"],
{linkopts}{deps}    visibility = ["//visibility:public"],
)
""".format(
        name = apparent_name,
        linkopts = linkopts_str,
        deps = deps_str,
    ))

    # Symlink ALL ROS2 headers so transitive deps are always available.
    # ROS2 Jazzy has a doubly-nested layout: include/<pkg>/<subdir>/headers.hpp
    # Some packages contain headers for MULTIPLE packages (e.g., rosidl_runtime_cpp
    # contains both rosidl_runtime_cpp/ and rosidl_typesupport_cpp/ subdirs).
    # We symlink ALL subdirs from ALL packages so every header is findable.
    repo_ctx.execute(["bash", "-c", """
        mkdir -p include
        ros_prefix="{ros_prefix}"

        # Iterate over all ROS2 package include directories
        for pkg_dir in "${{ros_prefix}}"/include/*/; do
            # Symlink ALL subdirectories inside this package dir
            for sub_dir in "$pkg_dir"/*/; do
                [ -d "$sub_dir" ] || continue
                sub_name=$(basename "$sub_dir")
                if [ ! -e "include/$sub_name" ]; then
                    ln -sf "$sub_dir" "include/$sub_name"
                fi
            done
            # Also handle non-nested packages (single header files directly in pkg_dir)
            pkg_name=$(basename "$pkg_dir")
            if [ ! -d "$pkg_dir/$pkg_name" ] && [ ! -e "include/$pkg_name" ]; then
                ln -sf "$pkg_dir" "include/$pkg_name"
            fi
        done
    """.format(ros_prefix = ros_prefix)])

ros2_package_repository = repository_rule(
    implementation = _ros2_package_repository,
    attrs = {
        "pkg_name": attr.string(mandatory = True, doc = "ROS2 package name"),
        "extra_linkopts": attr.string_list(default = [], doc = "Extra linker flags"),
        "extra_deps": attr.string_list(default = [], doc = "Extra Bazel deps"),
    },
    local = True,
    environ = ["ROS_DISTRO"],
)

# ==============================================================================
# Public function to register all ROS2 package repositories
# ==============================================================================

def register_ros2_packages():
    """Registers all ROS2 system package repositories."""

    # Core ROS2 client library (with all transitive runtime deps)
    ros2_package_repository(
        name = "ros2_rclcpp",
        pkg_name = "rclcpp",
        extra_linkopts = [
            "-lrclcpp",
            "-lrcl",
            "-lrcutils",
            "-lrmw",
            "-lrmw_implementation",
            "-lrcl_logging_interface",
            "-lrcl_logging_spdlog",
            "-lrosidl_runtime_c",
            "-lrosidl_typesupport_c",
            "-lrosidl_typesupport_cpp",
            "-llibstatistics_collector",
            "-lstatistics_msgs__rosidl_typesupport_cpp",
            "-ltracetools",
            "-lrcl_yaml_param_parser",
            "-lrcl_interfaces__rosidl_typesupport_cpp",
            "-lrcl_interfaces__rosidl_typesupport_c",
            "-lbuiltin_interfaces__rosidl_typesupport_cpp",
            "-lbuiltin_interfaces__rosidl_typesupport_c",
            "-lrosgraph_msgs__rosidl_typesupport_cpp",
            "-ltype_description_interfaces__rosidl_typesupport_cpp",
            "-lservice_msgs__rosidl_typesupport_cpp",
        ],
    )

    # ament_index_cpp
    ros2_package_repository(
        name = "ros2_ament_index_cpp",
        pkg_name = "ament_index_cpp",
        extra_linkopts = ["-lament_index_cpp"],
    )

    # Message packages
    ros2_package_repository(
        name = "ros2_std_msgs",
        pkg_name = "std_msgs",
    )

    ros2_package_repository(
        name = "ros2_geometry_msgs",
        pkg_name = "geometry_msgs",
    )

    ros2_package_repository(
        name = "ros2_visualization_msgs",
        pkg_name = "visualization_msgs",
        extra_linkopts = [
            "-lvisualization_msgs__rosidl_typesupport_cpp",
            "-lvisualization_msgs__rosidl_typesupport_c",
        ],
    )

    ros2_package_repository(
        name = "ros2_sensor_msgs",
        pkg_name = "sensor_msgs",
        extra_linkopts = [
            "-lsensor_msgs__rosidl_typesupport_cpp",
            "-lsensor_msgs__rosidl_typesupport_c",
        ],
    )

    ros2_package_repository(
        name = "ros2_rcl_interfaces",
        pkg_name = "rcl_interfaces",
    )

    ros2_package_repository(
        name = "ros2_builtin_interfaces",
        pkg_name = "builtin_interfaces",
    )

    ros2_package_repository(
        name = "ros2_unique_identifier_msgs",
        pkg_name = "unique_identifier_msgs",
    )

    # TF2
    ros2_package_repository(
        name = "ros2_tf2_ros",
        pkg_name = "tf2_ros",
        extra_linkopts = [
            "-ltf2_ros",
            "-ltf2",
            "-ltf2_msgs__rosidl_typesupport_cpp",
            "-lgeometry_msgs__rosidl_typesupport_cpp",
            "-lgeometry_msgs__rosidl_typesupport_c",
            "-lstd_msgs__rosidl_typesupport_cpp",
            "-lstd_msgs__rosidl_typesupport_c",
            "-laction_msgs__rosidl_typesupport_cpp",
            "-lrclcpp_action",
        ],
    )

    ros2_package_repository(
        name = "ros2_tf2_eigen",
        pkg_name = "tf2_eigen",
    )

    # KDL parser
    ros2_package_repository(
        name = "ros2_kdl_parser",
        pkg_name = "kdl_parser",
        extra_linkopts = ["-lkdl_parser"],
    )

    # Interactive markers
    ros2_package_repository(
        name = "ros2_interactive_markers",
        pkg_name = "interactive_markers",
        extra_linkopts = ["-linteractive_markers"],
    )

    # rosidl (for message generation support)
    ros2_package_repository(
        name = "ros2_rosidl",
        pkg_name = "rosidl_default_generators",
    )
