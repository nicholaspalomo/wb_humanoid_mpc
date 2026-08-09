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
    if "~" in apparent_name:
        apparent_name = apparent_name.split("~")[-1]

    repo_ctx.file("BUILD.bazel", content = """
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

    # Symlink headers from the ROS2 install
    # ROS2 Jazzy installs headers as /opt/ros/jazzy/include/<pkg>/<pkg>/headers.hpp
    # We need include/<pkg>/headers.hpp so that #include <pkg/header.hpp> works.
    repo_ctx.execute(["bash", "-c", """
        mkdir -p include
        ros_prefix="{ros_prefix}"
        pkg="{pkg_name}"

        # ROS2 Jazzy has a doubly-nested layout: include/<pkg>/<pkg>/
        # Symlink the inner directory so headers are at include/<pkg>/
        if [ -d "${{ros_prefix}}/include/${{pkg}}/${{pkg}}" ]; then
            ln -sf "${{ros_prefix}}/include/${{pkg}}/${{pkg}}" "include/${{pkg}}"
        elif [ -d "${{ros_prefix}}/include/${{pkg}}" ]; then
            # Fallback: single-level layout (Humble or non-nested packages)
            ln -sf "${{ros_prefix}}/include/${{pkg}}" "include/${{pkg}}"
        fi

        # Some packages have related include directories (e.g., pkg_detail)
        for inc_dir in $(find "${{ros_prefix}}/include" -maxdepth 1 -type d -name "${{pkg}}*" 2>/dev/null); do
            basename=$(basename "$inc_dir")
            # Check for doubly-nested first
            if [ -d "$inc_dir/$basename" ]; then
                if [ ! -e "include/$basename" ]; then
                    ln -sf "$inc_dir/$basename" "include/$basename"
                fi
            elif [ ! -e "include/$basename" ]; then
                ln -sf "$inc_dir" "include/$basename"
            fi
        done
    """.format(ros_prefix = ros_prefix, pkg_name = pkg_name)])

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

    # Core ROS2 client library
    ros2_package_repository(
        name = "ros2_rclcpp",
        pkg_name = "rclcpp",
        extra_linkopts = ["-lrclcpp"],
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
    )

    ros2_package_repository(
        name = "ros2_sensor_msgs",
        pkg_name = "sensor_msgs",
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
        extra_linkopts = ["-ltf2_ros"],
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
