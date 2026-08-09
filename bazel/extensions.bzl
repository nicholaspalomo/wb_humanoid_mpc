"""Module extensions for registering system library and ROS2 repositories."""

load("//bazel:system_libs.bzl", "register_system_libs")
load("//bazel:ros2.bzl", "register_ros2_packages")

def _system_libs_impl(module_ctx):
    """Registers all system library and ROS2 package repositories."""
    register_system_libs()
    register_ros2_packages()

system_libs = module_extension(
    implementation = _system_libs_impl,
)
