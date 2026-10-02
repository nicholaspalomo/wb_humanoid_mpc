"""Module extension for registering the system library repositories."""

load("//bazel:system_libs.bzl", "register_system_libs")

def _system_libs_impl(module_ctx):
    """Registers all system library repositories."""
    register_system_libs()

system_libs = module_extension(
    implementation = _system_libs_impl,
)
