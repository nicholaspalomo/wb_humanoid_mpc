"""robot_bundle: the robot side of the distributed runtime as one tar file, for the robot-runtime image.

The image (docker/Dockerfile, target robot-runtime) has no Bazel, so the binaries cannot find their runfiles through
it. The bundle lays them out as Bazel does next to a built binary, and the repository's files at their repository
paths, so that the paths of the launch files work in it as in the checkout:

    bin/<package>/<binary>                 every binary of `binaries`
    bin/<package>/<binary>.runfiles/...    its runfiles: _main/... for this repository, <repository>/... for others,
                                           and _repo_mapping
    <repository path>                      every file of `data` (robot_models/..., config/ipc/...)
    <directory>/<file name>                every file of a `layout` entry, under the directory it maps to

The tar is deterministic: sorted entries, owned by root, dated 0, mode 0755 for executables and 0644 otherwise, and
every symlink of the runfiles replaced by the file it points to.
"""

def _runfiles_path(ctx, file):
    """The path of `file` in a runfiles tree: _main/<path> for this repository, <repository>/<path> otherwise."""
    if file.short_path.startswith("../"):
        return file.short_path[len("../"):]
    return ctx.workspace_name + "/" + file.short_path

def _robot_bundle_impl(ctx):
    entries = {}  # path in the bundle -> (source File or None for an empty file, executable)

    def add(path, source, executable):
        if path in entries and entries[path][0] != source:
            fail("two files for %s in the bundle: %s and %s" % (path, entries[path][0], source))
        entries[path] = (source, executable)

    for binary in ctx.attr.binaries:
        info = binary[DefaultInfo]
        executable = info.files_to_run.executable
        if executable == None:
            fail("%s is not an executable target" % binary.label)
        if executable.short_path.startswith("../"):
            fail("%s: only binaries of this repository can be bundled" % binary.label)
        binary_path = "bin/" + executable.short_path
        add(binary_path, executable, True)
        runfiles_root = binary_path + ".runfiles/"
        runfiles = info.default_runfiles
        for file in runfiles.files.to_list():
            add(runfiles_root + _runfiles_path(ctx, file), file, False)
        for symlink in runfiles.symlinks.to_list():
            add(runfiles_root + ctx.workspace_name + "/" + symlink.path, symlink.target_file, False)
        for symlink in runfiles.root_symlinks.to_list():
            add(runfiles_root + symlink.path, symlink.target_file, False)
        for empty in runfiles.empty_filenames.to_list():
            add(runfiles_root + ctx.workspace_name + "/" + empty, None, False)
        repo_mapping = info.files_to_run.repo_mapping_manifest
        if repo_mapping != None:
            add(runfiles_root + "_repo_mapping", repo_mapping, False)

    for target in ctx.attr.data:
        for file in target[DefaultInfo].files.to_list():
            if file.short_path.startswith("../"):
                fail("%s: only files of this repository can be bundled at their repository path" % target.label)
            add(file.short_path, file, False)

    for target, directory in ctx.attr.layout.items():
        for file in target[DefaultInfo].files.to_list():
            add(directory.rstrip("/") + "/" + file.basename, file, True)

    manifest_lines = []
    inputs = []
    for path in sorted(entries):
        source, executable = entries[path]
        if source == None:
            manifest_lines.append("%s\t\t%s" % (path, "0"))
        else:
            # Otherwise ("auto") a file is executable when its source is: the packer reads the source's mode.
            manifest_lines.append("%s\t%s\t%s" % (path, source.path, "1" if executable else "auto"))
            inputs.append(source)
    manifest = ctx.actions.declare_file(ctx.label.name + ".manifest")
    ctx.actions.write(manifest, "\n".join(manifest_lines) + "\n")

    output = ctx.actions.declare_file(ctx.label.name + ".tar")
    arguments = ctx.actions.args()
    arguments.add("--manifest", manifest)
    arguments.add("--output", output)
    ctx.actions.run(
        executable = ctx.executable._packer,
        arguments = [arguments],
        inputs = depset([manifest] + inputs),
        outputs = [output],
        mnemonic = "RobotBundle",
        progress_message = "Packing the robot bundle %{output}",
    )
    return [DefaultInfo(files = depset([output]))]

robot_bundle = rule(
    implementation = _robot_bundle_impl,
    doc = "The robot binaries with their runfiles and the repository files they read, as one deterministic tar.",
    attrs = {
        "binaries": attr.label_list(
            doc = "Executables, each placed at bin/<repository path> with its runfiles tree next to it.",
            cfg = "target",
        ),
        "data": attr.label_list(
            doc = "Files placed at their repository paths.",
            allow_files = True,
        ),
        "layout": attr.label_keyed_string_dict(
            doc = "Files placed under the directory each label maps to, by file name; made executable.",
            allow_files = True,
        ),
        "_packer": attr.label(
            default = "//tools/deploy:pack_bundle",
            executable = True,
            cfg = "exec",
        ),
    },
)
