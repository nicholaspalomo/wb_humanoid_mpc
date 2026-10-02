"""Packs the robot bundle (robot_bundle.bzl) into a deterministic tar file.

    pack_bundle --manifest <bundle>.manifest --output <bundle>.tar

Every line of the manifest is `<path in the bundle>\\t<source file>\\t<mode>`: mode `1` makes the entry executable,
`auto` executable when the source is, and `0` with an empty source is an empty file. Sources are read through their
symlinks, so the tar holds no link. Entries are sorted, every directory has an entry of its own, everything is owned
by root and dated 0, and files are 0644 or 0755: two builds of the same inputs give the same bytes.
"""

import argparse
import io
import os
import stat
import sys
import tarfile
from typing import List, Optional, Sequence, Tuple

Entry = Tuple[str, str, str]


def read_manifest(path: str) -> List[Entry]:
    entries: List[Entry] = []
    with open(path, "r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 3 or not fields[0] or fields[2] not in ("0", "1", "auto"):
                raise ValueError(
                    f"{path}:{number}: not '<path>\\t<source>\\t<0|1|auto>': {line!r}"
                )
            if fields[0].startswith("/") or ".." in fields[0].split("/"):
                raise ValueError(f"{path}:{number}: {fields[0]} leaves the bundle")
            entries.append((fields[0], fields[1], fields[2]))
    return entries


def _info(name: str, kind: bytes, mode: int, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = mode
    info.size = size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


def pack(entries: Sequence[Entry], output: str) -> None:
    directories = set()
    for path, _, _ in entries:
        parent = os.path.dirname(path)
        while parent:
            directories.add(parent)
            parent = os.path.dirname(parent)
    with tarfile.open(output, "w", format=tarfile.GNU_FORMAT) as archive:
        for directory in sorted(directories):
            archive.addfile(_info(directory, tarfile.DIRTYPE, 0o755))
        for path, source, mode in sorted(entries):
            if mode == "0":
                archive.addfile(_info(path, tarfile.REGTYPE, 0o644), io.BytesIO(b""))
                continue
            status = os.stat(source)
            if not stat.S_ISREG(status.st_mode):
                raise ValueError(f"{path}: {source} is not a regular file")
            executable = mode == "1" or bool(status.st_mode & stat.S_IXUSR)
            with open(source, "rb") as stream:
                archive.addfile(
                    _info(
                        path,
                        tarfile.REGTYPE,
                        0o755 if executable else 0o644,
                        status.st_size,
                    ),
                    stream,
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pack_bundle", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        pack(read_manifest(args.manifest), args.output)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"pack_bundle: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
