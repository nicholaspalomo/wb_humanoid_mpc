# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Tests for pack_bundle.py: the manifest it reads and the deterministic tar it writes."""

import os
import tarfile
import tempfile
import unittest

import pack_bundle


class PackBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name
        self.plain = self.write("plain.txt", b"plain\n", 0o640)
        self.program = self.write("program", b"#!/bin/sh\n", 0o700)
        self.link = os.path.join(self.directory, "link")
        os.symlink(self.program, self.link)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write(self, name: str, content: bytes, mode: int) -> str:
        path = os.path.join(self.directory, name)
        with open(path, "wb") as stream:
            stream.write(content)
        os.chmod(path, mode)
        return path

    def pack(self, entries) -> str:
        output = os.path.join(self.directory, "bundle.tar")
        pack_bundle.pack(entries, output)
        return output

    def test_entries_are_sorted_regular_owned_by_root_and_dated_zero(self) -> None:
        output = self.pack(
            [
                ("b/plain.txt", self.plain, "auto"),
                ("a/run", self.link, "auto"),
                ("a/forced", self.plain, "1"),
                ("a/empty/__init__.py", "", "0"),
            ]
        )
        with tarfile.open(output) as archive:
            members = archive.getmembers()
            self.assertEqual(
                [member.name for member in members],
                [
                    "a",
                    "a/empty",
                    "b",
                    "a/empty/__init__.py",
                    "a/forced",
                    "a/run",
                    "b/plain.txt",
                ],
            )
            for member in members:
                self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))
                self.assertFalse(member.issym() or member.islnk(), member.name)
            modes = {member.name: member.mode for member in members if member.isfile()}
            self.assertEqual(
                modes,
                {
                    "a/empty/__init__.py": 0o644,
                    "a/forced": 0o755,
                    "a/run": 0o755,
                    "b/plain.txt": 0o644,
                },
            )
            # The symlink's target is in the tar, not the link.
            run = archive.extractfile("a/run")
            assert run is not None
            self.assertEqual(run.read(), b"#!/bin/sh\n")

    def test_the_same_inputs_give_the_same_bytes(self) -> None:
        entries = [("x/plain.txt", self.plain, "auto"), ("x/run", self.program, "auto")]
        first = self.pack(entries)
        with open(first, "rb") as stream:
            first_bytes = stream.read()
        os.utime(self.plain, (12345, 12345))
        with open(self.pack(list(reversed(entries))), "rb") as stream:
            self.assertEqual(stream.read(), first_bytes)

    def test_a_malformed_manifest_is_refused(self) -> None:
        for line in ("only-a-path", "a\tb\tmaybe", "/abs\tb\t1", "../up\tb\t1"):
            with self.subTest(line=line):
                manifest = self.write("manifest", (line + "\n").encode(), 0o644)
                with self.assertRaises(ValueError):
                    pack_bundle.read_manifest(manifest)

    def test_main_reports_a_missing_source(self) -> None:
        manifest = self.write("manifest", b"a\t/nonexistent\tauto\n", 0o644)
        output = os.path.join(self.directory, "out.tar")
        self.assertEqual(
            pack_bundle.main(["--manifest", manifest, "--output", output]), 1
        )


if __name__ == "__main__":
    unittest.main()
