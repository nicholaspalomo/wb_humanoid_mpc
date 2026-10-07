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

"""The registry of Rerun sinks: names, the save sink, and the native viewer of the wheel.

serve_web is tested in test_serve_web_sink.py, a process of its own. The connect sink without a viewer is tested on the
binary (test_end_to_end): rerun-sdk then waits without end when the process exits, which the binary avoids and a test
process could not.
"""

import os
import shutil
import tempfile
import unittest

import rerun as rr

from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import rerun_sinks
import rrd_contents


class SinkRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.recording = bridge.new_recording("test_rerun_sinks")
        self.addCleanup(self.recording.disconnect)
        self.layout = blueprint.build_blueprint()

    def test_the_names(self) -> None:
        self.assertEqual(
            rerun_sinks.sink_names(), ("spawn", "connect", "serve_web", "save")
        )
        self.assertIn(rerun_sinks.DEFAULT_SINK, rerun_sinks.sink_names())

    def test_an_unknown_name_lists_the_valid_ones(self) -> None:
        with self.assertRaises(ValueError) as context:
            rerun_sinks.attach_sink(
                "no_such_sink", self.recording, self.layout, rerun_sinks.SinkOptions()
            )
        for name in rerun_sinks.sink_names():
            self.assertIn(name, str(context.exception))

    def test_save_writes_the_recording_and_the_blueprint(self) -> None:
        path = os.path.join(self.directory, "nested", "out.rrd")
        description = rerun_sinks.attach_sink(
            "save",
            self.recording,
            self.layout,
            rerun_sinks.SinkOptions(rrd_path=path),
        )
        self.assertIn(path, description)
        self.recording.log("world/x", rr.Points3D([[0.0, 0.0, 0.0]]))
        self.recording.flush()
        self.recording.disconnect()
        contents = rrd_contents.RrdContents(path)
        self.assertIn("world/x", contents.entities)
        self.assertIn(blueprint.SCENE_VIEW_NAME, contents.view_names())

    def test_save_needs_a_path(self) -> None:
        with self.assertRaises(ValueError):
            rerun_sinks.attach_sink(
                "save", self.recording, self.layout, rerun_sinks.SinkOptions()
            )

    def test_the_wheel_ships_the_native_viewer(self) -> None:
        executable = rerun_sinks.viewer_executable()
        self.assertIsNotNone(executable)
        assert executable is not None
        self.assertTrue(os.access(executable, os.X_OK))


if __name__ == "__main__":
    unittest.main()
