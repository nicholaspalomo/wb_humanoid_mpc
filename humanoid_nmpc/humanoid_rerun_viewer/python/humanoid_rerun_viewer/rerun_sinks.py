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

"""Where the bridge's recording goes, selected by name (--rerun_sink): the registry of Rerun sinks.

- spawn: start the native viewer (the rerun binary of the rerun-sdk wheel) and stream to it.
- connect: stream to a viewer that is already running, at --rerun_url.
- serve_web: serve the recording over gRPC and the web viewer over HTTP (--web_port), for a browser on the host.
- save: write the recording to an .rrd file (--rrd_path), to open later with `rerun <file>`.

Each sink sends the blueprint with send_blueprint() (active and default), so that it is the layout also in a viewer
that already showed this application. None passes it as the sink's `default_blueprint`: in rerun-sdk 0.38 that makes
connect_grpc() block for minutes when no viewer is up, and serve_grpc() never return about once in twelve starts (a
race the stress runs of test_serve_web_sink found; send_blueprint() after the sink showed none in 100). connect_grpc()
itself still waits up to about 5 s for a viewer before it returns and keeps retrying in the background.
A new sink is added to SINKS and nowhere else.
"""

from collections.abc import Callable
import dataclasses
import os

import rerun as rr
import rerun.blueprint as rrb

# LINT.IfChange(sink_defaults)
DEFAULT_GRPC_PORT = 9876
DEFAULT_WEB_PORT = 9090
DEFAULT_CONNECT_URL = f"rerun+http://127.0.0.1:{DEFAULT_GRPC_PORT}/proxy"
# LINT.ThenChange(//docker-compose.bridge.yaml:rerun_ports)


@dataclasses.dataclass(frozen=True)
class SinkOptions:
    """The options of every sink; each sink reads the ones it needs.

    Attributes:
        url: connect: the viewer's gRPC proxy URL.
        rrd_path: save: the file to write.
        grpc_port: spawn: the port the spawned viewer listens on; serve_web: the gRPC server's port.
        web_port: serve_web: the HTTP port of the web viewer.
        open_browser: serve_web: open the web viewer in a browser of this machine.
        memory_limit: spawn: the viewer's memory limit, beyond which it drops the oldest data.
        server_memory_limit: serve_web: the gRPC server's buffer for viewers that connect late.
    """

    url: str = DEFAULT_CONNECT_URL
    rrd_path: str = ""
    grpc_port: int = DEFAULT_GRPC_PORT
    web_port: int = DEFAULT_WEB_PORT
    open_browser: bool = False
    memory_limit: str = "75%"
    server_memory_limit: str = "8MiB"


SinkFunction = Callable[[rr.RecordingStream, rrb.Blueprint, SinkOptions], str]


def viewer_executable() -> str | None:
    """The native viewer of the rerun-sdk wheel (rerun_cli/rerun), or None to let Rerun search PATH."""
    try:
        # pylint: disable-next=import-outside-toplevel  # An optional wheel: without it Rerun searches PATH.
        import rerun_cli
    except ImportError:
        return None
    candidate = os.path.join(os.path.dirname(rerun_cli.__file__), "rerun")
    return candidate if os.access(candidate, os.X_OK) else None


def _spawn(
    recording: rr.RecordingStream, blueprint: rrb.Blueprint, options: SinkOptions
) -> str:
    rr.spawn(
        port=options.grpc_port,
        connect=True,
        memory_limit=options.memory_limit,
        executable_path=viewer_executable(),
        recording=recording,
    )
    recording.send_blueprint(blueprint, make_active=True, make_default=True)
    return f"spawned the Rerun viewer on port {options.grpc_port}"


def _connect(
    recording: rr.RecordingStream, blueprint: rrb.Blueprint, options: SinkOptions
) -> str:
    recording.connect_grpc(options.url)
    recording.send_blueprint(blueprint, make_active=True, make_default=True)
    return f"streaming to the Rerun viewer at {options.url}"


def _serve_web(
    recording: rr.RecordingStream, blueprint: rrb.Blueprint, options: SinkOptions
) -> str:
    """The serve_web sink: the recording over gRPC, and the web viewer that shows it over HTTP."""
    grpc_url = recording.serve_grpc(
        grpc_port=options.grpc_port,
        server_memory_limit=options.server_memory_limit,
    )
    recording.send_blueprint(blueprint, make_active=True, make_default=True)
    rr.serve_web_viewer(
        web_port=options.web_port,
        open_browser=options.open_browser,
        connect_to=grpc_url,
    )
    return f"serving the web viewer at http://localhost:{options.web_port}/?url={grpc_url} (recording at {grpc_url})"


def _save(
    recording: rr.RecordingStream, blueprint: rrb.Blueprint, options: SinkOptions
) -> str:
    if not options.rrd_path:
        raise ValueError("the save sink needs an .rrd path (--rrd_path)")
    directory = os.path.dirname(os.path.abspath(options.rrd_path))
    os.makedirs(directory, exist_ok=True)
    recording.save(options.rrd_path)
    recording.send_blueprint(blueprint, make_active=True, make_default=True)
    return f"saving the recording to {options.rrd_path}"


# LINT.IfChange(sink_names)
SINKS: dict[str, SinkFunction] = {
    "spawn": _spawn,
    "connect": _connect,
    "serve_web": _serve_web,
    "save": _save,
}
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:sink_names)
DEFAULT_SINK = "spawn"


def sink_names() -> tuple[str, ...]:
    return tuple(SINKS)


def attach_sink(
    name: str,
    recording: rr.RecordingStream,
    blueprint: rrb.Blueprint,
    options: SinkOptions,
) -> str:
    """Sends `recording` to the sink `name` with `blueprint` as its layout.

    Args:
        name: the sink, a key of SINKS.
        recording: what to send.
        blueprint: the layout, sent as the active and the default one.
        options: the sink's options; each sink reads the ones it needs.

    Returns:
        What the sink did, for the log.

    Raises:
        ValueError: an unknown sink name (the message lists the valid ones), or a sink option it needs is missing.
    """
    sink = SINKS.get(name)
    if sink is None:
        raise ValueError(
            f"unknown Rerun sink '{name}' (valid: {', '.join(sink_names())})"
        )
    return sink(recording, blueprint, options)
