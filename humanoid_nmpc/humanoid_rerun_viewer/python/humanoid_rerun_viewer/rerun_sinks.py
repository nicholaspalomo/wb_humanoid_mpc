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

"""Where the bridge's recordings go, selected by name (--rerun_sink): the registry of Rerun sinks.

- spawn: start the native viewer (the rerun binary of the rerun-sdk wheel) and stream to it.
- connect: stream to a viewer that is already running, at --rerun_url.
- serve_web: serve the recording over gRPC and the web viewer over HTTP (--web_port), for a browser on the host.
- save: write the recording to an .rrd file (--rrd_path), to open later with `rerun <file>`.

    sink = make_sink("serve_web", layout, SinkOptions())
    sink.attach(recording)           # the first recording: starts the viewer, the server or the file
    sink.attach(next_run_recording)  # a later run of the robot: the same viewer or server, or a numbered file
    sink.reattach(recording)         # spawn, connect: the viewer is back, and rerun-sdk 0.38 never reconnects

Each sink sends the blueprint with send_blueprint() (active and default), so that it is the layout also in a viewer
that already showed this application. None passes it as the sink's `default_blueprint`: in rerun-sdk 0.38 that makes
connect_grpc() block for minutes when no viewer is up, and serve_grpc() never return about once in twelve starts (a
race the stress runs of test_serve_web_sink found; send_blueprint() after the sink showed none in 100). connect_grpc()
itself still waits up to about 5 s for a viewer before it returns, and retries in the background only until it first
connects: rerun-sdk never reconnects to a viewer that went away later, which is what reattach() is for.
A new sink is added to SINKS and nowhere else.
"""

import dataclasses
import os
import socket
import urllib.parse

import rerun as rr
import rerun.blueprint as rrb

# The GUI's "Open Rerun viewer" bridge spawns its viewer on a port of its own (remote_control's GUI_RERUN_GRPC_PORT),
# which these must not take.
# LINT.IfChange(sink_defaults)
DEFAULT_GRPC_PORT = 9876
DEFAULT_WEB_PORT = 9090
DEFAULT_CONNECT_URL = f"rerun+http://127.0.0.1:{DEFAULT_GRPC_PORT}/proxy"
# LINT.ThenChange(//docker-compose.bridge.yaml:rerun_ports, //humanoid_nmpc/remote_control/remote_control/rerun_viewer.py:gui_viewer_port, //humanoid_nmpc/remote_control/test/test_rerun_viewer.py:bridge_default_grpc_port, //tools/deploy/session.sh:views_urls)

# serve_web: how long the flush after the gRPC server started may take [s]; it takes milliseconds when the server runs.
_SERVER_START_TIMEOUT_S = 1.0
# serve_web: how many recordings of the latest runs stay attached besides the first, which owns the server. An older
# one is disconnected, which frees its threads and buffers in the bridge; the viewer keeps what it was sent.
MAX_ATTACHED_RUNS = 2


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
        viewer_executable: spawn: the viewer to start; empty: the native viewer of the rerun-sdk wheel.
    """

    url: str = DEFAULT_CONNECT_URL
    rrd_path: str = ""
    grpc_port: int = DEFAULT_GRPC_PORT
    web_port: int = DEFAULT_WEB_PORT
    open_browser: bool = False
    memory_limit: str = "75%"
    server_memory_limit: str = "8MiB"
    viewer_executable: str = ""


def viewer_executable() -> str | None:
    """The native viewer of the rerun-sdk wheel (rerun_cli/rerun), or None to let Rerun search PATH."""
    try:
        # pylint: disable-next=import-outside-toplevel  # An optional wheel: without it Rerun searches PATH.
        import rerun_cli
    except ImportError:
        return None
    candidate = os.path.join(os.path.dirname(rerun_cli.__file__), "rerun")
    return candidate if os.access(candidate, os.X_OK) else None


def url_endpoint(url: str) -> tuple[str, int] | None:
    """The (host, port) of a gRPC URL such as rerun+http://127.0.0.1:9876/proxy; None when it names no host or port."""
    try:
        parts = urllib.parse.urlsplit(url.removeprefix("rerun+"))
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname or port is None:
        return None
    return parts.hostname, port


def _check_port_free(port: int, kind: str, flag: str) -> None:
    """Raises OSError when a server listens on `port`, which a Rerun server could then not bind.

    SO_REUSEADDR, which Rerun's servers set too: without it a port that only holds connections in TIME_WAIT (a bridge
    that just stopped) is refused as well, while Linux still refuses one that a socket listens on.

    Args:
        port: The TCP port number to check.
        kind: The description of the port (e.g. 'web' or 'gRPC').
        flag: The command-line flag name to suggest if in use.
    """
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError as error:
            raise OSError(
                f"the Rerun {kind} port {port} is in use (another bridge or a"
                f" native Rerun viewer?); pass {flag}: {error}"
            ) from error


class Sink:
    """One sink of SINKS with its options and layout: sends the bridge's recordings to one destination.

    The first recording attach() is given starts the destination (the viewer, the server, the file); each later one, a
    new run of the robot, goes to the same viewer or server, or to a numbered file. Not thread-safe: one thread at a
    time attaches recordings (cli.run's, then the bridge's writer thread).

    Args:
        blueprint: the layout, sent to every recording as its active and default one.
        options: the sink's options; each sink reads the ones it needs.

    Attributes:
        options: The sink's options.
    """

    def __init__(self, blueprint: rrb.Blueprint, options: SinkOptions) -> None:
        self._blueprint = blueprint
        self.options = options
        # The recordings still attached, oldest first, and how many were attached in all (the runs).
        self._recordings: list[rr.RecordingStream] = []
        self._runs = 0

    @property
    def recordings(self) -> tuple[rr.RecordingStream, ...]:
        """The recordings still attached, oldest first (serve_web disconnects old runs': _release_old_runs())."""
        return tuple(self._recordings)

    @property
    def runs(self) -> int:
        """How many recordings were attached so far."""
        return self._runs

    @property
    def viewer_endpoint(self) -> tuple[str, int] | None:
        """The (host, port) of the viewer the recordings stream to, which may go away; None for a server or a file."""
        return None

    @property
    def owns_viewer(self) -> bool:
        """Whether the viewer was started for these recordings (spawn), so that nothing is left to draw once it is gone."""
        return False

    def attach(self, recording: rr.RecordingStream) -> str:
        """Sends `recording` to the sink with the layout; returns what the sink did, for the log.

        Args:
            recording: The Rerun recording stream to attach.

        Returns:
            A human-readable description of what the sink configured.

        Raises:
            ValueError: a sink option it needs is missing.
            OSError: serve_web: its gRPC or web port is in use (the message names the port and its flag).
            RuntimeError: another failure of rerun-sdk to start the sink.
        """
        if self._runs:
            description = self._attach_next(recording, run=self._runs + 1)
        else:
            description = self._attach_first(recording)
        self._recordings.append(recording)
        self._runs += 1
        self._release_old_runs()
        return description

    def reattach(self, recording: rr.RecordingStream) -> str:
        """Connects `recording`, attached before, to the viewer again after it was gone (spawn and connect only)."""
        return self._attach_next(recording, run=self._runs)

    def _release_old_runs(self) -> None:
        """Disconnects the recordings of old runs the sink no longer needs; by default none."""

    def disconnect_all(self) -> None:
        """Flushes and closes every recording, newest first, so that the one that owns a server goes last."""
        for recording in reversed(self._recordings):
            recording.disconnect()

    def _send_blueprint(self, recording: rr.RecordingStream) -> None:
        recording.send_blueprint(
            self._blueprint, make_active=True, make_default=True
        )

    def _attach_first(self, recording: rr.RecordingStream) -> str:
        raise NotImplementedError

    def _attach_next(self, recording: rr.RecordingStream, run: int) -> str:
        raise NotImplementedError


class _SpawnSink(Sink):
    """spawn: starts the native viewer on --grpc_port and streams to it; a later recording connects to it."""

    @property
    def viewer_endpoint(self) -> tuple[str, int] | None:
        return "127.0.0.1", self.options.grpc_port

    @property
    def owns_viewer(self) -> bool:
        return True

    def _attach_first(self, recording: rr.RecordingStream) -> str:
        rr.spawn(
            port=self.options.grpc_port,
            connect=True,
            memory_limit=self.options.memory_limit,
            executable_path=self.options.viewer_executable
            or viewer_executable(),
            recording=recording,
        )
        self._send_blueprint(recording)
        return f"spawned the Rerun viewer on port {self.options.grpc_port}"

    def _attach_next(self, recording: rr.RecordingStream, run: int) -> str:
        url = f"rerun+http://127.0.0.1:{self.options.grpc_port}/proxy"
        recording.connect_grpc(url)
        self._send_blueprint(recording)
        return f"streaming run {run} to the Rerun viewer at {url}"


class _ConnectSink(Sink):
    """connect: streams every recording to the viewer at --rerun_url."""

    @property
    def viewer_endpoint(self) -> tuple[str, int] | None:
        return url_endpoint(self.options.url)

    def _attach_first(self, recording: rr.RecordingStream) -> str:
        return self._attach_next(recording, run=1)

    def _attach_next(self, recording: rr.RecordingStream, run: int) -> str:
        recording.connect_grpc(self.options.url)
        self._send_blueprint(recording)
        return f"streaming run {run} to the Rerun viewer at {self.options.url}"


class _ServeWebSink(Sink):
    """serve_web: the first recording serves itself over gRPC and the web viewer shows it; later ones join its server.

    rerun-sdk 0.38's serve_grpc() returns its URL even when its server cannot bind the port, and the recording then
    drops every message; the ports are checked first, and a flush right after the server started catches a port taken
    in between. The first recording owns the server, so it stays attached until disconnect_all().
    """

    def __init__(self, blueprint: rrb.Blueprint, options: SinkOptions) -> None:
        super().__init__(blueprint, options)
        self._grpc_url = ""

    def _attach_first(self, recording: rr.RecordingStream) -> str:
        options = self.options
        _check_port_free(options.grpc_port, "gRPC", "--grpc_port")
        _check_port_free(options.web_port, "web viewer", "--web_port")
        grpc_url = recording.serve_grpc(
            grpc_port=options.grpc_port,
            server_memory_limit=options.server_memory_limit,
        )
        self._send_blueprint(recording)
        try:
            recording.flush(timeout_sec=_SERVER_START_TIMEOUT_S)
        except RuntimeError as error:
            raise OSError(
                f"the Rerun gRPC server on port {options.grpc_port} did not"
                f" start (the port is in use?); pass --grpc_port: {error}"
            ) from error
        rr.serve_web_viewer(
            web_port=options.web_port,
            open_browser=options.open_browser,
            connect_to=grpc_url,
        )
        self._grpc_url = grpc_url
        return (
            "serving the web viewer at"
            f" http://localhost:{options.web_port}/?url={grpc_url} (recording"
            f" at {grpc_url})"
        )

    def _attach_next(self, recording: rr.RecordingStream, run: int) -> str:
        recording.connect_grpc(self._grpc_url)
        self._send_blueprint(recording)
        return f"serving run {run} at {self._grpc_url}"

    def _release_old_runs(self) -> None:
        """Disconnects all but the first (the server's) and the newest MAX_ATTACHED_RUNS recordings.

        Each run's recording is a gRPC client of the bridge's own server, with threads and buffers of its own; without
        this, a bridge whose robot restarts often would keep every run's in memory. The server cannot go away, so the
        disconnect, which flushes, does not wait for a viewer.
        """
        while len(self._recordings) > MAX_ATTACHED_RUNS + 1:
            self._recordings.pop(1).disconnect()


class _SaveSink(Sink):
    """save: writes the first recording to --rrd_path, and run N after it to run_path(--rrd_path, N) next to it."""

    def _attach_first(self, recording: rr.RecordingStream) -> str:
        if not self.options.rrd_path:
            raise ValueError("the save sink needs an .rrd path (--rrd_path)")
        return self._save(recording, self.options.rrd_path)

    def _attach_next(self, recording: rr.RecordingStream, run: int) -> str:
        # The previous run's file is complete: close it now rather than at exit.
        self._recordings[-1].disconnect()
        return self._save(recording, run_path(self.options.rrd_path, run))

    def _release_old_runs(self) -> None:
        """Forgets the previous runs' recordings, whose files _attach_next() closed."""
        del self._recordings[:-1]

    def _save(self, recording: rr.RecordingStream, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        recording.save(path)
        self._send_blueprint(recording)
        return f"saving the recording to {path}"


def run_path(rrd_path: str, run: int) -> str:
    """The save sink's file of run `run` (from 1): `rrd_path` itself for the first, <stem>.run<N><extension> after."""
    if run <= 1:
        return rrd_path
    stem, extension = os.path.splitext(rrd_path)
    return f"{stem}.run{run}{extension}"


# LINT.IfChange(sink_names)
SINKS: dict[str, type[Sink]] = {
    "spawn": _SpawnSink,
    "connect": _ConnectSink,
    "serve_web": _ServeWebSink,
    "save": _SaveSink,
}
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:sink_names)
DEFAULT_SINK = "spawn"


def sink_names() -> tuple[str, ...]:
    return tuple(SINKS)


def make_sink(
    name: str, blueprint: rrb.Blueprint, options: SinkOptions
) -> Sink:
    """The sink `name` with `blueprint` as its layout; nothing is started before its first attach().

    Args:
        name: The sink name identifier (e.g. 'spawn', 'connect', 'serve_web', 'save').
        blueprint: The layout blueprint to apply to attached recordings.
        options: Configuration options for sink targets and endpoints.

    Returns:
        The configured Sink instance ready to attach recordings.

    Raises:
        ValueError: an unknown sink name (the message lists the valid ones).
    """
    sink = SINKS.get(name)
    if sink is None:
        raise ValueError(
            f"unknown Rerun sink '{name}' (valid: {', '.join(sink_names())})"
        )
    return sink(blueprint, options)


def attach_sink(
    name: str,
    recording: rr.RecordingStream,
    blueprint: rrb.Blueprint,
    options: SinkOptions,
) -> str:
    """Sends `recording` to the sink `name` with `blueprint` as its layout (make_sink(), then Sink.attach()).

    Args:
        name: the sink, a key of SINKS.
        recording: what to send.
        blueprint: the layout, sent as the active and the default one.
        options: the sink's options; each sink reads the ones it needs.

    Returns:
        What the sink did, for the log.

    Raises:
        ValueError: an unknown sink name (the message lists the valid ones), or a sink option it needs is missing.
        OSError: serve_web: its gRPC or web port is in use (the message names the port and its flag).
        RuntimeError: another failure of rerun-sdk to start the sink.
    """
    return make_sink(name, blueprint, options).attach(recording)
