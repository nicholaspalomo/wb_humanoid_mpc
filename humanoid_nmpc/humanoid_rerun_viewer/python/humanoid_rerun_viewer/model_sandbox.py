"""The model sandbox: a robot's URDF drawn in Rerun without an MPC, at its nominal joint positions or at those of a
slider per joint. It replaces RViz's display launch files with joint_state_publisher_gui.

    bazel run //humanoid_nmpc/humanoid_rerun_viewer:model_sandbox -- --urdf <robot.urdf> [--joint_source sliders]
    bazel run //humanoid_nmpc/humanoid_rerun_viewer -- --urdf <robot.urdf>              # the bridge that draws it

It computes the link poses itself (urdf_kinematics.py, the root link at the world origin) and publishes them as the
`measured` robot instance of a VisualizationScene on viz/scene, as the MPC's visualization publisher does, so the
Rerun bridge draws them unchanged. It publishes as the bus node of `--ipc_node` (default "mpc": in the sandbox no MPC
runs) whenever a joint moves and again every `--republish_period`, so that a bridge started later draws the model
too.

The joint positions come from a source chosen by name (`--joint_source`, JOINT_SOURCES): `nominal` holds every joint
at 0, clamped to its limits; `sliders` opens a Tk window with a slider per joint (its limits, or one turn for a joint
without limits) and a button that puts them back. Ctrl-C, SIGTERM or closing the window stops it. Exit status: 0, or 2
for a usage or configuration error.
"""

import argparse
import logging
import math
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import visualization_scene_pb2

from humanoid_rerun_viewer import cli
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import urdf_kinematics
from humanoid_rerun_viewer import urdf_model

_LOGGER = logging.getLogger("model_sandbox")

EXIT_OK = 0
EXIT_USAGE = 2

# LINT.IfChange(joint_sources)
JOINT_SOURCES: Tuple[str, ...] = ("nominal", "sliders")
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:joint_sources)

# The range of a slider whose joint has no limit: one turn for a rotation, a meter either way for a translation [m].
_UNLIMITED_ROTATION = (-math.pi, math.pi)
_UNLIMITED_TRANSLATION = (-1.0, 1.0)
# How often the main thread checks for a stop request [s].
_POLL_PERIOD_S = 0.05


def slider_range(joint: urdf_kinematics.Joint) -> Tuple[float, float]:
    """The range of a joint's slider: its limits, or one turn (one meter either way) where it has none."""
    unlimited = (
        _UNLIMITED_TRANSLATION
        if joint.type in urdf_kinematics.PRISMATIC_TYPES
        else _UNLIMITED_ROTATION
    )
    lower = joint.lower if joint.lower is not None else unlimited[0]
    upper = joint.upper if joint.upper is not None else unlimited[1]
    return lower, upper


class SandboxScene:
    """The joint positions of the sandbox and the scene they make. Thread-safe.

    Args:
        tree: the robot's kinematic tree.
        instance: the robot instance of the scene the bridge draws.
    """

    def __init__(
        self,
        tree: urdf_kinematics.KinematicTree,
        instance: str = scene_contract.MEASURED,
    ) -> None:
        self.tree = tree
        self.instance = instance
        self._joints = {joint.name: joint for joint in tree.movable_joints()}
        self._lock = threading.Lock()
        self._positions = tree.nominal_positions()
        self._version = 0

    @property
    def joints(self) -> Tuple[urdf_kinematics.Joint, ...]:
        return self.tree.movable_joints()

    def positions(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._positions)

    @property
    def version(self) -> int:
        """Counts the changes, so that a publisher sees that something moved."""
        with self._lock:
            return self._version

    def set_position(self, joint: str, position: float) -> float:
        """Moves `joint` to `position`, clamped to its limits; returns the position taken.

        Raises:
            KeyError: `joint` is not a settable joint of the robot.
            ValueError: `position` is not finite.
        """
        if joint not in self._joints:
            raise KeyError(
                f"'{joint}' is not a settable joint; the joints are: {', '.join(self._joints)}"
            )
        if not math.isfinite(position):
            raise ValueError(
                f"the position of '{joint}' must be finite, got {position}"
            )
        clamped = self._joints[joint].clamp(position)
        with self._lock:
            if self._positions[joint] != clamped:
                self._positions[joint] = clamped
                self._version += 1
        return clamped

    def reset(self) -> None:
        """Every joint back at its nominal position."""
        with self._lock:
            nominal = self.tree.nominal_positions()
            if nominal != self._positions:
                self._positions = nominal
                self._version += 1

    def scene(self, time_s: float) -> visualization_scene_pb2.VisualizationScene:
        """The scene of the current positions, at robot time `time_s`: one robot instance with every link."""
        poses = self.tree.link_poses(self.positions())
        message = visualization_scene_pb2.VisualizationScene(time=time_s)
        robot = message.robots.add(name=self.instance)
        for link in self.tree.links:
            translation, quaternion = poses[link]
            robot.link_names.append(link)
            pose = robot.link_poses.add()
            pose.position.x, pose.position.y, pose.position.z = translation
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = quaternion
        return message


class ScenePublisher:
    """Publishes the sandbox's scene when it changed, and again every `republish_period`.

    Args:
        scene: what to publish.
        publish: sends one message; returns False when the bus refused it.
        republish_period: [s] how often an unchanged scene goes out again.
        clock: monotonic seconds; the scene's time is the time since the first call of tick().
    """

    def __init__(
        self,
        scene: SandboxScene,
        publish: Callable[[visualization_scene_pb2.VisualizationScene], bool],
        republish_period: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not republish_period > 0.0:
            raise ValueError("republish_period must be positive")
        self._scene = scene
        self._publish = publish
        self._republish_period = republish_period
        self._clock = clock
        self._start: Optional[float] = None
        self._last_publish: Optional[float] = None
        self._published_version = -1
        self.published = 0

    def tick(self) -> bool:
        """Publishes if the scene changed or the period passed; True when it published."""
        now = self._clock()
        if self._start is None:
            self._start = now
        version = self._scene.version
        due = (
            self._last_publish is None
            or now - self._last_publish >= self._republish_period
        )
        if version == self._published_version and not due:
            return False
        if not self._publish(self._scene.scene(now - self._start)):
            return False
        self._published_version = version
        self._last_publish = now
        self.published += 1
        return True


class SliderWindow:  # pragma: no cover - drawn by test_model_sandbox where a display exists
    """The `sliders` source: a Tk window with a slider per settable joint of the scene, and a reset button."""

    TITLE = "Model Sandbox"

    def __init__(self, scene: SandboxScene, root=None) -> None:
        import tkinter

        self._scene = scene
        self.root = root if root is not None else tkinter.Tk()
        self.root.title(f"{self.TITLE}: {scene.tree.name}")
        frame = tkinter.Frame(self.root)
        frame.pack(fill="both", expand=True)
        canvas = tkinter.Canvas(frame, width=420, height=600)
        scrollbar = tkinter.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = tkinter.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.sliders: Dict[str, "tkinter.Scale"] = {}
        self._values: List["tkinter.DoubleVar"] = []
        positions = scene.positions()
        for joint in scene.joints:
            lower, upper = slider_range(joint)
            # A variable trace rather than the scale's -command, which Tk only runs while the slider is drawn.
            value = tkinter.DoubleVar(self.root, value=positions[joint.name])
            value.trace_add(
                "write",
                lambda *_, name=joint.name, value=value: self._moved(name, value.get()),
            )
            slider = tkinter.Scale(
                inner,
                label=joint.name,
                from_=lower,
                to=upper,
                resolution=(upper - lower) / 1000.0,
                orient="horizontal",
                length=380,
                variable=value,
            )
            slider.pack(fill="x")
            self.sliders[joint.name] = slider
            self._values.append(value)
        tkinter.Button(self.root, text="Nominal pose", command=self.reset).pack(
            fill="x"
        )

    def _moved(self, joint: str, value: float) -> None:
        self._scene.set_position(joint, value)

    def reset(self) -> None:
        self._scene.reset()
        positions = self._scene.positions()
        for name, slider in self.sliders.items():
            slider.set(positions[name])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_sandbox",
        description="Draws a URDF in Rerun at its nominal joint positions or at those of a slider per joint.",
    )
    parser.add_argument("--urdf", required=True, help="the robot's URDF")
    parser.add_argument(
        "--joint_source",
        default="sliders",
        choices=JOINT_SOURCES,
        help="where the joint positions come from (default: %(default)s)",
    )
    parser.add_argument(
        "--network_config",
        default=cli.DEFAULT_NETWORK_CONFIG,
        help="the bus's network file (default: %(default)s)",
    )
    parser.add_argument(
        "--ipc_node",
        default="mpc",
        help="the bus node the scene is published as (default: %(default)s: no MPC runs in the sandbox)",
    )
    parser.add_argument(
        "--republish_period",
        type=float,
        default=1.0,
        help="[s] how often an unchanged scene goes out again, for a bridge that starts later",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="[s] stop after this long; 0 runs until Ctrl-C or until the window closes",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="model_sandbox: %(levelname)s: %(message)s",
    )
    if not args.republish_period > 0.0:
        _LOGGER.error("--republish_period must be positive")
        return EXIT_USAGE
    try:
        tree = urdf_kinematics.load_kinematic_tree(cli.resolve_input_path(args.urdf))
        network = robot_ipc.load_network_config(
            cli.resolve_input_path(args.network_config)
        )
        bus = robot_ipc.Bus(args.ipc_node, network)
    except (
        urdf_model.UrdfError,
        robot_ipc.NetworkConfigError,
        robot_ipc.BusError,
        ValueError,
        OSError,
    ) as error:
        _LOGGER.error("%s", error)
        return EXIT_USAGE

    scene = SandboxScene(tree)
    publisher = ScenePublisher(
        scene,
        lambda message: bus.publish(topics.VIZ_SCENE, message),
        republish_period=args.republish_period,
    )
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    # Only the main thread may install signal handlers (a test runs main() on another one).
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, request_stop)
    deadline = time.monotonic() + args.duration if args.duration > 0.0 else None
    _LOGGER.info(
        "%s: %d links, %d joints to set, root '%s' at the origin; publishing %s as node '%s' (%s joints)",
        tree.name,
        len(tree.links),
        len(scene.joints),
        tree.root_link,
        topics.VIZ_SCENE,
        args.ipc_node,
        args.joint_source,
    )
    bus.start()
    try:
        if args.joint_source == "sliders":
            window = SliderWindow(scene)

            def poll() -> None:
                publisher.tick()
                if stop.is_set() or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    window.root.destroy()
                    return
                window.root.after(int(_POLL_PERIOD_S * 1000), poll)

            window.root.after(0, poll)
            window.root.mainloop()
        else:
            while not stop.wait(_POLL_PERIOD_S):
                publisher.tick()
                if deadline is not None and time.monotonic() >= deadline:
                    break
    finally:
        bus.close()
    _LOGGER.info("stopped after %d scenes", publisher.published)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
