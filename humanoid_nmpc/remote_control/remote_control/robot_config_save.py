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

"""The robot's copy of a saved configuration file: sent on operator/config_save, answered on robot/config_save_status.

The MPC runs on the laptop and reads the laptop's files; the robot process reads its own copies, from its persistent
configuration store (humanoid_nmpc/docs/distributed_runtime/README.md, "Saving the configuration"). A tuning tab's
Save therefore writes the laptop's file first (tuned_file.TunedFile.save()) and then hands the exact text it wrote to
RobotConfigSaver.send(), which publishes it as a humanoid_mpc_msgs.ConfigFileSave:

    saver = robot_config_save.RobotConfigSaver(bus.config_save, bus.config_save_statuses)
    sequence = saver.send(robot_config_save.KIND_TASK, path, result.text, robot_name)
    ...
    saver.poll(saver.clock())                # on the Tk timer, every 200 ms
    outcome = saver.outcome(sequence)        # SAVING until the robot answers, TIMED_OUT after 5 s without an answer
    robot_config_save.describe(outcome)      # the tab's status line

The robot checks the file as it checks it at start-up (its kind, the configuration's path from robot_models/ on, the
robot name, the file schema's fingerprint, a strict parse and the start-up checks), stores it atomically and answers
with the save's sequence. A status that arrives after the timeout still replaces TIMED_OUT. The robot answers a repeat
of one sequence with the status it recorded, so push_robot_config can send one save until it hears the answer.

Free of Tk, so that the tests exercise it headless; every call is made from one thread (Tk's, or the CLI's main
thread), and only the status mailbox is filled from the bus's receive thread.
"""

from collections.abc import Callable
import dataclasses
import enum
import os
import time
from typing import Any, Protocol

from google.protobuf import message as protobuf_message
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2
from humanoid_mpc_msgs import config_file_kind_pb2
from humanoid_mpc_msgs import config_file_save_pb2
from humanoid_mpc_msgs import config_file_save_status_pb2

import nproto_schema
import nproto_textproto
from remote_control import config_files

# The kinds of the files the robot stores (humanoid_mpc_msgs.ConfigFileKind).
KIND_TASK = config_file_kind_pb2.CONFIG_FILE_KIND_TASK
KIND_REFERENCE = config_file_kind_pb2.CONFIG_FILE_KIND_REFERENCE
KIND_JOINT_PD_GAINS = config_file_kind_pb2.CONFIG_FILE_KIND_JOINT_PD_GAINS

# How long a save waits for the robot's answer before it is shown as unknown [s].
SAVE_TIMEOUT_SECONDS = 5.0
# How many saves the saver remembers the outcome of; a status of an older one is ignored.
_REMEMBERED_SAVES = 64

_Status = config_file_save_status_pb2.ConfigFileSaveStatus


@dataclasses.dataclass(frozen=True)
class FileKind:
    """A configuration file the robot stores.

    Attributes:
      kind: Its humanoid_mpc_msgs.ConfigFileKind.
      message_class: The generated class of its schema, whose fingerprint the save carries.
      name: How a message names it, e.g. "task file".
    """

    kind: int
    message_class: type[protobuf_message.Message]
    name: str


# Every humanoid_mpc_msgs.ConfigFileKind but CONFIG_FILE_KIND_UNSPECIFIED, as the robot stores them.
# LINT.IfChange(file_kinds)
FILE_KINDS: tuple[FileKind, ...] = (
    FileKind(KIND_TASK, task_file_pb2.TaskFile, "task file"),
    FileKind(KIND_REFERENCE, reference_file_pb2.ReferenceFile, "reference file"),
    FileKind(
        KIND_JOINT_PD_GAINS,
        joint_pd_gains_file_pb2.JointPdGainsFile,
        "joint PD gains file",
    ),
)
# LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_msgs/config_file_kind.proto:config_file_kinds)


def file_kind(kind: int) -> FileKind:
    """The FILE_KINDS entry of `kind`; ValueError for a kind the robot does not store."""
    for entry in FILE_KINDS:
        if entry.kind == kind:
            return entry
    raise ValueError(f"the robot stores no configuration file of kind {kind}")


def file_kind_of_message(full_name: str) -> FileKind | None:
    """The FILE_KINDS entry whose schema is the message `full_name` (a file's `# proto-message:`); None for none."""
    for entry in FILE_KINDS:
        if entry.message_class.DESCRIPTOR.full_name == full_name:
            return entry
    return None


def schema_fingerprint(kind: int) -> str:
    """The fingerprint of the schema of the files of `kind` in this build, which the robot compares with its own."""
    return nproto_schema.schema_fingerprint(file_kind(kind).message_class.DESCRIPTOR)


def config_path_of(path: str) -> str:
    """The identity of a robot configuration file: its path from the last robot_models/ component on.

    The laptop's file and the robot's copy of it are the same configuration file wherever their checkouts or bundles
    lie, so the identity is the part of the path below them: robot_models/<robot>/<package>/config/<directory>/<file>,
    as config_files.resolve_source_path() re-roots a file into the checkout. The path is normalized lexically (`.` and
    `..` resolved, symbolic links not followed), as the C++ configFileIdentity() normalizes it.

    Args:
      path: The file.

    Returns:
      Its identity; the normalized path itself when no component of it but the last is robot_models.
    """
    # LINT.IfChange(config_path_of)
    normal = os.path.normpath(path)
    components = normal.split(os.sep)
    start = None
    for index in range(len(components) - 1):
        if components[index] == config_files.ROBOT_MODELS_DIR:
            start = index
    if start is None:
        return normal
    return "/".join(components[start:])
    # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/ConfigFiles.cpp:config_file_identity)


class RobotNameError(ValueError):
    """The robot a configuration file is for cannot be read: no task file beside it, or one that does not parse."""


def robot_name_of(kind: int, path: str) -> str:
    """The robot a configuration file is for: model_settings.robot_name of its configuration's task file.

    Args:
      kind: The file's kind.
      path: The file: the task file itself, or a file of the same config/ directory.

    Returns:
      The robot name.

    Raises:
      RobotNameError: The task file cannot be read or does not parse; the message names it.
    """
    task_file = path if kind == KIND_TASK else config_files.task_file_beside(path)
    try:
        task = nproto_textproto.load_textproto(task_file, task_file_pb2.TaskFile)
    except (OSError, nproto_textproto.TextprotoError) as error:
        raise RobotNameError(
            f"the robot of {path} is named by its task file {task_file}, which cannot be read: {error}"
        ) from error
    return str(task.model_settings.robot_name)


def config_file_save(
    kind: int, path: str, text: str, robot_name: str, sequence: int
) -> config_file_save_pb2.ConfigFileSave:
    """The ConfigFileSave of a file's text.

    Args:
      kind: The file's kind.
      path: The file, whose config_path_of() the save carries.
      text: The file's content: exactly what the laptop's copy holds.
      robot_name: The robot the file is for (robot_name_of()).
      sequence: The save's number, which the robot's answer carries.

    Returns:
      The message.
    """
    return config_file_save_pb2.ConfigFileSave(
        kind=kind,
        robot_name=robot_name,
        config_path=config_path_of(path),
        schema_fingerprint=schema_fingerprint(kind),
        text=text,
        sequence=sequence,
    )


class SaveState(enum.Enum):
    """Where a save to the robot stands."""

    SAVING = "saving"  # sent; no answer yet
    SAVED = "saved"  # the robot stored it
    UNCHANGED = "unchanged"  # the robot's copy is the same text: nothing written
    NOT_STORED = "not_stored"  # the robot has no store: it reads a file in place, the laptop's or its own
    REFUSED = "refused"  # the robot refused it: the file, the robot or the schema is not the robot's
    FAILED = "failed"  # the robot could not write it
    TIMED_OUT = "timed_out"  # no answer within the timeout: the robot's copy is unknown
    NOT_SENT = "not_sent"  # never sent: the bus dropped it, or the GUI has no bus


# The robot's results, as states; a result this build does not know is FAILED.
_STATES_OF_RESULTS = {
    _Status.RESULT_SAVED: SaveState.SAVED,
    _Status.RESULT_UNCHANGED: SaveState.UNCHANGED,
    _Status.RESULT_NOT_STORED: SaveState.NOT_STORED,
    _Status.RESULT_REFUSED: SaveState.REFUSED,
    _Status.RESULT_FAILED: SaveState.FAILED,
}
# The states the robot's copy may still change from: the robot has not answered (yet).
_WAITING_STATES = (SaveState.SAVING, SaveState.TIMED_OUT)
# The states that leave the two copies different or their equality unknown.
_PROBLEM_STATES = (
    SaveState.REFUSED,
    SaveState.FAILED,
    SaveState.TIMED_OUT,
    SaveState.NOT_SENT,
)


@dataclasses.dataclass(frozen=True)
class SaveOutcome:
    """What became of one save to the robot.

    Attributes:
      sequence: The save's number; 0 for a save that was never numbered (not_sent()).
      kind: The file's kind.
      config_path: The file's identity (config_path_of()).
      state: Where it stands.
      message: The robot's refusal or failure, or why it was not sent; empty otherwise.
      stored_path: Where the robot stored it, or for NOT_STORED the file it reads in place; empty until it says.
      path: The laptop's file the save was of.
      reads_laptop_file: For NOT_STORED: whether the file the robot reads in place is the laptop's file itself (a robot
        process of the same checkout), so that the save is in it.
    """

    sequence: int
    kind: int
    config_path: str
    state: SaveState
    message: str = ""
    stored_path: str = ""
    path: str = ""
    reads_laptop_file: bool = False

    @property
    def waiting(self) -> bool:
        """Whether the robot has not answered: a later status may still change the outcome."""
        return self.state in _WAITING_STATES

    @property
    def problem(self) -> bool:
        """Whether the robot's copy may differ from the laptop's: refused, failed, unanswered, not sent, or another file read."""
        if self.state == SaveState.NOT_STORED:
            return not self.reads_laptop_file
        return self.state in _PROBLEM_STATES


def not_sent(kind: int, path: str, reason: str) -> SaveOutcome:
    """The outcome of a save that is not sent at all, for `reason`."""
    return SaveOutcome(
        sequence=0,
        kind=kind,
        config_path=config_path_of(path),
        state=SaveState.NOT_SENT,
        message=reason,
        path=path,
    )


def describe(outcome: SaveOutcome, timeout: float = SAVE_TIMEOUT_SECONDS) -> str:
    """The status line of a tab after its Save: the laptop's copy, saved, and the robot's.

    Args:
      outcome: The robot's side of the save.
      timeout: The saver's timeout [s], which the unanswered state names.

    Returns:
      The line.
    """
    laptop = "Saved on the laptop"
    state = outcome.state
    if state == SaveState.SAVING:
        return f"{laptop} · saving on the robot…"
    if state == SaveState.SAVED:
        return f"{laptop} and on the robot ({outcome.stored_path})"
    if state == SaveState.UNCHANGED:
        return f"{laptop} and on the robot ({outcome.stored_path}) … robot unchanged: its copy was the same"
    if state == SaveState.NOT_STORED:
        if outcome.reads_laptop_file:
            return f"{laptop} · the robot has no store and reads this file ({outcome.stored_path})"
        read = outcome.stored_path or "its own copy"
        return f"{laptop} · the robot has no store: it reads {read} in place; this Save did not change it"
    if state == SaveState.REFUSED:
        return f"{laptop} · the robot refused it: {outcome.message}"
    if state == SaveState.FAILED:
        return f"{laptop} · the robot could not store it: {outcome.message}"
    if state == SaveState.TIMED_OUT:
        return (
            f"{laptop} · the robot did not answer in {timeout:g} s: the robot's copy is unknown; "
            "Save again to retry (a repeat answers unchanged)"
        )
    if state == SaveState.NOT_SENT:
        return f"{laptop} · not sent to the robot: {outcome.message}"
    raise ValueError(f"unknown save state {state!r}")


class SaveStatusSource(Protocol):
    """Where the robot's answers come from: operator_bus.ConfigSaveStatusMailbox, filled by the bus."""

    def take_all(self) -> list[Any]:
        """Every ConfigFileSaveStatus received since the last call, oldest first."""


class SavePublisher(Protocol):
    """Where the saves go: the operator_bus.TopicPublisher of operator/config_save."""

    def publish(self, message: protobuf_message.Message) -> bool:
        """Publishes `message`; False when the bus dropped it."""


@dataclasses.dataclass
class _PendingSave:
    """A save the saver remembers: its message, when it was sent and its outcome."""

    message: config_file_save_pb2.ConfigFileSave
    sent_at: float
    outcome: SaveOutcome


class RobotConfigSaver:
    """Sends saved configuration files to the robot and follows its answers, matched by the save's sequence.

    One saver serves every tab of the GUI, so that one status mailbox has one reader; each tab follows the sequence of
    its own last save (outcome()). Not thread-safe: send(), resend() and poll() are called from one thread.

    Args:
      publisher: The publisher of operator/config_save.
      statuses: The robot's answers (robot/config_save_status), drained by poll().
      clock: Monotonic seconds, which the timeout is measured in; the tests pass a fake one.
      sequence_clock: The wall clock in nanoseconds, which numbers the saves, so that the saves of a restarted GUI
        still count up from those of the run before it.
      timeout: How long a save waits for its answer before it is TIMED_OUT [s].

    Attributes:
      clock: The monotonic clock the timeout is measured on, which poll()'s `now` reads.
      timeout: How long a save waits for its answer [s].
    """

    def __init__(
        self,
        publisher: SavePublisher,
        statuses: SaveStatusSource,
        clock: Callable[[], float] = time.monotonic,
        sequence_clock: Callable[[], int] = time.time_ns,
        timeout: float = SAVE_TIMEOUT_SECONDS,
    ) -> None:
        # The not-form rejects NaN, which `timeout <= 0.0` would let through.
        if not timeout > 0.0:
            raise ValueError(f"the timeout must be positive, got {timeout}")
        self._publisher = publisher
        self._statuses = statuses
        self.clock = clock
        self._sequence_clock = sequence_clock
        self.timeout = timeout
        self._last_sequence = 0
        # The remembered saves by sequence, oldest first.
        self._saves: dict[int, _PendingSave] = {}

    def send(self, kind: int, path: str, text: str, robot_name: str) -> int:
        """Publishes a save of the file `path` with the content `text` and returns its sequence.

        Args:
          kind: The file's kind.
          path: The laptop's file.
          text: Exactly the text the laptop's file holds.
          robot_name: The robot it is for (robot_name_of()).

        Returns:
          The save's sequence: the wall clock in nanoseconds, above every earlier sequence of this saver.
        """
        sequence = max(int(self._sequence_clock()), self._last_sequence + 1)
        self._last_sequence = sequence
        message = config_file_save(kind, path, text, robot_name, sequence)
        state = SaveState.SAVING
        reason = ""
        if not self._publisher.publish(message):
            state = SaveState.NOT_SENT
            reason = "the bus dropped it (its send queue is full); Save again"
        self._saves[sequence] = _PendingSave(
            message=message,
            sent_at=self.clock(),
            outcome=SaveOutcome(
                sequence=sequence,
                kind=kind,
                config_path=message.config_path,
                state=state,
                message=reason,
                path=path,
            ),
        )
        while len(self._saves) > _REMEMBERED_SAVES:
            del self._saves[next(iter(self._saves))]
        return sequence

    def resend(self, sequence: int) -> bool:
        """Publishes the save `sequence` again, unchanged, as a sender does until the robot answers.

        A save the bus dropped is SAVING again once a repeat goes out; its timeout still counts from the first send.

        Args:
          sequence: A sequence send() returned.

        Returns:
          Whether the repeat went out; False for a save the saver does not remember or the bus dropped.
        """
        pending = self._saves.get(sequence)
        if pending is None or not self._publisher.publish(pending.message):
            return False
        if pending.outcome.state == SaveState.NOT_SENT:
            pending.outcome = dataclasses.replace(
                pending.outcome, state=SaveState.SAVING, message=""
            )
        return True

    def outcome(self, sequence: int) -> SaveOutcome | None:
        """The outcome of the save `sequence` as of the last poll(); None for one the saver does not remember."""
        pending = self._saves.get(sequence)
        return None if pending is None else pending.outcome

    def poll(self, now: float) -> list[SaveOutcome]:
        """Applies the robot's answers received since the last call, then the timeout.

        An answer is matched to its save by sequence; an answer to a save of another sender or one the saver no longer
        remembers is ignored. An answer replaces SAVING and TIMED_OUT alike, so a late answer is shown.

        Args:
          now: The time on the saver's clock.

        Returns:
          The outcomes that changed, in the order they changed.
        """
        changed: list[SaveOutcome] = []
        for status in self._statuses.take_all():
            pending = self._saves.get(int(status.sequence))
            if pending is None:
                continue
            answered = _answered(pending.outcome, status)
            if answered != pending.outcome:
                pending.outcome = answered
                changed.append(answered)
        for pending in self._saves.values():
            if (
                pending.outcome.state == SaveState.SAVING
                and now - pending.sent_at >= self.timeout
            ):
                pending.outcome = dataclasses.replace(
                    pending.outcome, state=SaveState.TIMED_OUT
                )
                changed.append(pending.outcome)
        return changed


def _same_file(first: str, second: str) -> bool:
    """Whether the paths `first` and `second` name one existing file (os.path.samefile()); False when either is not there."""
    if not first or not second:
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _answered(outcome: SaveOutcome, status: Any) -> SaveOutcome:
    """`outcome` as the robot's ConfigFileSaveStatus `status` answers it."""
    state = _STATES_OF_RESULTS.get(int(status.result), SaveState.FAILED)
    message = str(status.message)
    if state == SaveState.FAILED and int(status.result) not in _STATES_OF_RESULTS:
        message = f"the robot answered with result {int(status.result)}, which this GUI does not know: {message}"
    stored_path = str(status.stored_path)
    return dataclasses.replace(
        outcome,
        state=state,
        message=message,
        stored_path=stored_path,
        # Checked once, here: the robot's path is the laptop's only for a robot process of this machine's checkout.
        reads_laptop_file=state == SaveState.NOT_STORED
        and _same_file(stored_path, outcome.path),
    )


class SaveTracker:
    """A tab's view of its last save to the robot: the outcome to show, and when it changes.

    Args:
      saver: The GUI's saver; None: the tab has no bus, and its saves are not sent.
    """

    def __init__(self, saver: RobotConfigSaver | None) -> None:
        self._saver = saver
        self._sequence: int | None = None
        self._shown: SaveOutcome | None = None

    def outcome(self) -> SaveOutcome | None:
        """The outcome of the last save as last returned; None before the first save."""
        return self._shown

    @property
    def timeout(self) -> float:
        """How long a save waits for its answer [s]."""
        return SAVE_TIMEOUT_SECONDS if self._saver is None else self._saver.timeout

    def send(self, kind: int, path: str, text: str) -> SaveOutcome:
        """Sends the saved file `path` with the content `text` to the robot, for the robot its configuration names.

        Args:
          kind: The file's kind.
          path: The laptop's file, just saved.
          text: Exactly the text it holds (tuned_file.TunedFile.save()).

        Returns:
          The save's outcome now: SAVING, or NOT_SENT with the reason.
        """
        self._sequence = None
        if self._saver is None:
            self._shown = not_sent(kind, path, "the GUI is not connected to the bus")
            return self._shown
        try:
            robot_name = robot_name_of(kind, path)
        except RobotNameError as error:
            self._shown = not_sent(kind, path, str(error))
            return self._shown
        self._sequence = self._saver.send(kind, path, text, robot_name)
        outcome = self._saver.outcome(self._sequence)
        if outcome is None:
            raise RuntimeError(
                f"the saver does not remember the save {self._sequence} it just sent"
            )
        self._shown = outcome
        return outcome

    def poll(self) -> SaveOutcome | None:
        """Polls the saver, at its clock's time; returns the last save's outcome when it changed since the last call."""
        if self._saver is None or self._sequence is None:
            return None
        self._saver.poll(self._saver.clock())
        outcome = self._saver.outcome(self._sequence)
        if outcome is None or outcome == self._shown:
            return None
        self._shown = outcome
        return outcome
