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

"""The robot's copy of a saved file (robot_config_save, push_robot_config), headless: the message, the answers, the CLI.

The saver publishes over a bus that records (operator_test_support.RecordingBus) and takes the robot's answers from the
GUI's status mailbox, which the tests fill as the bus's receive thread would. A fake clock stands in for the time.
"""

import os
import shutil
import tempfile
import unittest

from google.protobuf import message as protobuf_message
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2
from humanoid_mpc_msgs import config_file_kind_pb2
from humanoid_mpc_msgs import config_file_save_pb2
from humanoid_mpc_msgs import config_file_save_status_pb2

from humanoid_mpc_ipc import topics
import nproto_schema
import nproto_textproto
import operator_test_support
from remote_control import operator_bus
from remote_control import push_robot_config
from remote_control import robot_config_save

_Status = config_file_save_status_pb2.ConfigFileSaveStatus
SaveState = robot_config_save.SaveState

ATLAS_IDENTITY = (
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto"
)
STORED = "/var/lib/wb-humanoid-robot/config/drc_atlas/mpc/task.textproto"


class FakeClock:
    """Monotonic seconds that move only when a test sleeps."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class DroppingBus(operator_test_support.RecordingBus):
    """A recording bus whose send queue is full for the first `drops` messages: publish() returns False for them."""

    def __init__(self, drops: int) -> None:
        super().__init__()
        self.drops = drops

    def publish(self, topic: str, message: protobuf_message.Message) -> bool:
        if self.drops > 0:
            self.drops -= 1
            return False
        return super().publish(topic, message)


class AnsweringPublisher(operator_bus.TopicPublisher):
    """The publisher of operator/config_save, with a robot that answers the `answer_at`-th message it hears."""

    def __init__(
        self,
        statuses: operator_bus.ConfigSaveStatusMailbox,
        answer_at: int | None,
        result: int = _Status.RESULT_SAVED,
    ) -> None:
        row = operator_bus.operator_topic(topics.OPERATOR_CONFIG_SAVE)
        super().__init__(
            operator_test_support.RecordingBus(), row.topic, row.message_class
        )
        self._statuses = statuses
        self._answer_at = answer_at
        self._result = result

    def publish(self, message: protobuf_message.Message) -> bool:
        published = super().publish(message)
        if len(self.messages) == self._answer_at:
            assert isinstance(message, config_file_save_pb2.ConfigFileSave)
            self._statuses.put(
                _Status(
                    sequence=message.sequence,
                    kind=message.kind,
                    config_path=message.config_path,
                    result=self._result,
                    stored_path=STORED,
                )
            )
        return published

    @property
    def messages(self) -> list[protobuf_message.Message]:
        return self.bus.messages_on(self.topic)


def _status(
    sequence: int, result: int, message: str = "", stored: str = STORED
) -> config_file_save_status_pb2.ConfigFileSaveStatus:
    return _Status(
        sequence=sequence,
        kind=robot_config_save.KIND_TASK,
        config_path=ATLAS_IDENTITY,
        result=result,
        message=message,
        stored_path=stored,
    )


class TestTheIdentityAndTheMessage(unittest.TestCase):
    def test_the_identity_is_the_path_from_robot_models_on(self):
        self.assertEqual(
            robot_config_save.config_path_of(
                os.path.join("/home/me/checkout", ATLAS_IDENTITY)
            ),
            ATLAS_IDENTITY,
        )
        # The robot's bundle and the checkout name the same file alike; the last robot_models/ component counts.
        self.assertEqual(
            robot_config_save.config_path_of(
                os.path.join("/opt/robot_models/bundle", ATLAS_IDENTITY)
            ),
            ATLAS_IDENTITY,
        )
        self.assertEqual(
            robot_config_save.config_path_of(
                os.path.join(
                    "/a/robot_models/b/..", "robot_models", "x", "config", "f.textproto"
                )
            ),
            "robot_models/x/config/f.textproto",
        )
        # A relative path that starts at robot_models/ is the identity already.
        self.assertEqual(
            robot_config_save.config_path_of(ATLAS_IDENTITY), ATLAS_IDENTITY
        )

    def test_a_file_outside_robot_models_is_its_normalized_path(self):
        # As configFileIdentity() (ConfigFiles.h) normalizes it: lexically, without the working directory.
        self.assertEqual(
            robot_config_save.config_path_of(
                "/tmp/x/./config/../config/task.textproto"
            ),
            "/tmp/x/config/task.textproto",
        )
        self.assertEqual(
            robot_config_save.config_path_of("config/mpc/task.textproto"),
            "config/mpc/task.textproto",
        )
        # A last component named robot_models has nothing below it: no identity starts there.
        self.assertEqual(
            robot_config_save.config_path_of("/x/robot_models"), "/x/robot_models"
        )

    def test_the_save_carries_kind_robot_identity_fingerprint_text_and_sequence(self):
        for kind, message_class in (
            (robot_config_save.KIND_TASK, task_file_pb2.TaskFile),
            (robot_config_save.KIND_REFERENCE, reference_file_pb2.ReferenceFile),
            (
                robot_config_save.KIND_JOINT_PD_GAINS,
                joint_pd_gains_file_pb2.JointPdGainsFile,
            ),
        ):
            with self.subTest(kind=kind):
                save = robot_config_save.config_file_save(
                    kind, os.path.join("/c", ATLAS_IDENTITY), "x: 1\n", "atlas", 5
                )
                self.assertEqual(save.kind, kind)
                self.assertEqual(save.robot_name, "atlas")
                self.assertEqual(save.config_path, ATLAS_IDENTITY)
                # The fingerprint of the file's schema, which the robot compares with its own build's.
                self.assertEqual(
                    save.schema_fingerprint,
                    nproto_schema.schema_fingerprint(message_class.DESCRIPTOR),
                )
                self.assertEqual(save.text, "x: 1\n")
                self.assertEqual(save.sequence, 5)

    def test_every_kind_the_robot_stores_is_known_and_no_other(self):
        # Every kind of the enum but the unspecified one, whatever the enum lists.
        self.assertEqual(
            {entry.kind for entry in robot_config_save.FILE_KINDS},
            set(config_file_kind_pb2.ConfigFileKind.values())
            - {config_file_kind_pb2.CONFIG_FILE_KIND_UNSPECIFIED},
        )
        self.assertEqual(
            len({entry.message_class for entry in robot_config_save.FILE_KINDS}),
            len(robot_config_save.FILE_KINDS),
            "two kinds share a schema",
        )
        with self.assertRaises(ValueError):
            robot_config_save.file_kind(0)
        self.assertEqual(
            robot_config_save.file_kind_of_message("humanoid_mpc_config.TaskFile"),
            robot_config_save.file_kind(robot_config_save.KIND_TASK),
        )
        self.assertIsNone(
            robot_config_save.file_kind_of_message(
                "humanoid_mpc_config.ContactPlanningFile"
            )
        )

    def test_the_robot_is_the_one_its_configurations_task_file_names(self):
        atlas = operator_test_support.ATLAS_CONFIG
        for kind, path in (
            (robot_config_save.KIND_TASK, operator_test_support.ATLAS_TASK_FILE),
            (
                robot_config_save.KIND_REFERENCE,
                operator_test_support.ATLAS_REFERENCE_FILE,
            ),
            (
                robot_config_save.KIND_JOINT_PD_GAINS,
                operator_test_support.ATLAS_PD_GAINS_FILE,
            ),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(robot_config_save.robot_name_of(kind, path), "atlas")
        with self.assertRaisesRegex(robot_config_save.RobotNameError, "task.textproto"):
            robot_config_save.robot_name_of(
                robot_config_save.KIND_REFERENCE, "/nowhere/config/command/x.textproto"
            )
        self.assertTrue(os.path.isdir(atlas))


class TestRobotConfigSaver(unittest.TestCase):
    def setUp(self):
        self.bus = operator_test_support.RecordingBus()
        self.statuses = operator_bus.ConfigSaveStatusMailbox()
        self.clock = FakeClock()
        self.saver = self._saver(self.bus)

    def _saver(
        self, bus: operator_test_support.RecordingBus
    ) -> robot_config_save.RobotConfigSaver:
        return robot_config_save.RobotConfigSaver(
            operator_bus.TopicPublisher.for_topic(bus, topics.OPERATOR_CONFIG_SAVE),
            self.statuses,
            clock=self.clock,
            sequence_clock=lambda: 1_000,
        )

    def _send(self, saver: robot_config_save.RobotConfigSaver | None = None) -> int:
        return (saver or self.saver).send(
            robot_config_save.KIND_TASK,
            os.path.join("/c", ATLAS_IDENTITY),
            "terrain_height: 0.0\n",
            "atlas",
        )

    def _state(self, sequence: int) -> SaveState:
        outcome = self.saver.outcome(sequence)
        assert outcome is not None, sequence
        return outcome.state

    def test_a_save_is_published_whole_with_a_rising_sequence(self):
        first, second = self._send(), self._send()
        self.assertEqual(first, 1_000)
        self.assertGreater(second, first)
        published = self.bus.messages_on(topics.OPERATOR_CONFIG_SAVE)
        self.assertEqual([message.sequence for message in published], [first, second])
        self.assertEqual(published[0].text, "terrain_height: 0.0\n")
        self.assertEqual(published[0].config_path, ATLAS_IDENTITY)
        self.assertEqual(self._state(first), SaveState.SAVING)

    def test_each_answer_is_matched_by_sequence(self):
        for result, state in (
            (_Status.RESULT_SAVED, SaveState.SAVED),
            (_Status.RESULT_UNCHANGED, SaveState.UNCHANGED),
            (_Status.RESULT_NOT_STORED, SaveState.NOT_STORED),
            (_Status.RESULT_REFUSED, SaveState.REFUSED),
            (_Status.RESULT_FAILED, SaveState.FAILED),
        ):
            with self.subTest(state=state):
                sequence = self._send()
                other = self._send()
                self.statuses.put(_status(sequence, result, "the message"))
                changed = self.saver.poll(self.clock())
                self.assertEqual([outcome.sequence for outcome in changed], [sequence])
                outcome = self.saver.outcome(sequence)
                assert outcome is not None
                self.assertEqual(
                    (outcome.state, outcome.message, outcome.stored_path),
                    (state, "the message", STORED),
                )
                self.assertEqual(self._state(other), SaveState.SAVING)

    def test_an_answer_to_another_senders_save_is_ignored(self):
        sequence = self._send()
        self.statuses.put(_status(sequence + 1_000_000, _Status.RESULT_REFUSED))
        self.assertEqual(self.saver.poll(self.clock()), [])
        self.assertEqual(self._state(sequence), SaveState.SAVING)

    def test_an_unanswered_save_is_unknown_after_the_timeout_and_a_late_answer_replaces_it(
        self,
    ):
        sequence = self._send()
        self.clock.sleep(robot_config_save.SAVE_TIMEOUT_SECONDS - 0.1)
        self.assertEqual(self.saver.poll(self.clock()), [])
        self.clock.sleep(0.2)
        changed = self.saver.poll(self.clock())
        self.assertEqual([outcome.state for outcome in changed], [SaveState.TIMED_OUT])
        timed_out = self.saver.outcome(sequence)
        assert timed_out is not None
        self.assertTrue(timed_out.waiting)
        self.assertTrue(timed_out.problem)
        text = robot_config_save.describe(timed_out)
        self.assertIn("did not answer in 5 s", text)
        self.assertIn("unknown", text)
        self.assertIn("Save again", text)
        self.clock.sleep(30.0)
        self.statuses.put(_status(sequence, _Status.RESULT_SAVED))
        self.saver.poll(self.clock())
        self.assertEqual(self._state(sequence), SaveState.SAVED)

    def test_a_save_the_bus_drops_is_not_sent_until_a_repeat_goes_out(self):
        dropping = DroppingBus(drops=1)
        saver = self._saver(dropping)
        sequence = self._send(saver)
        outcome = saver.outcome(sequence)
        assert outcome is not None
        self.assertEqual(outcome.state, SaveState.NOT_SENT)
        self.assertIn("not sent to the robot", robot_config_save.describe(outcome))
        self.assertEqual(dropping.published, [])
        self.assertTrue(saver.resend(sequence))
        repeated = saver.outcome(sequence)
        assert repeated is not None
        self.assertEqual(repeated.state, SaveState.SAVING)
        self.assertFalse(saver.resend(sequence + 1))

    def test_a_result_this_build_does_not_know_is_a_failure_that_says_so(self):
        sequence = self._send()
        self.statuses.put(_status(sequence, 99, "new"))
        self.saver.poll(self.clock())
        outcome = self.saver.outcome(sequence)
        assert outcome is not None
        self.assertEqual(outcome.state, SaveState.FAILED)
        self.assertIn("99", outcome.message)

    def test_the_saver_needs_a_positive_timeout(self):
        for timeout in (0.0, -1.0, float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                robot_config_save.RobotConfigSaver(
                    operator_bus.TopicPublisher.for_topic(
                        self.bus, topics.OPERATOR_CONFIG_SAVE
                    ),
                    self.statuses,
                    timeout=timeout,
                )


class TestTheStatusLines(unittest.TestCase):
    def _line(self, state: SaveState, message: str = "", stored_path: str = "") -> str:
        return robot_config_save.describe(
            robot_config_save.SaveOutcome(
                sequence=1,
                kind=robot_config_save.KIND_TASK,
                config_path=ATLAS_IDENTITY,
                state=state,
                message=message,
                stored_path=stored_path,
            )
        )

    def test_every_state_has_its_line(self):
        self.assertEqual(
            self._line(SaveState.SAVING), "Saved on the laptop · saving on the robot…"
        )
        self.assertEqual(
            self._line(SaveState.SAVED, stored_path=STORED),
            f"Saved on the laptop and on the robot ({STORED})",
        )
        self.assertIn(
            "robot unchanged", self._line(SaveState.UNCHANGED, stored_path=STORED)
        )
        self.assertEqual(
            self._line(SaveState.REFUSED, "robot_name 'g1' is not 'atlas'"),
            "Saved on the laptop · the robot refused it: robot_name 'g1' is not 'atlas'",
        )
        self.assertIn(
            "could not store it: disk full", self._line(SaveState.FAILED, "disk full")
        )
        self.assertEqual(
            self._line(SaveState.NOT_STORED, stored_path=STORED),
            f"Saved on the laptop · the robot has no store: it reads {STORED} in place; this Save did not change it",
        )
        self.assertEqual(
            self._line(SaveState.NOT_SENT, "the bus dropped it"),
            "Saved on the laptop · not sent to the robot: the bus dropped it",
        )
        for state in SaveState:
            with self.subTest(state=state):
                self.assertTrue(self._line(state).startswith("Saved on the laptop"))

    def test_not_stored_is_in_sync_only_when_the_robot_reads_the_laptops_file(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        laptop = os.path.join(directory, "task.textproto")
        with open(laptop, "w", encoding="utf-8") as file:
            file.write("x")
        other = os.path.join(directory, "other.textproto")
        with open(other, "w", encoding="utf-8") as file:
            file.write("x")
        sent = robot_config_save.SaveOutcome(
            sequence=7,
            kind=robot_config_save.KIND_TASK,
            config_path=ATLAS_IDENTITY,
            state=SaveState.SAVING,
            path=laptop,
        )
        for read, in_sync in (
            (laptop, True),
            (os.path.join(directory, ".", "task.textproto"), True),
            (other, False),
            ("/robot/only/task.textproto", False),
            ("", False),
        ):
            with self.subTest(read=read):
                answered = robot_config_save._answered(
                    sent, _status(7, _Status.RESULT_NOT_STORED, stored=read)
                )
                self.assertEqual(answered.state, SaveState.NOT_STORED)
                self.assertEqual(answered.reads_laptop_file, in_sync)
                self.assertEqual(answered.problem, not in_sync)
                line = robot_config_save.describe(answered)
                self.assertEqual("did not change it" in line, not in_sync, line)
                self.assertEqual(
                    push_robot_config.exit_status(answered),
                    (
                        push_robot_config.EXIT_IN_SYNC
                        if in_sync
                        else push_robot_config.EXIT_NOT_STORED
                    ),
                )


class TestSaveTracker(unittest.TestCase):
    def test_without_a_saver_nothing_is_sent_and_the_line_says_so(self):
        tracker = robot_config_save.SaveTracker(None)
        outcome = tracker.send(
            robot_config_save.KIND_TASK, operator_test_support.ATLAS_TASK_FILE, "x"
        )
        self.assertEqual(outcome.state, SaveState.NOT_SENT)
        self.assertIn("not connected", outcome.message)
        self.assertIsNone(tracker.poll())

    def test_a_file_whose_robot_cannot_be_read_is_not_sent(self):
        bus = operator_test_support.RecordingBus()
        saver = robot_config_save.RobotConfigSaver(
            operator_bus.TopicPublisher.for_topic(bus, topics.OPERATOR_CONFIG_SAVE),
            operator_bus.ConfigSaveStatusMailbox(),
        )
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        lonely = os.path.join(directory, "controller", "joint_pd_gains.textproto")
        outcome = robot_config_save.SaveTracker(saver).send(
            robot_config_save.KIND_JOINT_PD_GAINS, lonely, "x"
        )
        self.assertEqual(outcome.state, SaveState.NOT_SENT)
        self.assertIn("task file", outcome.message)
        self.assertEqual(bus.published, [])

    def test_the_tracker_reports_each_change_of_its_last_save_once(self):
        statuses = operator_bus.ConfigSaveStatusMailbox()
        clock = FakeClock()
        bus = operator_test_support.RecordingBus()
        saver = robot_config_save.RobotConfigSaver(
            operator_bus.TopicPublisher.for_topic(bus, topics.OPERATOR_CONFIG_SAVE),
            statuses,
            clock=clock,
        )
        tracker = robot_config_save.SaveTracker(saver)
        sent = tracker.send(
            robot_config_save.KIND_TASK, operator_test_support.ATLAS_TASK_FILE, "x"
        )
        self.assertEqual(sent.state, SaveState.SAVING)
        (message,) = bus.messages_on(topics.OPERATOR_CONFIG_SAVE)
        self.assertEqual(message.robot_name, "atlas")
        self.assertIsNone(tracker.poll())
        statuses.put(_status(sent.sequence, _Status.RESULT_SAVED))
        answered = tracker.poll()
        assert answered is not None
        self.assertEqual(answered.state, SaveState.SAVED)
        self.assertFalse(answered.waiting)
        self.assertIsNone(tracker.poll())


class TestPushRobotConfig(unittest.TestCase):
    """push_robot_config repeats one save, under one sequence, until the robot answers it."""

    def _push(
        self,
        publisher: AnsweringPublisher,
        statuses: operator_bus.ConfigSaveStatusMailbox,
    ) -> robot_config_save.SaveOutcome:
        """push() of the Atlas task file through `publisher`, on a fake clock that moves only while push() sleeps."""
        clock = FakeClock()
        saver = robot_config_save.RobotConfigSaver(
            publisher,
            statuses,
            clock=clock,
            timeout=push_robot_config.ANSWER_TIMEOUT_SECONDS,
        )
        return push_robot_config.push(
            saver,
            robot_config_save.KIND_TASK,
            operator_test_support.ATLAS_TASK_FILE,
            "terrain_height: 0.0\n",
            "atlas",
            clock=clock,
            sleep=clock.sleep,
        )

    def test_it_repeats_one_sequence_until_answered(self):
        statuses = operator_bus.ConfigSaveStatusMailbox()
        publisher = AnsweringPublisher(statuses, answer_at=3)
        outcome = self._push(publisher, statuses)
        self.assertEqual(outcome.state, SaveState.SAVED)
        messages = publisher.messages
        self.assertEqual(len(messages), 3)
        # The repeats are the save itself: one sequence, the same bytes.
        self.assertEqual(len({message.SerializeToString() for message in messages}), 1)
        self.assertEqual(messages[0].sequence, outcome.sequence)

    def test_without_an_answer_it_gives_up_after_the_timeout(self):
        statuses = operator_bus.ConfigSaveStatusMailbox()
        publisher = AnsweringPublisher(statuses, answer_at=None)
        outcome = self._push(publisher, statuses)
        self.assertEqual(outcome.state, SaveState.TIMED_OUT)
        expected = int(
            push_robot_config.ANSWER_TIMEOUT_SECONDS
            / push_robot_config.RESEND_PERIOD_SECONDS
        )
        self.assertEqual(len(publisher.messages), expected)
        self.assertEqual(len({message.sequence for message in publisher.messages}), 1)

    def test_a_refusal_ends_it_at_once(self):
        statuses = operator_bus.ConfigSaveStatusMailbox()
        publisher = AnsweringPublisher(
            statuses, answer_at=1, result=_Status.RESULT_REFUSED
        )
        self.assertEqual(self._push(publisher, statuses).state, SaveState.REFUSED)
        self.assertEqual(len(publisher.messages), 1)

    def test_the_file_decides_the_kind_and_must_parse(self):
        entry, text = push_robot_config.read_file(
            operator_test_support.ATLAS_PD_GAINS_FILE
        )
        self.assertEqual(entry.kind, robot_config_save.KIND_JOINT_PD_GAINS)
        with open(
            operator_test_support.ATLAS_PD_GAINS_FILE, encoding="utf-8", newline=""
        ) as file:
            self.assertEqual(text, file.read())
        nproto_textproto.parse_textproto(
            text, joint_pd_gains_file_pb2.JointPdGainsFile(), "gains"
        )
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        planner = os.path.join(directory, "contact_planning.textproto")
        with open(planner, "w", encoding="utf-8") as file:
            file.write("# proto-message: humanoid_mpc_config.ContactPlanningFile\n")
        with self.assertRaisesRegex(
            push_robot_config.PushError, "names no file the robot stores"
        ):
            push_robot_config.read_file(planner)
        broken = os.path.join(directory, "task.textproto")
        with open(broken, "w", encoding="utf-8") as file:
            file.write(
                "# proto-message: humanoid_mpc_config.TaskFile\nterrainHeight: 0.1\n"
            )
        with self.assertRaisesRegex(push_robot_config.PushError, "task.textproto:2"):
            push_robot_config.read_file(broken)
        with self.assertRaises(push_robot_config.PushError):
            push_robot_config.read_file(os.path.join(directory, "missing.textproto"))

    def test_a_file_it_cannot_send_exits_two_without_a_bus(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        self.assertEqual(
            push_robot_config.main([os.path.join(directory, "missing.textproto")]),
            push_robot_config.EXIT_CANNOT_SEND,
        )

    def test_the_terminal_line_names_the_answer(self):
        def line(state: SaveState) -> str:
            return push_robot_config.describe(
                robot_config_save.SaveOutcome(
                    sequence=1,
                    kind=robot_config_save.KIND_TASK,
                    config_path=ATLAS_IDENTITY,
                    state=state,
                    message="why",
                    stored_path=STORED,
                ),
                timeout=10.0,
            )

        self.assertIn(STORED, line(SaveState.SAVED))
        self.assertIn("unchanged", line(SaveState.UNCHANGED))
        self.assertIn("refused it: why", line(SaveState.REFUSED))
        self.assertIn("did not answer in 10 s", line(SaveState.TIMED_OUT))


if __name__ == "__main__":
    unittest.main()
