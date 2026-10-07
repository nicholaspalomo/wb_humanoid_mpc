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

"""The Python topic names are the C++ ones, unique, and safe for ZeroMQ's prefix filter.

The C++ header include/humanoid_mpc_ipc/Topics.h is data of this test, so that a topic changed on one side only fails
here as well as in the LINT.IfChange review.
"""

import os
import re
import unittest

from humanoid_mpc_ipc import topics

# The header is data of this test, next to its directory in the runfiles tree.
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS_HEADER = os.path.join(PACKAGE_DIR, "include", "humanoid_mpc_ipc", "Topics.h")

CPP_CONSTANT = re.compile(
    r'inline constexpr absl::string_view (k[A-Za-z0-9]+) = "([^"]*)";'
)
TOPIC = re.compile(r"^(robot|mpc|viz|operator)/[a-z0-9_]+$")


def python_name(cpp_name: str) -> str:
    """kRobotMpcObservation -> ROBOT_MPC_OBSERVATION."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", cpp_name[1:]).upper()


def cpp_topics() -> dict[str, str]:
    with open(TOPICS_HEADER, encoding="utf-8") as header:
        return {
            python_name(name): value
            for name, value in CPP_CONSTANT.findall(header.read())
        }


def python_topics() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(topics).items()
        if name.isupper() and isinstance(value, str)
    }


class TopicsTest(unittest.TestCase):
    def test_every_cpp_topic_has_an_equal_python_twin(self) -> None:
        expected = cpp_topics()
        self.assertTrue(expected, "no topic constants found in Topics.h")
        self.assertEqual(python_topics(), expected)

    def test_all_topics_lists_every_constant_once(self) -> None:
        self.assertEqual(sorted(topics.ALL_TOPICS), sorted(python_topics().values()))
        self.assertEqual(len(set(topics.ALL_TOPICS)), len(topics.ALL_TOPICS))

    def test_every_topic_is_a_publisher_prefix_and_a_snake_case_name(self) -> None:
        for topic in topics.ALL_TOPICS:
            self.assertRegex(topic, TOPIC)

    def test_the_save_and_its_status_are_listed_once_each(self) -> None:
        self.assertEqual(topics.OPERATOR_CONFIG_SAVE, "operator/config_save")
        self.assertEqual(topics.ROBOT_CONFIG_SAVE_STATUS, "robot/config_save_status")
        for topic in (topics.OPERATOR_CONFIG_SAVE, topics.ROBOT_CONFIG_SAVE_STATUS):
            self.assertEqual(topics.ALL_TOPICS.count(topic), 1)

    def test_no_topic_is_a_prefix_of_another(self) -> None:
        # ZeroMQ's SUB filter matches a prefix of the topic frame.
        for topic in topics.ALL_TOPICS:
            for other in topics.ALL_TOPICS:
                if topic != other:
                    self.assertFalse(other.startswith(topic), f"{topic} begins {other}")


if __name__ == "__main__":
    unittest.main()
