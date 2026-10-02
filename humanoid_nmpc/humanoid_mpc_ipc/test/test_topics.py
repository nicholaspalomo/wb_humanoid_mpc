"""The Python topic names are the C++ ones, unique, and safe for ZeroMQ's prefix filter.

The C++ header include/humanoid_mpc_ipc/Topics.h is data of this test, so that a topic changed on one side only fails
here as well as in the LINT.IfChange review.
"""

import os
import re
import unittest
from typing import Dict

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


def cpp_topics() -> Dict[str, str]:
    with open(TOPICS_HEADER, encoding="utf-8") as header:
        return {
            python_name(name): value
            for name, value in CPP_CONSTANT.findall(header.read())
        }


def python_topics() -> Dict[str, str]:
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

    def test_no_topic_is_a_prefix_of_another(self) -> None:
        # ZeroMQ's SUB filter matches a prefix of the topic frame.
        for topic in topics.ALL_TOPICS:
            for other in topics.ALL_TOPICS:
                if topic != other:
                    self.assertFalse(other.startswith(topic), f"{topic} begins {other}")


if __name__ == "__main__":
    unittest.main()
