"""Every package of operator_requirements.txt loads from the @operator_deps hub with the hermetic Python.

The operator tools (the remote_control GUI, teleoperation, the Rerun bridge, tools/ipc) depend on these packages and
on Tk, which the hermetic interpreter bundles.
"""

import importlib
import importlib.metadata
import os
import re
import unittest
from typing import Dict, List, NamedTuple, Optional

# pygame greets on import unless told not to.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REQUIREMENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "operator_requirements.txt"
)

# The module a distribution is imported as, where the two names differ. A requirement added to the file without an
# entry here is imported under its own name.
IMPORT_NAMES: Dict[str, str] = {
    "protobuf": "google.protobuf",
    "pyyaml": "yaml",
    "pyzmq": "zmq",
    "rerun-sdk": "rerun",
}

_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==\s*([^\s;]+))?")


class Requirement(NamedTuple):
    name: str
    pinned_version: Optional[str]


def read_requirements(path: str) -> List[Requirement]:
    """The requirements of a pip requirements file: name and, for `name==version`, the version."""
    requirements = []
    with open(path, encoding="utf-8") as requirements_file:
        for line in requirements_file:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            match = _REQUIREMENT.match(line)
            if match is None:
                raise ValueError(f"{path}: cannot parse requirement {line!r}")
            name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
            requirements.append(Requirement(name, match.group(2)))
    return requirements


class ReadRequirementsTest(unittest.TestCase):

    def test_reads_names_and_pins_and_skips_comments(self) -> None:
        path = os.path.join(os.environ["TEST_TMPDIR"], "requirements.txt")
        with open(path, "w", encoding="utf-8") as requirements_file:
            requirements_file.write(
                "# comment\n\npyzmq\nprotobuf==7.36.0  # pinned\nnumpy>=2\nRerun_SDK==0.38.1\n"
            )
        self.assertEqual(
            read_requirements(path),
            [
                Requirement("pyzmq", None),
                Requirement("protobuf", "7.36.0"),
                Requirement("numpy", None),
                Requirement("rerun-sdk", "0.38.1"),
            ],
        )


class OperatorDepsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.requirements = read_requirements(REQUIREMENTS_FILE)
        self.assertTrue(self.requirements)

    def test_every_requirement_is_installed_at_its_pinned_version(self) -> None:
        for requirement in self.requirements:
            with self.subTest(requirement=requirement.name):
                version = importlib.metadata.version(requirement.name)
                if requirement.pinned_version is not None:
                    self.assertEqual(version, requirement.pinned_version)

    def test_every_requirement_imports(self) -> None:
        for requirement in self.requirements:
            module_name = IMPORT_NAMES.get(requirement.name, requirement.name)
            with self.subTest(requirement=requirement.name, module=module_name):
                importlib.import_module(module_name)

    def test_tkinter_is_bundled_with_the_interpreter(self) -> None:
        # Importing loads the Tcl/Tk libraries; no display is needed until a Tk root is created.
        import tkinter

        self.assertTrue(tkinter.TkVersion)


if __name__ == "__main__":
    unittest.main()
