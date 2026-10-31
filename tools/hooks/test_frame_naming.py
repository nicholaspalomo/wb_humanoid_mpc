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

"""Tests for frame_naming.py: frame-dependent quantities are named by their frames."""

import re
import unittest

from tools.hooks import check_test_support
from tools.hooks import frame_naming
from tools.hooks import lint_code

_CPP = "humanoid_nmpc/humanoid_common_mpc/src/a.cpp"
_PY = "humanoid_nmpc/remote_control/remote_control/a.py"
_PROTO = "humanoid_nmpc/humanoid_mpc_msgs/a.proto"
_OCS2 = "lib/ocs2/pinocchio/centroidal_model/src/a.cpp"


def _names(source: str, path: str = _CPP) -> list[str]:
    """The identifiers the check flags in `source`, in order."""
    found = check_test_support.findings(frame_naming.NAME, source, path)
    return [f.message.split("`")[1] for f in found]


def _readme_column(label: str) -> list[str]:
    """The backticked first cells of the table rows in the README block `label`."""
    text = check_test_support.read_repository_file(frame_naming.README)
    start = text.index(f"<!-- LINT.IfChange({label}) -->")
    block = text[start : text.index("LINT.ThenChange", start)]
    return re.findall(r"^\| `([a-z0-9]+)` \|", block, re.MULTILINE)


class GrammarTest(unittest.TestCase):
    def test_well_formed_names_have_no_problem(self):
        for name in (
            "pos_lfoot_rt_world_ewrt_base",
            "linvel_base_rt_world_ewrt_base",
            "angvel_base_rt_world_ewrt_base_z",
            "quat_base_rt_world",
            "quat_base_rt_world_xyzw",
            "eulerzyx_base_rt_world_pitch",
            "wrench_lfoot_ewrt_world",
            "force_contacts_ewrt_world",
            "normal_terrain_ewrt_world",
            "gravity_ewrt_world",
            "des_pos_feet_rt_world_ewrt_world_traj",
            "ref_mnorm_cmom_com_rt_world_ewrt_world",
            "swing_meas_linvel_foot_rt_world_ewrt_world",
            "pos_base_rt_world_ewrt_world_",
            "posBaseRtWorldEwrtWorld_",
            "nomPosBaseRtComEwrtBase",
            "getPosBaseRtWorldEwrtWorld",
            "setLinvelBaseRtWorldEwrtBase",
            "getLinearApproximationPosEesRtWorldEwrtWorld",
            "quatBaseRtWorldXyzw",
            "kPosBaseRtWorldEwrtWorld",
            "kNomPosBaseRtWorldEwrtWorldZ",
            "POS_BASE_RT_WORLD_EWRT_WORLD",
            "_quat_base_rt_world",
            "mutable_pos_base_rt_world_ewrt_world",
            "wrench_contacts_ewrt_world_size",
        ):
            with self.subTest(name=name):
                self.assertEqual(frame_naming.grammar_problems(name), [])

    def test_the_form_of_each_quantity(self):
        cases = {
            "pos_base_rt_world": "needs the frame it is expressed in",
            "quat_base_rt_world_ewrt_world": "expressed in none",
            "force_lfoot_rt_world_ewrt_world": "names no relative frame",
            "pos_base_ewrt_world": "needs the frame it is relative to",
            "wrench_ewrt_world": "needs the frame it acts at",
        }
        for name, message in cases.items():
            with self.subTest(name=name):
                self.assertIn(
                    message, " ".join(frame_naming.grammar_problems(name))
                )

    def test_unknown_words(self):
        cases = {
            "position_base_rt_world_ewrt_world": "not a quantity",
            "pos_pelvis_rt_world_ewrt_world": "frame1 `pelvis`",
            "pos_base_rt_odom_ewrt_world": "frame2 `odom`",
            "pos_base_rt_world_ewrt_feet": "frame3 `feet` is a plural",
            "pos_base_rt_base_ewrt_world": "both `base`",
            "pos_base_rt_world_ewrt_world_w": "not a suffix",
            "quat_base_rt_world_x": "not a suffix",
            "desired_pos_base_rt_world_ewrt_world": "as `des`",
            "des_ref_pos_base_rt_world_ewrt_world": "out of order",
            "pos_a_rt_world_ewrt_world_minus_pos_b_rt_world": "one quantity",
        }
        for name, message in cases.items():
            with self.subTest(name=name):
                self.assertIn(
                    message, " ".join(frame_naming.grammar_problems(name))
                )

    def test_rt_without_a_quantity_is_another_word(self):
        self.assertEqual(frame_naming.grammar_problems("rt"), [])
        self.assertEqual(frame_naming.grammar_problems("rt_trigger"), [])

    def test_split_words(self):
        self.assertEqual(
            frame_naming.split_words("getBasePosition"),
            ["get", "base", "position"],
        )
        words = ["nom", "pos", "base", "rt", "com", "ewrt", "base"]
        for name in (
            "nomPosBaseRtComEwrtBase",
            "nomPosBaseRtComEwrtBase_",
            "kNomPosBaseRtComEwrtBase",
            "nom_pos_base_rt_com_ewrt_base",
            "_nom_pos_base_rt_com_ewrt_base",
            "NOM_POS_BASE_RT_COM_EWRT_BASE",
            "nomPos_base_rt_com_ewrt_base",
        ):
            with self.subTest(name=name):
                self.assertEqual(frame_naming.split_words(name), words)


class LegacyTest(unittest.TestCase):
    def test_legacy_names_are_flagged_in_cpp(self):
        source = (
            "vector3_t basePosition = model.getBasePosition(state);\n"
            "const vector3_t angularVelocityLocal = w;\n"
            "scalar_t minNormalForce_ = 0.0;\n"
            "vector6_t kDefaultWrench;\n"
        )
        self.assertEqual(
            _names(source),
            [
                "basePosition",
                "getBasePosition",
                "angularVelocityLocal",
                "minNormalForce_",
                "kDefaultWrench",
            ],
        )

    def test_legacy_names_are_flagged_in_python_and_proto(self):
        self.assertEqual(
            _names("base_position = sample.base_position_world\n", _PY),
            ["base_position", "base_position_world"],
        )
        self.assertEqual(
            _names(
                "message A {\n  Vector3 base_linear_velocity_local = 5;\n}\n",
                _PROTO,
            ),
            ["base_linear_velocity_local"],
        )

    def test_the_lint_machinery_is_not_read(self):
        self.assertEqual(_names("position = 3\n", "tools/hooks/a.py"), [])
        self.assertEqual(
            _names("position = 3\n", "tools/locomotion_heuristics/a.py"),
            ["position"],
        )

    def test_lib_ocs2_is_read_and_its_thirdparty_is_not(self):
        self.assertEqual(_names("int basePosition;\n", _OCS2), ["basePosition"])
        self.assertEqual(
            _names("int basePosition;\n", "lib/ocs2/thirdparty/a.cpp"), []
        )

    def test_exempt_names_are_not_flagged(self):
        source = (
            "class ContactWrenchConeConstraint;\n"
            "quaternion_t q;\n"
            "size_t i = model.getBaseOrientationStartIndex();\n"
            "vector_t jointPositions; vector_t jointVelocities;\n"
            "matrix_t positionWeights;\n"
            "constexpr size_t kPositionDim = 3;\n"
            "enum class Mode { kZeroTorque, kWrench };\n"
            "Mode m = Mode::kZeroTorque;\n"
            "matrix3_t r = pinocchio::getFrameVelocity(model, data, id);\n"
            "matrix3_t q = transform.rotation(); vector4_t c = quaternionProduct(a, b);\n"
            "vector3_t e = eulerZyxFromQuaternion(q);\n"
            "matrix3_t g = quaternionRotationMatrix(q);\n"
            "matrix_t a = getCentroidalMomentumMatrix(interface);\n"
            "matrix6_t b = computeFloatingBaseCentroidalMomentumMatrixInverse(a);\n"
            "#define HUMANOID_BASE_POSITION 1\n"
            "// basePosition in a comment\n"
            'const char kName[] = "base_position";\n'
            "vector3_t posBaseRtWorldEwrtWorld;\n"
        )
        self.assertEqual(_names(source), [])

    def test_python_imports_modules_and_tests_are_not_flagged(self):
        source = (
            "from humanoid_mpc_msgs import walking_velocity_command_pb2\n"
            "import numpy as np\n"
            "r = scipy.spatial.transform.Rotation.from_quat(q)\n"
            "def test_base_position_is_read(self):\n"
            "    pass\n"
            "x = np.rotation\n"
            "s = 'base_position'  # base_position\n"
        )
        self.assertEqual(_names(source, _PY), [])

    def test_one_finding_per_name_and_line(self):
        self.assertEqual(
            _names("basePosition = basePosition + basePosition;\n"),
            ["basePosition"],
        )

    def test_a_malformed_grammar_name_is_flagged_once(self):
        found = check_test_support.findings(
            frame_naming.NAME, "int posBaseRtWorld;\n", _CPP
        )
        self.assertEqual(len(found), 1)
        self.assertIn(
            "does not follow the frame-naming grammar", found[0].message
        )

    def test_the_check_behaves(self):
        for path, flagged, clean in (
            (
                _CPP,
                "vector3_t basePosition;\n",
                "vector3_t posBaseRtWorldEwrtWorld;\n",
            ),
            (_PY, "base_position = 1\n", "pos_base_rt_world_ewrt_world = 1\n"),
        ):
            with self.subTest(path=path):
                check_test_support.assert_check_behaves(
                    self, frame_naming.NAME, flagged, path, clean=clean
                )


class StyleTest(unittest.TestCase):
    def test_style_of(self):
        cases = {
            "posBaseRtWorldEwrtWorld": frame_naming.CAMEL,
            "posBaseRtWorldEwrtWorld_": frame_naming.CAMEL,
            "kPosBaseRtWorldEwrtWorld": frame_naming.CONSTANT,
            "pos_base_rt_world_ewrt_world": frame_naming.SNAKE,
            "_pos_base_rt_world_ewrt_world": frame_naming.SNAKE,
            "POS_BASE_RT_WORLD_EWRT_WORLD": frame_naming.UPPER_SNAKE,
            "getPos_base_rt_world_ewrt_world": frame_naming.MIXED,
        }
        for name, style in cases.items():
            with self.subTest(name=name):
                self.assertEqual(frame_naming.style_of(name), style)

    def test_spell_keeps_the_words_and_the_underscores(self):
        self.assertEqual(
            frame_naming.spell(
                "getPos_ees_rt_world_ewrt_world", frame_naming.CAMEL
            ),
            "getPosEesRtWorldEwrtWorld",
        )
        self.assertEqual(
            frame_naming.spell(
                "nom_pos_base_rt_com_ewrt_base", frame_naming.CAMEL
            ),
            "nomPosBaseRtComEwrtBase",
        )
        self.assertEqual(
            frame_naming.spell("quatBaseRtWorld_", frame_naming.SNAKE),
            "quat_base_rt_world_",
        )
        self.assertEqual(
            frame_naming.spell(
                "nomPosBaseRtComEwrtBase", frame_naming.CONSTANT
            ),
            "kNomPosBaseRtComEwrtBase",
        )
        self.assertEqual(
            frame_naming.spell(
                "_get_pos_base_rt_world", frame_naming.UPPER_SNAKE
            ),
            "_GET_POS_BASE_RT_WORLD",
        )

    def test_cpp_names_never_mix_the_styles(self):
        source = (
            "vector3_t v = model.getPos_base_rt_world_ewrt_world(state);\n"
            "vector3_t getPosBase_rt_world_ewrt_world();\n"
            "vector3_t nomPos_base_rt_com_ewrt_base_;\n"
            "vector3_t posBaseRtWorldEwrtWorld = getPosBaseRtWorldEwrtWorld(state);\n"
            "constexpr double kNomPosBaseRtComEwrtBaseZ = 1.0;\n"
            "quaternion_t quat_base_rt_world = robotState.getQuatBaseRtWorld();\n"
        )
        self.assertEqual(
            _names(source),
            [
                "getPos_base_rt_world_ewrt_world",
                "getPosBase_rt_world_ewrt_world",
                "nomPos_base_rt_com_ewrt_base_",
            ],
        )
        found = check_test_support.findings(
            frame_naming.NAME,
            "vector3_t nomPos_base_rt_com_ewrt_base_;\n",
            _CPP,
        )
        self.assertIn("write `nomPosBaseRtComEwrtBase_`", found[0].message)

    def test_cpp_snake_case_names_and_members_are_accepted(self):
        source = (
            "struct Sample {\n"
            "  vector3_t pos_base_rt_world_ewrt_world;\n"
            "};\n"
            "void f() {\n"
            "  vector3_t posBaseRtWorldEwrtWorld ="
            " toEigen(sample.pos_base_rt_world_ewrt_world());\n"
            "  sample.mutable_pos_base_rt_world_ewrt_world()->set_x(1.0);\n"
            "  int n = msg->wrench_contacts_ewrt_world_size();\n"
            "  Sample s{.pos_base_rt_world_ewrt_world = posBaseRtWorldEwrtWorld};\n"
            "}\n"
        )
        self.assertEqual(_names(source), [])

    def test_python_and_proto_names_are_snake_case(self):
        self.assertEqual(
            _names(
                "posBaseRtWorldEwrtWorld = 1\n"
                "get_pos_base_rt_world_ewrt_world(x)\n"
                "self._quat_base_rt_world = q\n"
                "NOM_POS_BASE_RT_WORLD_EWRT_WORLD_Z = 1.0\n"
                "kNomPosBaseRtWorldEwrtWorldZ = 1.0\n"
                "getPos_base_rt_world_ewrt_world(x)\n",
                _PY,
            ),
            [
                "posBaseRtWorldEwrtWorld",
                "kNomPosBaseRtWorldEwrtWorldZ",
                "getPos_base_rt_world_ewrt_world",
            ],
        )
        found = check_test_support.findings(
            frame_naming.NAME, "kNomPosBaseRtWorldEwrtWorldZ = 1.0\n", _PY
        )
        self.assertIn(
            "write `NOM_POS_BASE_RT_WORLD_EWRT_WORLD_Z`", found[0].message
        )
        self.assertEqual(
            _names(
                "message A {\n"
                "  Vector3 pos_base_rt_world_ewrt_world = 3;\n"
                "  Vector3 linvelBaseRtWorldEwrtBase = 4;\n"
                "}\n",
                _PROTO,
            ),
            ["linvelBaseRtWorldEwrtBase"],
        )

    def test_lib_ocs2_is_held_to_the_styles(self):
        self.assertEqual(
            _names(
                "vector3_t getMnormCmom_com_rt_world_ewrt_world();\n", _OCS2
            ),
            ["getMnormCmom_com_rt_world_ewrt_world"],
        )
        self.assertEqual(
            _names("vector3_t nomPosBaseRtComEwrtBase;\n", _OCS2), []
        )


class ScopeTest(unittest.TestCase):
    def test_the_code_nproto_generates_is_not_read(self):
        # Its names are the .proto fields', which the check reads.
        source = "Vector3 position;\nvoid set_position(double position);\n"
        generated = "humanoid_nmpc/humanoid_mpc_msgs/_virtual_imports/msgs/pose"
        for suffix in (".nproto.h", ".nproto.pb.h", ".nproto.pb.cc"):
            self.assertEqual(_names(source, generated + suffix), [], suffix)
        self.assertEqual(
            _names(source, "humanoid_nmpc/humanoid_mpc_msgs/src/Pose.cpp"),
            ["position", "set_position", "position"],
        )


class OptInTest(unittest.TestCase):
    def test_make_lint_leaves_it_out_and_only_runs_it(self):
        selected, _ = lint_code._parse_only(None)
        self.assertNotIn(frame_naming.NAME, selected)
        selected, steps = lint_code._parse_only(frame_naming.NAME)
        self.assertEqual(selected, frozenset({frame_naming.NAME}))
        self.assertEqual(steps, frozenset({"token-checks"}))


class VocabularyTest(unittest.TestCase):
    def test_the_vocabularies_are_disjoint_single_words(self):
        groups = {
            "quantities": set(frame_naming.QUANTITIES),
            "frames": set(frame_naming.FRAMES)
            | set(frame_naming.PLURAL_FRAMES)
            | set(frame_naming.FREE_SUBJECTS),
            "qualifiers": set(frame_naming.QUALIFIERS),
            "markers": {frame_naming.RT, frame_naming.EWRT},
        }
        names = list(groups)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                with self.subTest(first=first, second=second):
                    self.assertEqual(groups[first] & groups[second], set())
        for words in groups.values():
            for word in words:
                self.assertRegex(word, r"^[a-z][a-z0-9]*$")

    def test_the_readme_lists_the_vocabulary(self):
        self.assertEqual(
            sorted(_readme_column("quantities")),
            sorted(frame_naming.QUANTITIES),
        )
        self.assertEqual(
            sorted(_readme_column("frames")),
            sorted(
                set(frame_naming.FRAMES)
                | set(frame_naming.PLURAL_FRAMES)
                | set(frame_naming.FREE_SUBJECTS)
            ),
        )
        self.assertEqual(
            _readme_column("qualifiers"), list(frame_naming.QUALIFIERS)
        )


if __name__ == "__main__":
    unittest.main()
