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

"""nproto_layout breaks the generator's lines as clang-format 18 does under the repository's .clang-format.

Each expected layout is clang-format's own output for the line, one case per rule of clang-format the generated code
meets: the break after `=`, the alignment after an open parenthesis against the break after it, nested calls, template
arguments, scope resolution, conditions, case labels, and declarations with BinPackParameters off.
"""

from collections.abc import Sequence
import unittest

import nproto_layout

STATEMENT = nproto_layout.Kind.STATEMENT
FUNCTION = nproto_layout.Kind.FUNCTION
MEMBER = nproto_layout.Kind.MEMBER


class LayoutTestCase(unittest.TestCase):
    """The assertion the layout tests share."""

    def assertWraps(
        self,
        line: str,
        expected: Sequence[str],
        kind: nproto_layout.Kind = STATEMENT,
    ) -> None:
        self.assertEqual(nproto_layout.wrap(line, kind), list(expected))


class StatementTest(LayoutTestCase):
    def test_a_line_that_fits_is_left_alone(self):
        line = "  status = FromProto(proto.kind(), &value->kind);"
        self.assertWraps(line, [line])

    def test_the_right_hand_side_moves_to_the_next_line_when_it_fits_there(
        self,
    ):
        self.assertWraps(
            "  status = ::nproto::internal::MapFromProto(proto.name_by_id(),"
            " &value->name_by_id);",
            [
                "  status =",
                (
                    "      ::nproto::internal::MapFromProto(proto.name_by_id(),"
                    " &value->name_by_id);"
                ),
            ],
        )

    def test_arguments_align_after_the_parenthesis_when_that_takes_no_more_lines(
        self,
    ):
        self.assertWraps(
            "  status ="
            " ::nproto::internal::ConvertOptionalFromProto(proto.has_ch(),"
            " proto.ch(), &value->ch);",
            [
                (
                    "  status ="
                    " ::nproto::internal::ConvertOptionalFromProto(proto.has_ch(),"
                ),
                (
                    "                                                       "
                    " proto.ch(), &value->ch);"
                ),
            ],
        )

    def test_arguments_move_after_the_parenthesis_when_that_takes_fewer_lines(
        self,
    ):
        self.assertWraps(
            "  status ="
            " ::nproto::internal::ConvertOptionalFromProto(proto.has_child(),"
            " proto.child(), &value->child);",
            [
                "  status = ::nproto::internal::ConvertOptionalFromProto(",
                "      proto.has_child(), proto.child(), &value->child);",
            ],
        )

    def test_a_return_statement_breaks_after_the_call(self):
        self.assertWraps(
            "      return"
            " ::nproto::internal::UnknownEnumValueError(static_cast<int32_t>(proto),"
            ' "pkg.Color");',
            [
                "      return ::nproto::internal::UnknownEnumValueError(",
                '          static_cast<int32_t>(proto), "pkg.Color");',
            ],
        )

    def test_a_nested_call_of_a_call_with_two_arguments_indents_from_its_own_start(
        self,
    ):
        self.assertWraps(
            "  ToProto(std::get<Kxaqhyoprhlhvhy::kKjanrudfuxjdxkxIndex>(value.umzgdpa_mntyyawoixzhsdkaaauramv),"
            " proto->mutable_szycidpyo());",
            [
                "  ToProto(std::get<Kxaqhyoprhlhvhy::kKjanrudfuxjdxkxIndex>(",
                "              value.umzgdpa_mntyyawoixzhsdkaaauramv),",
                "          proto->mutable_szycidpyo());",
            ],
        )

    def test_a_nested_call_of_a_call_with_one_argument_indents_from_the_statement(
        self,
    ):
        self.assertWraps(
            "  ToProto(std::get<Kxaqhyoprhlhvhy::kKjanrudfuxjdxkxIndex>(value.umzgdpa_mntyyawoixzhsdkaaauramv));",
            [
                "  ToProto(std::get<Kxaqhyoprhlhvhy::kKjanrudfuxjdxkxIndex>(",
                "      value.umzgdpa_mntyyawoixzhsdkaaauramv));",
            ],
        )

    def test_template_arguments_break_after_their_angle_bracket(self):
        self.assertWraps(
            "  proto->set_szycidpyo(static_cast<::srgqnp_lnlar::tztkotazhufrsfczrz::Kibvccaoayyihidztfljcffiqfvi>"
            "(*value.szycidpyo));",
            [
                "  proto->set_szycidpyo(",
                "      static_cast<",
                (
                    "         "
                    " ::srgqnp_lnlar::tztkotazhufrsfczrz::Kibvccaoayyihidztfljcffiqfvi>("
                ),
                "          *value.szycidpyo));",
            ],
        )

    def test_a_qualified_name_breaks_after_a_scope_resolution_as_a_last_resort(
        self,
    ):
        self.assertWraps(
            "    proto->set_szycidpyo(static_cast<::kppdajmknzgidixqgtn_aha::ebxfowqvnr_hu::"
            "Kwqohquamvszkvunbxjegbjccjj>(std::get<Kxaqhyoprhlhvhy::kKjanrudfuxjdxkxIndex>("
            "value.umzgdpa_mntyyawoixzhsdkaaauramv)));",
            [
                (
                    "    proto->set_szycidpyo(static_cast<::kppdajmknzgidixqgtn_aha::ebxfowqvnr_hu::"
                ),
                (
                    "                                        "
                    " Kwqohquamvszkvunbxjegbjccjj>("
                ),
                "        std::get<Kxaqhyoprhlhvhy::kKjanrudfuxjdxkxIndex>(",
                "            value.umzgdpa_mntyyawoixzhsdkaaauramv)));",
            ],
        )

    def test_an_assignment_breaks_after_its_operator(self):
        self.assertWraps(
            "  value->some_very_long_field_name_abcdefg ="
            " proto.some_very_long_field_name_abcdefg();",
            [
                "  value->some_very_long_field_name_abcdefg =",
                "      proto.some_very_long_field_name_abcdefg();",
            ],
        )

    def test_a_comparison_breaks_after_its_operator_and_aligns_its_operands(
        self,
    ):
        self.assertWraps(
            "  if (lhs.a_very_very_very_very_very_long_name !="
            " rhs.a_very_very_very_very_very_long_name) {",
            [
                "  if (lhs.a_very_very_very_very_very_long_name !=",
                "      rhs.a_very_very_very_very_very_long_name) {",
            ],
        )

    def test_an_else_if_condition_breaks_after_its_parenthesis(self):
        self.assertWraps(
            "  } else if"
            " (proto.has_some_very_long_field_name_abcdefghij_klmnopqrstu_vwxyz_ab()) {",
            [
                "  } else if (",
                (
                    "     "
                    " proto.has_some_very_long_field_name_abcdefghij_klmnopqrstu_vwxyz_ab()) {"
                ),
            ],
        )

    def test_a_case_label_breaks_inside_its_cast(self):
        self.assertWraps(
            "    case"
            " static_cast<int32_t>(Color::kDeepDeepDeepDeepDeepDeepDeepDeepDeepDeepDeepDeepRed):",
            [
                "    case static_cast<int32_t>(",
                (
                    "       "
                    " Color::kDeepDeepDeepDeepDeepDeepDeepDeepDeepDeepDeepDeepRed):"
                ),
            ],
        )

    def test_a_line_that_cannot_fit_keeps_its_cheapest_layout(self):
        name = "x" * 90
        lines = nproto_layout.wrap(f"  value->{name} = proto.{name}();")
        # With nothing that fits, clang-format even breaks before the member accesses.
        self.assertEqual(
            lines,
            [
                "  value",
                f"      ->{name} =",
                "      proto",
                f"          .{name}();",
            ],
        )
        self.assertFalse(nproto_layout.fits(lines))
        self.assertTrue(nproto_layout.fits(["x" * nproto_layout.COLUMN_LIMIT]))


class FunctionTest(LayoutTestCase):
    def test_parameters_align_after_the_parenthesis_one_per_line(self):
        self.assertWraps(
            "void ToProto(const ConfigFileSaveStatusResultVeryLongName& value, "
            "::humanoid_mpc_msgs::ConfigFileSave_Result* absl_nonnull proto) {",
            [
                (
                    "void ToProto(const ConfigFileSaveStatusResultVeryLongName&"
                    " value,"
                ),
                (
                    "             ::humanoid_mpc_msgs::ConfigFileSave_Result*"
                    " absl_nonnull proto) {"
                ),
            ],
            FUNCTION,
        )

    def test_parameters_move_after_the_parenthesis_when_they_do_not_align(self):
        self.assertWraps(
            "absl::Status FromProto(const"
            " ::humanoid_mpc_msgs::TargetContactPatch_Pose& proto,"
            " TargetContactPatch::Pose* absl_nonnull value);",
            [
                "absl::Status FromProto(",
                (
                    "    const ::humanoid_mpc_msgs::TargetContactPatch_Pose&"
                    " proto,"
                ),
                "    TargetContactPatch::Pose* absl_nonnull value);",
            ],
            FUNCTION,
        )

    def test_a_parameter_too_long_for_a_line_breaks_before_its_name(self):
        self.assertWraps(
            "absl::Status FromProto(const"
            " ::hmc::Abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz_abcdefg&"
            " proto, Config* absl_nonnull value) {",
            [
                "absl::Status FromProto(",
                (
                    "    const"
                    " ::hmc::Abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz_abcdefg&"
                ),
                "        proto,",
                "    Config* absl_nonnull value) {",
            ],
            FUNCTION,
        )

    def test_an_empty_body_closes_on_the_next_line_when_it_does_not_fit(self):
        self.assertWraps(
            "void ToProto(const Node::AVeryLongVeryLongVeryLongVeryLongName&"
            " /*value*/, ::pkg::Node_AVeryLongVeryLongVeryLongVeryLongName*"
            " absl_nonnull /*proto*/) {}",
            [
                "void ToProto(",
                (
                    "    const Node::AVeryLongVeryLongVeryLongVeryLongName&"
                    " /*value*/,"
                ),
                (
                    "    ::pkg::Node_AVeryLongVeryLongVeryLongVeryLongName*"
                    " absl_nonnull /*proto*/) {"
                ),
                "}",
            ],
            FUNCTION,
        )

    def test_an_operator_aligns_its_parameters(self):
        self.assertWraps(
            "inline bool operator==(const"
            " ContactPlanningFile::SharedConfig::Slack& lhs, const"
            " ContactPlanningFile::SharedConfig::Slack& rhs) {",
            [
                (
                    "inline bool operator==(const"
                    " ContactPlanningFile::SharedConfig::Slack& lhs,"
                ),
                (
                    "                       const"
                    " ContactPlanningFile::SharedConfig::Slack& rhs) {"
                ),
            ],
            FUNCTION,
        )


class MemberTest(LayoutTestCase):
    def test_template_arguments_of_a_declaration_go_one_per_line(self):
        self.assertWraps(
            "  std::variant<std::monostate, double, double, std::string,"
            " Scalars, Severity, Outer::Inner, std::string> shape;",
            [
                "  std::variant<std::monostate,",
                "               double,",
                "               double,",
                "               std::string,",
                "               Scalars,",
                "               Severity,",
                "               Outer::Inner,",
                "               std::string>",
                "      shape;",
            ],
            MEMBER,
        )

    def test_an_initializer_moves_to_the_next_line(self):
        self.assertWraps(
            "  double"
            " a_very_very_very_very_very_very_very_very_very_very_long_tolerance"
            " = kDefaultTolerance;",
            [
                (
                    "  double"
                    " a_very_very_very_very_very_very_very_very_very_very_long_tolerance ="
                ),
                "      kDefaultTolerance;",
            ],
            MEMBER,
        )

    def test_a_constant_too_long_before_its_initializer_breaks_before_its_name(
        self,
    ):
        self.assertWraps(
            "  static constexpr double"
            " kDefaultAVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongTolerance ="
            " 1.0e-06;",
            [
                "  static constexpr double",
                (
                    "     "
                    " kDefaultAVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongTolerance"
                    " = 1.0e-06;"
                ),
            ],
            MEMBER,
        )

    def test_a_template_type_breaks_one_argument_per_line_and_before_the_name(
        self,
    ):
        self.assertWraps(
            "  std::map<int32_t,"
            " Kxawsoqsaqqifealmuubdc::Kxxeparhlduxeakfqnvzqq::Kxpaswqpqzglsj>"
            " xwmvauisdgammfqqvxcasfxhizlhw_fpqrpwkak ="
            ' "xdrco_tfrzssdszjnlqlesjdcmsjcywtu";',
            [
                "  std::map<int32_t,",
                (
                    "          "
                    " Kxawsoqsaqqifealmuubdc::Kxxeparhlduxeakfqnvzqq::Kxpaswqpqzglsj>"
                ),
                "      xwmvauisdgammfqqvxcasfxhizlhw_fpqrpwkak =",
                '          "xdrco_tfrzssdszjnlqlesjdcmsjcywtu";',
            ],
            MEMBER,
        )


if __name__ == "__main__":
    unittest.main()
