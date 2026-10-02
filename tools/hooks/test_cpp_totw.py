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

"""Tests for cpp_totw.py: the Abseil Tips of the Week that clang-tidy and GCC do not cover."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import check_types
from tools.hooks import checks
from tools.hooks import cpp_totw as totw


def _count(
    check: check_types.CheckFunction, source: str, path: str = "src/a.cpp"
) -> int:
    return len(check(source + "\n", path))


class TotwTest(unittest.TestCase):
    def assert_cases(
        self,
        check: check_types.CheckFunction,
        flagged: list[str],
        clean: list[str],
        path: str = "src/a.cpp",
    ):
        """Asserts one finding for each source of `flagged` and none for each of `clean`."""
        for source in flagged:
            with self.subTest(flagged=source):
                self.assertEqual(_count(check, source, path), 1)
        for source in clean:
            with self.subTest(clean=source):
                self.assertEqual(_count(check, source, path), 0)

    def test_unordered_container(self):
        self.assert_cases(
            totw.check_unordered,
            ["std::unordered_map<int, int> m;", "std::unordered_set<int> s;"],
            [
                "absl::flat_hash_map<int, int> m;",
                "foo::std::unordered_map<int, int> m;",
            ],
        )

    def test_header_constant(self):
        self.assert_cases(
            totw.check_header_constant,
            [
                "constexpr double kTolerance = 1.0e-6;",
                "namespace a { const int kCount = 3; }",
                "static constexpr int kN = 2;",
            ],
            [
                "inline constexpr double kTolerance = 1.0e-6;",
                "extern const int kCount;",
                "constexpr int Square(int x) { return x * x; }",
                "class A { static constexpr int kN = 2; };",
                "void f() { constexpr int kLocal = 2; }",
            ],
            path="src/a.h",
        )
        self.assertEqual(
            _count(totw.check_header_constant, "constexpr int kN = 2;", "src/a.cpp"), 0
        )

    def test_string_constant(self):
        self.assert_cases(
            totw.check_string_constant,
            [
                'constexpr const char* kName = "a";',
                'inline constexpr const char* kName = "a";',
                'class A { static constexpr const char* kName = "a"; };',
                'const char* const kName = "a";',
            ],
            [
                'inline constexpr char kName[] = "a";',
                'void f() { const char* local = "a"; }',
                "const char* Name();",
            ],
        )

    def test_enum_switch_default(self):
        flagged = "switch (mode) {\n  case Mode::kWalk:\n    break;\n  default:\n    break;\n}"
        self.assertEqual(
            _count(totw.check_enum_switch_default, "void f() {\n" + flagged + "\n}"), 1
        )
        for clean in [
            "void f() { switch (mode) { case Mode::kWalk: break; case Mode::kStand: break; } }",
            "void f() { switch (c) { case 'a': break; default: break; } }",
            "void f() { switch (proto) { case Proto::WALK: break; default: break; } }",
        ]:
            with self.subTest(clean=clean):
                self.assertEqual(_count(totw.check_enum_switch_default, clean), 0)

    def test_at(self):
        self.assert_cases(
            totw.check_at,
            ["int x = v.at(3);", "int x = p->at(3);"],
            ["int x = v[3];", "int at = 3;"],
        )
        check = check_test_support.assert_registered(self, "totw-at", pending=True)
        self.assertFalse(check.applies_to("src/test/testA.cpp"))

    def test_raw_new(self):
        self.assert_cases(
            totw.check_raw_new,
            [
                "void f() { Foo* p = new Foo(); }",
                "void f() { ptr_.reset(new Foo(1)); }",
            ],
            [
                "Foo* clone() const override { return new Foo(*this); }",
                "auto p = absl::WrapUnique(new Foo());",
                "static const Foo& kFoo = *new Foo();",
                "void* operator new(size_t n);",
                "auto p = std::make_unique<Foo>();",
            ],
        )

    def test_printf(self):
        self.assert_cases(
            totw.check_printf,
            [
                'printf("%d", x);',
                'std::snprintf(b, n, "%d", x);',
                "out << std::setw(4) << x;",
                "out << std::fixed;",
            ],
            [
                'absl::PrintF("%d", x);',
                'auto s = absl::StrFormat("%d", x);',
                "logger.printf(x);",
            ],
        )

    def test_small_by_const_ref(self):
        self.assert_cases(
            totw.check_small_by_const_ref,
            [
                "void f(const double& x);",
                "void f(int a, const scalar_t& x);",
                "void f(const int64_t& x);",
            ],
            [
                "void f(double x);",
                "void f(const SCALAR_T& x);",
                "void f(const vector_t& x);",
                "void f() { for (const double& x : xs) {} }",
                "void f() { const double& r = x; }",
            ],
        )

    def test_view_param(self):
        self.assert_cases(
            totw.check_view_param,
            [
                "void f(const absl::string_view& s);",
                "void f(const absl::Span<const int>& s);",
                "void f(const absl::Duration& d);",
            ],
            ["void f(absl::string_view s);", "void f(absl::Span<const int> s);"],
        )

    def test_optional_ref_param(self):
        self.assert_cases(
            totw.check_optional_ref_param,
            ["void f(const std::optional<Foo>& foo);"],
            [
                "void f(std::optional<int> x);",
                "void f() { for (const std::optional<int>& x : xs) {} }",
            ],
        )

    def test_smart_ptr_ref_param(self):
        self.assert_cases(
            totw.check_smart_ptr_ref_param,
            [
                "void f(const std::shared_ptr<Foo>& foo);",
                "void f(int a, const std::unique_ptr<Foo>& foo);",
            ],
            [
                "void f(std::shared_ptr<Foo> foo);",
                "void f() { for (const std::unique_ptr<Foo>& foo : foos) {} }",
            ],
        )

    def test_brace_literal_init(self):
        self.assert_cases(
            totw.check_brace_literal_init,
            [
                "bool done_{false};",
                "double x{-1.0};",
                'std::string mode_{"WB_MPC"};',
                "size_t n{0};",
            ],
            ["bool done_ = false;", "vector_t x{3};", "Foo foo{1};", "int x{y};"],
        )
        self.assertEqual(
            totw.fix_brace_literal_init(
                'bool done_{false};\nstd::string mode_{"WB_MPC"};\n', "a.h"
            ),
            'bool done_ = false;\nstd::string mode_ = "WB_MPC";\n',
        )

    def test_namespace_name(self):
        self.assert_cases(
            totw.check_namespace_name,
            [
                "namespace robot { namespace testing {} }",
                "namespace robot::model::testing {}",
                "namespace a { namespace util {} }",
            ],
            [
                "namespace testing {}",
                "namespace robot { namespace internal {} }",
                "namespace fs = std::filesystem;",
            ],
        )

    def test_unscoped_enum(self):
        self.assert_cases(
            totw.check_unscoped_enum,
            ["enum Index : int { kA };", "enum { kB };"],
            [
                "enum class Mode { kWalk };",
                "enum struct Mode { kWalk };",
                "void f(enum Mode m);",
            ],
        )

    def test_view_member(self):
        self.assert_cases(
            totw.check_view_member,
            [
                "struct A { absl::string_view name; };",
                "class A { absl::Span<const int> values_; };",
            ],
            [
                "struct A { std::string name; };",
                'struct A { static constexpr absl::string_view kName = "a"; };',
                "struct A { absl::string_view name() const; };",
                "void f() { absl::string_view local; }",
            ],
        )

    def test_friend_test(self):
        self.assert_cases(
            totw.check_friend_test,
            [
                "class A { FRIEND_TEST(ATest, Works); };",
                "class A { friend class ATest; };",
            ],
            ["class A { friend class Builder; };"],
        )

    def test_std_specialization(self):
        self.assert_cases(
            totw.check_std_specialization,
            ["namespace std { void f(); }", "template <> struct hash<Foo> {};"],
            ["namespace stdx {}", "std::hash<int> h;"],
        )

    def test_flag_location(self):
        self.assertEqual(
            _count(
                totw.check_flag_location,
                'ABSL_FLAG(int, x, 1, "x");',
                "src/Library.cpp",
            ),
            1,
        )
        self.assertEqual(
            _count(
                totw.check_flag_location,
                'ABSL_FLAG(int, x, 1, "x");',
                "src/RobotMain.cpp",
            ),
            0,
        )
        self.assertEqual(
            _count(
                totw.check_flag_location,
                'ABSL_FLAG(int, x, 1, "x");',
                "src/RobotAppFlags.h",
            ),
            0,
        )
        self.assertEqual(
            _count(
                totw.check_flag_location,
                'ABSL_FLAG(int, x, 1, "x");',
                "src/test/testA.cpp",
            ),
            0,
        )
        self.assertEqual(
            _count(totw.check_flag_location, "ABSL_DECLARE_FLAG(int, x);", "src/A.cpp"),
            1,
        )
        self.assertEqual(
            _count(
                totw.check_flag_location, "ABSL_DECLARE_FLAG(int, x);", "src/AppFlags.h"
            ),
            0,
        )

    def test_size_minus(self):
        self.assert_cases(
            totw.check_size_minus,
            [
                "for (size_t i = 0; i < v.size() - 1; ++i) {}",
                "if (i <= values.size() - 2) {}",
            ],
            [
                "for (size_t i = 0; i + 1 < v.size(); ++i) {}",
                "auto s = static_cast<scalar_t>(values.size() - 1);",
            ],
        )

    def test_c_str_stored(self):
        self.assert_cases(
            totw.check_c_str_stored,
            [
                'const char* name = std::string("a").c_str();',
                "const char* name = Name().c_str();",
            ],
            ["const char* name = name_.c_str();", "f(Name().c_str());"],
        )

    def test_reader_lock(self):
        self.assert_cases(
            totw.check_reader_lock,
            [
                "std::shared_mutex mutex_;",
                "std::shared_lock<Mutex> lock;",
                "absl::ReaderMutexLock lock(&m);",
            ],
            ["absl::Mutex mutex_;", "std::mutex mutex_;"],
        )


class RegistryTest(unittest.TestCase):
    def test_every_check_behaves(self):
        for name, flagged, path in [
            (
                "totw-unordered-container",
                "std::unordered_map<int, int> m;\n",
                "src/a.cpp",
            ),
            ("totw-header-constant", "constexpr int kN = 2;\n", "src/a.h"),
            (
                "totw-string-constant",
                'constexpr const char* kName = "a";\n',
                "src/a.cpp",
            ),
            ("totw-at", "int x = v.at(3);\n", "src/a.cpp"),
            ("totw-raw-new", "Foo* p = new Foo();\n", "src/a.cpp"),
            ("totw-printf", 'int n = printf("%d", x);\n', "src/a.cpp"),
            ("totw-small-by-const-ref", "void f(const double& x);\n", "src/a.cpp"),
            ("totw-view-param", "void f(const absl::string_view& s);\n", "src/a.cpp"),
            (
                "totw-optional-ref-param",
                "void f(const std::optional<Foo>& foo);\n",
                "src/a.cpp",
            ),
            (
                "totw-smart-ptr-ref-param",
                "void f(const std::shared_ptr<Foo>& foo);\n",
                "src/a.cpp",
            ),
            ("totw-brace-literal-init", "bool done_{false};\n", "src/a.h"),
            ("totw-namespace-name", "namespace robot::testing {}\n", "src/a.cpp"),
            ("totw-unscoped-enum", "enum Index : int { kA };\n", "src/a.cpp"),
            (
                "totw-view-member",
                "struct A {\n  absl::string_view name;\n};\n",
                "src/a.h",
            ),
            (
                "totw-friend-test",
                "class A {\n  FRIEND_TEST(ATest, Works);\n};\n",
                "src/a.h",
            ),
            ("totw-std-specialization", "namespace std {\n}\n", "src/a.cpp"),
            ("totw-flag-location", 'ABSL_FLAG(int, x, 1, "x");\n', "src/Library.cpp"),
            ("totw-size-minus", "bool b = i < v.size() - 1;\n", "src/a.cpp"),
            ("totw-c-str-stored", "const char* name = Name().c_str();\n", "src/a.cpp"),
            ("totw-reader-lock", "std::shared_mutex mutex_;\n", "src/a.cpp"),
        ]:
            with self.subTest(check=name):
                check_test_support.assert_check_behaves(
                    self, name, flagged, path, pending=name in totw_pending()
                )

    def test_the_enum_switch_default_check_behaves(self):
        check_test_support.assert_check_behaves(
            self,
            "totw-enum-switch-default",
            "void f() {\n  switch (mode) {\n    case Mode::kWalk:\n      break;\n    default:\n      break;\n  }\n}\n",
            "src/a.cpp",
            pending=True,
        )


def totw_pending() -> frozenset[str]:
    """The totw-* checks that are still PENDING."""
    return frozenset(name for name in checks.PENDING if name.startswith("totw-"))


if __name__ == "__main__":
    unittest.main()
