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

"""Tests for cpp_google_style.py: the Google C++ Style Guide rules cpplint and clang-tidy do not cover."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import check_types
from tools.hooks import cpp_google_style as style


def _lines(
    check: check_types.CheckFunction, source: str, path: str = "src/a.cpp"
) -> list[int]:
    return [f.line for f in check(source, path)]


class FloatLiteralTest(unittest.TestCase):
    def test_flagged_and_fixed(self):
        for literal, fixed in [
            ("1e-6", "1.0e-6"),
            ("5e3", "5.0e3"),
            ("1.f", "1.0f"),
            (".5", "0.5"),
            ("1.", "1.0"),
            ("1.e5", "1.0e5"),
            ("2E-3f", "2.0E-3f"),
        ]:
            with self.subTest(literal=literal):
                source = f"double x = {literal};\n"
                self.assertEqual(_lines(style.check_float_literal, source), [1])
                self.assertEqual(
                    style.fix_float_literal(source, "a.cpp"), f"double x = {fixed};\n"
                )

    def test_accepted(self):
        source = "double a = 1.0e-6, b = 0.5, c = 1.0f; int d = 42, e = 0x1F, f = 1'000; auto g = 100ms; h = 1.5_m;\n"
        self.assertEqual(style.check_float_literal(source, "a.cpp"), [])
        self.assertEqual(
            style.check_float_literal('const char s[] = "1e-6"; // 1e-6\n', "a.cpp"), []
        )

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "float-literal",
            "double x = 1e-6;\n",
            "src/a.cpp",
            clean="double x = 1.0e-6;\n",
        )


class PostfixIncrementTest(unittest.TestCase):
    def test_unused_values_are_flagged_and_fixed(self):
        for source, fixed in [
            ("void g() {\n  i++;\n}\n", "void g() {\n  ++i;\n}\n"),
            (
                "void g() { for (int i = 0; i < n; i++) {} }\n",
                "void g() { for (int i = 0; i < n; ++i) {} }\n",
            ),
            (
                "void g() { for (;; it++, j--) {} }\n",
                "void g() { for (;; ++it, --j) {} }\n",
            ),
            ("void g() { if (x) count_++; }\n", "void g() { if (x) ++count_; }\n"),
            ("void g() { a.b[i]++; }\n", "void g() { ++a.b[i]; }\n"),
            ("void g() { (*it)++; }\n", "void g() { ++(*it); }\n"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(
                    len(style.check_postfix_increment(source, "a.cpp")),
                    source.count("++") + source.count("--"),
                )
                self.assertEqual(style.fix_postfix_increment(source, "a.cpp"), fixed)

    def test_used_values_are_not(self):
        for source in [
            "void g() { x = i++; }",
            "void g() { f(i++); }",
            "void g() { f(a, i++); }",
            "int g() { return i++; }",
            "void g() { while (n--) {} }",
            "void g() { v[i++] = 0; }",
            "void g() { ++i; }",
        ]:
            with self.subTest(source=source):
                self.assertEqual(style.check_postfix_increment(source, "a.cpp"), [])

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "postfix-increment",
            "void g() {\n  i++;\n}\n",
            "src/a.cpp",
            clean="void g() {\n  ++i;\n}\n",
        )


class StdIntegerTypeTest(unittest.TestCase):
    def test_flagged_and_fixed(self):
        source = "std::size_t n; std::int64_t m; ::std::uint8_t b; std::ptrdiff_t d;\n"
        self.assertEqual(len(style.check_std_integer_type(source, "a.cpp")), 4)
        self.assertEqual(
            style.fix_std_integer_type(source, "a.cpp"),
            "size_t n; int64_t m; uint8_t b; ptrdiff_t d;\n",
        )

    def test_other_names_are_fine(self):
        self.assertEqual(
            style.check_std_integer_type(
                "std::string s; size_t n; foo::std::size_t x;\n", "a.cpp"
            ),
            [],
        )

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "std-integer-type",
            "std::size_t n = 0;\n",
            "src/a.cpp",
            clean="size_t n = 0;\n",
        )


class ForbiddenConstructTest(unittest.TestCase):
    def test_flagged(self):
        for source in [
            "long double x;",
            "std::auto_ptr<int> p;",
            'int operator""_m(unsigned long long x);',
            "using namespace std::chrono_literals;",
            "auto t = 100ms;",
            "inline namespace v1 {}",
            "decltype(auto) f();",
            "void f() { co_return; }",
            "#include <ratio>",
            "#include <cfenv>",
            "using Ms = std::chrono::duration<double, std::milli>;",
            "void f() { char* p = (char*)alloca(8); }",
            "int x = ({ int y = 1; y; });",
            "int f() __attribute__((pure));",
            "bool b = __builtin_expect(x, 0);",
            "#pragma pack(1)",
            "thread_local std::string tls;",
            'void f() { strtok(s, ","); }',
            "int x = a ?: b;",
            "double d = 1.0L;",
            "double d = 1e3l;",
            "double d = 0x1.8p3L;",
        ]:
            with self.subTest(source=source):
                self.assertGreaterEqual(
                    len(style.check_forbidden_construct(source + "\n", "a.cpp")), 1
                )

    def test_accepted(self):
        for source in [
            "#pragma once",
            "#pragma GCC diagnostic push",
            "#include <filesystem>",
            "void f() { thread_local int counter = 0; }",
            "constinit thread_local int counter = 0;",
            "void f() { g({1, 2}); }",
            "void f() { EXPECT_NO_THROW({ g(); }); }",
            "std::chrono::milliseconds t(5);",
            "double x = 1.0;",
            "long count = 1L;",
            "int x = a ? b : c;",
            "auto x = c ? ::f() : g();",
        ]:
            with self.subTest(source=source):
                self.assertEqual(
                    style.check_forbidden_construct(source + "\n", "a.cpp"), []
                )

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self, "forbidden-construct", "long double x;\n", "src/a.cpp"
        )


class ExceptionsAndRttiTest(unittest.TestCase):
    def test_exceptions(self):
        source = "void f() {\n  try {\n    throw 1;\n  } catch (...) {\n  }\n}\n"
        self.assertEqual(_lines(style.check_exceptions, source), [2, 3, 4])
        self.assertEqual(
            style.check_exceptions(
                "void try_lock(); bool b = noexcept(f());\n", "a.cpp"
            ),
            [],
        )

    def test_rtti(self):
        self.assertEqual(
            len(
                style.check_rtti(
                    "auto* d = dynamic_cast<D*>(b); auto& t = typeid(x);\n", "a.cpp"
                )
            ),
            2,
        )

    def test_rtti_library_facilities(self):
        for source in [
            "void e() { std::dynamic_pointer_cast<Foo>(p); }\n",
            "void e() { dynamic_pointer_cast<Foo>(p); }\n",
            "void e() { std::any_cast<int>(a); }\n",
            "std::type_index key = std::type_index(k);\n",
        ]:
            with self.subTest(source=source):
                self.assertGreaterEqual(len(style.check_rtti(source, "a.cpp")), 1)
        for source in [
            "void e() { std::static_pointer_cast<Foo>(p); }\n",
            "void e() { static_cast<Foo*>(p); }\n",
            "// dynamic_pointer_cast in a comment\n",
        ]:
            with self.subTest(source=source):
                self.assertEqual(style.check_rtti(source, "a.cpp"), [])

    def test_tests_may_use_them(self):
        check = check_test_support.assert_registered(self, "exceptions")
        self.assertFalse(check.applies_to("src/test/testFoo.cpp"))
        self.assertFalse(
            check_test_support.assert_registered(self, "rtti").applies_to(
                "test/testFoo.cpp"
            )
        )

    def test_the_registry_runs_them(self):
        check_test_support.assert_check_behaves(
            self, "exceptions", "void f() {\n  throw 1;\n}\n", "src/a.cpp"
        )
        check_test_support.assert_check_behaves(
            self, "rtti", "auto* d = dynamic_cast<D*>(b);\n", "src/a.cpp"
        )


class DebugOnlyCheckTest(unittest.TestCase):
    def test_flagged(self):
        for source in [
            "void f(int* p) { assert(p != nullptr); }",
            "void f(int* p) { ABSL_DCHECK(p != nullptr); }",
            "void f(int n) { ABSL_DCHECK_GT(n, 0); }",
            "void f(int* p) { DCHECK(p); }",
            "void f(int n) { DCHECK_EQ(n, 1); }",
            "void f(int n) { ABSL_ASSERT(n > 0); }",
        ]:
            with self.subTest(source=source):
                self.assertEqual(
                    len(style.check_debug_only_check(source + "\n", "a.cpp")), 1
                )

    def test_accepted(self):
        for source in [
            'static_assert(sizeof(int) == 4, "int");',
            "void f(int* p) { ABSL_CHECK(p != nullptr); }",
            "void f(int n) { CHECK_EQ(n, 1); }",
            "void f(Checker& c) { c.assert(1); }",
            "// assert(x) in a comment",
        ]:
            with self.subTest(source=source):
                self.assertEqual(
                    style.check_debug_only_check(source + "\n", "a.cpp"), []
                )

    def test_tests_may_assert(self):
        check = check_test_support.assert_registered(self, "debug-only-check")
        self.assertFalse(check.applies_to("src/test/testFoo.cpp"))

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "debug-only-check",
            "void f(int n) {\n  assert(n > 0);\n}\n",
            "src/a.cpp",
        )


class CtadTest(unittest.TestCase):
    def test_flagged(self):
        for source in [
            "std::vector v = {1, 2};",
            "std::pair p(a, b);",
            "std::array a{1, 2, 3};",
            "std::lock_guard lock(mutex_);",
            "absl::flat_hash_map m = other;",
            "Eigen::Matrix m = n;",
        ]:
            with self.subTest(source=source):
                self.assertEqual(len(style.check_ctad(source + "\n", "a.cpp")), 1)

    def test_accepted(self):
        for source in [
            "std::vector<int> v = {1, 2};",
            "std::lock_guard<std::mutex> lock(mutex_);",
            "absl::Cleanup cleanup = [] {};",
            "return std::pair(a, b);",
            "using std::vector;",
        ]:
            with self.subTest(source=source):
                self.assertEqual(style.check_ctad(source + "\n", "a.cpp"), [])

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self, "ctad", "std::vector v = {1, 2};\n", "src/a.cpp"
        )


class MacroNamingTest(unittest.TestCase):
    def test_header_macros_need_the_prefix(self):
        self.assertEqual(
            _lines(style.check_macro_naming, "#define FOO 1\n", "src/a.h"), [1]
        )
        self.assertEqual(
            _lines(
                style.check_macro_naming,
                "#ifndef A_H_\n#define A_H_\n#endif\n",
                "src/a.h",
            ),
            [2],
        )
        for allowed in [
            "ROBOT_IPC_RETURN_IF_ERROR(x) x",
            "RETURN_IF_ERROR(x) x",
            "STATUS_MACROS_CONCAT(x, y) x##y",
        ]:
            with self.subTest(macro=allowed):
                self.assertEqual(
                    style.check_macro_naming(f"#define {allowed}\n", "src/a.h"), []
                )

    def test_include_guards(self):
        for source in [
            "#ifndef HUMANOID_NMPC_FOO_INCLUDE_FOO_FOO_H_\n#define HUMANOID_NMPC_FOO_INCLUDE_FOO_FOO_H_\nint x;\n#endif\n",
            "// A comment first.\n#if !defined(HUMANOID_FOO_H_)\n#define HUMANOID_FOO_H_\n#endif\n",
        ]:
            with self.subTest(source=source):
                findings = style.check_macro_naming(source, "src/a.h")
                self.assertEqual(len(findings), 1, findings)
                self.assertIn("#pragma once", findings[0].message)
        for source in [
            "#pragma once\n#define HUMANOID_FOO 1\n",
            "#pragma once\n#ifndef HUMANOID_FOO\n#define HUMANOID_FOO 1\n#endif\n",
        ]:
            with self.subTest(source=source):
                self.assertEqual(style.check_macro_naming(source, "src/a.h"), [])

    def test_cpp_macros_are_undefined(self):
        self.assertEqual(
            _lines(style.check_macro_naming, "#define LOCAL(x) x\nint a = LOCAL(1);\n"),
            [1],
        )
        self.assertEqual(
            style.check_macro_naming(
                "#define LOCAL(x) x\nint a = LOCAL(1);\n#undef LOCAL\n", "a.cpp"
            ),
            [],
        )

    def test_abseil_names(self):
        self.assertEqual(
            _lines(style.check_macro_naming, "#define CHECK(x) x\n#undef CHECK\n"), [1]
        )

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self, "macro-naming", "#define FOO 1\n", "src/a.h"
        )


class StaticStorageTest(unittest.TestCase):
    def test_flagged(self):
        for source in [
            'const std::string kName = "x";',
            "static std::vector<int> cache;",
            "namespace a { const absl::flat_hash_map<int, int> kMap = {}; }",
            'void f() { static const std::vector<std::string> names = {"a"}; }',
            "class A { static std::string name_; };",
            "std::string A::name_;",
            "const std::unique_ptr<int> kOwned = nullptr;",
            "static Eigen::MatrixXd gM;",
            "static Eigen::Matrix<double, Eigen::Dynamic, 1> gD;",
            "static ocs2::matrix_t gOm;",
            "static ocs2::vector_t gOv;",
            "static vector_t gU;",
            "static scalar_array_t gS;",
            "static std::array<std::string, 3> gA;",
            "static std::optional<std::string> gO;",
            "static std::variant<int, std::string> gV;",
            "static std::tuple<int, ocs2::vector_t> gT;",
            "static absl::Status gStatus;",
            "static std::thread gThread;",
        ]:
            with self.subTest(source=source):
                self.assertEqual(
                    len(style.check_static_storage(source + "\n", "a.cpp")), 1
                )

    def test_accepted(self):
        for source in [
            'inline constexpr char kName[] = "x";',
            "constexpr std::array<int, 2> kSizes = {1, 2};",
            "void f() { std::string local; }",
            "class A { std::string name_; };",
            'static const std::string& kName = *new std::string("x");',
            'static const absl::NoDestructor<std::string> kName("x");',
            "std::string Name();",
            "static std::string Describe(const Foo& foo);",
            "extern const std::string kName;",
            "using Names = std::vector<std::string>;",
            "static Eigen::Vector3d gV3;",
            "static Eigen::Matrix<double, 3, 3> gM3;",
            "static std::array<int, 3> gI;",
            "static std::optional<int> gOi;",
            "static std::pair<int, double> gP;",
            "class A { static absl::Status Create(const B& b); };",
            "class A { static absl::StatusOr<std::unique_ptr<A>> Create(); };",
            "static vector_t compute();",
            "absl::Status FromProto(::msgs::ControllerType proto, ControllerType* absl_nonnull value);",
        ]:
            with self.subTest(source=source):
                self.assertEqual(style.check_static_storage(source + "\n", "a.cpp"), [])

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "static-storage",
            'const std::string kName = "x";\n',
            "src/a.cpp",
        )


class ClassCommentTest(unittest.TestCase):
    def test_flagged(self):
        self.assertEqual(
            _lines(
                style.check_class_comment,
                "namespace a {\n\nclass Foo {\n};\n\n}\n",
                "x/Foo.h",
            ),
            [3],
        )
        self.assertEqual(
            _lines(
                style.check_class_comment,
                "template <typename T>\nstruct Bar {};\n",
                "x/Bar.h",
            ),
            [2],
        )

    def test_accepted(self):
        for source in [
            "// What Foo is for.\nclass Foo {\n};\n",
            "/* What Foo is for. */\nclass Foo final : public Base {\n};\n",
            "// A template.\ntemplate <typename T>\nclass Foo {\n};\n",
            "class Forward;\n",
            "// Outer.\nclass Outer {\n  struct Nested {};\n};\n",
            "enum class Mode { kA };\n",
        ]:
            with self.subTest(source=source):
                self.assertEqual(style.check_class_comment(source, "x/Foo.h"), [])
        self.assertEqual(
            style.check_class_comment("class Foo {};\n", "x/Foo.cpp"),
            [],
            "headers only",
        )

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self, "class-comment", "class Foo {};\n", "src/Foo.h"
        )


class IfElseBracesTest(unittest.TestCase):
    def test_flagged(self):
        self.assertEqual(
            _lines(
                style.check_if_else_braces,
                "void f() {\n  if (a) b();\n  else c();\n}\n",
            ),
            [2, 3],
        )
        self.assertEqual(
            _lines(
                style.check_if_else_braces, "void f() {\n  if (a) {\n  } else c();\n}\n"
            ),
            [3],
        )

    def test_accepted(self):
        for source in [
            "void f() { if (a) return; }",
            "void f() { if (a) { b(); } else { c(); } }",
            "void f() { if (a) { b(); } else if (d) { c(); } }",
        ]:
            with self.subTest(source=source):
                self.assertEqual(style.check_if_else_braces(source + "\n", "a.cpp"), [])

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "if-else-braces",
            "void f() {\n  if (a) {\n  } else c();\n}\n",
            "src/a.cpp",
        )


if __name__ == "__main__":
    unittest.main()
