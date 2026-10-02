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

"""Tests for pointer_nullability.py: every raw-pointer declarator carries absl_nonnull or absl_nullable."""

import os
import shutil
import subprocess
import tempfile
import unittest

from tools.hooks import check_test_support
from tools.hooks import pointer_nullability

INCLUDE = '#include "absl/base/nullability.h"\n'


def _messages(source: str) -> list[str]:
    return [f.message for f in pointer_nullability.check_source(source, "src/a.cpp")]


def _count(source: str) -> int:
    return len(pointer_nullability.check_source(INCLUDE + source, "src/a.cpp"))


class MissingAnnotationTest(unittest.TestCase):
    def test_one_finding_per_unannotated_declarator(self):
        for source, expected in [
            ("void f(Foo* foo);", 1),
            ("Foo* make();", 1),
            ("class A { Foo* foo_; };", 1),
            ("void g() { Foo* local = nullptr; }", 1),
            ("void f(const Foo* foo);", 1),
            ("void f(Foo* const foo);", 1),
            ("void f(Foo** foo);", 2),
            ("void f(Foo* absl_nonnull* foo);", 1),
            ("std::vector<Foo*> v;", 1),
            ("std::map<K, const V*> m;", 1),
            ("absl::StatusOr<const T*> Find();", 1),
            ("std::function<void(const Foo*)> callback;", 1),
            ("std::atomic<T*> sink;", 1),
            ("void (*fn)(int);", 1),
            ("using F = R (*)(A);", 1),
            ("int Foo::*member;", 1),
            ("auto lambda = [](Foo* f) { return f; };", 1),
            ("void g() { for (Foo* f : foos) {} }", 1),
            ("void g() { if (T* p = f()) {} }", 1),
            ("void g() { auto* p = f(); }", 1),
            ("struct A { operator const char*() const; };", 1),
            ("struct A { T* operator->(); };", 1),
            ("int main(int argc, char** argv);", 2),
            ("int main(int argc, char* argv[]);", 2),
            ("void f(const double p[3]);", 1),
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(source), expected)

    def test_a_right_aligned_declarator(self):
        # The first declarator of `Foo *a, *b` is reported; the second looks like a dereference in an argument list.
        messages = _messages(INCLUDE + "Foo *a, *b;\n")
        self.assertEqual(len(messages), 1)
        self.assertIn("right-aligned", messages[0])
        self.assertIn("right-aligned", _messages(INCLUDE + "void f(Foo *a);\n")[0])
        self.assertEqual(
            _messages(INCLUDE + "int x = a *b;\n"),
            [],
            "an expression is not a declaration",
        )


class NotFlaggedTest(unittest.TestCase):
    def test_annotated_forms(self):
        for source in [
            "void f(Foo* absl_nonnull foo);",
            "void f(const Foo* absl_nullable foo);",
            "void f(Foo* absl_nonnull const foo);",
            "std::vector<Foo* absl_nonnull> v;",
            "void f(char* absl_nonnull* absl_nonnull argv);",
            "void (*absl_nonnull fn)(int);",
            "int Foo::*absl_nullable member;",
            "void f(const double p[absl_nonnull 3]);",
            "absl_nonnull std::unique_ptr<Foo> p;",
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(source), 0)

    def test_expressions(self):
        for source in [
            "int x = a * b;",
            "int x = *p;",
            "int x = **pp;",
            "int x = (*it).x;",
            "int x = (*fn)(3);",
            "struct A { A& f() { return *this; } };",
            "bool b = (*opt)[0] && (*opt)[1];",
            "x = 1'000 * y;",
            "f(0.5 * t[0] + 0.5 * t[1]);",
            "if (i > ends[0]) {}",
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(source), 0)

    def test_operators(self):
        for source in [
            "T operator*() const;",
            "T& operator*=(T x);",
            "int x = a->*m;",
            "int x = a.*m;",
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(source), 0)

    def test_casts_and_type_expressions(self):
        for source in [
            "auto p = static_cast<Foo*>(q);",
            "auto p = reinterpret_cast<const char*>(q);",
            "auto p = dynamic_cast<Foo*>(q);",
            "auto p = (const char*)buf;",
            "size_t n = sizeof(Foo*);",
            "size_t n = alignof(Foo*);",
            "auto& t = typeid(Foo*);",
            "auto p = new Foo*[3];",
            "constexpr bool b = std::is_pointer_v<T*>;",
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(source), 0)

    def test_comments_literals_and_the_preprocessor(self):
        source = '// Foo* p\n/* Foo* q */\nconst char s[] = "Foo* x";\nauto r = R"(Foo* y)";\nchar c = \'*\';\n#define X(T) T*\n'
        self.assertEqual(_count(source), 0)

    def test_a_local_array_is_not_a_parameter(self):
        self.assertEqual(_count("void g() { double buf[3]; }"), 0)


class RulesTest(unittest.TestCase):
    def test_misplaced_annotations(self):
        self.assertIn(
            "misplaced", _messages(INCLUDE + "void f(absl_nonnull Foo* p);")[0]
        )
        self.assertTrue(
            any(
                "misplaced" in m
                for m in _messages(INCLUDE + "void f(Foo* const absl_nonnull p);")
            )
        )

    def test_nonnull_initialized_with_null(self):
        for source in [
            "Foo* absl_nonnull p = nullptr;",
            "void f(Foo* absl_nonnull p = nullptr);",
            "struct A { Foo* absl_nonnull p_{nullptr}; };",
            "Foo* absl_nonnull p = NULL;",
        ]:
            with self.subTest(source=source):
                messages = _messages(INCLUDE + source)
                self.assertEqual(len(messages), 1)
                self.assertIn("initialized with", messages[0])
        self.assertEqual(_count("Foo* absl_nullable p = nullptr;"), 0)

    def test_banned_spellings(self):
        for source in [
            "Foo* absl_nullability_unknown p;",
            "ABSL_POINTERS_DEFAULT_NONNULL",
            "absl::Nonnull<int*> p;",
            "absl::Nullable<int*> p;",
            "int* _Nonnull p;",
        ]:
            with self.subTest(source=source):
                self.assertGreaterEqual(_count(source), 1)

    def test_the_include(self):
        self.assertEqual(len(_messages("void f(Foo* absl_nonnull p);")), 1)
        self.assertIn("nullability.h", _messages("void f(Foo* absl_nonnull p);")[0])
        self.assertEqual(_count("void f(Foo* absl_nonnull p);"), 0)
        self.assertEqual(_messages("void f(int x);"), [])


class WiringTest(unittest.TestCase):
    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "pointer-nullability",
            INCLUDE + "void f(Foo* foo);\n",
            "src/a.cpp",
            clean=INCLUDE + "void f(Foo* absl_nonnull foo);\n",
            pending=True,
        )

    def test_the_scope_is_first_party_and_lib_ocs2(self):
        check = check_test_support.assert_registered(
            self, "pointer-nullability", pending=True
        )
        self.assertTrue(check.applies_to("lib/ocs2/core/include/ocs2_core/Types.h"))
        self.assertFalse(
            check.applies_to("lib/ocs2/thirdparty/include/cppad/cppad.hpp")
        )
        self.assertTrue(check.applies_to("robot_runtime/robot_core/src/A.cpp"))

    def test_the_annotations_are_clang_format_attribute_macros(self):
        config = check_test_support.read_repository_file(".clang-format")
        for annotation in pointer_nullability.ANNOTATIONS:
            self.assertIn(annotation, config)

    @unittest.skipIf(shutil.which("clang-format") is None, "no clang-format")
    def test_clang_format_keeps_annotated_pointers(self):
        probe = (
            INCLUDE
            + "void f(std::vector<Foo* absl_nonnull> v, const int* absl_nullable p, char* absl_nonnull* absl_nonnull argv);\n"
            + "int (*absl_nonnull fn)(int);\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy(
                check_test_support.repository_file(".clang-format"),
                os.path.join(directory, ".clang-format"),
            )
            path = os.path.join(directory, "probe.cpp")
            with open(path, "w", encoding="utf-8") as f:
                f.write(probe)
            formatted = subprocess.run(
                ["clang-format", "--style=file", path],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        self.assertEqual(formatted, probe)


if __name__ == "__main__":
    unittest.main()
