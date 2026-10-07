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

"""Tests for no_auto.py: `auto` only for std::make_unique / std::make_shared initializers."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import no_auto


def _count(source: str) -> int:
    return len(no_auto.check_source(source, "src/a.cpp"))


class NoAutoTest(unittest.TestCase):
    def test_every_other_auto_is_flagged(self):
        for source in [
            "void g() { auto x = f(); }",
            "void g() { const auto& y = f(); }",
            "void g() { for (const auto& e : v) {} }",
            "void g() { for (const auto& [key, value] : map) {} }",
            "void g() { auto lambda = [] {}; }",
            "auto l = [](auto x) { return x; };",
            "decltype(auto) f();",
            "auto f() -> int;",
            "auto f() { return 1; }",
            "template <auto N> struct A {};",
            "void g() { auto p = std::make_unique; }",
            "void g() { auto* p = std::make_unique<T>(); }",
            "void c() { auto p = std::make_unique<Foo>().get(); }",
            "void c() { auto p = std::make_unique<Foo>()->member; }",
            "void c() { auto p = std::make_unique<Foo>().release(); }",
            "void c() { auto p = std::make_shared<Foo>(1), q = 3; }",
            "void c() { auto p = std::make_shared<Foo>(1) + q; }",
        ]:
            with self.subTest(source=source):
                self.assertGreaterEqual(_count(source), 1)
        self.assertEqual(_count("auto l = [](auto x) { return x; };"), 2)

    def test_the_factory_initializers_are_allowed(self):
        for source in [
            "void g() { auto term = std::make_unique<Term>(model, /*weight=*/0.5); }",
            "void g() { const auto shared = std::make_shared<Model>(); }",
            "void g() { auto map = std::make_unique<std::map<int, std::vector<int>>>(f(a, b)); }",
            "void g() { auto queue = std::make_unique<SpscQueue<Sample>>(/*capacity=*/8); }",
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(source), 0)

    def test_comments_strings_and_identifiers(self):
        self.assertEqual(
            _count('// auto x\nconst char kText[] = "auto";\nint automatic = 0;\n'), 0
        )

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "no-auto",
            "void g() {\n  auto x = f();\n}\n",
            "src/a.cpp",
            clean="void g() {\n  int x = f();\n}\n",
        )


if __name__ == "__main__":
    unittest.main()
