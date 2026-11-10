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

"""Tests for cpp_abseil.py: the Abseil types chosen over their standard library counterparts."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import check_types
from tools.hooks import cpp_abseil as abseil


def _count(check: check_types.CheckFunction, source: str) -> int:
    return len(check(f"{source}\n", "src/a.cpp"))


class AbseilTest(unittest.TestCase):
    def assert_counts(
        self, check: check_types.CheckFunction, cases: list[tuple[str, int]]
    ):
        """Asserts that each source of `cases` has its number of findings."""
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(_count(check, source), expected)

    def test_span_param(self):
        self.assert_counts(
            abseil.check_span_param,
            [
                ("void f(const std::vector<int>& v);", 1),
                ("void f(int x, const ::std::vector<Foo<int>>& v) const;", 1),
                ("explicit Foo(const vector_array_t& values);", 1),
                ("void f(const ocs2::scalar_array_t& times);", 1),
                (
                    "bool operator==(const std::vector<int>& a,"
                    " const std::vector<int>& b);",
                    2,
                ),
                ("void f(const std::vector<bool>& v);", 0),
                ("void f(absl::Span<const int> v);", 0),
                ("void Foo::f(const std::vector<int>& v) {}", 0),
                ("void f(const std::vector<int>& v) const override;", 0),
                ("void f(const std::vector<int>& v) final {}", 0),
                ("auto g = [](const std::vector<int>& v) { return 1; };", 0),
                ("std::function<void(const std::vector<int>&)> callback;", 0),
                ("for (const std::vector<int>& v : vectors) {}", 0),
                ("const std::vector<int>& x = y;", 0),
                ("const std::vector<int>& values() const;", 0),
                (
                    "MOCK_METHOD(void, f, (const std::vector<int>&),"
                    " (override));",
                    0,
                ),
                ("void f(const robot::vector<int>& v);", 0),
            ],
        )

    def test_string_view_param(self):
        self.assert_counts(
            abseil.check_string_view_param,
            [
                ("void f(const std::string& name);", 1),
                ("Foo(int id, const std::string& name = {});", 1),
                ("void f(absl::string_view name);", 0),
                ("void f(const std::string& name) final;", 0),
                ("absl::Status Foo::f(const std::string& name) {", 0),
                ("const std::string& name() const;", 0),
                ("void f(const std::vector<std::string>& names);", 0),
            ],
        )

    def test_ordered_container(self):
        self.assert_counts(
            abseil.check_ordered_container,
            [
                ("std::map<int, int> m;", 1),
                ("std::set<int> s; std::multimap<int, int> m;", 2),
                ("absl::btree_map<int, int> m;", 0),
                ("x.map(1); robot::set<int> s;", 0),
            ],
        )

    def test_mutex(self):
        self.assert_counts(
            abseil.check_mutex,
            [
                ("std::mutex m;", 1),
                ("std::lock_guard<std::mutex> lock(m);", 2),
                ("std::condition_variable ready;", 1),
                ("absl::Mutex m; absl::MutexLock lock(&m);", 0),
            ],
        )

    def test_function(self):
        self.assert_counts(
            abseil.check_function,
            [
                ("std::function<void()> callback;", 1),
                ("absl::AnyInvocable<void()> callback;", 0),
                ("void f(absl::FunctionRef<int(int)> g);", 0),
            ],
        )

    def test_number_conversion(self):
        self.assert_counts(
            abseil.check_number_conversion,
            [
                ("int n = std::stoi(text);", 1),
                ("std::string s = std::to_string(1);", 1),
                ("double d = std::stod(text);", 1),
                ("bool ok = absl::SimpleAtoi(text, &n);", 0),
            ],
        )

    def test_string_stream(self):
        self.assert_counts(
            abseil.check_string_stream,
            [
                ("std::ostringstream os;", 1),
                ("std::stringstream ss(text);", 1),
                ("std::string s = absl::StrCat(a, b);", 0),
            ],
        )

    def test_view_type(self):
        self.assert_counts(
            abseil.check_view_type,
            [
                ("void f(std::string_view name);", 1),
                ("void f(std::span<const int> values);", 1),
                ("void f(absl::string_view name, absl::Span<int> v);", 0),
            ],
        )


class RegistryTest(unittest.TestCase):
    def test_every_check_behaves(self):
        for name, flagged, path in [
            (
                "absl-span-param",
                "void f(const std::vector<int>& v);\n",
                "src/a.h",
            ),
            (
                "absl-string-view-param",
                "void f(const std::string& s);\n",
                "src/a.h",
            ),
            ("absl-ordered-container", "std::map<int, int> m;\n", "src/a.cpp"),
            ("absl-mutex", "std::mutex m;\n", "src/a.cpp"),
            ("absl-function", "std::function<void()> f;\n", "src/a.cpp"),
            (
                "absl-number-conversion",
                "std::string s = std::to_string(1);\n",
                "src/a.cpp",
            ),
            ("absl-string-stream", "std::ostringstream out;\n", "src/a.cpp"),
            ("absl-view-type", "std::string_view v;\n", "src/a.cpp"),
        ]:
            with self.subTest(check=name):
                check_test_support.assert_check_behaves(
                    self, name, flagged, path
                )


if __name__ == "__main__":
    unittest.main()
