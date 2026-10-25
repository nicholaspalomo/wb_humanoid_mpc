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

"""Tests for cpp_source.py, the C++ tokenizer and masker the C++ checks share."""

import unittest

from tools.hooks import cpp_source


def _texts(
    source: str, keep_preprocessor: bool = False
) -> list[tuple[str, str]]:
    return [
        (token.kind, token.text)
        for token in cpp_source.tokenize(source, keep_preprocessor)
    ]


class TokenizeTest(unittest.TestCase):
    def test_kinds_lines_and_columns(self):
        tokens = cpp_source.tokenize("int x = 1;\n  foo(a, 2.5f);\n")
        self.assertEqual(
            [(t.kind, t.text, t.line, t.col) for t in tokens[:5]],
            [
                (cpp_source.IDENTIFIER, "int", 1, 1),
                (cpp_source.IDENTIFIER, "x", 1, 5),
                (cpp_source.PUNCTUATOR, "=", 1, 7),
                (cpp_source.NUMBER, "1", 1, 9),
                (cpp_source.PUNCTUATOR, ";", 1, 10),
            ],
        )
        foo = tokens[5]
        self.assertEqual((foo.text, foo.line, foo.col), ("foo", 2, 3))
        self.assertIn(
            (cpp_source.NUMBER, "2.5f"), [(t.kind, t.text) for t in tokens]
        )

    def test_whitespace_before(self):
        tokens = cpp_source.tokenize("Foo* x = a * b;")
        star, x = tokens[1], tokens[2]
        self.assertEqual(star.text, "*")
        self.assertFalse(star.ws_before)
        self.assertTrue(x.ws_before)
        self.assertTrue(
            tokens[0].ws_before, "the start of the file counts as whitespace"
        )
        multiply = tokens[5]
        self.assertEqual(multiply.text, "*")
        self.assertTrue(multiply.ws_before)

    def test_multi_character_punctuators(self):
        self.assertEqual(
            [
                t.text
                for t in cpp_source.tokenize("a->b::c <=> d ... e->*f x.*y")
            ],
            [
                "a",
                "->",
                "b",
                "::",
                "c",
                "<=>",
                "d",
                "...",
                "e",
                "->*",
                "f",
                "x",
                ".*",
                "y",
            ],
        )

    def test_numbers(self):
        for number in [
            "1'000'000",
            "0x1F",
            "1e-3",
            "1.0e-6",
            ".5",
            "0b1010",
            "42u",
            "0x1p+4",
            "100ms",
        ]:
            with self.subTest(number=number):
                self.assertEqual(_texts(number), [(cpp_source.NUMBER, number)])

    def test_comments_and_literals_are_tokens_of_their_own(self):
        source = "f(/*index=*/0, \"a, b\", 'x'); // done\n"
        self.assertEqual(
            _texts(source),
            [
                (cpp_source.IDENTIFIER, "f"),
                (cpp_source.PUNCTUATOR, "("),
                (cpp_source.COMMENT, "/*index=*/"),
                (cpp_source.NUMBER, "0"),
                (cpp_source.PUNCTUATOR, ","),
                (cpp_source.STRING, '"a, b"'),
                (cpp_source.PUNCTUATOR, ","),
                (cpp_source.STRING, "'x'"),
                (cpp_source.PUNCTUATOR, ")"),
                (cpp_source.PUNCTUATOR, ";"),
                (cpp_source.COMMENT, "// done"),
            ],
        )

    def test_raw_strings(self):
        source = 'auto s = R"delim(a ) " b)delim"; int y;'
        texts = _texts(source)
        self.assertIn((cpp_source.STRING, 'R"delim(a ) " b)delim"'), texts)
        self.assertIn((cpp_source.IDENTIFIER, "y"), texts)
        self.assertIn((cpp_source.STRING, 'u8R"(x)"'), _texts('u8R"(x)"'))

    def test_a_backslash_continues_a_line_comment(self):
        tokens = cpp_source.tokenize("// a \\\n still the comment\nint x;\n")
        self.assertEqual(tokens[0].kind, cpp_source.COMMENT)
        self.assertEqual(tokens[1].text, "int")
        self.assertEqual(tokens[1].line, 3)

    def test_preprocessor_directives(self):
        source = "#define X(T) \\\n  T* y\n# include <vector> // why\nint z;\n"
        self.assertEqual(
            _texts(source),
            [
                (cpp_source.IDENTIFIER, "int"),
                (cpp_source.IDENTIFIER, "z"),
                (cpp_source.PUNCTUATOR, ";"),
            ],
        )
        kept = cpp_source.tokenize(source, keep_preprocessor=True)
        self.assertEqual(
            [t.kind for t in kept[:2]],
            [cpp_source.PREPROCESSOR, cpp_source.PREPROCESSOR],
        )
        self.assertEqual(kept[0].text, "#define X(T) \\\n  T* y")
        self.assertEqual(kept[2].line, 4)

    def test_a_hash_inside_a_line_is_not_a_directive(self):
        self.assertIn((cpp_source.PUNCTUATOR, "#"), _texts("x # y"))

    def test_an_unterminated_literal_ends_with_its_line(self):
        tokens = cpp_source.tokenize('"abc\nint x;')
        self.assertEqual(tokens[0].text, '"abc')
        self.assertEqual(tokens[1].text, "int")

    def test_code_tokens_drop_comments(self):
        self.assertEqual(
            [t.text for t in cpp_source.code_tokens("a /* b */ c // d\n")],
            ["a", "c"],
        )


class CallArgumentsTest(unittest.TestCase):
    def _arguments(self, source: str) -> list[list[str]] | None:
        tokens = cpp_source.code_tokens(source)
        opening = next(
            k for k, token in enumerate(tokens) if token.text in "({"
        )
        arguments = cpp_source.call_arguments(tokens, opening)
        if arguments is None:
            return None
        return [[token.text for token in argument] for argument in arguments]

    def test_top_level_commas_split(self):
        self.assertEqual(
            self._arguments("f(a, g(b, c), {d, e}, x[1, 2]);"),
            [
                ["a"],
                ["g", "(", "b", ",", "c", ")"],
                ["{", "d", ",", "e", "}"],
                ["x", "[", "1", ",", "2", "]"],
            ],
        )

    def test_braces_open_a_list_too(self):
        self.assertEqual(self._arguments("T t{1.0, 2.0};"), [["1.0"], ["2.0"]])

    def test_an_empty_list_has_no_argument_and_an_open_one_is_none(self):
        self.assertEqual(self._arguments("f();"), [])
        self.assertIsNone(self._arguments("f(a, b"))

    def test_an_empty_argument_is_kept(self):
        self.assertEqual(self._arguments("f(a, );"), [["a"], []])


class MaskTest(unittest.TestCase):
    def test_comments_and_literals_are_blanked_in_place(self):
        source = 'int a; // x\nconst char* s = "y"; /* z */\n'
        without_comments, code = cpp_source.mask(source)
        self.assertEqual(len(without_comments), len(source))
        self.assertEqual(without_comments.count("\n"), source.count("\n"))
        self.assertNotIn("x", without_comments)
        self.assertIn('"y"', without_comments)
        self.assertNotIn('"y"', code)
        self.assertIn("const char* s", code)

    def test_digit_separators_and_raw_strings(self):
        _, code = cpp_source.mask("int n = 1'000; auto r = R\"(a 'b)\"; int m;")
        self.assertIn("int m;", code)
        self.assertNotIn("'b", code)

    def test_comment_text(self):
        source = 'f(); // NOLINT(x): y\n"// not a comment";\n'
        without_comments, _ = cpp_source.mask(source)
        comments = cpp_source.comment_text(source, without_comments)
        first, second = comments.split("\n")[:2]
        self.assertEqual(first.strip(), "// NOLINT(x): y")
        self.assertEqual(second.strip(), "")

    def test_line_starts(self):
        self.assertEqual(cpp_source.line_starts("ab\ncd\n"), [0, 3, 6])


class ScopesTest(unittest.TestCase):
    def _number_scopes(self, source: str) -> list[tuple[str, ...]]:
        tokens = cpp_source.code_tokens(source)
        scopes = cpp_source.scopes(source)
        return [
            scopes[k]
            for k, token in enumerate(tokens)
            if token.kind == cpp_source.NUMBER
        ]

    def test_an_enum_body_is_an_enum_scope(self):
        source = (
            "enum class Mode : int { kA = 3 };\n[[nodiscard]] enum Flag { kB ="
            " 4 };\n"
        )
        self.assertEqual(
            self._number_scopes(source),
            [(cpp_source.ENUM_SCOPE,), (cpp_source.ENUM_SCOPE,)],
        )

    def test_a_function_with_an_elaborated_enum_parameter_has_a_code_body(self):
        source = "void set(enum Mode mode) {\n  gain = 3;\n}\n"
        self.assertEqual(
            self._number_scopes(source), [(cpp_source.CODE_SCOPE,)]
        )

    def test_a_function_returning_an_elaborated_enum_has_a_code_body(self):
        source = (
            "enum Color parse(int x) {\n  return static_cast<Color>(x +"
            " 3);\n}\n"
        )
        self.assertEqual(
            self._number_scopes(source), [(cpp_source.CODE_SCOPE,)]
        )

    def test_an_enum_with_a_decltype_underlying_type_is_an_enum_scope(self):
        source = (
            "enum E : decltype(kX) { kA = 2 };\nenum F :"
            " std::underlying_type_t<G> { kB = 3 };\n"
        )
        self.assertEqual(
            self._number_scopes(source),
            [(cpp_source.ENUM_SCOPE,), (cpp_source.ENUM_SCOPE,)],
        )

    def test_a_constexpr_lambda_has_a_code_body(self):
        source = (
            "const std::function<int()> kMake = []() constexpr { return 3; };\n"
        )
        self.assertEqual(
            self._number_scopes(source), [(cpp_source.CODE_SCOPE,)]
        )


class CodeBodiesInInitializersTest(unittest.TestCase):
    def _bodies(self, source: str) -> list[str]:
        tokens = cpp_source.code_tokens(source)
        return [
            " ".join(t.text for t in tokens[first : last + 1])
            for first, last in cpp_source.code_bodies_in_initializers(source)
        ]

    def test_the_lambda_bodies_of_a_braced_initializer_are_found(self):
        self.assertEqual(
            self._bodies(
                "const std::vector<Fn> kFns = {[](double x) { return 3.0 * x;"
                " }, [] { return 2; }};\n"
            ),
            ["{ return 3.0 * x ; }", "{ return 2 ; }"],
        )

    def test_the_lambda_bodies_after_lambda_specifiers_are_found(self):
        self.assertEqual(
            self._bodies(
                "const Foo kFoo{.a = []() constexpr { return 3; }, .b = []()"
                " noexcept(true) { return 4; }, .c = []() -> int { return 5; },"
                " .d = [](int x) mutable { return x; }};\n"
            ),
            [
                "{ return 3 ; }",
                "{ return 4 ; }",
                "{ return 5 ; }",
                "{ return x ; }",
            ],
        )

    def test_nested_braced_initializers_are_not_bodies(self):
        self.assertEqual(
            self._bodies(
                "const Foo kFoo{{1, 2}, Bar{3}, std::array<int, 1>{4}};\n"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
