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


def _texts(source: str, keep_preprocessor: bool = False) -> list[tuple[str, str]]:
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
        self.assertIn((cpp_source.NUMBER, "2.5f"), [(t.kind, t.text) for t in tokens])

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
            [t.text for t in cpp_source.tokenize("a->b::c <=> d ... e->*f x.*y")],
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
            [t.text for t in cpp_source.code_tokens("a /* b */ c // d\n")], ["a", "c"]
        )


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


if __name__ == "__main__":
    unittest.main()
