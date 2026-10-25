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

"""Tests of tools/embed_page.py, the generator that compiles the viewer's page into the robot binary."""

import os
import re
import tempfile
import unittest

import embed_page

# The page as the BUILD file's data puts it in the test's runfiles.
_PAGE = "robot_runtime/web_viewer/page/index.html"


def _embedded_page(source: str) -> str:
    """The text of the raw string literal of a generated source."""
    match = re.search(
        r'R"(?P<delimiter>[^(]*)\((?P<page>.*)\)(?P=delimiter)";',
        source,
        re.DOTALL,
    )
    assert match is not None, "the source has no raw string literal"
    return match.group("page")


class EmbedPageTest(unittest.TestCase):

    def test_round_trips_a_page(self) -> None:
        page = (
            '<html><body><script>const s = "a\\\\b"; //'
            ' )html"</script></body></html>\n'
        )
        source = embed_page.render_source(page, "page.html")
        self.assertEqual(_embedded_page(source), page)
        self.assertIn("absl::string_view webViewerPage()", source)
        self.assertIn('#include "web_viewer/WebViewerPage.h"', source)

    def test_skips_a_delimiter_whose_terminator_the_page_contains(self) -> None:
        page = 'x )page" y'
        self.assertEqual(
            embed_page.choose_delimiter(page), embed_page.DELIMITERS[1]
        )
        self.assertEqual(
            _embedded_page(embed_page.render_source(page, "p.html")), page
        )

    def test_refuses_a_page_containing_every_terminator(self) -> None:
        page = " ".join(f'){delimiter}"' for delimiter in embed_page.DELIMITERS)
        with self.assertRaisesRegex(embed_page.EmbedError, "every delimiter"):
            embed_page.render_source(page, "p.html")

    def test_delimiters_are_valid_raw_string_delimiters(self) -> None:
        for delimiter in embed_page.DELIMITERS:
            self.assertLessEqual(len(delimiter), 16)
            self.assertRegex(delimiter, r"^[A-Za-z0-9_]+$")

    def test_refuses_a_page_too_long_for_a_literal(self) -> None:
        with self.assertRaisesRegex(embed_page.EmbedError, "bytes"):
            embed_page.render_source(
                "x" * (embed_page.MAX_PAGE_BYTES + 1), "p.html"
            )

    def test_refuses_a_nul(self) -> None:
        with self.assertRaisesRegex(embed_page.EmbedError, "NUL"):
            embed_page.render_source("a\0b", "p.html")

    def test_the_viewer_page_embeds(self) -> None:
        with open(_PAGE, encoding="utf-8") as page_file:
            page = page_file.read()
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "WebViewerPage.cpp")
            self.assertEqual(
                embed_page.main(["--input", _PAGE, "--output", output]), 0
            )
            with open(output, encoding="utf-8") as source_file:
                self.assertEqual(_embedded_page(source_file.read()), page)
        # The routes the server answers (WebViewerServer.cpp:web_viewer_routes) are the ones the page uses.
        for route in (
            '"/frame.jpg"',
            '"/state"',
            '"/input"',
            '"/settings"',
            '"X-MuJoCo-Viewer"',
        ):
            self.assertIn(route, page)

    def test_main_reports_a_page_it_cannot_embed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "page.html")
            with open(page, "w", encoding="utf-8") as page_file:
                page_file.write("\0")
            output = os.path.join(directory, "out.cpp")
            self.assertEqual(
                embed_page.main(["--input", page, "--output", output]), 1
            )
            self.assertFalse(os.path.exists(output))


if __name__ == "__main__":
    unittest.main()
