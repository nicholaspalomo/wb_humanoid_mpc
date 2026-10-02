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

"""Inclusive language: no `master` / `slave`, `blacklist` / `whitelist` or `redline` (Google C++ Style Guide).

The words are matched as whole words and as the parts of a camelCase or snake_case identifier (`masterScale`,
`kWhitelist`), in every text file the spelling check reads, never inside a URL: a link to another project's `master`
branch cannot be renamed here. Write `main`, `primary` / `secondary`, `leader` / `follower`, `allowlist` / `denylist`,
`global` or `limit` instead. Gendered pronouns for unspecified people are left to the agent rules (AGENTS.md): a word
list cannot tell them apart from quotations.
"""

import os
import re

from tools.hooks import check_types
from tools.hooks import lint_files

NAME = "inclusive-language"

WORDS = {
    "master": "main, primary, leader or global",
    "masters": "mains, primaries or leaders",
    "slave": "secondary, follower or replica",
    "slaves": "secondaries, followers or replicas",
    "blacklist": "denylist or blocklist",
    "blacklists": "denylists",
    "blacklisted": "denied or blocked",
    "whitelist": "allowlist",
    "whitelists": "allowlists",
    "whitelisted": "allowed",
    "redline": "limit",
    "redlines": "limits",
    "redlined": "limited",
}
# The checker and its tests name the words on purpose.
_OWN_FILES = frozenset(
    {"tools/hooks/inclusive_language.py", "tools/hooks/test_inclusive_language.py"}
)
_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")
_URL = re.compile(r"[a-z][a-z0-9+.-]*://\S+")


def check_source(source: str, path: str = "<source>") -> list[check_types.Finding]:
    """The non-inclusive words of `source`, outside URLs."""
    if (
        path in _OWN_FILES
        or os.path.splitext(path)[1].lower() in check_types.MODEL_EXTENSIONS
    ):
        return []
    findings = []
    for number, line in enumerate(source.splitlines(), start=1):
        urls = [(m.start(), m.end()) for m in _URL.finditer(line)]
        for part in _PART.finditer(line):
            word = part.group(0).lower()
            if word not in WORDS or any(
                start <= part.start() < end for start, end in urls
            ):
                continue
            findings.append(
                check_types.Finding(
                    path,
                    number,
                    part.start() + 1,
                    NAME,
                    f"`{part.group(0)}`: write {WORDS[word]} (Google C++ style, Inclusive language).",
                )
            )
    return findings


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.TEXT}),
        scope=lint_files.Scope.TEXT,
        check_source=check_source,
        description="no master/slave, blacklist/whitelist or redline (G: Inclusive language).",
    )
]
