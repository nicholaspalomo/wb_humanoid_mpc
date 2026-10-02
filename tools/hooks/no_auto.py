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

"""No `auto`: spell the type, except for `auto x = std::make_unique<T>(...)` and `std::make_shared<T>(...)` (AGENTS.md).

The repository's rule is stricter than the Google C++ Style Guide's "Type deduction" and ToTW #232: the only `auto` is a
variable initialized by `std::make_unique` or `std::make_shared`, where the type is already written on the line
(`const auto` included). Everything else is flagged as `no-auto`: range-for and structured bindings, lambdas stored in
an `auto` variable, generic-lambda `auto` parameters, `decltype(auto)`, deduced and trailing return types, and
`template <auto N>`. Spell map elements `std::pair<const K, V>`.
"""

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "no-auto"

_FACTORIES = frozenset({"make_unique", "make_shared"})


def _is_factory_initializer(code: tuple[cpp_source.Token, ...], index: int) -> bool:
    """True for the `auto` of `[const] auto name = std::make_unique<` (or make_shared)."""
    texts = [t.text for t in code[index + 1 : index + 7]]
    return (
        len(texts) >= 6
        and code[index + 1].kind == cpp_source.IDENTIFIER
        and texts[1] == "="
        and texts[2] == "std"
        and texts[3] == "::"
        and texts[4] in _FACTORIES
        and texts[5] == "<"
    )


def check_source(source: str, path: str = "<source>") -> list[check_types.Finding]:
    """The findings of `no-auto` in C++ `source`, the file at `path`."""
    code = cpp_source.code_tokens(source)
    findings = []
    for index, token in enumerate(code):
        if token.text != "auto" or token.kind != cpp_source.IDENTIFIER:
            continue
        if _is_factory_initializer(code, index):
            continue
        findings.append(
            check_types.Finding(
                path,
                token.line,
                token.col,
                NAME,
                "`auto`: write the type (AGENTS.md: no `auto` except `auto x = std::make_unique<T>(...)` or "
                "`std::make_shared<T>(...)`; map elements are `std::pair<const K, V>`).",
            )
        )
    return findings


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=check_source,
        description="no `auto` except `auto x = std::make_unique<T>(...)` / `std::make_shared<T>(...)` (AGENTS.md; "
        "stricter than ToTW #232).",
        hint="Spell out every type: range-for elements, structured bindings, lambdas, iterators and return types.",
    )
]
