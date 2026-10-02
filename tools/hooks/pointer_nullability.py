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

"""Raw pointers carry nullability: `Foo* absl_nonnull` or `Foo* absl_nullable` (AGENTS.md).

Every raw-pointer declarator in first-party code and in lib/ocs2 (not lib/ocs2/thirdparty) carries `absl_nonnull` or
`absl_nullable` directly after its `*`, the way `const` qualifies a const pointer. GCC, and clang before 19, expand the
annotations to nothing, so this check is the only one that sees them. It reports, as `pointer-nullability`:

- a declarator `*`, `(*`, `::*`, or an array parameter `[`, not directly followed by an annotation: parameters,
  returns, members, locals, template arguments (`std::vector<Foo*>`), function and member pointers;
- a misplaced annotation: `absl_nonnull Foo* p` (it qualifies Foo) or `Foo* const absl_nonnull p`; write
  `Foo* absl_nonnull const p`. An annotation in front of a smart pointer type (`absl_nonnull std::unique_ptr<T>`) is fine;
- `absl_nonnull` on a declarator initialized or defaulted to `nullptr`, `NULL` or `0`;
- the spellings that are not used: `absl_nullability_unknown`, `ABSL_POINTERS_DEFAULT_NONNULL`, the removed
  `absl::Nonnull<>` / `absl::Nullable<>` / `absl::NullabilityUnknown<>` templates, and raw `_Nonnull` / `_Nullable` /
  `_Null_unspecified`;
- a right-aligned declarator, `Foo *a` (clang-format's fallback for `Foo *a, *b`): one declarator per declaration;
- a file that uses an annotation without `#include "absl/base/nullability.h"`.

Exempt: casts (`static_cast<Foo*>`, C-style), `sizeof` / `alignof` / `typeid` / `decltype`, `new T*[n]`, type-trait
arguments (`std::is_pointer_v<T*>`), and everything that is not code (comments, literals, preprocessor lines). An
annotated cast is an assertion that silences the analyzer, not documentation.

The classification relies on clang-format's PointerAlignment: Left layout: a declarator `*` is glued to the type on its
left; a binary `*` has spaces on both sides; a unary `*` has none after it. It cannot see a pointer behind an alias
(`FooPtr`), a template parameter instantiated with a pointer, `auto p = &x` (lib/ocs2), a macro, or generated code
(the nproto generator emits annotated signatures instead).
"""

import re

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "pointer-nullability"

# LINT.IfChange(rules)
ANNOTATIONS = frozenset({"absl_nonnull", "absl_nullable", "absl_nullability_unknown"})
BANNED = frozenset(
    {
        "absl_nullability_unknown",
        "ABSL_POINTERS_DEFAULT_NONNULL",
        "_Nonnull",
        "_Nullable",
        "_Null_unspecified",
    }
)
BANNED_TEMPLATES = frozenset({"Nonnull", "Nullable", "NullabilityUnknown"})
INCLUDE = "absl/base/nullability.h"
# LINT.ThenChange(//.clang-format:attribute_macros, //AGENTS.md:nullability)

# Smart pointer and callable types an annotation may precede (Abseil annotates them as classes).
_SMART_POINTERS = frozenset(
    {"unique_ptr", "shared_ptr", "function", "AnyInvocable", "weak_ptr"}
)
_NOT_TYPE_WORDS = frozenset(
    {
        "return",
        "case",
        "throw",
        "sizeof",
        "delete",
        "and",
        "or",
        "not",
        "co_return",
        "co_yield",
        "co_await",
        "else",
        "do",
        "if",
        "while",
        "for",
        "switch",
        "alignof",
        "typeid",
        "new",
        "goto",
        "default",
        "operator",
        "using",
        "typedef",
    }
)
_CASTS = frozenset(
    {
        "static_cast",
        "reinterpret_cast",
        "const_cast",
        "dynamic_cast",
        "bit_cast",
        "implicit_cast",
        "down_cast",
    }
)
_TYPE_EXPRESSIONS = frozenset(
    {"sizeof", "alignof", "typeid", "decltype", "alignas", "offsetof"}
)
_TRAIT = re.compile(
    r"^(is_\w+|remove_\w+|add_\w+|decay(_t)?|enable_if(_t)?|conditional(_t)?|common_type(_t)?|invoke_result(_t)?|"
    r"result_of(_t)?|type_identity(_t)?|underlying_type(_t)?|void_t|integral_constant|pointer_traits|iterator_traits)$"
)
# The tokens a declarator `*` may be followed by, directly (Foo*>, Foo*, Foo*) ...).
_AFTER_DECLARATOR = frozenset(
    {">", ">>", ",", ")", "&", "&&", "*", "[", "...", ";", "=", "{", ":"}
)
_NULL_VALUES = frozenset({"nullptr", "NULL", "0"})
_C_CAST_CONTEXT = frozenset(
    {"=", "(", ",", "return", "+", "-", "?", ":", "{", "<<", "&&", "||", "!"}
)
_INCLUDE_LINE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*"absl/base/nullability\.h"', re.MULTILINE
)
_ADVICE = "write `T* absl_nonnull` or `T* absl_nullable` (AGENTS.md, 'Raw pointers carry nullability')"


def _is_typeish(token: cpp_source.Token) -> bool:
    if token.kind == cpp_source.IDENTIFIER:
        return token.text not in _NOT_TYPE_WORDS
    return token.text in (">", ">>")


def _enclosing_owner(code: tuple[cpp_source.Token, ...], index: int) -> str | None:
    """The name before the innermost unclosed `<` around code[index], or `(` + a type-expression keyword."""
    angle = 0
    paren = 0
    k = index - 1
    while k >= 0:
        text = code[k].text
        if text in (";", "{", "}"):
            return None
        if text == ")":
            paren += 1
        elif text == "(":
            if paren == 0:
                if k > 0 and code[k - 1].text in _TYPE_EXPRESSIONS:
                    return "(" + code[k - 1].text
            else:
                paren -= 1
        elif text == ">":
            angle += 1
        elif text == ">>":
            angle += 2
        elif text == "<":
            if angle == 0:
                return code[k - 1].text if k > 0 else None
            angle -= 1
        k -= 1
    return None


def _is_exempt(code: tuple[cpp_source.Token, ...], index: int) -> bool:
    """True for a `*` in a cast, a type expression, a trait argument or a new-expression."""
    owner = _enclosing_owner(code, index)
    if owner is not None and (
        owner in _CASTS or owner.startswith("(") or _TRAIT.match(owner)
    ):
        return True
    k = index - 1
    while k >= 0 and (
        code[k].kind == cpp_source.IDENTIFIER
        or code[k].text in ("::", "<", ">", ",", "*")
    ):
        if code[k].text == "new":
            return True
        k -= 1
    return False


def _is_c_style_cast(code: tuple[cpp_source.Token, ...], star: int) -> bool:
    """True for the `*` of `(const char*)buf`: `(` type `*` `)` followed by an operand, after an operator."""
    if star + 2 >= len(code) or code[star + 1].text != ")":
        return False
    depth = 0
    k = star
    while k >= 0:
        if code[k].text == ")":
            depth += 1
        elif code[k].text == "(":
            if depth == 0:
                break
            depth -= 1
        k -= 1
    if k <= 0:
        return False
    before = code[k - 1]
    following = code[star + 2]
    return (
        before.text in _C_CAST_CONTEXT
        or (code[k].ws_before and before.kind == cpp_source.PUNCTUATOR)
    ) and (
        following.kind in (cpp_source.IDENTIFIER, cpp_source.NUMBER, cpp_source.STRING)
        or following.text in ("(", "&", "*")
    )


def _is_conversion_operator(code: tuple[cpp_source.Token, ...], star: int) -> bool:
    """True for the `*` of `operator const char*()`."""
    k = star - 1
    while k >= 0 and (code[k].kind == cpp_source.IDENTIFIER or code[k].text == "::"):
        if code[k].text == "operator":
            return True
        k -= 1
    return False


def _declaration_context(code: tuple[cpp_source.Token, ...], index: int) -> bool:
    """True when the type-ish run ending at code[index] starts a declaration, not an expression (`a *b`)."""
    k = index
    while k >= 0:
        token = code[k]
        if token.kind == cpp_source.IDENTIFIER and token.text not in _NOT_TYPE_WORDS:
            k -= 1
        elif token.text == "::":
            k -= 1
        elif token.text in (">", ">>"):
            depth = 0
            while k >= 0:
                if code[k].text == ">":
                    depth += 1
                elif code[k].text == ">>":
                    depth += 2
                elif code[k].text == "<":
                    depth -= 1
                    if depth <= 0:
                        break
                elif code[k].text in (";", "{", "}"):
                    return False
                k -= 1
            k -= 1
        else:
            break
    return k < 0 or code[k].text in (";", "{", "}", "(", ",", ":", ">", "<")


class _Finder:
    """Collects the findings of one file."""

    def __init__(self, source: str, path: str) -> None:
        self.source = source
        self.path = path
        self.code = cpp_source.code_tokens(source)
        self.findings: list[check_types.Finding] = []

    def add(self, token: cpp_source.Token, message: str) -> None:
        self.findings.append(
            check_types.Finding(self.path, token.line, token.col, NAME, message)
        )

    def _annotation_after(self, index: int) -> bool:
        return index + 1 < len(self.code) and self.code[index + 1].text in ANNOTATIONS

    def _stars(self, index: int) -> tuple[list[int], int]:
        """The glued run of `*` from code[index] (`**`, `* absl_nonnull*`, `* const*`), and its last index."""
        code = self.code
        stars = [index]
        last = index
        while True:
            k = last + 1
            while k < len(code) and (
                code[k].text in ANNOTATIONS or code[k].text in ("const", "volatile")
            ):
                k += 1
            if k < len(code) and code[k].text == "*" and not code[k].ws_before:
                stars.append(k)
                last = k
                continue
            return stars, last

    def _missing(self, stars: list[int], what: str) -> None:
        for star in stars:
            if not self._annotation_after(star):
                self.add(self.code[star], f"{what} without nullability: {_ADVICE}.")

    def run(self) -> list[check_types.Finding]:
        """Finds every finding of the file."""
        code = self.code
        n = len(code)
        i = 0
        while i < n:
            token = code[i]
            if token.text == "*":
                i = self._star(i) + 1
                continue
            if token.text == "[":
                self._array_parameter(i)
            elif token.text in ANNOTATIONS:
                self._annotation(i)
            if token.text in BANNED:
                self.add(token, f"`{token.text}` is not used here: {_ADVICE}.")
            elif (
                token.text in BANNED_TEMPLATES
                and i >= 2
                and code[i - 1].text == "::"
                and code[i - 2].text == "absl"
                and i + 1 < n
                and code[i + 1].text == "<"
            ):
                self.add(
                    token, f"`absl::{token.text}<>` was removed from Abseil: {_ADVICE}."
                )
            i += 1
        self._include()
        return self.findings

    def _star(self, i: int) -> int:
        """Judges the `*` at code[i] and returns the index of the last `*` of its run."""
        code = self.code
        n = len(code)
        token = code[i]
        previous = code[i - 1] if i > 0 else None
        stars, last = self._stars(i)
        if previous is None or previous.text == "operator":
            return last
        if not token.ws_before and previous.text == "::":
            self._missing(stars, "member pointer")
            return last
        if not token.ws_before and (
            _is_typeish(previous) or previous.text in ("const", "volatile")
        ):
            after = code[i + 1] if i + 1 < n else None
            if after is None:
                return last
            if (
                after.ws_before
                or after.text in _AFTER_DECLARATOR
                or (after.text == "(" and _is_conversion_operator(code, i))
            ):
                if _is_exempt(code, i) or _is_c_style_cast(code, i):
                    return last
                self._missing(stars, f"raw pointer `{previous.text}*`")
                self._contradiction(last)
            return last
        if not token.ws_before and previous.text == "(" and i >= 2:
            before = code[i - 2]
            if previous.ws_before and _is_typeish(before):
                k = i + 1
                while k < n and (
                    code[k].text in ANNOTATIONS or code[k].text == "const"
                ):
                    k += 1
                if k < n and code[k].kind == cpp_source.IDENTIFIER:
                    k += 1
                if k + 1 < n and code[k].text == ")" and code[k + 1].text in ("(", "["):
                    self._missing(stars, "function or array pointer")
            return last
        if (
            token.ws_before
            and previous.kind == cpp_source.IDENTIFIER
            and _is_typeish(previous)
            and i + 1 < n
            and not code[i + 1].ws_before
            and code[i + 1].kind == cpp_source.IDENTIFIER
            and _declaration_context(code, i - 1)
            and not _is_exempt(code, i)
        ):
            self.add(
                token,
                f"right-aligned pointer declarator `{previous.text} *{code[i + 1].text}`: declare one variable per "
                f"declaration as `{previous.text}* absl_nonnull {code[i + 1].text}` (or absl_nullable).",
            )
        return last

    def _contradiction(self, last: int) -> None:
        """Reports `absl_nonnull` on a declarator that is initialized or defaulted to null."""
        code = self.code
        n = len(code)
        k = last + 1
        if k >= n or code[k].text != "absl_nonnull":
            return
        k += 1
        if k < n and code[k].text == "const":
            k += 1
        if k < n and code[k].kind == cpp_source.IDENTIFIER:
            k += 1
        if k + 1 < n and code[k].text == "=" and code[k + 1].text in _NULL_VALUES:
            null = code[k + 1]
        elif (
            k + 2 < n
            and code[k].text == "{"
            and code[k + 1].text in _NULL_VALUES
            and code[k + 2].text == "}"
        ):
            null = code[k + 1]
        else:
            return
        self.add(
            code[last + 1],
            f"`absl_nonnull` pointer initialized with `{null.text}`: it is nullable (`absl_nullable`), or it needs a "
            "non-null value.",
        )

    def _array_parameter(self, i: int) -> None:
        """Reports an array parameter (`const double p[3]`) whose `[` carries no annotation."""
        code = self.code
        if i < 2 or code[i].ws_before:
            return
        name = code[i - 1]
        type_token = code[i - 2]
        if (
            name.kind != cpp_source.IDENTIFIER
            or name.text in _NOT_TYPE_WORDS
            or not name.ws_before
        ):
            return
        # The type before the name: `double p[3]`, `T* p[3]`, `T* absl_nonnull p[3]`; never a binary `a * p[3]`.
        if type_token.text in ("*", "&", ">", ">>"):
            if type_token.ws_before:
                return
        elif not (_is_typeish(type_token) or type_token.text in ANNOTATIONS):
            return
        close = cpp_source.matching(code, i)
        if close + 1 >= len(code) or code[close + 1].text not in (",", ")"):
            return
        # The innermost enclosing bracket is the `(` of a parameter list.
        depth = 0
        k = i - 1
        while k >= 0:
            text = code[k].text
            if text in (")", "]", "}"):
                depth += 1
            elif text in ("(", "[", "{"):
                if depth == 0:
                    break
                depth -= 1
            elif text == ";":
                return
            k -= 1
        if k < 0 or code[k].text != "(":
            return
        if i + 1 < len(code) and code[i + 1].text in ANNOTATIONS:
            return
        self.add(
            code[i],
            f"array parameter `{name.text}[]` without nullability: write `{name.text}[absl_nonnull N]` or "
            f"`{name.text}[absl_nullable N]` (AGENTS.md, 'Raw pointers carry nullability').",
        )

    def _annotation(self, i: int) -> None:
        """Reports an annotation that does not directly follow its `*` or `[`."""
        code = self.code
        previous = code[i - 1] if i > 0 else None
        if previous is not None and previous.text in ("*", "["):
            return
        following = code[i + 1 : i + 4]
        texts = [t.text for t in following]
        if (
            texts[:2] == ["std", "::"]
            and len(texts) > 2
            and texts[2] in _SMART_POINTERS
        ):
            return
        if (
            texts[:2] == ["absl", "::"]
            and len(texts) > 2
            and texts[2] in _SMART_POINTERS
        ):
            return
        if texts[:1] and texts[0] in _SMART_POINTERS:
            return
        self.add(
            code[i],
            f"misplaced `{code[i].text}`: it qualifies the pointer when it directly follows the `*` "
            f"(`Foo* {code[i].text} const p`), not elsewhere.",
        )

    def _include(self) -> None:
        used = [t for t in self.code if t.text in ANNOTATIONS]
        if not used:
            return
        without_comments, _ = cpp_source.mask(self.source)
        if _INCLUDE_LINE.search(without_comments):
            return
        self.add(
            used[0],
            f'`{used[0].text}` without `#include "{INCLUDE}"` (and a dependency on @abseil-cpp//absl/base:nullability).',
        )


def check_source(source: str, path: str = "<source>") -> list[check_types.Finding]:
    """The findings of `pointer-nullability` in C++ `source`, the file at `path`."""
    return _Finder(source, path).run()


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        scope=lint_files.Scope.FIRST_PARTY_AND_OCS2,
        check_source=check_source,
        description="every raw pointer carries absl_nonnull or absl_nullable after its `*` (AGENTS.md, 'Raw "
        "pointers carry nullability').",
        hint="Annotate each `*` from the implementation and every caller: absl_nonnull when it is dereferenced "
        "unchecked, absl_nullable when any path compares it with nullptr (AGENTS.md).",
    )
]
