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
- `absl_nonnull` on a pointer given a null value: initialized or defaulted to `nullptr`, `NULL` or `0` (`= nullptr`,
  `(nullptr)`, `{nullptr}`, `= {nullptr}`), value-initialized (`{}`, `= {}`), given a null branch of a conditional
  (`= c ? &a : nullptr`), set to null in a member initializer list (`p_(nullptr)`, `p_{}`) or by an assignment
  (`p_ = nullptr;`), or returned as null from a function declared to return `T* absl_nonnull`. A name declared both
  `absl_nonnull` and `absl_nullable` in one file is left alone;
- an annotation in a cast, `sizeof`, `alignof`, `typeid`, `decltype`, a type-trait argument or `new T*[n]`: an
  annotated cast is an assertion that silences the analyzer, not documentation;
- a declaration of more than one annotated pointer (`Foo* absl_nonnull a, * absl_nonnull b;`): one declarator per
  declaration;
- the spellings that are not used: `absl_nullability_unknown`, `ABSL_POINTERS_DEFAULT_NONNULL`, the removed
  `absl::Nonnull<>` / `absl::Nullable<>` / `absl::NullabilityUnknown<>` templates, and raw `_Nonnull` / `_Nullable` /
  `_Null_unspecified`;
- a right-aligned declarator, `Foo *a` (clang-format's fallback for `Foo *a, *b`): one declarator per declaration;
- a file that uses an annotation without `#include "absl/base/nullability.h"`.

Exempt from the annotation: casts (`static_cast<Foo*>`, C-style), `sizeof` / `alignof` / `typeid` / `decltype`,
`new T*[n]`, type-trait arguments (`std::is_pointer_v<T*>`), and everything that is not code (comments, literals,
preprocessor lines).

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
# What may stand between the `)` of a function's parameters and its body: qualifiers, specifiers, attributes' macros.
_FUNCTION_SPECIFIERS = frozenset(
    {"const", "volatile", "noexcept", "override", "final", "&", "&&", "mutable"}
)
_CAST_ANNOTATION = (
    "in a cast, a type expression, a type-trait argument or a new-expression the pointer stays unannotated: an "
    "annotated cast is an assertion that silences the analyzer, not documentation (AGENTS.md, 'Raw pointers carry "
    "nullability')"
)
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
    """True for the `*` of `(const char*)buf`: `(` type `*` `)` followed by an operand, after an operator.

    An annotation between the `*` and the `)` (`(Foo* absl_nonnull)buf`) is skipped: the cast is still one.

    Args:
      code: The code tokens of the file.
      star: The index of the `*`.

    Returns:
      Whether the `*` is the last token of a C-style cast's type.
    """
    close = star + 1
    while close < len(code) and code[close].text in ANNOTATIONS:
        close += 1
    if close + 1 >= len(code) or code[close].text != ")":
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
    following = code[close + 1]
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
        # The data members declared `T* absl_nonnull name_`, and every name declared `absl_nullable` (any scope): a
        # member set to null in an initializer list or by an assignment is reported unless the file declares the same
        # name nullable too (two classes, or a member and a local).
        self.nonnull_members: set[str] = set()
        self.nullable_names: set[str] = set()

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
        self._null_member_assignments()
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
                self._declarator(last)
            return last
        if not token.ws_before and previous.text == "(" and i >= 2:
            before = code[i - 2]
            # The return type ends in a name (`R (*)`), or in a `&`, `&&` or `*` glued to it (`R& (*)`, `R* (*)`).
            returns_type = _is_typeish(before) or (
                before.text in ("&", "&&", "*") and not before.ws_before
            )
            if previous.ws_before and returns_type:
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

    def _text(self, k: int) -> str:
        return self.code[k].text if 0 <= k < len(self.code) else ""

    def _name_end(self, k: int) -> int:
        """The index after the declarator name, possibly qualified (`Foo::get`), at code[k]; k when there is none."""
        code = self.code
        n = len(code)
        if (
            k >= n
            or code[k].kind != cpp_source.IDENTIFIER
            or code[k].text in _NOT_TYPE_WORDS
        ):
            return k
        k += 1
        while k < n:
            if code[k].text == "<" and not code[k].ws_before:
                close = cpp_source.matching_angle(code, k)
                if close >= n or self._text(close + 1) != "::":
                    break
                k = close + 1
            if (
                code[k].text == "::"
                and k + 1 < n
                and code[k + 1].kind == cpp_source.IDENTIFIER
            ):
                k += 2
                continue
            break
        return k

    def _declarator(self, last: int) -> None:
        """Judges the annotated declarator whose last `*` is code[last]: what it is given, and what it declares."""
        code = self.code
        annotation = self._text(last + 1)
        if annotation not in ("absl_nonnull", "absl_nullable"):
            return
        k = last + 2
        if self._text(k) == "const":
            k += 1
        name_end = self._name_end(k)
        has_name = name_end > k
        if has_name and name_end == k + 1 and self._text(name_end) != "(":
            if annotation == "absl_nullable":
                self.nullable_names.add(code[k].text)
            elif cpp_source.scopes(self.source)[k][-1:] == (cpp_source.CLASS_SCOPE,):
                self.nonnull_members.add(code[k].text)
        body = None
        if self._text(name_end) == "(" and has_name:
            body = self._function_body(name_end)
        elif not has_name and self._text(name_end) == "{":
            # A trailing return type: `[]() -> Foo* absl_nonnull {` or `auto f() -> Foo* absl_nonnull {`.
            body = name_end
        if (
            body is None
            and has_name
            and self._text(self._declarator_end(name_end)) == ","
            and self._enclosing_bracket(last) in ("{", "")
        ):
            self.add(
                code[k],
                f"`{code[k].text}` is declared with another declarator after it: declare one variable per "
                f"declaration (`T* {annotation} a; T* {annotation} b;`).",
            )
        if annotation != "absl_nonnull":
            return
        null = self._null_initializer(name_end)
        if null is not None:
            self.add(
                code[last + 1],
                f"`absl_nonnull` pointer initialized with `{null.text}`: it is nullable (`absl_nullable`), or it needs "
                "a non-null value.",
            )
        if body is not None:
            self._null_returns(body)

    def _null_initializer(self, k: int) -> cpp_source.Token | None:
        """The null that the initializer starting at code[k] gives a pointer (`= nullptr`, `{}`, `(0)`, ...), or None."""
        if self._text(k) == "=":
            k += 1
            if self._text(k) in _NULL_VALUES:
                return self.code[k]
            if self._text(k) == "{":
                return self._null_braces(k)
            return self._null_branch(k)
        if self._text(k) == "(":
            if self._text(k + 1) in _NULL_VALUES and self._text(k + 2) == ")":
                return self.code[k + 1]
            return None
        if self._text(k) == "{":
            return self._null_braces(k)
        return None

    def _null_braces(self, k: int) -> cpp_source.Token | None:
        """For the `{` at code[k]: the null of `{}` (value-initialization) or `{nullptr}`, or None."""
        if self._text(k + 1) == "}":
            return self.code[k]
        if self._text(k + 1) in _NULL_VALUES and self._text(k + 2) == "}":
            return self.code[k + 1]
        return None

    def _null_branch(self, k: int) -> cpp_source.Token | None:
        """The null branch of a conditional initializer starting at code[k] (`c ? &a : nullptr`), or None."""
        code = self.code
        n = len(code)
        question = -1
        colon = -1
        nested = 0
        j = k
        while j < n:
            text = code[j].text
            if text in ("(", "[", "{"):
                j = cpp_source.matching(code, j) + 1
                continue
            if text in (")", "]", "}", ";", ","):
                break
            if text == "?":
                if question < 0:
                    question = j
                else:
                    nested += 1
            elif text == ":" and question >= 0:
                if nested:
                    nested -= 1
                elif colon < 0:
                    colon = j
            j += 1
        if question < 0 or colon < 0:
            return None
        if colon == question + 2 and code[question + 1].text in _NULL_VALUES:
            return code[question + 1]
        if j == colon + 2 and code[colon + 1].text in _NULL_VALUES:
            return code[colon + 1]
        return None

    def _declarator_end(self, k: int) -> int:
        """The index of the `,`, `;` or closing bracket that ends the declarator whose initializer starts at code[k]."""
        code = self.code
        n = len(code)
        while self._text(k) == "[":
            k = cpp_source.matching(code, k) + 1
        if self._text(k) in ("(", "{"):
            return cpp_source.matching(code, k) + 1
        if self._text(k) != "=":
            return k
        k += 1
        while k < n:
            text = code[k].text
            if text in ("(", "[", "{"):
                k = cpp_source.matching(code, k) + 1
                continue
            if (
                text == "<"
                and not code[k].ws_before
                and code[k - 1].kind == cpp_source.IDENTIFIER
            ):
                close = cpp_source.matching_angle(code, k)
                if close < n:
                    k = close + 1
                    continue
            if text in (",", ";", ")", "]", "}"):
                return k
            k += 1
        return n

    def _enclosing_bracket(self, index: int) -> str:
        """The innermost unclosed `(`, `[`, `{` or template `<` before code[index]; "" at file scope."""
        code = self.code
        depth = 0
        k = index - 1
        while k >= 0:
            text = code[k].text
            if text in (")", "]", "}"):
                depth += 1
            elif text in ("(", "[", "{"):
                if depth == 0:
                    return text
                depth -= 1
            elif (
                text == "<"
                and depth == 0
                and (not code[k].ws_before or self._text(k - 1) == "template")
            ):
                close = cpp_source.matching_angle(code, k)
                if close >= index:
                    return text
            k -= 1
        return ""

    def _function_body(self, k: int) -> int | None:
        """For the `(` of a parameter list at code[k], the index of the `{` of the function's body, or None."""
        code = self.code
        n = len(code)
        j = cpp_source.matching(code, k) + 1
        while j < n:
            text = code[j].text
            if text in _FUNCTION_SPECIFIERS or text in ANNOTATIONS:
                j += 1
            elif code[j].kind == cpp_source.IDENTIFIER:
                # A specifier macro, `noexcept(...)`, or a name of a trailing return type.
                j += 1
                if self._text(j) == "(":
                    j = cpp_source.matching(code, j) + 1
            elif text in ("->", "::", "*"):
                j += 1
            else:
                break
        return j if self._text(j) == "{" else None

    def _is_lambda_body(self, brace: int) -> bool:
        """True when the `{` at code[brace] opens a lambda's body: `[...] {`, or `[...](...) [specifiers] {`."""
        code = self.code
        k = brace - 1
        while k >= 0 and (
            code[k].kind == cpp_source.IDENTIFIER
            or code[k].text in ("->", "::", "*", "&", "&&", "<", ">")
        ):
            k -= 1
        if self._text(k) == "]":
            return True
        if self._text(k) != ")":
            return False
        depth = 0
        while k >= 0:
            if code[k].text == ")":
                depth += 1
            elif code[k].text == "(":
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        return self._text(k - 1) == "]"

    def _null_returns(self, body: int) -> None:
        """Reports `return nullptr;` in the body, opened at code[body], of a function that returns `T* absl_nonnull`."""
        code = self.code
        end = cpp_source.matching(code, body)
        j = body + 1
        while j < end:
            if code[j].text == "{" and self._is_lambda_body(j):
                j = cpp_source.matching(code, j) + 1
                continue
            if (
                code[j].text == "return"
                and self._text(j + 1) in _NULL_VALUES
                and self._text(j + 2) == ";"
            ):
                self.add(
                    code[j + 1],
                    f"`return {code[j + 1].text};` from a function that returns `absl_nonnull`: the return type is "
                    "`absl_nullable`, or the function needs a non-null value.",
                )
            j += 1

    def _in_member_initializer_list(self, name: int) -> bool:
        """True when the name at code[name] begins an entry of a constructor's member initializer list."""
        code = self.code
        k = name - 1
        while k >= 0:
            text = code[k].text
            if text == ":":
                # The `:` after a constructor's parameters: `Foo(int a) : p_(...)`, `Foo() noexcept : ...`.
                return self._text(k - 1) in (")", "noexcept")
            if text in (")", "}"):
                # A previous entry's arguments: `a_(x), b_{y}`.
                depth = 0
                while k >= 0:
                    if code[k].text in (")", "}"):
                        depth += 1
                    elif code[k].text in ("(", "{"):
                        depth -= 1
                        if depth == 0:
                            break
                    k -= 1
                k -= 1
                continue
            if (
                text == ","
                or code[k].kind == cpp_source.IDENTIFIER
                or text in ("::", "<", ">")
            ):
                k -= 1
                continue
            return False
        return False

    def _null_member_assignments(self) -> None:
        """Reports an `absl_nonnull` data member set to null: `p_(nullptr)`, `p_{}` in an initializer list, `p_ = nullptr;`."""
        members = self.nonnull_members - self.nullable_names
        if not members:
            return
        code = self.code
        for j, token in enumerate(code):
            if token.kind != cpp_source.IDENTIFIER or token.text not in members:
                continue
            previous = self._text(j - 1)
            following = self._text(j + 1)
            null = None
            if (
                following == "="
                and previous not in ANNOTATIONS
                and previous not in ("const", "*")
            ):
                if self._text(j + 2) in _NULL_VALUES and self._text(j + 3) in (
                    ";",
                    ",",
                    ")",
                    "}",
                ):
                    null = code[j + 2]
            elif (
                previous in (":", ",")
                and following in ("(", "{")
                and self._in_member_initializer_list(j)
            ):
                if following == "(":
                    if self._text(j + 2) in _NULL_VALUES and self._text(j + 3) == ")":
                        null = code[j + 2]
                else:
                    null = self._null_braces(j + 1)
            if null is not None:
                self.add(
                    null,
                    f"`{token.text}` is an `absl_nonnull` member and is set to `{null.text}`: it is nullable "
                    "(`absl_nullable`), or it needs a non-null value.",
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
        if previous is not None and previous.text == "*":
            star = i - 1
            # A cast's `*` is glued to its type (`(const Foo* absl_nonnull)buf`); `R (*absl_nonnull)(A)` is a type.
            typed = star > 0 and (
                _is_typeish(code[star - 1])
                or code[star - 1].text in ("const", "volatile")
            )
            if _is_exempt(code, star) or (typed and _is_c_style_cast(code, star)):
                self.add(code[i], f"`{code[i].text}` {_CAST_ANNOTATION}.")
            return
        if previous is not None and previous.text == "[":
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
