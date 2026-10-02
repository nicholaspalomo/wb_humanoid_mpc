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

"""Token checks for the rules of the Google C++ Style Guide that cpplint and clang-tidy do not cover.

Each check names the guide section it enforces:

- `float-literal` (Floating-point literals): a floating literal has digits on both sides of its radix point - `1.0e-6`,
  `1.0f`, `0.5`, not `1e-6`, `1.f` or `.5`. Fixed by `make format`.
- `postfix-increment` (Preincrement and predecrement): `++i`, not `i++`, where the value is unused. Fixed by
  `make format`.
- `std-integer-type` (Integer types): `int64_t` and `size_t`, without the `std::` prefix. Fixed by `make format`.
- `forbidden-construct` (Nonstandard extensions, Exceptions to naming rules, ...): `long double`, `std::auto_ptr`,
  user-defined literals, inline namespaces, `decltype(auto)`, coroutines, C++20 modules, `<ratio>` / `<cfenv>`,
  `alloca`, statement expressions, `__attribute__` / `__builtin_`, `#pragma` other than `once` and `GCC diagnostic`, a
  namespace- or class-scope `thread_local` without `constinit`, and `strtok`. `<filesystem>` is allowed (AGENTS.md).
- `exceptions` (Exceptions): `throw`, `try` and `catch` outside tests. A boundary with a library that throws keeps one
  `try` / `catch` with `// NOLINT(exceptions): <reason>`.
- `rtti` (Run-time type information): `typeid` and `dynamic_cast` outside tests.
- `ctad` (Class template argument deduction; stricter, the explicit-types rule): `std::vector v = {...}`; write the
  template arguments. `absl::Cleanup` is exempt.
- `macro-naming` (Preprocessor macros): a header macro without the library's prefix, a `.cpp` macro that is not
  `#undef`ined at the end of the file, and a redefinition of an Abseil macro name.
- `static-storage` (Static and global variables): a non-trivially-destructible object with static storage duration.
  Use `absl::NoDestructor<T>`, `constexpr`, or `static const T& x = *new T(...)`.
- `class-comment` (Class comments): a namespace-scope class or struct defined in a header without a comment above it.
- `if-else-braces` (Looping and branching statements): an `if ... else` without braces.
"""

import re

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

CPP = frozenset({check_types.Language.CPP})


def _finding(
    path: str, token: cpp_source.Token, name: str, message: str
) -> check_types.Finding:
    return check_types.Finding(path, token.line, token.col, name, message)


def _replace(source: str, edits: list[tuple[int, int, str]]) -> str:
    """`source` with each (start, end, text) edit applied; the edits do not overlap."""
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source


# ----------------------------------------------------------------------------------------------------------------------
# float-literal
# ----------------------------------------------------------------------------------------------------------------------
FLOAT_LITERAL = "float-literal"
_DECIMAL = re.compile(
    r"^(?P<integer>[\d']*)(?P<point>\.?)(?P<fraction>[\d']*)(?P<exponent>[eE][+-]?\d+)?(?P<suffix>[a-zA-Z_]*)$"
)


def _fixed_float(text: str) -> str | None:
    """The literal written with digits on both sides of a radix point, or None when it is fine or not a float."""
    if text.lower().startswith(("0x", "0b")):
        return None
    match = _DECIMAL.match(text)
    if not match or (not match.group("point") and not match.group("exponent")):
        return None
    suffix = match.group("suffix")
    if not match.group("point") and suffix.lower() not in ("", "f", "l"):
        return None  # an integer with a user-defined literal suffix
    integer = match.group("integer") or "0"
    fraction = match.group("fraction") or "0"
    fixed = f"{integer}.{fraction}{match.group('exponent') or ''}{suffix}"
    return None if fixed == text else fixed


def _float_literals(source: str) -> list[tuple[cpp_source.Token, str]]:
    return [
        (token, fixed)
        for token in cpp_source.code_tokens(source)
        if token.kind == cpp_source.NUMBER
        and (fixed := _fixed_float(token.text)) is not None
    ]


def check_float_literal(source: str, path: str) -> list[check_types.Finding]:
    """Floating literals without digits on both sides of the radix point."""
    return [
        _finding(
            path,
            token,
            FLOAT_LITERAL,
            f"floating literal `{token.text}`: write `{fixed}`, with digits on both sides of the radix point "
            "(Google C++ style, Floating-point literals).",
        )
        for token, fixed in _float_literals(source)
    ]


def fix_float_literal(source: str, path: str) -> str:
    """Rewrites each floating literal with digits on both sides of its radix point."""
    del path  # Unused.
    literals = _float_literals(source)
    if not literals:
        return source
    starts = cpp_source.line_starts(source)
    edits = []
    for token, fixed in literals:
        start = starts[token.line - 1] + token.col - 1
        edits.append((start, start + len(token.text), fixed))
    return _replace(source, edits)


# ----------------------------------------------------------------------------------------------------------------------
# postfix-increment
# ----------------------------------------------------------------------------------------------------------------------
POSTFIX_INCREMENT = "postfix-increment"
_STATEMENT_STARTS = frozenset({";", "{", "}", "else", "do", ":"})


def _operand_start(code: tuple[cpp_source.Token, ...], end: int) -> int | None:
    """The index of the first token of the operand that ends at code[end] (`a.b->c[i]`, `(*it)`), or None."""
    k = end
    while k >= 0:
        token = code[k]
        if token.text in ("]", ")"):
            depth = 0
            while k >= 0:
                if code[k].text in ("]", ")"):
                    depth += 1
                elif code[k].text in ("[", "("):
                    depth -= 1
                    if depth == 0:
                        break
                k -= 1
            if k < 0:
                return None
            if code[k].text == "(" and (
                k == 0
                or code[k - 1].kind != cpp_source.IDENTIFIER
                and code[k - 1].text not in ("]", ")")
            ):
                return k  # a parenthesized operand, `(*it)++`
            k -= 1
            continue
        if token.kind == cpp_source.IDENTIFIER and token.text not in (
            "return",
            "else",
            "do",
            "case",
        ):
            if k > 0 and code[k - 1].text in (".", "->", "::"):
                k -= 2
                continue
            return k
        return None
    return None


def _in_for_increment(code: tuple[cpp_source.Token, ...], index: int) -> bool:
    """True when code[index] is in the third clause of a `for (...; ...; ...)` header."""
    depth = 0
    semicolons = 0
    k = index - 1
    while k >= 0:
        text = code[k].text
        if text in (")", "]", "}"):
            depth += 1
        elif text in ("[", "{"):
            if depth == 0:
                return False
            depth -= 1
        elif text == "(":
            if depth == 0:
                return semicolons == 2 and k > 0 and code[k - 1].text == "for"
            depth -= 1
        elif text == ";" and depth == 0:
            semicolons += 1
        k -= 1
    return False


def _postfix_increments(source: str) -> list[tuple[cpp_source.Token, int]]:
    """The `x++` / `x--` tokens whose value is unused, with the index of their operand's first token."""
    code = cpp_source.code_tokens(source)
    found = []
    for index, token in enumerate(code):
        if (
            token.text not in ("++", "--")
            or index == 0
            or token.ws_before
            or index + 1 >= len(code)
        ):
            continue
        after = code[index + 1].text
        start = _operand_start(code, index - 1)
        if start is None:
            continue
        before = code[start - 1].text if start > 0 else ";"
        if after == ";" and (
            before in _STATEMENT_STARTS
            or (before == ")" and _closes_condition(code, start - 1))
        ):
            found.append((token, start))
        elif (
            after in (")", ",")
            and before in (";", ",")
            and _in_for_increment(code, index)
        ):
            found.append((token, start))
    return found


def _closes_condition(code: tuple[cpp_source.Token, ...], close: int) -> bool:
    """True when the `)` at code[close] closes the condition of an `if`, `while` or `for`."""
    depth = 0
    k = close
    while k >= 0:
        if code[k].text == ")":
            depth += 1
        elif code[k].text == "(":
            depth -= 1
            if depth == 0:
                return k > 0 and code[k - 1].text in ("if", "while", "for")
        k -= 1
    return False


def check_postfix_increment(source: str, path: str) -> list[check_types.Finding]:
    """Postfix increments and decrements whose value is unused."""
    return [
        _finding(
            path,
            token,
            POSTFIX_INCREMENT,
            f"postfix `{token.text}` whose value is unused: write the prefix form (Google C++ style, Preincrement "
            "and predecrement).",
        )
        for token, _ in _postfix_increments(source)
    ]


def fix_postfix_increment(source: str, path: str) -> str:
    """Moves each unused postfix `++` / `--` in front of its operand."""
    del path  # Unused.
    found = _postfix_increments(source)
    if not found:
        return source
    code = cpp_source.code_tokens(source)
    starts = cpp_source.line_starts(source)

    def offset(token: cpp_source.Token) -> int:
        return starts[token.line - 1] + token.col - 1

    edits = []
    for token, start in found:
        operator_offset = offset(token)
        edits.append((operator_offset, operator_offset + 2, ""))
        operand_offset = offset(code[start])
        edits.append((operand_offset, operand_offset, token.text))
    return _replace(source, edits)


# ----------------------------------------------------------------------------------------------------------------------
# std-integer-type
# ----------------------------------------------------------------------------------------------------------------------
STD_INTEGER_TYPE = "std-integer-type"
_INTEGER_TYPES = re.compile(
    r"^(u?int(8|16|32|64|max|ptr)_t|u?int_(fast|least)(8|16|32|64)_t|size_t|ptrdiff_t|ssize_t|max_align_t)$"
)


def _std_integer_types(source: str) -> list[tuple[cpp_source.Token, cpp_source.Token]]:
    """The (`std`, type) token pairs of `std::int64_t` and the like; `::std::size_t` included."""
    code = cpp_source.code_tokens(source)
    found = []
    for index in range(len(code) - 2):
        if (
            code[index].text == "std"
            and code[index + 1].text == "::"
            and _INTEGER_TYPES.match(code[index + 2].text)
            and not (index > 0 and code[index - 1].text in (".", "->"))
            and not (
                index > 1
                and code[index - 1].text == "::"
                and code[index - 2].kind == cpp_source.IDENTIFIER
            )
        ):
            found.append((code[index], code[index + 2]))
    return found


def check_std_integer_type(source: str, path: str) -> list[check_types.Finding]:
    """Integer types written with the `std::` prefix."""
    return [
        _finding(
            path,
            std,
            STD_INTEGER_TYPE,
            f"`std::{name.text}`: write `{name.text}` (Google C++ style, Integer types: omit the std:: prefix).",
        )
        for std, name in _std_integer_types(source)
    ]


def fix_std_integer_type(source: str, path: str) -> str:
    """Removes the `std::` (and a leading `::`) in front of each integer type."""
    del path  # Unused.
    found = _std_integer_types(source)
    if not found:
        return source
    starts = cpp_source.line_starts(source)
    edits = []
    for std, name in found:
        start = starts[std.line - 1] + std.col - 1
        end = starts[name.line - 1] + name.col - 1
        if start >= 2 and source[start - 2 : start] == "::":
            start -= 2
        edits.append((start, end, ""))
    return _replace(source, edits)


# ----------------------------------------------------------------------------------------------------------------------
# forbidden-construct
# ----------------------------------------------------------------------------------------------------------------------
FORBIDDEN_CONSTRUCT = "forbidden-construct"
_FORBIDDEN_HEADERS = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*[<"](ratio|cfenv|fenv\.h)[>"]', re.MULTILINE
)
_PRAGMA = re.compile(r"^[ \t]*#[ \t]*pragma[ \t]+(?P<rest>.*)$", re.MULTILINE)
_STANDARD_NUMBER_SUFFIX = re.compile(
    r"^(?:[uU](?:ll|LL|l|L|z|Z)?|(?:ll|LL|l|L|z|Z)[uU]?|[fFlL])?$"
)
_NUMBER_BODY = re.compile(
    r"^(?:0[xX][0-9a-fA-F'.]+(?:[pP][+-]?\d+)?|0[bB][01']+|[\d'.]+(?:[eE][+-]?\d+)?)"
)
_RATIOS = frozenset(
    {"ratio", "milli", "micro", "nano", "pico", "kilo", "mega", "giga", "centi", "deci"}
)


def _is_statement_expression(code: tuple[cpp_source.Token, ...], brace: int) -> bool:
    """True when the `{` at code[brace], right after a `(`, holds statements (`({ int x = f(); x; })`)."""
    close = cpp_source.matching(code, brace)
    depth = 0
    for k in range(brace + 1, min(close, len(code))):
        text = code[k].text
        if text in ("(", "[", "{"):
            depth += 1
        elif text in (")", "]", "}"):
            depth -= 1
        elif text == ";" and depth == 0:
            return True
    return False


def check_forbidden_construct(source: str, path: str) -> list[check_types.Finding]:
    """Constructs the guide does not allow."""
    code = cpp_source.code_tokens(source)
    scopes = cpp_source.scopes(source)
    findings = []

    def add(token: cpp_source.Token, what: str) -> None:
        findings.append(
            _finding(
                path, token, FORBIDDEN_CONSTRUCT, f"{what} (Google C++ Style Guide)."
            )
        )

    for index, token in enumerate(code):
        text = token.text
        following = code[index + 1].text if index + 1 < len(code) else ""
        previous = code[index - 1].text if index > 0 else ""
        if text == "long" and following == "double":
            add(token, "`long double` is not portable: use `double`")
        elif text == "auto_ptr" and previous == "::":
            add(token, "`std::auto_ptr` was removed: use `std::unique_ptr`")
        elif text == "operator" and following.startswith('""'):
            add(token, "user-defined literals are not used")
        elif (
            text == "namespace"
            and previous == "using"
            and index + 2 < len(code)
            and any(
                t.text.endswith("literals")
                for t in code[index + 1 : index + 8]
                if t.text != ";"
            )
        ):
            add(
                token,
                "user-defined literals are not used (`using namespace ...literals`)",
            )
        elif token.kind == cpp_source.NUMBER and not _STANDARD_NUMBER_SUFFIX.match(
            _NUMBER_BODY.sub("", text)
        ):
            add(token, f"user-defined literal `{text}`: use a function or a constant")
        elif text == "inline" and following == "namespace":
            add(token, "`inline namespace` is not used")
        elif (
            text == "decltype"
            and following == "("
            and index + 3 < len(code)
            and code[index + 2].text == "auto"
        ):
            add(token, "`decltype(auto)` is not used: spell the type")
        elif text in ("co_await", "co_yield", "co_return"):
            add(token, f"coroutines (`{text}`) are not used")
        elif (
            text in ("import", "module")
            and previous in (";", "}", "", "export")
            and (
                index + 1 < len(code)
                and (
                    code[index + 1].kind == cpp_source.IDENTIFIER
                    or following in ("<", ";", ":")
                )
            )
        ):
            add(token, f"C++20 modules (`{text}`) are not used")
        elif (
            text in _RATIOS
            and previous == "::"
            and index >= 2
            and code[index - 2].text == "std"
        ):
            add(
                token,
                f"`std::{text}` and <ratio> are not used: use absl::Duration or a plain constant",
            )
        elif text == "alloca" and following == "(":
            add(token, "`alloca` is not used: use a std::vector or absl::InlinedVector")
        elif (
            text == "("
            and following == "{"
            and not re.match(
                r"^[A-Z][A-Z0-9_]*$", previous
            )  # a block as a macro argument: EXPECT_NO_THROW({ ... })
            and _is_statement_expression(code, index + 1)
        ):
            add(token, "statement expressions `({ ... })` are a GNU extension")
        elif text == "__attribute__" or text.startswith("__builtin_"):
            add(
                token,
                f"`{text}` is a compiler extension: use the standard attribute or an Abseil macro",
            )
        elif text == "thread_local" and "constinit" not in [
            t.text for t in code[max(0, index - 3) : index + 4]
        ]:
            if all(
                kind in (cpp_source.NAMESPACE_SCOPE, cpp_source.CLASS_SCOPE)
                for kind in scopes[index]
            ):
                add(
                    token,
                    "a namespace- or class-scope `thread_local` must be `constinit`",
                )
        elif text == "strtok" and following == "(":
            add(token, "`strtok` is not reentrant: use absl::StrSplit")
    without_comments, _ = cpp_source.mask(source)
    starts = cpp_source.line_starts(without_comments)
    for match in _FORBIDDEN_HEADERS.finditer(without_comments):
        line = without_comments.count("\n", 0, match.start()) + 1
        findings.append(
            check_types.Finding(
                path,
                line,
                1,
                FORBIDDEN_CONSTRUCT,
                f"<{match.group(1)}> is not used (Google C++ Style Guide).",
            )
        )
    for match in _PRAGMA.finditer(without_comments):
        rest = match.group("rest").strip()
        if rest == "once" or rest.startswith(("GCC diagnostic", "clang diagnostic")):
            continue
        line = without_comments.count("\n", 0, match.start()) + 1
        column = match.start() - starts[line - 1] + 1
        findings.append(
            check_types.Finding(
                path,
                line,
                max(column, 1),
                FORBIDDEN_CONSTRUCT,
                f"`#pragma {rest.split()[0] if rest else ''}`: only `#pragma once` and `#pragma GCC diagnostic` are "
                "used (Google C++ Style Guide).",
            )
        )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
# exceptions, rtti
# ----------------------------------------------------------------------------------------------------------------------
EXCEPTIONS = "exceptions"
RTTI = "rtti"


def check_exceptions(source: str, path: str) -> list[check_types.Finding]:
    """`throw`, `try` and `catch`."""
    return [
        _finding(
            path,
            token,
            EXCEPTIONS,
            f"`{token.text}`: first-party code returns absl::Status / absl::StatusOr instead of throwing (Google C++ "
            "style, Exceptions; ToTW #76). A boundary with a throwing library converts once, with "
            "`// NOLINT(exceptions): <reason>`.",
        )
        for token in cpp_source.code_tokens(source)
        if token.text in ("throw", "try", "catch")
        and token.kind == cpp_source.IDENTIFIER
    ]


def check_rtti(source: str, path: str) -> list[check_types.Finding]:
    """`typeid` and `dynamic_cast`."""
    return [
        _finding(
            path,
            token,
            RTTI,
            f"`{token.text}`: avoid run-time type information; use a virtual function, a visitor or a type tag "
            "(Google C++ style, Run-time type information).",
        )
        for token in cpp_source.code_tokens(source)
        if token.text in ("typeid", "dynamic_cast")
    ]


# ----------------------------------------------------------------------------------------------------------------------
# ctad
# ----------------------------------------------------------------------------------------------------------------------
CTAD = "ctad"
_CTAD_TEMPLATES = {
    "std": frozenset(
        {
            "array",
            "pair",
            "tuple",
            "vector",
            "deque",
            "list",
            "map",
            "set",
            "optional",
            "variant",
            "function",
            "unique_ptr",
            "shared_ptr",
            "span",
            "basic_string_view",
            "lock_guard",
            "scoped_lock",
            "unique_lock",
        }
    ),
    "absl": frozenset(
        {"flat_hash_map", "flat_hash_set", "InlinedVector", "FixedArray", "Span"}
    ),
    "Eigen": frozenset({"Matrix", "Array"}),
}


def check_ctad(source: str, path: str) -> list[check_types.Finding]:
    """Declarations of a known class template without its template arguments."""
    code = cpp_source.code_tokens(source)
    findings = []
    for index in range(len(code) - 4):
        namespace = code[index].text
        if namespace not in _CTAD_TEMPLATES or code[index + 1].text != "::":
            continue
        if index > 0 and code[index - 1].text in ("::", ".", "->"):
            continue
        name = code[index + 2]
        variable = code[index + 3]
        if (
            name.text not in _CTAD_TEMPLATES[namespace]
            or variable.kind != cpp_source.IDENTIFIER
        ):
            continue
        if code[index + 4].text not in ("=", "{", "("):
            continue
        findings.append(
            _finding(
                path,
                name,
                CTAD,
                f"`{namespace}::{name.text} {variable.text}` deduces its template arguments: write them out "
                "(AGENTS.md: explicit types; Google C++ style, Class template argument deduction).",
            )
        )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
# macro-naming
# ----------------------------------------------------------------------------------------------------------------------
MACRO_NAMING = "macro-naming"
MACRO_PREFIXES = ("ROBOT_IPC_", "NPROTO_", "HUMANOID_", "WBMPC_")
# The status macros keep the names Google uses (AGENTS.md); STATUS_MACROS_ and ASSIGN_OR_RETURN_IMPL are their helpers.
_ALLOWED_MACROS = re.compile(
    r"^(RETURN_IF_ERROR|ASSIGN_OR_RETURN|ASSIGN_OR_RETURN_IMPL|STATUS_MACROS_\w+)$"
)
_ABSEIL_MACROS = frozenset(
    {"CHECK", "LOG", "DCHECK", "VLOG", "QCHECK", "PLOG", "LOG_IF", "CHECK_OK"}
)
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(?P<name>\w+)", re.MULTILINE)
_UNDEF = re.compile(r"^[ \t]*#[ \t]*undef[ \t]+(?P<name>\w+)", re.MULTILINE)


def check_macro_naming(source: str, path: str) -> list[check_types.Finding]:
    """Macro names: the library prefix in headers, an `#undef` in .cpp files, no Abseil names."""
    without_comments, _ = cpp_source.mask(source)
    header = path.endswith(lint_files.CPP_HEADER_EXTENSIONS)
    undefined_after: dict[str, int] = {}
    for match in _UNDEF.finditer(without_comments):
        undefined_after[match.group("name")] = match.start()
    findings = []
    for match in _DEFINE.finditer(without_comments):
        name = match.group("name")
        line = without_comments.count("\n", 0, match.start()) + 1
        column = match.start("name") - without_comments.rfind(
            "\n", 0, match.start("name")
        )
        message = None
        if name in _ABSEIL_MACROS:
            message = (
                f"`{name}` redefines an Abseil macro name: give it the library's prefix"
            )
        elif (
            header
            and not _ALLOWED_MACROS.match(name)
            and not name.startswith(MACRO_PREFIXES)
        ):
            message = (
                f"header macro `{name}` without the library's prefix ({', '.join(MACRO_PREFIXES)}); headers use "
                "#pragma once, not include guards"
            )
        elif not header and undefined_after.get(name, -1) < match.start():
            message = f"macro `{name}` defined in a .cpp file is not `#undef`ined at the end of it"
        if message:
            findings.append(
                check_types.Finding(
                    path,
                    line,
                    column,
                    MACRO_NAMING,
                    message + " (Google C++ style, Preprocessor macros).",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
# static-storage
# ----------------------------------------------------------------------------------------------------------------------
STATIC_STORAGE = "static-storage"
_NON_TRIVIAL = {
    "std": frozenset(
        {
            "string",
            "vector",
            "map",
            "multimap",
            "set",
            "multiset",
            "unordered_map",
            "unordered_set",
            "unordered_multimap",
            "unordered_multiset",
            "deque",
            "list",
            "forward_list",
            "function",
            "unique_ptr",
            "shared_ptr",
            "regex",
            "wstring",
        }
    ),
    "absl": frozenset(
        {
            "flat_hash_map",
            "flat_hash_set",
            "node_hash_map",
            "node_hash_set",
            "btree_map",
            "btree_set",
            "btree_multimap",
            "btree_multiset",
            "InlinedVector",
            "FixedArray",
        }
    ),
}
_STATIC_SPECIFIERS = frozenset(
    {"static", "inline", "const", "constexpr", "thread_local", "extern", "constinit"}
)


def _qualified_type(
    code: tuple[cpp_source.Token, ...], index: int
) -> tuple[str, str] | None:
    """(`std`, `string`) for a `[::]std::string` at code[index], else None."""
    if index < len(code) and code[index].text == "::":
        index += 1
    if (
        index + 2 < len(code)
        and code[index].text in _NON_TRIVIAL
        and code[index + 1].text == "::"
    ):
        return code[index].text, code[index + 2].text
    return None


def check_static_storage(source: str, path: str) -> list[check_types.Finding]:
    """Non-trivially-destructible objects with static storage duration."""
    code = cpp_source.code_tokens(source)
    findings = []
    for statement in cpp_source.statements(source):
        if statement.body is not None:
            continue
        tokens = code[statement.start : statement.end]
        texts = [t.text for t in tokens]
        if (
            not texts
            or texts[0]
            in ("using", "typedef", "template", "friend", "return", "namespace")
            or "extern" in texts
        ):
            continue
        specifiers = set()
        k = 0
        while k < len(texts) and texts[k] in _STATIC_SPECIFIERS:
            specifiers.add(texts[k])
            k += 1
        if "constexpr" in specifiers:
            continue
        namespace_scope = cpp_source.at_namespace_scope(statement.scopes)
        innermost = (
            statement.scopes[-1] if statement.scopes else cpp_source.NAMESPACE_SCOPE
        )
        if not (
            namespace_scope or "static" in specifiers or "thread_local" in specifiers
        ):
            continue
        if innermost == cpp_source.ENUM_SCOPE:
            continue
        found = _qualified_type(tokens, k)
        if found is None or found[1] not in _NON_TRIVIAL[found[0]]:
            continue
        # The declarator: past the template arguments, a name; a reference (`= *new T`) and functions are fine.
        j = k + (1 if texts[k] == "::" else 0) + 3
        if j < len(texts) and texts[j] == "<":
            j = cpp_source.matching_angle(tokens, j) + 1
        if j >= len(texts) or texts[j] in ("&", "&&", "*", "::", "(", ")"):
            continue
        if tokens[j].kind != cpp_source.IDENTIFIER:
            continue
        if j + 1 < len(texts) and texts[j + 1] == "(":
            inside = texts[j + 2 : j + 3]
            if (
                not inside
                or inside[0] == ")"
                or tokens[j + 2].kind == cpp_source.IDENTIFIER
            ):
                continue  # a function declaration
        findings.append(
            _finding(
                path,
                tokens[j],
                STATIC_STORAGE,
                f"`{found[0]}::{found[1]}` with static storage duration is not trivially destructible: use "
                "absl::NoDestructor<T>, a constexpr type (`inline constexpr char kName[]`), or "
                "`static const T& x = *new T(...)` (Google C++ style, Static and global variables).",
            )
        )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
# class-comment
# ----------------------------------------------------------------------------------------------------------------------
CLASS_COMMENT = "class-comment"


def check_class_comment(source: str, path: str) -> list[check_types.Finding]:
    """Namespace-scope class and struct definitions in a header without a comment directly above them."""
    if not path.endswith(lint_files.CPP_HEADER_EXTENSIONS):
        return []
    code = cpp_source.code_tokens(source)
    every = cpp_source.tokenize(source)
    position = {(t.line, t.col): i for i, t in enumerate(every)}
    findings = []
    for statement in cpp_source.statements(source):
        if (
            statement.body != cpp_source.CLASS_SCOPE
            or not cpp_source.at_namespace_scope(statement.scopes)
        ):
            continue
        tokens = code[statement.start : statement.end]
        key = next(
            (
                k
                for k, t in enumerate(tokens)
                if t.text in ("class", "struct", "union")
                and (k == 0 or tokens[k - 1].text not in ("<", ",", "enum"))
            ),
            None,
        )
        if key is None or key + 1 >= len(tokens):
            continue
        name = tokens[key + 1]
        first = tokens[0]
        index = position.get((first.line, first.col))
        previous = every[index - 1] if index else None
        if previous is not None and previous.kind == cpp_source.COMMENT:
            end_line = previous.line + previous.text.count("\n")
            if end_line >= first.line - 1:
                continue
        findings.append(
            _finding(
                path,
                name,
                CLASS_COMMENT,
                f"`{tokens[key].text} {name.text}` has no comment: say what it is for, how to use it and whether it is "
                "thread-safe (Google C++ style, Class comments).",
            )
        )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
# if-else-braces
# ----------------------------------------------------------------------------------------------------------------------
IF_ELSE_BRACES = "if-else-braces"


def _statement_end(code: tuple[cpp_source.Token, ...], start: int) -> int:
    """The index of the `;` that ends the braceless statement starting at code[start]."""
    k = start
    while k < len(code):
        text = code[k].text
        if text in ("(", "[", "{"):
            k = cpp_source.matching(code, k) + 1
            continue
        if text == ";":
            return k
        k += 1
    return len(code)


def check_if_else_braces(source: str, path: str) -> list[check_types.Finding]:
    """`if ... else` without braces."""
    code = cpp_source.code_tokens(source)
    findings = []
    for index, token in enumerate(code):
        following = code[index + 1].text if index + 1 < len(code) else ""
        if token.text == "else" and following not in ("{", "if"):
            findings.append(
                _finding(
                    path,
                    token,
                    IF_ELSE_BRACES,
                    "`else` without braces: an `if ... else` always has braces (AGENTS.md; Google C++ style, Looping and "
                    "branching statements).",
                )
            )
        elif token.text == "if" and following in ("(", "constexpr"):
            open_paren = index + 1 if following == "(" else index + 2
            close = cpp_source.matching(code, open_paren)
            if close + 1 >= len(code) or code[close + 1].text == "{":
                continue
            end = _statement_end(code, close + 1)
            if end + 1 < len(code) and code[end + 1].text == "else":
                findings.append(
                    _finding(
                        path,
                        token,
                        IF_ELSE_BRACES,
                        "`if` without braces before an `else`: an `if ... else` always has braces (AGENTS.md).",
                    )
                )
    return findings


NON_TEST = lint_files.Scope.FIRST_PARTY_NON_TEST
FIRST_PARTY = lint_files.Scope.FIRST_PARTY

CHECKS = [
    check_types.Check(
        name=FLOAT_LITERAL,
        languages=CPP,
        scope=FIRST_PARTY,
        check_source=check_float_literal,
        fix_source=fix_float_literal,
        fixed_by_format=True,
        description="floating literals have digits on both sides of the radix point (G: Floating-point literals).",
        hint="`make format` (or lint_code --fix --only float-literal) rewrites them.",
    ),
    check_types.Check(
        name=POSTFIX_INCREMENT,
        languages=CPP,
        scope=FIRST_PARTY,
        check_source=check_postfix_increment,
        fix_source=fix_postfix_increment,
        fixed_by_format=True,
        description="`++i`, not `i++`, where the value is unused (G: Preincrement and predecrement).",
        hint="`make format` (or lint_code --fix --only postfix-increment) rewrites them.",
    ),
    check_types.Check(
        name=STD_INTEGER_TYPE,
        languages=CPP,
        scope=FIRST_PARTY,
        check_source=check_std_integer_type,
        fix_source=fix_std_integer_type,
        fixed_by_format=True,
        description="integer types without the std:: prefix: `int64_t`, `size_t` (G: Integer types).",
        hint="`make format` (or lint_code --fix --only std-integer-type) rewrites them.",
    ),
    check_types.Check(
        name=FORBIDDEN_CONSTRUCT,
        languages=CPP,
        scope=FIRST_PARTY,
        check_source=check_forbidden_construct,
        description="no long double, user-defined literals, inline namespaces, coroutines, modules, <ratio>, alloca, "
        "GNU extensions or non-constinit global thread_local (Google C++ Style Guide).",
    ),
    check_types.Check(
        name=EXCEPTIONS,
        languages=CPP,
        scope=NON_TEST,
        check_source=check_exceptions,
        description="first-party code does not throw; it returns absl::Status (G: Exceptions; ToTW #76).",
        hint="Return absl::Status / absl::StatusOr; a boundary with a throwing library (OCS2, yaml-cpp, cppzmq) converts "
        "in one try/catch with `// NOLINT(exceptions): <reason>` (AGENTS.md).",
    ),
    check_types.Check(
        name=RTTI,
        languages=CPP,
        scope=NON_TEST,
        check_source=check_rtti,
        description="no typeid or dynamic_cast outside tests (G: Run-time type information).",
    ),
    check_types.Check(
        name=CTAD,
        languages=CPP,
        scope=FIRST_PARTY,
        check_source=check_ctad,
        description="class templates are declared with their template arguments (AGENTS.md; G: Class template "
        "argument deduction).",
    ),
    check_types.Check(
        name=MACRO_NAMING,
        languages=CPP,
        scope=FIRST_PARTY,
        check_source=check_macro_naming,
        description="header macros carry the library's prefix, .cpp macros are #undef'ined, Abseil names are not "
        "redefined (G: Preprocessor macros).",
    ),
    check_types.Check(
        name=STATIC_STORAGE,
        languages=CPP,
        scope=FIRST_PARTY,
        check_source=check_static_storage,
        description="no non-trivially-destructible objects with static storage duration (G: Static and global "
        "variables).",
    ),
    check_types.Check(
        name=CLASS_COMMENT,
        languages=CPP,
        scope=NON_TEST,
        check_source=check_class_comment,
        description="every namespace-scope class or struct in a header has a comment (G: Class comments).",
    ),
    check_types.Check(
        name=IF_ELSE_BRACES,
        languages=CPP,
        scope=FIRST_PARTY,
        check_source=check_if_else_braces,
        description="an `if ... else` always has braces (AGENTS.md; G: Looping and branching statements).",
    ),
]
