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

"""Token checks for the Abseil C++ Tips of the Week (https://abseil.io/tips/) that clang-tidy and GCC do not cover.

Each check is named `totw-<topic>` after what it enforces and cites its tips:

- `totw-unordered-container` (#136): `absl::flat_hash_map` / `flat_hash_set`, not `std::unordered_*`.
- `totw-header-constant` (#140, #168): a namespace-scope constant in a header is `inline constexpr` (or `extern`).
- `totw-string-constant` (#140): a string constant is `inline constexpr char kName[] = "...";`, not a `const char*`.
- `totw-enum-switch-default` (#147): no `default:` in a switch over an owned `enum class`, so -Wswitch sees a new
  enumerator. A switch over a proto enum, which is open, takes a NOLINT.
- `totw-at` (#224): no `.at()` outside tests; check the index once and use `operator[]`.
- `totw-raw-new` (#126, #134): `std::make_unique` / `make_shared`; `absl::WrapUnique(new T)` only for private
  constructors, `return new T(*this)` in `clone()`, and `static const T& x = *new T(...)`.
- `totw-printf` (#124): `absl::StrFormat`, not the printf family or iomanip manipulators.
- `totw-small-by-const-ref` (#234): numbers, `bool` and `char` by value, not `const double&`.
- `totw-view-param` (#1, #93): `absl::string_view`, `absl::Span`, `absl::Duration` and `absl::Time` by value.
- `totw-optional-ref-param` (#163): not `const std::optional<T>&`: `std::optional<T>` by value, or `const T* absl_nullable`.
- `totw-smart-ptr-ref-param` (#188): not `const std::unique_ptr<T>&` / `const std::shared_ptr<T>&`.
- `totw-brace-literal-init` (#88): `bool done = false;`, not `bool done{false};`. Fixed by `make format`.
- `totw-namespace-name` (#130): no nested namespace named `testing`, `std`, `absl` or `util` (nor `Eigen`, `pinocchio`).
- `totw-unscoped-enum` (#86): `enum class`.
- `totw-view-member` (#180): no `string_view`, `Span`, `Eigen::Ref` or `Eigen::Map` data members.
- `totw-friend-test` (#135): no `FRIEND_TEST` or befriended test fixtures; test through the public API or a `*Peer`.
- `totw-std-specialization` (#99, #152, #218): nothing added to `namespace std`, and no specialization of a std or absl
  template (`struct std::hash<Foo>`, `std::formatter`, `std::tuple_size`), qualified or not.
- `totw-flag-location` (#45, #103): `ABSL_FLAG` only in `*Main.cpp`, `*AppFlags.{h,cpp}` and tests;
  `ABSL_DECLARE_FLAG` only in headers.
- `totw-size-minus` (#227): `i + 1 < v.size()`, never `i < v.size() - 1`.
- `totw-c-str-stored` (#5): no `.c_str()` of a temporary stored in a pointer.
- `totw-reader-lock` (#197): no reader locks (`std::shared_mutex`, `std::shared_lock`).
"""

import os
import re

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

CPP = frozenset({check_types.Language.CPP})
Token = cpp_source.Token


def _finding(path: str, token: Token, name: str, message: str) -> check_types.Finding:
    return check_types.Finding(path, token.line, token.col, name, message)


def _texts(code: tuple[Token, ...], start: int, count: int) -> list[str]:
    return [t.text for t in code[start : start + count]]


def _qualified(
    code: tuple[Token, ...], index: int, namespace: str, names: frozenset[str]
) -> str | None:
    """The name of `namespace::name` starting at code[index] (`::` in front allowed), when it is one of `names`."""
    if index > 0 and code[index - 1].text in (".", "->"):
        return None
    if (
        index > 1
        and code[index - 1].text == "::"
        and code[index - 2].kind == cpp_source.IDENTIFIER
    ):
        return None
    texts = _texts(code, index, 3)
    if (
        len(texts) == 3
        and texts[0] == namespace
        and texts[1] == "::"
        and texts[2] in names
    ):
        return texts[2]
    return None


def _parameter_start(code: tuple[Token, ...], index: int) -> bool:
    """True when code[index] starts a parameter: after `(` or `,` of a parameter list, not of a range-for."""
    if index == 0 or code[index - 1].text not in ("(", ","):
        return False
    depth = 0
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
                return not (
                    k > 0
                    and code[k - 1].text in ("for", "if", "while", "switch", "return")
                )
            depth -= 1
        elif text == ";" and depth == 0:
            return False
        k -= 1
    return False


# ----------------------------------------------------------------------------------------------------------------------
TOTW_UNORDERED = "totw-unordered-container"
_UNORDERED = frozenset(
    {"unordered_map", "unordered_set", "unordered_multimap", "unordered_multiset"}
)


def check_unordered(source: str, path: str) -> list[check_types.Finding]:
    """`std::unordered_*` containers."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i + 2],
            TOTW_UNORDERED,
            f"`std::{name}`: use absl::flat_hash_{name.split('_')[1].replace('multi', '')} (ToTW #136); never let its "
            "iteration order reach numerics, golden data, logs or serialized output.",
        )
        for i in range(len(code))
        if (name := _qualified(code, i, "std", _UNORDERED)) is not None
    ]


# ----------------------------------------------------------------------------------------------------------------------
TOTW_HEADER_CONSTANT = "totw-header-constant"
TOTW_STRING_CONSTANT = "totw-string-constant"


def _declaration_specifiers(texts: list[str]) -> tuple[set[str], int]:
    """The leading declaration specifiers of a statement, and the index of the first token after them."""
    specifiers = set()
    k = 0
    while k < len(texts) and texts[k] in (
        "static",
        "inline",
        "const",
        "constexpr",
        "extern",
        "constinit",
        "thread_local",
    ):
        specifiers.add(texts[k])
        k += 1
    return specifiers, k


def check_header_constant(source: str, path: str) -> list[check_types.Finding]:
    """Namespace-scope `const` / `constexpr` variables in a header without `inline`."""
    if not path.endswith(lint_files.CPP_HEADER_EXTENSIONS):
        return []
    code = cpp_source.code_tokens(source)
    findings = []
    for statement in cpp_source.statements(source):
        if statement.body is not None or not cpp_source.at_namespace_scope(
            statement.scopes
        ):
            continue
        tokens = code[statement.start : statement.end]
        texts = [t.text for t in tokens]
        specifiers, k = _declaration_specifiers(texts)
        if not ({"const", "constexpr"} & specifiers or "const" in texts[k : k + 4]):
            continue
        if (
            "inline" in specifiers
            or "extern" in specifiers
            or not texts
            or texts[0] in ("using", "typedef", "template")
        ):
            continue
        if "(" in texts and ("=" not in texts or texts.index("(") < texts.index("=")):
            continue  # a function declaration
        if "=" not in texts and "{" not in texts:
            continue
        findings.append(
            _finding(
                path,
                tokens[0],
                TOTW_HEADER_CONSTANT,
                "namespace-scope constant in a header without `inline`: write `inline constexpr` (ToTW #140, #168), "
                "so that every translation unit shares one definition.",
            )
        )
    return findings


def check_string_constant(source: str, path: str) -> list[check_types.Finding]:
    """`const char* kName = "...";` at namespace or class scope."""
    code = cpp_source.code_tokens(source)
    findings = []
    for statement in cpp_source.statements(source):
        if statement.body is not None or (
            statement.scopes
            and statement.scopes[-1]
            not in (
                cpp_source.NAMESPACE_SCOPE,
                cpp_source.CLASS_SCOPE,
            )
        ):
            continue
        tokens = code[statement.start : statement.end]
        texts = [t.text for t in tokens]
        if "char" not in texts:
            continue
        star = texts.index("char") + 1
        if (
            star >= len(texts)
            or texts[star] != "*"
            or not ({"const", "constexpr"} & set(texts[:star]))
        ):
            continue
        k = star + 1
        while k < len(texts) and texts[k] in ("absl_nonnull", "absl_nullable", "const"):
            k += 1
        if (
            k + 2 < len(texts)
            and tokens[k].kind == cpp_source.IDENTIFIER
            and texts[k + 1] == "="
            and tokens[k + 2].kind == cpp_source.STRING
        ):
            findings.append(
                _finding(
                    path,
                    tokens[k],
                    TOTW_STRING_CONSTANT,
                    f"string constant `{texts[k]}` as a `const char*`: write `inline constexpr char {texts[k]}[] = "
                    '"...";` (in a .cpp: `constexpr char` in an unnamed namespace) or `constexpr absl::string_view` '
                    "(ToTW #140).",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_ENUM_SWITCH_DEFAULT = "totw-enum-switch-default"
_ENUMERATOR = re.compile(r"^k[A-Z0-9]")


def check_enum_switch_default(source: str, path: str) -> list[check_types.Finding]:
    """`default:` in a switch whose case labels are all enumerators of a scoped enum (`case Mode::kWalk:`)."""
    code = cpp_source.code_tokens(source)
    findings = []
    for index, token in enumerate(code):
        if (
            token.text != "switch"
            or index + 1 >= len(code)
            or code[index + 1].text != "("
        ):
            continue
        close = cpp_source.matching(code, index + 1)
        if close + 1 >= len(code) or code[close + 1].text != "{":
            continue
        if [t.text for t in code[close - 3 : close]] == ["index", "(", ")"]:
            # `switch (value.shape.index())`: a std::variant's size_t index, whose labels are index constants
            # (`case Oneofs::kRadiusIndex:`), not enumerators; nproto generates these.
            continue
        body_end = cpp_source.matching(code, close + 1)
        depth = 0
        labels: list[bool] = []
        default: Token | None = None
        k = close + 2
        while k < body_end:
            text = code[k].text
            if text in ("{", "(", "["):
                depth += 1
            elif text in ("}", ")", "]"):
                depth -= 1
            elif text == "switch" and k + 1 < body_end and code[k + 1].text == "(":
                # A nested switch has labels of its own: skip its condition and its body.
                k = cpp_source.matching(code, cpp_source.matching(code, k + 1) + 1)
            elif text == "case" and depth == 0:
                colon = k + 1
                while colon < body_end and code[colon].text != ":":
                    colon += 1
                label = [t.text for t in code[k + 1 : colon]]
                labels.append(
                    len(label) >= 3
                    and label[-2] == "::"
                    and bool(_ENUMERATOR.match(label[-1]))
                )
                k = colon
            elif (
                text == "default"
                and depth == 0
                and k + 1 < body_end
                and code[k + 1].text == ":"
            ):
                default = code[k]
            k += 1
        if default is not None and labels and all(labels):
            findings.append(
                _finding(
                    path,
                    default,
                    TOTW_ENUM_SWITCH_DEFAULT,
                    "`default:` in a switch over an enum class: list every enumerator, so -Wswitch reports a new one, "
                    "and handle an impossible value after the switch (ToTW #147).",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_AT = "totw-at"


def check_at(source: str, path: str) -> list[check_types.Finding]:
    """`.at(` and `->at(` calls."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i],
            TOTW_AT,
            "`.at()` throws: check the index once and use `operator[]` (or `contains()` and `find()` for maps) "
            "(ToTW #224).",
        )
        for i in range(1, len(code) - 1)
        if code[i].text == "at"
        and code[i - 1].text in (".", "->")
        and code[i + 1].text == "("
    ]


# ----------------------------------------------------------------------------------------------------------------------
TOTW_RAW_NEW = "totw-raw-new"


def check_raw_new(source: str, path: str) -> list[check_types.Finding]:
    """`new T` outside the allowed forms."""
    code = cpp_source.code_tokens(source)
    findings = []
    for i, token in enumerate(code):
        if token.text != "new" or token.kind != cpp_source.IDENTIFIER:
            continue
        previous = _texts(code, max(0, i - 4), min(i, 4))
        if (
            previous[-1:] == ["operator"]
            or previous[-1:] == ["::"]
            and previous[-2:-1] == ["operator"]
        ):
            continue
        if previous[-3:] == ["WrapUnique", "("] or previous[-2:] == ["WrapUnique", "("]:
            continue
        if previous[-1:] == ["("] and i >= 2 and code[i - 2].text == "WrapUnique":
            continue
        if previous[-1:] == ["return"] and _is_clone_return(code, i):
            continue
        if (
            previous[-1:] == ["*"]
            and i >= 2
            and code[i - 2].text == "="
            and _statement_starts_with_static(code, i)
        ):
            continue
        findings.append(
            _finding(
                path,
                token,
                TOTW_RAW_NEW,
                "raw `new`: use std::make_unique / std::make_shared, absl::WrapUnique(new T) for a private constructor, "
                "or absl::NoDestructor (ToTW #126, #134).",
            )
        )
    return findings


def _is_clone_return(code: tuple[Token, ...], new: int) -> bool:
    """True for `return new T(*this);`."""
    k = new + 1
    while k < len(code) and code[k].text not in ("(", ";", "{"):
        k += 1
    return _texts(code, k, 4) == ["(", "*", "this", ")"]


def _statement_starts_with_static(code: tuple[Token, ...], index: int) -> bool:
    k = index
    while k > 0 and code[k - 1].text not in (";", "{", "}"):
        k -= 1
    return code[k].text == "static"


# ----------------------------------------------------------------------------------------------------------------------
TOTW_PRINTF = "totw-printf"
_PRINTF = re.compile(r"^(?:s|sn|f|v|vs|vsn|vf|d|vd)?printf$")
_IOMANIP = frozenset(
    {
        "setw",
        "setprecision",
        "setfill",
        "fixed",
        "scientific",
        "hexfloat",
        "defaultfloat",
    }
)


def check_printf(source: str, path: str) -> list[check_types.Finding]:
    """printf-family calls and iomanip manipulators."""
    code = cpp_source.code_tokens(source)
    findings = []
    for i, token in enumerate(code):
        previous = code[i - 1].text if i else ""
        if (
            _PRINTF.match(token.text)
            and i + 1 < len(code)
            and code[i + 1].text == "("
            and previous not in (".", "->")
        ):
            if previous == "::" and i >= 2 and code[i - 2].text not in ("std",):
                continue
            findings.append(
                _finding(
                    path,
                    token,
                    TOTW_PRINTF,
                    f"`{token.text}`: use absl::StrFormat / absl::PrintF (ToTW #124).",
                )
            )
        elif (
            token.text in _IOMANIP
            and previous == "::"
            and i >= 2
            and code[i - 2].text == "std"
        ):
            findings.append(
                _finding(
                    path,
                    token,
                    TOTW_PRINTF,
                    f"`std::{token.text}`: format with absl::StrFormat, not stream manipulators (ToTW #124).",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_SMALL_BY_CONST_REF = "totw-small-by-const-ref"
_SMALL = re.compile(r"^(scalar_t|double|float|int|bool|size_t|u?int\d+_t|char)$")


def check_small_by_const_ref(source: str, path: str) -> list[check_types.Finding]:
    """Parameters `const double&` and the like."""
    code = cpp_source.code_tokens(source)
    findings = []
    for i in range(len(code) - 2):
        if (
            code[i].text != "const"
            or not _SMALL.match(code[i + 1].text)
            or code[i + 2].text != "&"
        ):
            continue
        if not _parameter_start(code, i):
            continue
        findings.append(
            _finding(
                path,
                code[i + 1],
                TOTW_SMALL_BY_CONST_REF,
                f"`const {code[i + 1].text}&` parameter: pass `{code[i + 1].text}` by value (ToTW #234).",
            )
        )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_VIEW_PARAM = "totw-view-param"
TOTW_OPTIONAL_REF_PARAM = "totw-optional-ref-param"
TOTW_SMART_PTR_REF_PARAM = "totw-smart-ptr-ref-param"


def _const_reference_to(
    code: tuple[Token, ...], namespace: str, names: frozenset[str]
) -> list[tuple[int, str]]:
    """(index of `const`, name) for each `const namespace::name[<...>]&`."""
    found = []
    for i in range(len(code) - 4):
        if code[i].text != "const":
            continue
        name = _qualified(code, i + 1, namespace, names)
        if name is None:
            continue
        k = i + 4
        if k < len(code) and code[k].text == "<":
            k = cpp_source.matching_angle(code, k) + 1
        if k < len(code) and code[k].text == "&":
            found.append((i, name))
    return found


def check_view_param(source: str, path: str) -> list[check_types.Finding]:
    """Views, spans, durations and times passed by const reference."""
    code = cpp_source.code_tokens(source)
    findings = []
    for namespace, names in (
        ("absl", frozenset({"string_view", "Span", "Duration", "Time"})),
        ("std", frozenset({"string_view", "span"})),
    ):
        for i, name in _const_reference_to(code, namespace, names):
            findings.append(
                _finding(
                    path,
                    code[i],
                    TOTW_VIEW_PARAM,
                    f"`const {namespace}::{name}&`: pass `{namespace}::{name}` by value (ToTW #1, #93).",
                )
            )
    return findings


def check_optional_ref_param(source: str, path: str) -> list[check_types.Finding]:
    """`const std::optional<T>&` parameters."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i],
            TOTW_OPTIONAL_REF_PARAM,
            "`const std::optional<T>&` parameter: take `std::optional<T>` by value if T is small, otherwise "
            "`const T* absl_nullable` (ToTW #163).",
        )
        for i, _ in _const_reference_to(code, "std", frozenset({"optional"}))
        if _parameter_start(code, i)
    ]


def check_smart_ptr_ref_param(source: str, path: str) -> list[check_types.Finding]:
    """`const std::unique_ptr<T>&` and `const std::shared_ptr<T>&` parameters."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i],
            TOTW_SMART_PTR_REF_PARAM,
            f"`const std::{name}<T>&` parameter: take `const T&` (or `T* absl_nullable`), or the smart pointer by value "
            "to share or transfer ownership (ToTW #188).",
        )
        for i, name in _const_reference_to(
            code, "std", frozenset({"unique_ptr", "shared_ptr"})
        )
        if _parameter_start(code, i)
    ]


# ----------------------------------------------------------------------------------------------------------------------
TOTW_BRACE_LITERAL_INIT = "totw-brace-literal-init"
_BRACE_INIT_TYPES = frozenset({"bool", "int", "double", "float", "scalar_t", "size_t"})


def _brace_literal_inits(source: str) -> list[tuple[Token, Token, Token, Token]]:
    """(type, name, `{`, `}`) for each `Type name{literal}` of a scalar or std::string."""
    code = cpp_source.code_tokens(source)
    found = []
    for i in range(len(code) - 3):
        token = code[i]
        is_string = (
            token.text == "string"
            and i >= 2
            and code[i - 1].text == "::"
            and code[i - 2].text == "std"
        )
        if token.text not in _BRACE_INIT_TYPES and not is_string:
            continue
        if not is_string and i > 0 and code[i - 1].text in ("::", ".", "->"):
            continue
        name, brace = code[i + 1], code[i + 2]
        if name.kind != cpp_source.IDENTIFIER or brace.text != "{" or brace.ws_before:
            continue
        k = i + 3
        if code[k].text in ("-", "+"):
            k += 1
        if k + 1 >= len(code):
            continue
        literal = code[k]
        is_literal = literal.kind in (
            cpp_source.NUMBER,
            cpp_source.STRING,
        ) or literal.text in ("true", "false", "nullptr")
        if is_literal and code[k + 1].text == "}":
            found.append((token, name, brace, code[k + 1]))
    return found


def check_brace_literal_init(source: str, path: str) -> list[check_types.Finding]:
    """`bool done{false};`: write `bool done = false;`."""
    return [
        _finding(
            path,
            name,
            TOTW_BRACE_LITERAL_INIT,
            f"`{name.text}{{...}}`: initialize a scalar or string from a literal with `=` (`{name.text} = ...`) "
            "(ToTW #88).",
        )
        for _, name, _, _ in _brace_literal_inits(source)
    ]


def fix_brace_literal_init(source: str, path: str) -> str:
    """Rewrites `name{literal}` to `name = literal`."""
    del path  # Unused.
    found = _brace_literal_inits(source)
    if not found:
        return source
    starts = cpp_source.line_starts(source)
    edits = []
    for _, _, brace, close in found:
        open_offset = starts[brace.line - 1] + brace.col - 1
        close_offset = starts[close.line - 1] + close.col - 1
        edits.append((close_offset, close_offset + 1, ""))
        edits.append((open_offset, open_offset + 1, " = "))
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source


# ----------------------------------------------------------------------------------------------------------------------
TOTW_NAMESPACE_NAME = "totw-namespace-name"
_RESERVED_NAMESPACES = frozenset(
    {"testing", "std", "absl", "util", "Eigen", "pinocchio"}
)


def check_namespace_name(source: str, path: str) -> list[check_types.Finding]:
    """Nested namespaces named like a top-level one."""
    code = cpp_source.code_tokens(source)
    scopes = cpp_source.scopes(source)
    findings = []
    for i, token in enumerate(code):
        if token.text != "namespace" or (i > 0 and code[i - 1].text == "using"):
            continue
        components = []
        k = i + 1
        while k < len(code) and code[k].kind == cpp_source.IDENTIFIER:
            components.append(code[k])
            if k + 1 < len(code) and code[k + 1].text == "::":
                k += 2
                continue
            break
        if k + 1 >= len(code) or code[k + 1].text != "{":
            continue  # an alias or a using-directive
        nested = any(kind == cpp_source.NAMESPACE_SCOPE for kind in scopes[i])
        for position, component in enumerate(components):
            if (nested or position > 0) and component.text in _RESERVED_NAMESPACES:
                findings.append(
                    _finding(
                        path,
                        component,
                        TOTW_NAMESPACE_NAME,
                        f"nested namespace `{component.text}` shadows the top-level `{component.text}` for code inside "
                        "it: pick another name (ToTW #130).",
                    )
                )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_UNSCOPED_ENUM = "totw-unscoped-enum"


def check_unscoped_enum(source: str, path: str) -> list[check_types.Finding]:
    """`enum Name` that is not an `enum class`."""
    code = cpp_source.code_tokens(source)
    findings = []
    for i, token in enumerate(code):
        if (
            token.text != "enum"
            or i + 1 >= len(code)
            or code[i + 1].text in ("class", "struct")
        ):
            continue
        # An elaborated type specifier (`enum Foo x;`) declares nothing new.
        k = i + 1
        while k < len(code) and code[k].text not in ("{", ";", "=", "(", ")"):
            k += 1
        if k < len(code) and code[k].text == "{":
            findings.append(
                _finding(
                    path,
                    token,
                    TOTW_UNSCOPED_ENUM,
                    "unscoped `enum`: write `enum class` (ToTW #86); an Eigen-style index constant is "
                    "`inline constexpr int`.",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_VIEW_MEMBER = "totw-view-member"
_VIEWS = {
    "absl": frozenset({"string_view", "Span"}),
    "std": frozenset({"string_view", "span"}),
    "Eigen": frozenset({"Ref", "Map"}),
}


def check_view_member(source: str, path: str) -> list[check_types.Finding]:
    """Data members of a view type."""
    code = cpp_source.code_tokens(source)
    findings = []
    for statement in cpp_source.statements(source):
        if (
            statement.body is not None
            or not statement.scopes
            or statement.scopes[-1] != cpp_source.CLASS_SCOPE
        ):
            continue
        tokens = code[statement.start : statement.end]
        texts = [t.text for t in tokens]
        specifiers, k = _declaration_specifiers(texts)
        if (
            {"static", "constexpr"} & specifiers
            or not texts
            or texts[0] in ("using", "typedef", "friend")
        ):
            continue
        namespace = texts[k] if k < len(texts) else ""
        if (
            namespace not in _VIEWS
            or _qualified(tokens, k, namespace, _VIEWS[namespace]) is None
        ):
            continue
        j = k + 3
        if j < len(texts) and texts[j] == "<":
            j = cpp_source.matching_angle(tokens, j) + 1
        if (
            j < len(texts)
            and tokens[j].kind == cpp_source.IDENTIFIER
            and (j + 1 >= len(texts) or texts[j + 1] != "(")
        ):
            findings.append(
                _finding(
                    path,
                    tokens[j],
                    TOTW_VIEW_MEMBER,
                    f"data member `{texts[j]}` of view type `{namespace}::{texts[k + 2]}`: store the data it views, or "
                    "document what outlives it (ToTW #180).",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_FRIEND_TEST = "totw-friend-test"


def check_friend_test(source: str, path: str) -> list[check_types.Finding]:
    """`FRIEND_TEST` and befriended test fixtures."""
    code = cpp_source.code_tokens(source)
    findings = []
    for i, token in enumerate(code):
        befriended = (
            token.text == "friend"
            and i + 2 < len(code)
            and code[i + 1].text in ("class", "struct")
            and re.search(r"Tests?$", code[i + 2].text)
        )
        if token.text == "FRIEND_TEST" or befriended:
            findings.append(
                _finding(
                    path,
                    token,
                    TOTW_FRIEND_TEST,
                    "a test befriended: test through the public API, or give the class a `*Peer` (ToTW #135).",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_STD_SPECIALIZATION = "totw-std-specialization"


# The std templates a type would be hooked into by a specialization, in any spelling: write AbslHashValue,
# AbslStringify or operators next to the type instead.
_STD_HOOKS = frozenset({"hash", "formatter", "tuple_size", "tuple_element"})


def _is_std_specialization(code: tuple[cpp_source.Token, ...], i: int) -> bool:
    """True for a `struct` / `class` at code[i] that specializes a std or absl template, or a std hook template.

    The forms: `struct hash<Foo>` inside `namespace std`, the C++17 qualified `struct std::hash<Foo>` (also
    `::std::`), `struct absl::X<Foo>`, and `struct formatter<Foo>` / `tuple_size<Foo>` / `tuple_element<...>`.

    Args:
      code: The code tokens of the file.
      i: The index of a token.

    Returns:
      Whether code[i] begins such a specialization.
    """
    if code[i].text not in ("struct", "class"):
        return False
    k = i + 1
    if k < len(code) and code[k].text == "::":
        k += 1
    texts = _texts(code, k, 2)
    if len(texts) == 2 and texts[0] in _STD_HOOKS and texts[1] == "<":
        return True
    if not texts or texts[0] not in ("std", "absl"):
        return False
    # `std::hash<`, `absl::hash_internal::HashImpl<`: a qualified name whose last part takes template arguments.
    k += 1
    while (
        k + 2 < len(code)
        and code[k].text == "::"
        and code[k + 1].kind == cpp_source.IDENTIFIER
    ):
        k += 2
        if code[k].text == "<":
            return True
    return False


def check_std_specialization(source: str, path: str) -> list[check_types.Finding]:
    """`namespace std {`, and specializations of std and absl templates (`struct std::hash<Foo>`, `struct hash<`)."""
    code = cpp_source.code_tokens(source)
    findings = []
    for i in range(len(code) - 2):
        texts = _texts(code, i, 3)
        if texts == ["namespace", "std", "{"] or _is_std_specialization(code, i):
            findings.append(
                _finding(
                    path,
                    code[i],
                    TOTW_STD_SPECIALIZATION,
                    "nothing is added to namespace std: define AbslHashValue, AbslStringify or operators next to the "
                    "type (ToTW #99, #152, #218).",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_FLAG_LOCATION = "totw-flag-location"
_FLAG_FILE = re.compile(r"(Main\.cpp|AppFlags\.(h|cpp))$")


def check_flag_location(source: str, path: str) -> list[check_types.Finding]:
    """`ABSL_FLAG` outside main files and `ABSL_DECLARE_FLAG` outside headers."""
    code = cpp_source.code_tokens(source)
    name = os.path.basename(path)
    findings = []
    for i, token in enumerate(code):
        if i + 1 >= len(code) or code[i + 1].text != "(":
            continue
        if token.text in ("ABSL_FLAG", "ABSL_RETIRED_FLAG") and not (
            _FLAG_FILE.search(name) or lint_files.is_test_path(path)
        ):
            findings.append(
                _finding(
                    path,
                    token,
                    TOTW_FLAG_LOCATION,
                    "ABSL_FLAG outside a *Main.cpp or *AppFlags.{h,cpp}: define flags where the binary is, and pass "
                    "the values to libraries as parameters (ToTW #45, #103).",
                )
            )
        elif token.text == "ABSL_DECLARE_FLAG" and not path.endswith(
            lint_files.CPP_HEADER_EXTENSIONS
        ):
            findings.append(
                _finding(
                    path,
                    token,
                    TOTW_FLAG_LOCATION,
                    "ABSL_DECLARE_FLAG outside a header: declare a flag once, in the header next to its definition "
                    "(ToTW #45).",
                )
            )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_SIZE_MINUS = "totw-size-minus"
# A comparison: clang-format puts spaces around a binary `<`, and none after the `<` of a template argument list.
_SIZE_MINUS = re.compile(r"\s<=?\s+[\w.\->\[\]():]*\bsize\s*\(\s*\)\s*-\s*\w")


def check_size_minus(source: str, path: str) -> list[check_types.Finding]:
    """`i < v.size() - 1`, which wraps around for an empty container."""
    _, code = cpp_source.mask(source)
    findings = []
    for match in _SIZE_MINUS.finditer(code):
        line = code.count("\n", 0, match.start()) + 1
        column = match.start() - (code.rfind("\n", 0, match.start()) + 1) + 1
        findings.append(
            check_types.Finding(
                path,
                line,
                column,
                TOTW_SIZE_MINUS,
                "`< v.size() - N` wraps around when the container has fewer than N elements: write `i + N < v.size()` "
                "(ToTW #227).",
            )
        )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
TOTW_C_STR_STORED = "totw-c-str-stored"
_C_STR_STORED = re.compile(
    r"\bchar\s*\*\s*(?:absl_\w+\s+)?(?:const\s+)?\w+\s*=\s*[^;]*\)\s*\.\s*c_str\s*\(\s*\)\s*;"
)


def check_c_str_stored(source: str, path: str) -> list[check_types.Finding]:
    """`const char* p = f().c_str();`, a pointer into a destroyed temporary."""
    _, code = cpp_source.mask(source)
    return [
        check_types.Finding(
            path,
            code.count("\n", 0, match.start()) + 1,
            match.start() - (code.rfind("\n", 0, match.start()) + 1) + 1,
            TOTW_C_STR_STORED,
            "the `.c_str()` of a temporary is stored in a pointer that dangles at the end of the statement (ToTW #5).",
        )
        for match in _C_STR_STORED.finditer(code)
    ]


# ----------------------------------------------------------------------------------------------------------------------
TOTW_READER_LOCK = "totw-reader-lock"
_READER_LOCKS = frozenset({"shared_mutex", "shared_lock", "shared_timed_mutex"})


def check_reader_lock(source: str, path: str) -> list[check_types.Finding]:
    """Reader locks."""
    code = cpp_source.code_tokens(source)
    findings = []
    for i, token in enumerate(code):
        if _qualified(code, i, "std", _READER_LOCKS) is not None or token.text in (
            "ReaderMutexLock",
            "ReaderLock",
        ):
            target = code[i + 2] if token.text == "std" else token
            findings.append(
                _finding(
                    path,
                    target,
                    TOTW_READER_LOCK,
                    f"reader lock `{target.text}`: use absl::Mutex with an exclusive lock; reader locks are rarely faster "
                    "(ToTW #197).",
                )
            )
    return findings


def _check(
    name: str,
    function: check_types.CheckFunction,
    description: str,
    *,
    non_test: bool = False,
    fix_source: check_types.FixFunction | None = None,
    fixed_by_format: bool = False,
    hint: str = "",
) -> check_types.Check:
    return check_types.Check(
        name=name,
        languages=CPP,
        scope=(
            lint_files.Scope.FIRST_PARTY_NON_TEST
            if non_test
            else lint_files.Scope.FIRST_PARTY
        ),
        check_source=function,
        description=description,
        fix_source=fix_source,
        fixed_by_format=fixed_by_format,
        hint=hint,
    )


CHECKS = [
    _check(
        TOTW_UNORDERED,
        check_unordered,
        "absl::flat_hash_map / flat_hash_set, not std::unordered_* (ToTW #136).",
    ),
    _check(
        TOTW_HEADER_CONSTANT,
        check_header_constant,
        "header constants are inline constexpr (ToTW #140, #168).",
    ),
    _check(
        TOTW_STRING_CONSTANT,
        check_string_constant,
        "string constants are `constexpr char kName[]` (ToTW #140).",
    ),
    _check(
        TOTW_ENUM_SWITCH_DEFAULT,
        check_enum_switch_default,
        "no default: in a switch over an owned enum class (ToTW #147).",
    ),
    _check(TOTW_AT, check_at, "no .at() outside tests (ToTW #224).", non_test=True),
    _check(
        TOTW_RAW_NEW,
        check_raw_new,
        "std::make_unique / make_shared, not raw new (ToTW #126, #134).",
        non_test=True,
    ),
    _check(
        TOTW_PRINTF, check_printf, "absl::StrFormat, not printf or iomanip (ToTW #124)."
    ),
    _check(
        TOTW_SMALL_BY_CONST_REF,
        check_small_by_const_ref,
        "numbers, bool and char by value (ToTW #234).",
    ),
    _check(
        TOTW_VIEW_PARAM,
        check_view_param,
        "views, spans, durations and times by value (ToTW #1, #93).",
    ),
    _check(
        TOTW_OPTIONAL_REF_PARAM,
        check_optional_ref_param,
        "no const std::optional<T>& parameters (ToTW #163).",
    ),
    _check(
        TOTW_SMART_PTR_REF_PARAM,
        check_smart_ptr_ref_param,
        "no const std::unique_ptr<T>& / shared_ptr<T>& parameters (ToTW #188).",
    ),
    _check(
        TOTW_BRACE_LITERAL_INIT,
        check_brace_literal_init,
        "`=` for values, not `bool x{false};` (ToTW #88).",
        fix_source=fix_brace_literal_init,
        fixed_by_format=True,
        hint="`make format` (or lint_code --fix --only totw-brace-literal-init) rewrites them.",
    ),
    _check(
        TOTW_NAMESPACE_NAME,
        check_namespace_name,
        "no nested testing / std / absl / util namespaces (ToTW #130).",
    ),
    _check(TOTW_UNSCOPED_ENUM, check_unscoped_enum, "enum class, not enum (ToTW #86)."),
    _check(
        TOTW_VIEW_MEMBER, check_view_member, "no view-typed data members (ToTW #180)."
    ),
    _check(
        TOTW_FRIEND_TEST,
        check_friend_test,
        "no FRIEND_TEST or befriended test fixtures (ToTW #135).",
    ),
    _check(
        TOTW_STD_SPECIALIZATION,
        check_std_specialization,
        "nothing added to namespace std (ToTW #99, #152, #218).",
    ),
    _check(
        TOTW_FLAG_LOCATION,
        check_flag_location,
        "ABSL_FLAG only in main files (ToTW #45, #103).",
    ),
    _check(
        TOTW_SIZE_MINUS,
        check_size_minus,
        "`i + 1 < v.size()`, never `i < v.size() - 1` (ToTW #227).",
    ),
    _check(
        TOTW_C_STR_STORED,
        check_c_str_stored,
        "no c_str() of a temporary stored in a pointer (ToTW #5).",
    ),
    _check(TOTW_READER_LOCK, check_reader_lock, "no reader locks (ToTW #197)."),
]
