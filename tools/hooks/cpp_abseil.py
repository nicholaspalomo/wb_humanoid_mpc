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

"""Token checks that choose the Abseil type where the standard library has a counterpart (AGENTS.md, "Abseil first").

- `absl-span-param` (ToTW #93): a read-only `const std::vector<T>&` parameter, or one of OCS2's `std::vector` aliases
  (`scalar_array_t`, `vector_array_t`, ...), is `absl::Span<const T>` by value. A function that keeps a copy takes
  the vector by value and moves it (ToTW #117).
- `absl-string-view-param` (ToTW #1): a read-only `const std::string&` parameter is `absl::string_view` by value; a
  function that keeps a copy takes `std::string` by value and moves it.
- `absl-ordered-container`: `absl::btree_map` / `btree_set` (and the multi forms), not `std::map` / `std::set`.
- `absl-mutex`: `absl::Mutex`, `absl::MutexLock` and `absl::CondVar`, not `std::mutex`, its locks and
  `std::condition_variable`.
- `absl-function`: `absl::AnyInvocable` for a callable that is stored, `absl::FunctionRef` for one that is only called
  during the call that receives it, not `std::function`.
- `absl-number-conversion`: `absl::StrCat` and `absl::SimpleAtoi` / `SimpleAtod` / `SimpleAtob`, not `std::to_string`
  and the throwing `std::stoi` family.
- `absl-string-stream`: `absl::StrCat` / `StrFormat` / `StrSplit`, not the `std::stringstream` family.
- `absl-view-type`: `absl::string_view` and `absl::Span`, not `std::string_view` and `std::span`.

The two parameter checks read declarations only, so a definition needs no marker of its own: they skip out-of-line
definitions (`Foo::f(...)`, whose declaration is checked), overrides (`override`, `final`), lambdas and function types
(`std::function<void(const std::vector<int>&)>`), and `std::vector<bool>`, which a span cannot view. A parameter
whose type a third-party API fixes (an OCS2 virtual declared without `override` in a first-party base, a callback
signature, a template whose argument is deduced from the vector) takes a NOLINT naming that API, as does a container
or callable whose Abseil counterpart lacks a property the code needs: pointer stability across an insertion for
`std::map`, copyability for `std::function`, `std::condition_variable` waits on a mutex that a third-party API owns.

`std::chrono` stays: Abseil has no monotonic clock, and the realtime loop times itself with `std::chrono::steady_clock`.
`std::array` parameters stay too, because their length is part of the contract.
"""

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import cpp_totw
from tools.hooks import lint_files

CPP = frozenset({check_types.Language.CPP})
Token = cpp_source.Token


def _finding(
    path: str, token: Token, name: str, message: str
) -> check_types.Finding:
    return check_types.Finding(path, token.line, token.col, name, message)


def _std_names(
    code: tuple[Token, ...], names: frozenset[str]
) -> list[tuple[int, str]]:
    """(index of the name, name) for each `std::name` in `code`."""
    return [
        (i + 2, name)
        for i in range(len(code))
        if (name := cpp_totw.qualified(code, i, "std", names)) is not None
    ]


# ----------------------------------------------------------------------------------------------------------------------
ABSL_SPAN_PARAM = "absl-span-param"
ABSL_STRING_VIEW_PARAM = "absl-string-view-param"

# OCS2's aliases of std::vector (lib/ocs2/core/include/ocs2_core/Types.h).
_OCS2_VECTOR_ALIASES = frozenset(
    {
        "size_array_t",
        "size_array2_t",
        "scalar_array_t",
        "scalar_array2_t",
        "scalar_array3_t",
        "vector_array_t",
        "vector_array2_t",
        "vector_array3_t",
        "matrix_array_t",
        "matrix_array2_t",
        "matrix_array3_t",
    }
)
_FUNCTION_TAIL_STOPS = frozenset({"{", ";", "=", ":"})
_OVERRIDE_SPECIFIERS = frozenset({"override", "final"})


def _enclosing_open_paren(code: tuple[Token, ...], index: int) -> int:
    """The index of the `(` whose list holds code[index], or -1."""
    depth = 0
    for k in range(index - 1, -1, -1):
        text = code[k].text
        if text in (")", "]", "}"):
            depth += 1
        elif text in ("(", "[", "{"):
            if depth == 0:
                return k if text == "(" else -1
            depth -= 1
    return -1


def _function_name_index(code: tuple[Token, ...], open_paren: int) -> int:
    """The index of the name before a parameter list (`f`, or `operator` of `operator==` / `operator()`), or -1."""
    k = open_paren - 1
    if k < 0:
        return -1
    if (
        k >= 2
        and code[k].text == ")"
        and code[k - 1].text == "("
        and code[k - 2].text == "operator"
    ):
        return k - 2
    if k >= 1 and code[k - 1].text == "operator":
        return k - 1
    if code[k].kind != cpp_source.IDENTIFIER:
        return -1
    return k


def _is_checked_declaration(code: tuple[Token, ...], const_index: int) -> bool:
    """True when the parameter starting at code[const_index] belongs to a declaration the parameter checks read."""
    if not cpp_totw.parameter_start(code, const_index):
        return False
    open_paren = _enclosing_open_paren(code, const_index)
    name = _function_name_index(code, open_paren) if open_paren >= 0 else -1
    if name < 0:
        return False  # A lambda, a function pointer or a macro argument list.
    if name >= 1 and code[name - 1].text in ("::", "<", ","):
        return False  # An out-of-line definition, or a function type.
    close = cpp_source.matching(code, open_paren)
    k = close + 1
    while k < len(code) and code[k].text not in _FUNCTION_TAIL_STOPS:
        if code[k].text in _OVERRIDE_SPECIFIERS:
            return False
        k += 1
    return True


def _const_reference_parameters(
    code: tuple[Token, ...],
) -> list[tuple[int, int]]:
    """Finds the `const T&` parameter declarations of a token stream.

    The type is everything between `const` and the `&`: a qualified name with template arguments.

    Args:
        code: the code tokens of one source file.

    Returns:
        (index of `const`, index of the type's first name token) for each declaration, in source order.
    """
    found = []
    for i, token in enumerate(code):
        if token.text != "const" or not _is_checked_declaration(code, i):
            continue
        k = i + 1
        if k < len(code) and code[k].text == "::":
            k += 1
        while k + 1 < len(code) and code[k + 1].text == "::":
            k += 2
        end = k + 1
        if end < len(code) and code[end].text == "<":
            end = cpp_source.matching_angle(code, end) + 1
        if end < len(code) and code[end].text == "&":
            found.append((i, k))
    return found


def _is_std(code: tuple[Token, ...], const_index: int, name_index: int) -> bool:
    texts = [t.text for t in code[const_index + 1 : name_index]]
    return texts in (["std", "::"], ["::", "std", "::"])


def _is_vector_of_bool(code: tuple[Token, ...], name_index: int) -> bool:
    return (
        name_index + 3 < len(code)
        and code[name_index + 1].text == "<"
        and code[name_index + 2].text == "bool"
        and code[name_index + 3].text == ">"
    )


def check_span_param(source: str, path: str) -> list[check_types.Finding]:
    """`const std::vector<T>&` parameters (and OCS2's vector aliases) of function declarations."""
    code = cpp_source.code_tokens(source)
    findings = []
    for const_index, name_index in _const_reference_parameters(code):
        name = code[name_index].text
        if name == "vector":
            if not _is_std(code, const_index, name_index) or _is_vector_of_bool(
                code, name_index
            ):
                continue
            spelled = "std::vector<T>"
        elif name in _OCS2_VECTOR_ALIASES:
            spelled = name
        else:
            continue
        findings.append(
            _finding(
                path,
                code[const_index],
                ABSL_SPAN_PARAM,
                f"`const {spelled}&` parameter: take `absl::Span<const T>` by"
                " value, or the vector by value and move it when the"
                " function keeps a copy (ToTW #93, #117).",
            )
        )
    return findings


def check_string_view_param(
    source: str, path: str
) -> list[check_types.Finding]:
    """`const std::string&` parameters of function declarations."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[const_index],
            ABSL_STRING_VIEW_PARAM,
            "`const std::string&` parameter: take `absl::string_view` by"
            " value, or `std::string` by value and move it when the function"
            " keeps a copy (ToTW #1, #117).",
        )
        for const_index, name_index in _const_reference_parameters(code)
        if code[name_index].text == "string"
        and _is_std(code, const_index, name_index)
    ]


# ----------------------------------------------------------------------------------------------------------------------
ABSL_ORDERED_CONTAINER = "absl-ordered-container"
_ORDERED_CONTAINERS = {
    "map": "absl::btree_map",
    "set": "absl::btree_set",
    "multimap": "absl::btree_multimap",
    "multiset": "absl::btree_multiset",
}


def check_ordered_container(
    source: str, path: str
) -> list[check_types.Finding]:
    """`std::map`, `std::set` and their multi forms."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i],
            ABSL_ORDERED_CONTAINER,
            f"`std::{name}`: use {_ORDERED_CONTAINERS[name]}"
            ' ("absl/container/btree_map.h" / "btree_set.h"). A btree'
            " invalidates iterators and pointers on insert and erase; code"
            " that needs them stable keeps the std container with a NOLINT.",
        )
        for i, name in _std_names(code, frozenset(_ORDERED_CONTAINERS))
    ]


# ----------------------------------------------------------------------------------------------------------------------
ABSL_MUTEX = "absl-mutex"
_MUTEX_TYPES = {
    "mutex": "absl::Mutex",
    "recursive_mutex": "absl::Mutex (without recursion)",
    "timed_mutex": "absl::Mutex",
    "recursive_timed_mutex": "absl::Mutex (without recursion)",
    "lock_guard": "absl::MutexLock",
    "scoped_lock": "absl::MutexLock",
    "unique_lock": "absl::MutexLock, or absl::Mutex::Lock / Unlock",
    "condition_variable": "absl::CondVar, or absl::Mutex::Await",
    "condition_variable_any": "absl::CondVar, or absl::Mutex::Await",
}


def check_mutex(source: str, path: str) -> list[check_types.Finding]:
    """`std::mutex`, its locks and `std::condition_variable`."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i],
            ABSL_MUTEX,
            f"`std::{name}`: use {_MUTEX_TYPES[name]}"
            ' ("absl/synchronization/mutex.h"), and annotate what it'
            " guards with ABSL_GUARDED_BY.",
        )
        for i, name in _std_names(code, frozenset(_MUTEX_TYPES))
    ]


# ----------------------------------------------------------------------------------------------------------------------
ABSL_FUNCTION = "absl-function"


def check_function(source: str, path: str) -> list[check_types.Finding]:
    """`std::function`."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i],
            ABSL_FUNCTION,
            "`std::function`: use absl::AnyInvocable for a callable that is"
            " stored, absl::FunctionRef for one only called during the call"
            " that receives it. Keep std::function with a NOLINT where an"
            " OCS2 or third-party API takes one or the callable is copied.",
        )
        for i, _ in _std_names(code, frozenset({"function"}))
    ]


# ----------------------------------------------------------------------------------------------------------------------
ABSL_NUMBER_CONVERSION = "absl-number-conversion"
_NUMBER_CONVERSIONS = {
    "to_string": "absl::StrCat",
    "stoi": "absl::SimpleAtoi",
    "stol": "absl::SimpleAtoi",
    "stoll": "absl::SimpleAtoi",
    "stoul": "absl::SimpleAtoi",
    "stoull": "absl::SimpleAtoi",
    "stof": "absl::SimpleAtof",
    "stod": "absl::SimpleAtod",
    "stold": "absl::SimpleAtod",
}


def check_number_conversion(
    source: str, path: str
) -> list[check_types.Finding]:
    """`std::to_string` and the `std::stoi` family."""
    code = cpp_source.code_tokens(source)
    findings = []
    for i, name in _std_names(code, frozenset(_NUMBER_CONVERSIONS)):
        reason = ""
        if name != "to_string":
            reason = (
                ", which reports failure by its bool result instead of"
                " throwing"
            )
        findings.append(
            _finding(
                path,
                code[i],
                ABSL_NUMBER_CONVERSION,
                f"`std::{name}`: use {_NUMBER_CONVERSIONS[name]}{reason}.",
            )
        )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
ABSL_STRING_STREAM = "absl-string-stream"
_STRING_STREAMS = frozenset(
    {
        "stringstream",
        "ostringstream",
        "istringstream",
        "basic_stringstream",
        "basic_ostringstream",
        "basic_istringstream",
    }
)


def check_string_stream(source: str, path: str) -> list[check_types.Finding]:
    """The `std::stringstream` family."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i],
            ABSL_STRING_STREAM,
            f"`std::{name}`: build strings with absl::StrCat / StrFormat /"
            " StrJoin and parse them with absl::StrSplit and the"
            " absl::SimpleAto* functions.",
        )
        for i, name in _std_names(code, _STRING_STREAMS)
    ]


# ----------------------------------------------------------------------------------------------------------------------
ABSL_VIEW_TYPE = "absl-view-type"
_VIEW_TYPES = {"string_view": "absl::string_view", "span": "absl::Span"}


def check_view_type(source: str, path: str) -> list[check_types.Finding]:
    """`std::string_view` and `std::span`."""
    code = cpp_source.code_tokens(source)
    return [
        _finding(
            path,
            code[i],
            ABSL_VIEW_TYPE,
            f"`std::{name}`: use {_VIEW_TYPES[name]}.",
        )
        for i, name in _std_names(code, frozenset(_VIEW_TYPES))
    ]


# ----------------------------------------------------------------------------------------------------------------------
def _check(
    name: str, function: check_types.CheckFunction, description: str
) -> check_types.Check:
    return check_types.Check(
        name=name,
        languages=CPP,
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=function,
        description=description,
    )


CHECKS = [
    _check(
        ABSL_SPAN_PARAM,
        check_span_param,
        "absl::Span<const T>, not const std::vector<T>& parameters (ToTW #93).",
    ),
    _check(
        ABSL_STRING_VIEW_PARAM,
        check_string_view_param,
        "absl::string_view, not const std::string& parameters (ToTW #1).",
    ),
    _check(
        ABSL_ORDERED_CONTAINER,
        check_ordered_container,
        "absl::btree_map / btree_set, not std::map / std::set.",
    ),
    _check(
        ABSL_MUTEX,
        check_mutex,
        "absl::Mutex / MutexLock / CondVar, not std::mutex and its locks.",
    ),
    _check(
        ABSL_FUNCTION,
        check_function,
        "absl::AnyInvocable / FunctionRef, not std::function.",
    ),
    _check(
        ABSL_NUMBER_CONVERSION,
        check_number_conversion,
        "absl::StrCat / SimpleAto*, not std::to_string / std::stoi.",
    ),
    _check(
        ABSL_STRING_STREAM,
        check_string_stream,
        "absl::StrCat / StrFormat / StrSplit, not std::stringstream.",
    ),
    _check(
        ABSL_VIEW_TYPE,
        check_view_type,
        "absl::string_view / Span, not std::string_view / std::span.",
    ),
]
