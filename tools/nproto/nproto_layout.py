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

"""The line breaking of clang-format, for the lines of C++ that nproto generates.

nproto's generated code must be one that clang-format leaves unchanged under the repository's .clang-format (Google
style at 80 columns, BinPackParameters off), but the generator runs inside a Bazel action without clang-format. So it
breaks its long lines itself, the way clang-format 18 does: this module is a port of clang-format's optimizing line
formatter (clang/lib/Format: TokenAnnotator's split penalties, fake parentheses and break rules, ContinuationIndenter's
indentation rules and UnwrappedLineFormatter's search for the cheapest layout), cut down to the tokens and the three kinds
of line the generator writes:

- a statement or a control line (`if (...) {`, `case ...:`) in a function body (Kind.STATEMENT);
- a function declaration or definition at namespace scope (Kind.FUNCTION);
- a member declaration in a struct (Kind.MEMBER).

The port keeps clang-format's names in its comments, so that a change of clang-format's behavior can be traced to the
function it comes from. tools/nproto/test/test_generated_code.py checks the result with the real clang-format.
"""

from collections.abc import Sequence
import dataclasses
import enum
import heapq
import re
from typing import Any

# The repository's .clang-format.
# LINT.IfChange(column_limit)
COLUMN_LIMIT = 80
# LINT.ThenChange(//.clang-format:column_limit)

# Google style (clang/lib/Format/Format.cpp: getGoogleStyle()).
_CONTINUATION_INDENT_WIDTH = 4
_INDENT_WIDTH = 2
_PENALTY_EXCESS_CHARACTER = 1_000_000
_PENALTY_BREAK_ASSIGNMENT = 2
_PENALTY_BREAK_BEFORE_FIRST_CALL_PARAMETER = 1
_PENALTY_BREAK_SCOPE_RESOLUTION = 500
_PENALTY_RETURN_TYPE_ON_ITS_OWN_LINE = 200
# The extra penalty of the first line break inside a parenthesis level (ContinuationIndenter::addTokenOnNewLine()).
_PENALTY_FIRST_BREAK_ON_LEVEL = 15
# SplitPenalty grows by this much per BindingStrength (TokenAnnotator::calculateFormattingInformation()).
_PENALTY_PER_BINDING_STRENGTH = 20

# Operator precedences (clang/Basic/OperatorPrecedence.h) and the two of TokenAnnotator's ExpressionParser.
_PREC_UNKNOWN = 0
_PREC_COMMA = 1
_PREC_ASSIGNMENT = 2
_PREC_CONDITIONAL = 3
_PREC_BITWISE_AND = 8
_PREC_EQUALITY = 9
_PREC_RELATIONAL = 10
_PREC_SPACESHIP = 11
_PREC_ADDITIVE = 13
_PREC_MULTIPLICATIVE = 14
_PREC_UNARY_OPERATOR = 16
_PREC_ARROW_AND_PERIOD = 17

# The binding strength each bracket adds to the tokens inside it (AnnotatingParser's ScopedContextCreator).
_BINDING_OF = {"(": 1, "{": 1, "<": 12}

_KEYWORDS = frozenset(
    """bool case class const constexpr default double else enum false float for if inline int namespace operator
    return static static_cast struct switch true void while""".split()
)
# The attribute macros of .clang-format (AttributeMacros).
# LINT.IfChange(attribute_macros)
_ATTRIBUTE_MACROS = frozenset(
    {
        "__capability",
        "absl_nonnull",
        "absl_nullable",
        "absl_nullability_unknown",
    }
)
# LINT.ThenChange(//.clang-format:attribute_macros)
_LEXEME = re.compile(
    r'"(?:[^"\\]|\\.)*"|/\*.*?\*/|[A-Za-z_][A-Za-z_0-9]*|[0-9][A-Za-z_0-9.]*|::|->|!=|==|\S'
)


class Kind(enum.Enum):
    """The kind of a line, which decides how clang-format reads its tokens."""

    STATEMENT = enum.auto()
    FUNCTION = enum.auto()
    MEMBER = enum.auto()


class _Type(enum.Enum):
    """The token types of clang-format (FormatToken.h's TT_*) that the generated code has."""

    NONE = enum.auto()
    TEMPLATE_OPENER = enum.auto()
    TEMPLATE_CLOSER = enum.auto()
    BINARY_OPERATOR = enum.auto()
    UNARY_OPERATOR = enum.auto()
    POINTER_OR_REFERENCE = enum.auto()
    START_OF_NAME = enum.auto()
    FUNCTION_DECLARATION_NAME = enum.auto()
    OVERLOADED_OPERATOR = enum.auto()
    OVERLOADED_OPERATOR_LPAREN = enum.auto()
    ATTRIBUTE_MACRO = enum.auto()
    BLOCK_BRACE = enum.auto()


@dataclasses.dataclass
class _Token:  # pylint: disable=too-many-instance-attributes  # One field per FormatToken member the port reads.
    """One token of the line with what TokenAnnotator computes for it."""

    text: str
    spaces: int
    type: _Type = _Type.NONE
    nesting_level: int = 0
    binding_strength: int = 1
    matching: int | None = None
    parameter_count: int = 0
    parent_bracket: str = ""
    fake_lparens: list[int] = dataclasses.field(default_factory=list)
    fake_rparens: int = 0
    operator_index: int = 0
    next_operator: int | None = None
    starts_binary_expression: bool = False
    ends_binary_expression: bool = False
    can_break_before: bool = False
    split_penalty: int = 0
    total_length: int = 0

    @property
    def width(self) -> int:
        """The columns of the token."""
        return len(self.text)

    def precedence(self) -> int:
        """FormatToken::getPrecedence(): the binary-operator precedence of the token's kind."""
        return {
            ",": _PREC_COMMA,
            "=": _PREC_ASSIGNMENT,
            "!=": _PREC_EQUALITY,
            "==": _PREC_EQUALITY,
            "<": _PREC_RELATIONAL,
            ">": _PREC_RELATIONAL,
            "&": _PREC_BITWISE_AND,
            "*": _PREC_MULTIPLICATIVE,
            "-": _PREC_ADDITIVE,
        }.get(self.text, _PREC_UNKNOWN)

    def is_binary_operator(self) -> bool:
        """FormatToken::isBinaryOperator()."""
        return self.precedence() > _PREC_COMMA

    def opens_scope(self) -> bool:
        """FormatToken::opensScope()."""
        return self.text in ("(", "{") or self.type == _Type.TEMPLATE_OPENER

    def closes_scope(self) -> bool:
        """FormatToken::closesScope()."""
        return self.text in (")", "}") or self.type == _Type.TEMPLATE_CLOSER

    def is_member_access(self) -> bool:
        """FormatToken::isMemberAccess()."""
        return self.text in (".", "->")

    def is_comment(self) -> bool:
        """Whether the token is a /* ... */ comment."""
        return self.text.startswith("/*")

    def is_string(self) -> bool:
        """Whether the token is a string literal."""
        return self.text.startswith('"')

    def is_identifier(self) -> bool:
        """Whether the token is an identifier, not a keyword."""
        return (
            bool(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.text))
            and self.text not in _KEYWORDS
        )


class _Line:
    """The annotated tokens of one line (TokenAnnotator::annotate() and calculateFormattingInformation())."""

    def __init__(self, text: str, kind: Kind):
        self.indent = len(text) - len(text.lstrip(" "))
        self.kind = kind
        self.must_be_declaration = kind != Kind.STATEMENT
        self.might_be_function_declaration = kind == Kind.FUNCTION
        self.tokens: list[_Token] = []
        for match in _LEXEME.finditer(text, self.indent):
            spaced = (
                match.start() > self.indent and text[match.start() - 1] == " "
            )
            self.tokens.append(_Token(match.group(), int(spaced)))
        if not self.tokens:
            return
        self._match_brackets()
        self._assign_types()
        _ExpressionParser(self.tokens).parse()
        self._calculate_formatting_information()

    def previous(self, index: int) -> _Token | None:
        """The token before the one at `index`, or None for the first."""
        return self.tokens[index - 1] if index > 0 else None

    def _match_brackets(self) -> None:
        """MatchingParen, NestingLevel, BindingStrength, ParameterCount and ParentBracket of every token."""
        stack: list[int] = []
        binding = 1
        for index, token in enumerate(self.tokens):
            if (
                token.text in (")", "}", ">")
                and stack
                and self.tokens[stack[-1]].text
                == {")": "(", "}": "{", ">": "<"}[token.text]
            ):
                token.nesting_level = len(stack)
                token.binding_strength = binding
                opener = stack.pop()
                binding -= _BINDING_OF[self.tokens[opener].text]
                token.matching = opener
                self.tokens[opener].matching = index
                if token.text == ">":
                    token.type = _Type.TEMPLATE_CLOSER
                    self.tokens[opener].type = _Type.TEMPLATE_OPENER
                continue
            token.nesting_level = len(stack)
            token.binding_strength = binding
            if stack:
                opener_token = self.tokens[stack[-1]]
                if token.text == ",":
                    opener_token.parameter_count += 1
                elif (
                    opener_token.parameter_count == 0 and not token.is_comment()
                ):
                    opener_token.parameter_count = 1
            if token.text in ("(", "{", "<"):
                token.parent_bracket = (
                    self.tokens[stack[-1]].text if stack else ""
                )
                stack.append(index)
                binding += _BINDING_OF[token.text]

    def _assign_types(self) -> None:
        """The token types of AnnotatingParser::determineTokenType(), for the generated code's lines."""
        tokens = self.tokens
        function_name = None
        if self.kind == Kind.FUNCTION:
            for index, token in enumerate(tokens):
                if token.text == "(" and token.nesting_level == 0:
                    function_name = index - 1
                    if index >= 2 and tokens[index - 2].text == "operator":
                        function_name = index - 2
                    break
        for index, token in enumerate(tokens):
            previous = self.previous(index)
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if token.text in _ATTRIBUTE_MACROS:
                token.type = _Type.ATTRIBUTE_MACRO
            elif index == function_name:
                token.type = _Type.FUNCTION_DECLARATION_NAME
            elif (
                previous is not None
                and previous.text == "operator"
                and token.text in ("==", "!=")
            ):
                token.type = _Type.OVERLOADED_OPERATOR
            elif (
                previous is not None
                and previous.type == _Type.OVERLOADED_OPERATOR
                and token.text == "("
            ):
                token.type = _Type.OVERLOADED_OPERATOR_LPAREN
            elif token.text in ("=", "!=", "==") and token.type == _Type.NONE:
                token.type = _Type.BINARY_OPERATOR
            elif token.text in ("&", "*", "!", "-"):
                token.type = self._star_amp_usage(index)
            elif token.text == "{" and (
                following is None or following.text == "}"
            ):
                token.type = _Type.BLOCK_BRACE
            elif (
                token.text == "}"
                and previous is not None
                and previous.type == _Type.BLOCK_BRACE
            ):
                token.type = _Type.BLOCK_BRACE
            elif token.is_identifier() and self._is_start_of_name(index):
                token.type = _Type.START_OF_NAME

    def _star_amp_usage(self, index: int) -> _Type:
        """A `&`, `*`, `!` or `-`: a pointer or reference in a declaration, a unary operator in an expression."""
        token = self.tokens[index]
        previous = self.previous(index)
        if (
            token.text in ("&", "*")
            and self.must_be_declaration
            and previous is not None
        ):
            if (
                previous.is_identifier()
                or previous.type == _Type.TEMPLATE_CLOSER
                or previous.text in _KEYWORDS - {"return"}
            ):
                return _Type.POINTER_OR_REFERENCE
        return _Type.UNARY_OPERATOR

    def _is_start_of_name(self, index: int) -> bool:
        """The declared name of a declaration: a member, or a parameter of a function."""
        if not self.must_be_declaration:
            return False
        tokens = self.tokens
        previous = self.previous(index)
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if previous is None or following is None:
            return False
        named = (
            previous.is_identifier()
            or previous.type
            in (
                _Type.POINTER_OR_REFERENCE,
                _Type.TEMPLATE_CLOSER,
                _Type.ATTRIBUTE_MACRO,
            )
            or previous.text in ("double", "float", "bool", "int")
        )
        if self.kind == Kind.MEMBER:
            return (
                named
                and _ends_declarator(following)
                and tokens[index].nesting_level == 0
            )
        return (
            named
            and following.text in (",", ")")
            and tokens[index].nesting_level == 1
        )

    def _calculate_formatting_information(self) -> None:
        """CanBreakBefore, TotalLength and SplitPenalty of every token."""
        tokens = self.tokens
        tokens[0].total_length = self.indent + tokens[0].width
        for index in range(1, len(tokens)):
            token = tokens[index]
            token.can_break_before = self._can_break_before(index)
            token.total_length = (
                tokens[index - 1].total_length + token.width + token.spaces
            )
            token.split_penalty = (
                self._split_penalty(index)
                + _PENALTY_PER_BINDING_STRENGTH * token.binding_strength
            )

    def _can_break_before(self, index: int) -> bool:
        """TokenAnnotator::canBreakBefore()."""
        right = self.tokens[index]
        left = self.tokens[index - 1]
        if right.type == _Type.POINTER_OR_REFERENCE:
            return False  # PointerAlignment: Left
        if (
            right.type in (_Type.START_OF_NAME, _Type.FUNCTION_DECLARATION_NAME)
            or right.text == "operator"
        ):
            return True
        if left.type == _Type.POINTER_OR_REFERENCE:
            return False
        if right.text == ":":
            return False
        if right.type in (
            _Type.OVERLOADED_OPERATOR_LPAREN,
            _Type.OVERLOADED_OPERATOR,
        ):
            return False
        if (
            left.type in (_Type.TEMPLATE_CLOSER, _Type.UNARY_OPERATOR)
            or left.text == "operator"
        ):
            return False
        if (
            left.text == "("
            and index >= 2
            and self.tokens[index - 2].type == _Type.BINARY_OPERATOR
        ):
            return False
        if right.type == _Type.TEMPLATE_CLOSER:
            return False
        if right.text == "}":
            return right.type == _Type.BLOCK_BRACE
        if right.text == ")":
            return False
        if right.type == _Type.ATTRIBUTE_MACRO:
            return True
        if left.is_identifier() and right.is_string():
            return True
        if (
            left.is_binary_operator() or left.type == _Type.BINARY_OPERATOR
        ) and left.text != "<<":
            return True
        return (
            left.text in (",", "::", ";", "{", "class", "struct")
            or left.is_comment()
            or right.is_member_access()
            or right.text in (":", "[")
            or (
                left.text == ")"
                and (right.is_identifier() or right.text == "const")
            )
            or (left.text == "(" and right.text != ")")
            # A template closer, which cannot follow a break, returned above.
            or left.type == _Type.TEMPLATE_OPENER
        )

    def _split_penalty(self, index: int) -> int:
        """TokenAnnotator::splitPenalty(), without the BindingStrength term."""
        right = self.tokens[index]
        left = self.tokens[index - 1]
        in_function_declaration = self.might_be_function_declaration
        if left.text == ";":
            return 0
        if left.text == "::":
            return _PENALTY_BREAK_SCOPE_RESOLUTION
        if (
            right.type in (_Type.START_OF_NAME, _Type.FUNCTION_DECLARATION_NAME)
            or right.text == "operator"
        ):
            if left.type == _Type.START_OF_NAME:
                return 110
            if in_function_declaration and right.nesting_level == 0:
                return _PENALTY_RETURN_TYPE_ON_ITS_OWN_LINE
            return 200
        if right.type == _Type.POINTER_OR_REFERENCE:
            return 190
        if left.text in ("class", "struct"):
            return 5000
        if left.is_comment():
            return 1000
        if right.is_member_access():
            following = right.next_operator
            closes = (
                following is not None
                and self.tokens[following - 1].closes_scope()
            )
            return 35 if closes else 150
        if left.text == "(" and in_function_declaration:
            return 100
        if (
            left.text == "("
            and index >= 2
            and self.tokens[index - 2].text in ("for", "if")
        ):
            return 1000
        if left.text == "=" and in_function_declaration:
            return 110
        if right.text == "}":
            return 1
        if left.type == _Type.TEMPLATE_OPENER:
            return 100
        if left.opens_scope():
            return (
                _PENALTY_BREAK_BEFORE_FIRST_CALL_PARAMETER
                if left.parameter_count > 1
                else 19
            )
        if left.type == _Type.UNARY_OPERATOR:
            return 60
        if left.text == ",":
            return 1
        level = left.precedence() or right.precedence()
        if level == _PREC_ASSIGNMENT:
            return _PENALTY_BREAK_ASSIGNMENT
        if level != _PREC_UNKNOWN:
            return level
        return 3


def _ends_declarator(token: _Token) -> bool:
    """Whether `token` ends the declarator of a member: its initializer or its semicolon."""
    return token.text in ("=", ";")


class _ExpressionParser:
    """TokenAnnotator's ExpressionParser: the fake parentheses around the operands of each operator."""

    def __init__(self, tokens: list[_Token]):
        self.tokens = tokens
        self.current = 0

    def _token(self) -> _Token | None:
        """The current token, or None at the end of the line."""
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return None

    def _current_precedence(self) -> int:
        """ExpressionParser::getCurrentPrecedence(): the operator precedence of the current token, -1 for none."""
        token = self._token()
        if token is None:
            return -1
        if token.text == ";":
            return 0
        if token.type == _Type.BINARY_OPERATOR or token.text == ",":
            return token.precedence()
        if token.is_member_access():
            return _PREC_ARROW_AND_PERIOD
        return -1

    def parse(self, precedence: int = 0) -> None:
        """ExpressionParser::parse(): the operands of the operators from `precedence` up, from the current token."""
        while (token := self._token()) is not None and token.text == "return":
            self.current += 1
        if self._token() is None or precedence > _PREC_ARROW_AND_PERIOD:
            return
        if precedence == _PREC_CONDITIONAL:
            # No conditional expressions: parseConditionalExpr() parses the next level.
            self.parse(precedence + 1)
            return
        if precedence == _PREC_UNARY_OPERATOR:
            self._parse_unary_operator()
            return
        start = self.current
        latest_operator: int | None = None
        operator_index = 0
        while self._token() is not None:
            self.parse(precedence + 1)
            current_precedence = self._current_precedence()
            token = self._token()
            if (
                token is None
                or (token.closes_scope() and token.matching is not None)
                or -1 < current_precedence < precedence
            ):
                break
            if token.opens_scope():
                self._parse_scope()
                continue
            if current_precedence == precedence:
                if latest_operator is not None:
                    self.tokens[latest_operator].next_operator = self.current
                latest_operator = self.current
                token.operator_index = operator_index
                operator_index += 1
            self.current += 1
        if latest_operator is not None and (
            self._token() is not None or precedence > 0
        ):
            # Call expressions have no binary operator precedence.
            if precedence == _PREC_ARROW_AND_PERIOD:
                precedence = _PREC_UNKNOWN
            self._add_fake_parenthesis(start, precedence)

    def _parse_scope(self) -> None:
        """Parses the brackets the current token opens, through the token that closes them."""
        while (token := self._token()) is not None and (
            not token.closes_scope() or token.opens_scope()
        ):
            self.current += 1
            self.parse()
        self.current += 1

    def _add_fake_parenthesis(self, start: int, precedence: int) -> None:
        """ExpressionParser::addFakeParenthesis(): the operand from `start` to the token before the current one."""
        self.tokens[start].fake_lparens.append(precedence)
        if precedence > _PREC_UNKNOWN:
            self.tokens[start].starts_binary_expression = True
        if self.current >= len(self.tokens):
            return  # The end of the line closes it: no token does.
        end = self.current - 1
        while end > start and self.tokens[end].is_comment():
            end -= 1
        self.tokens[end].fake_rparens += 1
        if precedence > _PREC_UNKNOWN:
            self.tokens[end].ends_binary_expression = True

    def _parse_unary_operator(self) -> None:
        """ExpressionParser::parseUnaryOperator(): a fake parenthesis from each unary operator on."""
        unary: list[int] = []
        while (
            token := self._token()
        ) is not None and token.type == _Type.UNARY_OPERATOR:
            unary.append(self.current)
            self.current += 1
        self.parse(_PREC_ARROW_AND_PERIOD)
        for start in reversed(unary):
            self._add_fake_parenthesis(start, _PREC_UNKNOWN)


@dataclasses.dataclass(frozen=True)
class _ParenState:  # pylint: disable=too-many-instance-attributes  # ContinuationIndenter.h's ParenState.
    """One level of the paren stack: a bracket of the line or a fake parenthesis around an operand."""

    opener: int | None
    indent: int
    last_space: int
    nested_block_indent: int
    avoid_bin_packing: bool
    no_line_break: bool
    is_aligned: bool = False
    break_before_parameter: bool = False
    no_line_break_in_operand: bool = False
    last_operator_wrapped: bool = True
    contains_line_break: bool = False
    contains_unwrapped_builder: bool = False
    nested_block_inlined: bool = False
    start_of_function_call: int = 0
    call_continuation: int = 0
    variable_pos: int = 0
    break_before_closing_brace: bool = False

    def key(self) -> tuple[object, ...]:
        """The fields ParenState::operator< compares (not the opener or NoLineBreakInOperand)."""
        return (
            self.indent,
            self.last_space,
            self.nested_block_indent,
            self.is_aligned,
            self.break_before_closing_brace,
            self.avoid_bin_packing,
            self.break_before_parameter,
            self.no_line_break,
            self.last_operator_wrapped,
            self.start_of_function_call,
            self.call_continuation,
            self.variable_pos,
            self.contains_line_break,
            self.contains_unwrapped_builder,
            self.nested_block_inlined,
        )


def _new_paren_state(
    opener: int | None,
    indent: int,
    last_space: int,
    avoid_bin_packing: bool,
    no_line_break: bool,
) -> _ParenState:
    """ParenState's constructor: a level whose nested blocks indent from `indent`."""
    return _ParenState(
        opener, indent, last_space, indent, avoid_bin_packing, no_line_break
    )


@dataclasses.dataclass
class _LineState:
    """ContinuationIndenter.h's LineState."""

    column: int
    next_token: int
    stack: list[_ParenState]
    first_indent: int
    start_of_string_literal: int = 0
    start_of_line_level: int = 0
    lowest_level_on_line: int = 0

    def copy(self) -> "_LineState":
        """A copy whose stack can change without changing this one's."""
        return dataclasses.replace(self, stack=list(self.stack))

    def key(self) -> tuple[object, ...]:
        """The fields LineState::operator< compares."""
        return (
            self.next_token,
            self.column,
            self.start_of_line_level,
            self.lowest_level_on_line,
            self.start_of_string_literal,
            tuple(state.key() for state in self.stack),
        )

    @property
    def top(self) -> _ParenState:
        """The innermost level of the paren stack."""
        return self.stack[-1]

    def update_top(self, **changes: Any) -> None:
        """Replaces the fields `changes` names of the innermost level."""
        self.stack[-1] = dataclasses.replace(self.stack[-1], **changes)


class _Indenter:
    """ContinuationIndenter: places one token after another, on the current line or a new one."""

    def __init__(self, line: _Line, limit: int):
        self.line = line
        self.tokens = line.tokens
        self.limit = limit

    def initial_state(self) -> _LineState:
        """ContinuationIndenter::getInitialState(): the state after the first token."""
        indent = self.line.indent
        state = _LineState(
            indent,
            0,
            [_new_paren_state(None, indent, indent, False, False)],
            indent,
        )
        self.move_state_to_next_token(state, newline=False)
        return state

    # ContinuationIndenter::canBreak()
    def can_break(self, state: _LineState) -> bool:
        """Whether the next token may start a new line."""
        current = self.tokens[state.next_token]
        previous = self.tokens[state.next_token - 1]
        top = state.top
        if not current.can_break_before:
            return False
        if (
            previous.opens_scope()
            and previous.text != "{"
            and state.lowest_level_on_line < state.start_of_line_level
            and state.lowest_level_on_line < current.nesting_level
        ):
            return False
        if current.is_member_access() and top.contains_unwrapped_builder:
            return False
        if current.type == _Type.FUNCTION_DECLARATION_NAME and state.column < 6:
            return False
        if (
            not (current.type == _Type.BINARY_OPERATOR or current.text == ",")
            and top.no_line_break_in_operand
        ):
            return False
        return not top.no_line_break

    def _starts_next_parameter(self, index: int) -> bool:
        """startsNextParameter(): whether the token at `index` follows a comma."""
        return self.tokens[index - 1].text == ","

    def _length_to_next_operator(self, index: int) -> int:
        """getLengthToNextOperator(): the columns from the token at `index` to the next operator of its level."""
        token = self.tokens[index]
        if token.next_operator is None:
            return 0
        return (
            self.tokens[token.next_operator].total_length - token.total_length
        )

    def _starts_segment_of_builder_type_call(self, index: int) -> bool:
        """startsSegmentOfBuilderTypeCall(): whether the token at `index` is a `.` or `->` after a closing bracket."""
        return (
            self.tokens[index].is_member_access()
            and self.tokens[index - 1].closes_scope()
        )

    # ContinuationIndenter::mustBreak()
    def must_break(self, state: _LineState) -> bool:
        """Whether the next token must start a new line."""
        index = state.next_token
        current = self.tokens[index]
        previous = self.tokens[index - 1]
        top = state.top
        if (
            (
                self._starts_next_parameter(index)
                or previous.text == ";"
                or (
                    previous.type == _Type.TEMPLATE_CLOSER
                    and current.type == _Type.START_OF_NAME
                    and previous.nesting_level == 1
                )
            )
            and top.break_before_parameter
            and current.text not in (")", "}")
        ):
            return True
        new_line_column = self.new_line_column(state)
        if (
            current.is_member_access()
            and state.column + self._length_to_next_operator(index) > self.limit
            and (
                state.column > new_line_column
                or current.nesting_level < state.start_of_line_level
            )
        ):
            return True
        if (
            self._starts_segment_of_builder_type_call(index)
            and (top.call_continuation != 0 or top.break_before_parameter)
            and not (
                previous.type == _Type.BLOCK_BRACE
                and state.column <= new_line_column
            )
        ):
            return True
        if state.column <= new_line_column:
            return False
        if previous.type == _Type.BINARY_OPERATOR and current.can_break_before:
            precedence = previous.precedence()
            if precedence != _PREC_ASSIGNMENT and top.break_before_parameter:
                before = self.tokens[index - 2] if index >= 2 else None
                if before is not None and before.ends_binary_expression:
                    return True
                is_comparison = (
                    precedence
                    in (_PREC_RELATIONAL, _PREC_EQUALITY, _PREC_SPACESHIP)
                    and before is not None
                    and before.type != _Type.BINARY_OPERATOR
                )
                if not is_comparison:
                    return True
        elif (
            current.type == _Type.BINARY_OPERATOR
            and current.can_break_before
            and top.break_before_parameter
        ):
            return True
        if (
            current.type == _Type.FUNCTION_DECLARATION_NAME
            and top.break_before_parameter
        ):
            return True
        if (
            previous.type == _Type.BLOCK_BRACE
            and previous.text == "{"
            and current.text != "}"
        ):
            return True
        return False

    # ContinuationIndenter::getNewLineColumn()
    def new_line_column(self, state: _LineState) -> int:
        """The column the next token starts at on a new line."""
        index = state.next_token
        current = self.tokens[index]
        previous = self.tokens[index - 1]
        top = state.top
        continuation_indent = (
            max(top.last_space, top.indent) + _CONTINUATION_INDENT_WIDTH
        )
        if current.text == "}" and len(state.stack) > 1:
            return state.stack[-2].nested_block_indent
        if current.is_string() and state.start_of_string_literal != 0:
            return state.start_of_string_literal
        if current.is_member_access():
            return (
                continuation_indent
                if top.call_continuation == 0
                else top.call_continuation
            )
        if previous.text == "," and top.variable_pos != 0:
            return top.variable_pos
        if (
            (previous.type == _Type.ATTRIBUTE_MACRO and current.text != "(")
            or current.type == _Type.FUNCTION_DECLARATION_NAME
            or current.text == "operator"
        ):
            return max(top.last_space, top.indent)
        if current.type in (
            _Type.START_OF_NAME,
            _Type.POINTER_OR_REFERENCE,
        ) or previous.text in ("::", "="):
            return continuation_indent
        if (
            previous.text == ")"
            and not current.is_binary_operator()
            and current.text != ":"
            and not current.is_comment()
        ):
            return continuation_indent
        if top.indent == state.first_indent and previous.text != "}":
            return top.indent + _CONTINUATION_INDENT_WIDTH
        return top.indent

    # ContinuationIndenter::addTokenToState()
    def add_token_to_state(self, state: _LineState, newline: bool) -> int:
        """Places the next token on a new line or the current one; returns the penalty."""
        penalty = (
            self.add_token_on_new_line(state)
            if newline
            else self.add_token_on_current_line(state)
        )
        return self.move_state_to_next_token(state, newline) + penalty

    # ContinuationIndenter::addTokenOnCurrentLine()
    def add_token_on_current_line(self, state: _LineState) -> int:
        """Places the next token on the current line; returns the penalty, none."""
        index = state.next_token
        current = self.tokens[index]
        previous = self.tokens[index - 1]
        top = state.top
        changes: dict[str, Any] = {}
        if (
            current.text == "="
            and current.nesting_level == 0
            and top.variable_pos == 0
        ):
            variable_pos = state.column
            back = index - 1
            while back >= 0 and variable_pos >= self.tokens[back].width:
                variable_pos -= self.tokens[back].width
                if self.tokens[back].spaces != 0:
                    break
                back -= 1
            changes["variable_pos"] = variable_pos
        spaces = current.spaces
        if previous.opens_scope():
            changes["indent"] = state.column + spaces
            changes["is_aligned"] = True
        if top.avoid_bin_packing and self._starts_next_parameter(index):
            changes["no_line_break"] = True
        if self._starts_segment_of_builder_type_call(
            index
        ) and state.column > self.new_line_column(state):
            changes["contains_unwrapped_builder"] = True
        if (
            current.is_member_access()
            and previous.text == ")"
            and previous.matching is not None
            and previous.total_length
            - self.tokens[previous.matching].total_length
            > 10
        ):
            changes["no_line_break"] = True
        if (
            not current.is_comment()
            and (previous.type == _Type.BINARY_OPERATOR or previous.text == ",")
            and previous.precedence()
            not in (_PREC_ASSIGNMENT, _PREC_RELATIONAL, _PREC_SPACESHIP)
        ):
            has_two_operands = (
                previous.operator_index == 0 and previous.next_operator is None
            )
            if not has_two_operands:
                changes["no_line_break_in_operand"] = True
        state.column += spaces
        before = self.tokens[index - 2] if index >= 2 else None
        if (
            not current.is_comment()
            and previous.text == "("
            and before is not None
            and before.text in ("for", "if")
        ):
            changes["last_space"] = state.column
            changes["nested_block_indent"] = state.column
        elif (
            not current.is_comment()
            and previous.text == ","
            and previous.type != _Type.OVERLOADED_OPERATOR
        ):
            changes["last_space"] = state.column
        elif previous.type == _Type.BINARY_OPERATOR and (
            previous.precedence() != _PREC_ASSIGNMENT
            or current.starts_binary_expression
        ):
            changes["last_space"] = state.column
        elif previous.opens_scope() and previous.matching is not None:
            after = previous.matching + 1
            if (
                after < len(self.tokens)
                and self.tokens[after].is_member_access()
                and len(state.stack) > 1
                and state.stack[-2].call_continuation == 0
            ):
                changes["last_space"] = state.column
        state.update_top(**changes)
        return 0

    # ContinuationIndenter::addTokenOnNewLine()
    def add_token_on_new_line(self, state: _LineState) -> int:
        """Places the next token on a new line; returns the penalty of the break."""
        index = state.next_token
        current = self.tokens[index]
        previous = self.tokens[index - 1]
        top = state.top
        penalty = (
            0 if top.contains_line_break else _PENALTY_FIRST_BREAK_ON_LEVEL
        )
        penalty += current.split_penalty
        state.column = self.new_line_column(state)
        changes: dict[str, Any] = {
            "contains_line_break": True,
            "nested_block_indent": state.column,
        }
        if current.is_member_access() and top.call_continuation == 0:
            changes["call_continuation"] = state.column
        break_before_parameter = top.break_before_parameter
        if (
            previous.text in (",", ";") and not top.avoid_bin_packing
        ) or previous.type == _Type.BINARY_OPERATOR:
            break_before_parameter = False
        if (
            previous.type == _Type.TEMPLATE_CLOSER
            and current.nesting_level == 0
        ):
            break_before_parameter = False
        if current.type == _Type.BINARY_OPERATOR and current.can_break_before:
            break_before_parameter = False
        changes["last_space"] = state.column
        state.start_of_line_level = current.nesting_level
        state.lowest_level_on_line = current.nesting_level
        state.stack[:-1] = [
            dataclasses.replace(parent, break_before_parameter=True)
            for parent in state.stack[:-1]
        ]
        if (
            previous.text not in (",", ":", ";")
            and (
                previous.type != _Type.TEMPLATE_CLOSER
                or current.nesting_level != 0
            )
            and previous.type != _Type.BINARY_OPERATOR
            and current.type != _Type.BINARY_OPERATOR
            and not previous.opens_scope()
        ):
            break_before_parameter = True
        if previous.text == "{":
            changes["break_before_closing_brace"] = True
        if top.avoid_bin_packing and not (
            previous.text in ("(", "{")
            or previous.type == _Type.BINARY_OPERATOR
        ):
            break_before_parameter = True
        changes["break_before_parameter"] = break_before_parameter
        state.update_top(**changes)
        return penalty

    # ContinuationIndenter::moveStateToNextToken()
    def move_state_to_next_token(self, state: _LineState, newline: bool) -> int:
        """Moves past the next token, already placed; returns the penalty of the columns past the limit."""
        index = state.next_token
        current = self.tokens[index]
        changes: dict[str, Any] = {}
        if current.text == "," or current.type == _Type.BINARY_OPERATOR:
            changes["no_line_break_in_operand"] = False
        if current.type == _Type.BINARY_OPERATOR:
            changes["last_operator_wrapped"] = newline
        if (
            not current.opens_scope()
            and not current.closes_scope()
            and current.type != _Type.POINTER_OR_REFERENCE
        ):
            state.lowest_level_on_line = min(
                state.lowest_level_on_line, current.nesting_level
            )
        if current.is_member_access():
            changes["start_of_function_call"] = (
                0 if current.next_operator is None else state.column
            )
        if current.type == _Type.BINARY_OPERATOR and newline:
            changes["nested_block_indent"] = state.column + current.width + 1
        state.update_top(**changes)
        previous = self.tokens[index - 1] if index > 0 else None
        if (
            not current.is_comment()
            and previous is not None
            and previous.text == "{"
            and len(state.stack) > 1
        ):
            if state.stack[-2].nested_block_inlined and newline:
                state.stack[:-1] = [
                    dataclasses.replace(parent, no_line_break=True)
                    for parent in state.stack[:-1]
                ]
            state.stack[-2] = dataclasses.replace(
                state.stack[-2], nested_block_inlined=False
            )
        if previous is not None and (
            previous.type == _Type.BINARY_OPERATOR
            or previous.text in ("(", ",", ":")
        ):
            inlined = previous.text != "(" or previous.parameter_count > 1
            state.update_top(nested_block_inlined=not newline and inlined)
        self.move_state_past_fake_lparens(state, newline)
        self.move_state_past_scope_closer(state)
        self.move_state_past_scope_opener(state)
        self.move_state_past_fake_rparens(state)
        if current.is_string() and state.start_of_string_literal == 0:
            state.start_of_string_literal = state.column
        elif (
            not current.is_comment()
            and not current.is_identifier()
            and not current.is_string()
        ):
            state.start_of_string_literal = 0
        state.column += current.width
        state.next_token += 1
        if state.column > self.limit:
            return _PENALTY_EXCESS_CHARACTER * (state.column - self.limit)
        return 0

    # ContinuationIndenter::moveStatePastFakeLParens()
    def move_state_past_fake_lparens(
        self, state: _LineState, newline: bool
    ) -> None:
        """Pushes a level for each fake parenthesis the next token opens."""
        index = state.next_token
        current = self.tokens[index]
        if not current.fake_lparens:
            return
        previous = self.tokens[index - 1] if index > 0 else None
        skip_first_extra_indent = previous is not None and (
            previous.opens_scope()
            or previous.text in (";", "return")
            or previous.precedence() == _PREC_ASSIGNMENT
        )
        for precedence in reversed(current.fake_lparens):
            top = state.top
            new = dataclasses.replace(
                top,
                opener=None,
                contains_line_break=False,
                last_operator_wrapped=True,
                no_line_break=top.no_line_break or top.no_line_break_in_operand,
            )
            if precedence > _PREC_COMMA:
                new = dataclasses.replace(new, avoid_bin_packing=False)
            if previous is None or previous.text != "return" or precedence > 0:
                new = dataclasses.replace(
                    new, indent=max(state.column, new.indent, top.last_space)
                )
            if (
                previous is not None
                and (
                    previous.precedence() == _PREC_ASSIGNMENT
                    or previous.text == "return"
                )
                and not newline
            ):
                new = dataclasses.replace(new, is_aligned=True)
            if precedence > _PREC_UNKNOWN:
                new = dataclasses.replace(
                    new, last_space=max(new.last_space, state.column)
                )
            if current.type != _Type.UNARY_OPERATOR:
                new = dataclasses.replace(
                    new, start_of_function_call=state.column
                )
            if not skip_first_extra_indent and precedence > _PREC_ASSIGNMENT:
                new = dataclasses.replace(
                    new, indent=new.indent + _CONTINUATION_INDENT_WIDTH
                )
            if (
                previous is not None and not previous.opens_scope()
            ) or precedence != _PREC_COMMA:
                new = dataclasses.replace(new, break_before_parameter=False)
            state.stack.append(new)
            skip_first_extra_indent = False

    # ContinuationIndenter::moveStatePastFakeRParens()
    def move_state_past_fake_rparens(self, state: _LineState) -> None:
        """Pops the levels of the fake parentheses the next token closes."""
        for _ in range(self.tokens[state.next_token].fake_rparens):
            if len(state.stack) == 1:
                break
            variable_pos = state.top.variable_pos
            state.stack.pop()
            state.update_top(variable_pos=variable_pos)

    # ContinuationIndenter::moveStatePastScopeOpener()
    def move_state_past_scope_opener(self, state: _LineState) -> None:
        """Pushes the level of the bracket the next token opens."""
        index = state.next_token
        current = self.tokens[index]
        if not current.opens_scope():
            return
        top = state.top
        if current.text == "{" and current.matching is not None:
            # moveStateToNewBlock()
            nested_block_indent = top.nested_block_indent
            new = _new_paren_state(
                index,
                nested_block_indent + _INDENT_WIDTH,
                top.last_space,
                True,
                False,
            )
            state.stack.append(
                dataclasses.replace(
                    new,
                    nested_block_indent=nested_block_indent,
                    break_before_parameter=True,
                )
            )
            return
        nested_block_indent = max(
            top.start_of_function_call, top.nested_block_indent
        )
        last_space = top.last_space
        if current.text == "{":
            new_indent = _INDENT_WIDTH + min(
                state.column, top.nested_block_indent
            )
            avoid_bin_packing = False
        else:
            new_indent = _CONTINUATION_INDENT_WIDTH + max(
                top.last_space, top.start_of_function_call
            )
            if current.text == "<" and current.parent_bracket == "(":
                new_indent = max(new_indent, top.indent)
                last_space = max(last_space, top.indent)
            avoid_bin_packing = self.line.must_be_declaration
        no_line_break = (
            top.no_line_break
            or top.no_line_break_in_operand
            or (
                current.type == _Type.TEMPLATE_OPENER
                and top.contains_unwrapped_builder
            )
        )
        new = _new_paren_state(
            index, new_indent, last_space, avoid_bin_packing, no_line_break
        )
        state.stack.append(
            dataclasses.replace(new, nested_block_indent=nested_block_indent)
        )

    # ContinuationIndenter::moveStatePastScopeCloser()
    def move_state_past_scope_closer(self, state: _LineState) -> None:
        """Pops the level of the bracket the next token closes."""
        index = state.next_token
        current = self.tokens[index]
        if not current.closes_scope():
            return
        if len(state.stack) > 1 and (current.text != "}" or index != 0):
            state.stack.pop()


@dataclasses.dataclass(order=True)
class _Node:
    """A state of the search, ordered by its penalty and then by when it was found (UnwrappedLineFormatter)."""

    penalty: int
    count: int
    state: _LineState = dataclasses.field(compare=False)
    newline: bool = dataclasses.field(compare=False)
    parent: "_Node | None" = dataclasses.field(compare=False)


def _solve(line: _Line, limit: int) -> list[bool] | None:
    """UnwrappedLineFormatter's analyzeSolutionSpace(): the newline decisions of the cheapest layout, per token."""
    indenter = _Indenter(line, limit)
    count = 0
    queue: list[_Node] = [
        _Node(0, count, indenter.initial_state(), False, None)
    ]
    count += 1
    seen: set[tuple[object, ...]] = set()
    while queue:
        node = heapq.heappop(queue)
        if node.state.next_token == len(line.tokens):
            decisions: list[bool] = []
            while node.parent is not None:
                decisions.append(node.newline)
                node = node.parent
            return decisions[::-1]
        key = node.state.key()
        if key in seen:
            continue
        seen.add(key)
        for newline in (False, True):
            if newline and not indenter.can_break(node.state):
                continue
            if not newline and indenter.must_break(node.state):
                continue
            state = node.state.copy()
            penalty = node.penalty + indenter.add_token_to_state(state, newline)
            heapq.heappush(queue, _Node(penalty, count, state, newline, node))
            count += 1
    return None


def _render(line: _Line, limit: int, decisions: Sequence[bool]) -> list[str]:
    """The lines of `line` with a new line before each token whose decision is True."""
    indenter = _Indenter(line, limit)
    state = indenter.initial_state()
    lines = [" " * line.indent + line.tokens[0].text]
    for newline, token in zip(decisions, line.tokens[1:]):
        column = state.column
        indenter.add_token_to_state(state, newline)
        if newline:
            lines.append(" " * (state.column - token.width) + token.text)
        else:
            lines[-1] += (
                " " * (state.column - token.width - column) + token.text
            )
    return lines


def wrap(
    text: str, kind: Kind = Kind.STATEMENT, limit: int = COLUMN_LIMIT
) -> list[str]:
    """`text`, one line of C++ with its indentation, broken into lines as clang-format breaks it.

    Args:
      text: The line, its tokens separated as clang-format separates them (one space after a comma, around `=`, ...).
      kind: What the line is, which decides how clang-format reads its tokens.
      limit: The column limit.

    Returns:
      The lines; `text` itself when it fits. A line may still be longer than `limit` when no layout fits (fits()).
    """
    if len(text) <= limit:
        return [text]
    line = _Line(text, kind)
    if len(line.tokens) < 2:
        return [text]
    decisions = _solve(line, limit)
    if decisions is None:
        return [text]
    return _render(line, limit, decisions)


def fits(lines: Sequence[str], limit: int = COLUMN_LIMIT) -> bool:
    """Whether every line of `lines` is within `limit` columns."""
    return all(len(line) <= limit for line in lines)
