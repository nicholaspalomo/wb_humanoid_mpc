/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include "nproto/Textproto.h"

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "nproto/retired_field.pb.h"
#include "nproto/retired_field_options.pb.h"

namespace nproto {
namespace {

// The protobuf tokenizer counts lines and columns from 0; editors and compilers count them from 1.
constexpr int kFirstLine = 1;
constexpr int kFirstColumn = 1;

// The problems TextFormat::Parser notices only after it has consumed the token at fault, and so reports at the token
// after it: an unknown field name at the ':' that follows it, a bad bool or enum value at the next field.
constexpr absl::string_view kReportedAfterTheirToken[] = {
    "Message type \"",                     // ... has no field named "x".
    "Invalid value for boolean field \"",  // ... Value: "maybe".
    "Unknown enumeration value of \"",     // ... for field "x".
    "Non-repeated field \"",               // ... is specified multiple times.
    "text format contains deprecated field \"",
};

bool isReportedAfterItsToken(absl::string_view message) {
  for (const absl::string_view prefix : kReportedAfterTheirToken) {
    if (absl::StartsWith(message, prefix)) {
      return true;
    }
  }
  // Field "y" is specified along with field "x", another member of oneof "o".
  return absl::StartsWith(message, "Field \"") && absl::StrContains(message, "\" is specified along with field \"");
}

// The tokenizer below reports nothing: the parser has already reported every problem of the text.
class SilentErrorCollector final : public google::protobuf::io::ErrorCollector {
 public:
  void RecordError(int /*line*/, google::protobuf::io::ColumnNumber /*column*/, absl::string_view /*message*/) override {}
};

struct Position {
  int line = 0;
  int column = 0;
};

// The start of the last token of `text` that starts before `position`, tokenized as TextFormat::Parser does; `position`
// itself when no token does.
Position previousTokenStart(absl::string_view text, Position position) {
  google::protobuf::io::ArrayInputStream input(text.data(), static_cast<int>(text.size()));
  SilentErrorCollector silent;
  google::protobuf::io::Tokenizer tokenizer(&input, &silent);
  tokenizer.set_allow_f_after_float(/*value=*/true);
  tokenizer.set_comment_style(google::protobuf::io::Tokenizer::SH_COMMENT_STYLE);
  Position previous = position;
  while (tokenizer.Next()) {
    const google::protobuf::io::Tokenizer::Token& token = tokenizer.current();
    if (token.line > position.line || (token.line == position.line && token.column >= position.column)) {
      break;
    }
    previous = Position{.line = token.line, .column = token.column};
  }
  return previous;
}

struct Problem {
  int line = 0;  // negative: a problem of the whole message, which has no position
  int column = 0;
  std::string message;
};

// The header line a configuration file names its message with (tools/nproto/README.md, "Textproto configuration
// files"), in its leading comment block.
// LINT.IfChange(header_lines)
constexpr absl::string_view kProtoMessageHeader = "proto-message:";
// LINT.ThenChange(//tools/nproto/nproto_textproto.py:header_lines, //tools/hooks/textproto_headers.py:header_lines)

struct HeaderMessage {
  int line = 0;  // 1-based
  std::string fullName;
};

// The message the leading comment block of `text` names with `# proto-message: <full name>`, if it names one.
std::optional<HeaderMessage> headerMessage(absl::string_view text) {
  int line = 0;
  absl::string_view rest = text;
  while (!rest.empty()) {
    const size_t end = rest.find('\n');
    absl::string_view current = rest.substr(0, end);
    rest = end == absl::string_view::npos ? absl::string_view() : rest.substr(end + 1);
    ++line;
    current = absl::StripAsciiWhitespace(current);
    if (current.empty()) {
      continue;
    }
    if (!absl::ConsumePrefix(&current, "#")) {
      return std::nullopt;  // The leading comment block has ended.
    }
    current = absl::StripLeadingAsciiWhitespace(current);
    if (absl::ConsumePrefix(&current, kProtoMessageHeader)) {
      return HeaderMessage{.line = line, .fullName = std::string(absl::StripAsciiWhitespace(current))};
    }
  }
  return std::nullopt;
}

// `name` snake-cased and lower-cased, as tools/hooks/proto_file_layout.py names files: "useDcmTerminalCost" ->
// "use_dcm_terminal_cost", "HTTPRequest" -> "http_request". The form in which a retired name and an unknown field name
// are compared.
std::string snakeCase(absl::string_view name) {
  std::string result;
  result.reserve(name.size() + 4);
  for (size_t i = 0; i < name.size(); ++i) {
    const char character = name[i];
    if (absl::ascii_isupper(character) && i > 0) {
      const char previous = name[i - 1];
      const bool nextIsLower = i + 1 < name.size() && absl::ascii_islower(name[i + 1]);
      if (absl::ascii_islower(previous) || absl::ascii_isdigit(previous) || (absl::ascii_isupper(previous) && nextIsLower)) {
        result.push_back('_');
      }
    }
    result.push_back(absl::ascii_tolower(character));
  }
  return result;
}

// The message type and the field name of TextFormat::Parser's `Message type "M" has no field named "x".`
struct UnknownField {
  std::string messageType;
  std::string fieldName;
};

std::optional<UnknownField> unknownField(absl::string_view problem) {
  constexpr absl::string_view kMiddle = "\" has no field named \"";
  if (!absl::ConsumePrefix(&problem, "Message type \"")) {
    return std::nullopt;
  }
  const size_t middle = problem.find(kMiddle);
  if (middle == absl::string_view::npos) {
    return std::nullopt;
  }
  const absl::string_view field = problem.substr(middle + kMiddle.size());
  const size_t close = field.find('"');
  if (close == absl::string_view::npos) {
    return std::nullopt;
  }
  return UnknownField{.messageType = std::string(problem.substr(0, middle)), .fieldName = std::string(field.substr(0, close))};
}

// The (nproto.retired_field) entry of `type` that retires `fieldName`, or null.
// LINT.IfChange(retired_options)
const RetiredField* absl_nullable findRetiredField(const google::protobuf::Descriptor& type, absl::string_view fieldName) {
  const std::string key = snakeCase(fieldName);
  const google::protobuf::MessageOptions& options = type.options();
  for (int i = 0; i < options.ExtensionSize(retired_field); ++i) {
    const RetiredField& retired = options.GetExtension(retired_field, i);
    if (snakeCase(retired.name()) == key) {
      return &retired;
    }
  }
  return nullptr;
}

// An unknown-field problem answered from the schema: "'x' is retired: <replacement>" for a retired name; otherwise the
// parser's message, with "Did you mean "y"?" when the snake-cased name is a field and the message's layout hint.
std::string explainProblem(const google::protobuf::DescriptorPool& pool, const std::string& message) {
  const std::optional<UnknownField> unknown = unknownField(message);
  if (!unknown.has_value()) {
    return message;
  }
  const google::protobuf::Descriptor* absl_nullable type = pool.FindMessageTypeByName(unknown->messageType);
  if (type == nullptr) {
    return message;
  }
  const RetiredField* absl_nullable retired = findRetiredField(*type, unknown->fieldName);
  if (retired != nullptr) {
    return absl::StrCat("'", unknown->fieldName, "' is retired: ", retired->replacement());
  }
  std::string explained = message;
  const std::string snake = snakeCase(unknown->fieldName);
  if (snake != unknown->fieldName && type->FindFieldByName(snake) != nullptr) {
    absl::StrAppend(&explained, " Did you mean \"", snake, "\"?");
  }
  const std::string& hint = type->options().GetExtension(retired_layout_hint);
  if (!hint.empty()) {
    absl::StrAppend(&explained, " ", hint);
  }
  return explained;
}
// LINT.ThenChange(//tools/nproto/retired_field_options.proto:retired_options, //tools/nproto/nproto_textproto.py:retired_options)

// Collects the parser's problems instead of letting it log them.
class TextprotoErrorCollector final : public google::protobuf::io::ErrorCollector {
 public:
  void RecordError(int line, google::protobuf::io::ColumnNumber column, absl::string_view message) override {
    problems_.push_back(Problem{.line = line, .column = column, .message = std::string(message)});
  }

  // Strict: what the parser would only warn about is an error of the file too.
  void RecordWarning(int line, google::protobuf::io::ColumnNumber column, absl::string_view message) override {
    problems_.push_back(Problem{.line = line, .column = column, .message = std::string(message)});
  }

  const std::vector<Problem>& problems() const { return problems_; }

 private:
  std::vector<Problem> problems_;
};

// "<source>:<line>:<column>: <problem>", the position being the token at fault, 1-based, the problem explained from
// the schema (explainProblem()).
std::string formatProblem(absl::string_view text,
                          absl::string_view source,
                          const google::protobuf::DescriptorPool& pool,
                          const Problem& problem) {
  const std::string explained = explainProblem(pool, problem.message);
  if (problem.line < 0) {
    return absl::StrCat(source, ": ", explained);
  }
  Position position{.line = problem.line, .column = problem.column};
  if (isReportedAfterItsToken(problem.message)) {
    position = previousTokenStart(text, position);
  }
  return absl::StrCat(source, ":", position.line + kFirstLine, ":", position.column + kFirstColumn, ": ", explained);
}

}  // namespace

absl::Status ParseTextprotoInto(absl::string_view text, absl::string_view sourceName, google::protobuf::Message* absl_nonnull message) {
  const google::protobuf::Descriptor* absl_nonnull descriptor = message->GetDescriptor();
  const std::optional<HeaderMessage> header = headerMessage(text);
  if (header.has_value() && header->fullName != descriptor->full_name()) {
    return absl::InvalidArgumentError(absl::StrCat(sourceName, ":", header->line, ":1: ", sourceName, " is a ", header->fullName,
                                                   " (its '# proto-message:' header), not a ", descriptor->full_name()));
  }
  TextprotoErrorCollector errors;
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&errors);
  // These are the parser's defaults; set them anyway, so that the strictness does not hang on a default.
  parser.AllowUnknownField(/*allow=*/false);
  parser.AllowUnknownExtension(/*allow=*/false);
  parser.AllowPartialMessage(/*allow=*/false);
  parser.AllowFieldNumber(/*allow=*/false);
  parser.AllowCaseInsensitiveField(/*allow=*/false);
  const bool parsed = parser.ParseFromString(text, message);
  if (!errors.problems().empty()) {
    std::vector<std::string> lines;
    lines.reserve(errors.problems().size());
    for (const Problem& problem : errors.problems()) {
      lines.push_back(formatProblem(text, sourceName, *descriptor->file()->pool(), problem));
    }
    return absl::InvalidArgumentError(absl::StrJoin(lines, "\n"));
  }
  if (!parsed) {
    return absl::InvalidArgumentError(absl::StrCat(sourceName, ": not a valid textproto of ", descriptor->full_name()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ReadTextFile(absl::string_view path) {
  const std::string pathString(path);
  std::ifstream file(pathString, std::ios::binary);
  if (!file) {
    const int error = errno;
    return absl::ErrnoToStatus(error == 0 ? ENOENT : error, absl::StrCat("cannot open '", path, "'"));
  }
  std::ostringstream text;
  text << file.rdbuf();
  if (file.bad()) {
    return absl::DataLossError(absl::StrCat("cannot read '", path, "'"));
  }
  return text.str();
}

std::string WriteTextproto(const google::protobuf::Message& message) {
  google::protobuf::TextFormat::Printer printer;
  printer.SetUseShortRepeatedPrimitives(/*use_short_repeated_primitives=*/true);
  printer.SetUseUtf8StringEscaping(/*as_utf8=*/true);
  std::string text;
  if (!printer.PrintToString(message, &text)) {
    text.clear();
  }
  return text;
}

}  // namespace nproto
