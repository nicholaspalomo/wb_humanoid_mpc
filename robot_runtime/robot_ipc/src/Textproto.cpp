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

#include "robot_runtime/robot_ipc/src/Textproto.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/text_format.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace robot::ipc {
namespace {

// The protobuf tokenizer counts lines and columns from 0; editors and compilers count them from 1.
constexpr int kFirstLine = 1;
constexpr int kFirstColumn = 1;

// Collects the parser's errors as "<source>:<line>:<column>: <problem>" instead of letting it log them.
class TextprotoErrorCollector final : public google::protobuf::io::ErrorCollector {
 public:
  explicit TextprotoErrorCollector(absl::string_view source) : source_(source) {}

  void RecordError(int line, google::protobuf::io::ColumnNumber column, absl::string_view message) override {
    record(line, column, message);
  }

  // Strict: whatever the parser would only warn about is an error of the file too.
  void RecordWarning(int line, google::protobuf::io::ColumnNumber column, absl::string_view message) override {
    record(line, column, message);
  }

  const std::vector<std::string>& errors() const { return errors_; }

 private:
  void record(int line, google::protobuf::io::ColumnNumber column, absl::string_view message) {
    // A problem of the whole message (such as a missing proto2 required field) has no position.
    if (line < 0) {
      errors_.push_back(absl::StrCat(source_, ": ", message));
    } else {
      errors_.push_back(absl::StrCat(source_, ":", line + kFirstLine, ":", column + kFirstColumn, ": ", message));
    }
  }

  const std::string source_;
  std::vector<std::string> errors_;
};

}  // namespace

absl::Status parseTextproto(absl::string_view text, absl::string_view source, google::protobuf::Message& message) {
  TextprotoErrorCollector errors(source);
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&errors);
  // The defaults already are the strict ones; set them anyway, so that the strictness does not hang on a default.
  parser.AllowUnknownField(/*allow=*/false);
  parser.AllowPartialMessage(/*allow=*/false);
  const bool parsed = parser.ParseFromString(text, &message);
  if (!errors.errors().empty()) {
    return absl::InvalidArgumentError(absl::StrJoin(errors.errors(), "\n"));
  }
  if (!parsed) {
    return absl::InvalidArgumentError(absl::StrCat(source, ": not a valid textproto of ", message.GetDescriptor()->full_name()));
  }
  return absl::OkStatus();
}

absl::Status loadTextproto(const std::string& path, google::protobuf::Message& message) {
  std::ifstream file(path);
  if (!file) {
    return absl::NotFoundError(absl::StrCat("cannot open '", path, "'"));
  }
  std::stringstream text;
  text << file.rdbuf();
  if (file.bad()) {
    return absl::DataLossError(absl::StrCat("cannot read '", path, "'"));
  }
  return parseTextproto(text.str(), path, message);
}

}  // namespace robot::ipc
