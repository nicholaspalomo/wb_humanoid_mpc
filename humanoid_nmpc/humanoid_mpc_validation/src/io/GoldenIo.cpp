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

#include "humanoid_mpc_validation/io/GoldenIo.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"

#include "humanoid_mpc_validation/io/Sha256.h"

namespace ocs2::humanoid::validation {
namespace {

// LINT.IfChange(golden_format)
constexpr absl::string_view kMagicLine = "# wb_humanoid_mpc golden v1";
constexpr absl::string_view kGitCommitPrefix = "# git_commit: ";
constexpr absl::string_view kWorktreePrefix = "# worktree: ";
constexpr absl::string_view kConfigPrefix = "# config_sha256: ";
constexpr absl::string_view kNotePrefix = "# note ";
constexpr absl::string_view kMatrixPrefix = "matrix ";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/io/GoldenIo.h)

bool hasWhitespace(absl::string_view text) {
  for (const char c : text) {
    if (absl::ascii_isspace(static_cast<unsigned char>(c))) return true;
  }
  return false;
}

bool hasLineBreak(absl::string_view text) {
  return absl::StrContains(text, '\n') || absl::StrContains(text, '\r');
}

/** %.17g, with the non-finite values spelled as strtod reads them back. */
std::string formatValue(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value > 0.0 ? "inf" : "-inf";
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return buffer;
}

absl::Status lineError(size_t lineNumber, absl::string_view what) {
  return absl::InvalidArgumentError(absl::StrCat("[parseGoldenFile] line ", lineNumber, ": ", what));
}

}  // namespace

const golden_matrix_t* GoldenFile::find(absl::string_view label) const {
  for (const GoldenEntry& entry : entries) {
    if (entry.label == label) return &entry.value;
  }
  return nullptr;
}

absl::StatusOr<GoldenProvenance> makeGoldenProvenance(std::string gitCommit,
                                                      std::string worktreeState,
                                                      const std::vector<std::string>& configurationFiles) {
  GoldenProvenance provenance;
  provenance.gitCommit = std::move(gitCommit);
  provenance.worktreeState = std::move(worktreeState);
  for (const std::string& file : configurationFiles) {
    absl::StatusOr<std::string> hash = sha256HexOfFile(file);
    if (!hash.ok()) return hash.status();
    provenance.configurationHashes.emplace_back(file, *std::move(hash));
  }
  return provenance;
}

absl::StatusOr<std::string> formatGoldenFile(const GoldenFile& golden) {
  const GoldenProvenance& provenance = golden.provenance;
  if (hasLineBreak(provenance.gitCommit) || hasLineBreak(provenance.worktreeState)) {
    return absl::InvalidArgumentError("[formatGoldenFile] the commit and the worktree state must be single lines");
  }
  std::string text = absl::StrCat(kMagicLine, "\n");
  absl::StrAppend(&text, kGitCommitPrefix, provenance.gitCommit, "\n");
  absl::StrAppend(&text, kWorktreePrefix, provenance.worktreeState, "\n");
  for (const std::pair<std::string, std::string>& file : provenance.configurationHashes) {
    if (hasLineBreak(file.first) || hasWhitespace(file.second) || file.second.empty()) {
      return absl::InvalidArgumentError(absl::StrCat("[formatGoldenFile] invalid configuration hash entry for '", file.first, "'"));
    }
    absl::StrAppend(&text, kConfigPrefix, file.second, " ", file.first, "\n");
  }
  for (const std::pair<std::string, std::string>& note : provenance.notes) {
    if (note.first.empty() || hasWhitespace(note.first) || absl::StrContains(note.first, ':') || hasLineBreak(note.second)) {
      return absl::InvalidArgumentError(absl::StrCat("[formatGoldenFile] invalid note '", note.first, "'"));
    }
    absl::StrAppend(&text, kNotePrefix, note.first, ": ", note.second, "\n");
  }
  absl::flat_hash_set<std::string> labels;
  for (const GoldenEntry& entry : golden.entries) {
    if (entry.label.empty() || hasWhitespace(entry.label)) {
      return absl::InvalidArgumentError(absl::StrCat("[formatGoldenFile] the label '", entry.label, "' is empty or holds whitespace"));
    }
    if (!labels.insert(entry.label).second) {
      return absl::InvalidArgumentError(absl::StrCat("[formatGoldenFile] the label '", entry.label, "' is used twice"));
    }
    absl::StrAppend(&text, kMatrixPrefix, entry.label, " ", entry.value.rows(), " ", entry.value.cols(), "\n");
    if (entry.value.cols() == 0) continue;
    for (Eigen::Index row = 0; row < entry.value.rows(); ++row) {
      for (Eigen::Index col = 0; col < entry.value.cols(); ++col) {
        if (col > 0) text.push_back(' ');
        text += formatValue(entry.value(row, col));
      }
      text.push_back('\n');
    }
  }
  return text;
}

absl::StatusOr<GoldenFile> parseGoldenFile(absl::string_view text) {
  const std::vector<absl::string_view> lines = absl::StrSplit(text, '\n');
  GoldenFile golden;
  size_t index = 0;
  if (lines.empty() || lines[0] != kMagicLine) return lineError(/*lineNumber=*/1, absl::StrCat("expected '", kMagicLine, "'"));
  ++index;
  absl::flat_hash_set<std::string> labels;
  while (index < lines.size()) {
    const absl::string_view line = lines[index];
    const size_t lineNumber = index + 1;
    ++index;
    if (line.empty()) continue;
    absl::string_view rest = line;
    if (absl::ConsumePrefix(&rest, kGitCommitPrefix)) {
      golden.provenance.gitCommit = std::string(rest);
    } else if (absl::ConsumePrefix(&rest, kWorktreePrefix)) {
      golden.provenance.worktreeState = std::string(rest);
    } else if (absl::ConsumePrefix(&rest, kConfigPrefix)) {
      const size_t space = rest.find(' ');
      if (space == absl::string_view::npos) return lineError(lineNumber, "a configuration hash without a path");
      golden.provenance.configurationHashes.emplace_back(std::string(rest.substr(space + 1)), std::string(rest.substr(0, space)));
    } else if (absl::ConsumePrefix(&rest, kNotePrefix)) {
      const size_t colon = rest.find(": ");
      if (colon == absl::string_view::npos) return lineError(lineNumber, "a note without ': '");
      golden.provenance.notes.emplace_back(std::string(rest.substr(0, colon)), std::string(rest.substr(colon + 2)));
    } else if (absl::StartsWith(line, "#")) {
      continue;  // a comment of a later version
    } else if (absl::ConsumePrefix(&rest, kMatrixPrefix)) {
      const std::vector<absl::string_view> fields = absl::StrSplit(rest, ' ');
      int64_t rows = 0;
      int64_t cols = 0;
      if (fields.size() != 3 || !absl::SimpleAtoi(fields[1], &rows) || !absl::SimpleAtoi(fields[2], &cols) || rows < 0 || cols < 0) {
        return lineError(lineNumber, "expected 'matrix <label> <rows> <cols>'");
      }
      GoldenEntry entry;
      entry.label = std::string(fields[0]);
      if (!labels.insert(entry.label).second) return lineError(lineNumber, absl::StrCat("the label '", entry.label, "' is used twice"));
      entry.value.resize(rows, cols);
      if (cols > 0) {
        for (int64_t row = 0; row < rows; ++row) {
          if (index >= lines.size())
            return lineError(lineNumber, absl::StrCat("'", entry.label, "' ends after ", row, " of ", rows, " rows"));
          const std::vector<absl::string_view> values = absl::StrSplit(lines[index], ' ');
          if (static_cast<int64_t>(values.size()) != cols) {
            return lineError(index + 1, absl::StrCat("expected ", cols, " values of '", entry.label, "', found ", values.size()));
          }
          for (int64_t col = 0; col < cols; ++col) {
            const std::string value(values[col]);
            char* end = nullptr;
            entry.value(row, col) = std::strtod(value.c_str(), &end);
            if (value.empty() || end != value.c_str() + value.size()) {
              return lineError(index + 1, absl::StrCat("'", value, "' is not a number"));
            }
          }
          ++index;
        }
      }
      golden.entries.push_back(std::move(entry));
    } else {
      return lineError(lineNumber, absl::StrCat("unexpected line '", line, "'"));
    }
  }
  return golden;
}

absl::Status writeGoldenFile(const std::string& path, const GoldenFile& golden) {
  absl::StatusOr<std::string> text = formatGoldenFile(golden);
  if (!text.ok()) return text.status();
  const std::filesystem::path parent = std::filesystem::path(path).parent_path();
  std::error_code error;
  if (!parent.empty()) std::filesystem::create_directories(parent, error);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return absl::InternalError(absl::StrCat("[writeGoldenFile] cannot write '", path, "'"));
  file << *text;
  file.close();
  if (!file) return absl::InternalError(absl::StrCat("[writeGoldenFile] writing '", path, "' failed"));
  return absl::OkStatus();
}

absl::StatusOr<GoldenFile> readGoldenFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return absl::NotFoundError(absl::StrCat("[readGoldenFile] cannot read '", path, "'"));
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return parseGoldenFile(text);
}

}  // namespace ocs2::humanoid::validation
