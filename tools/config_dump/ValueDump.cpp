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

#include "tools/config_dump/ValueDump.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::config_dump {
namespace {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

/** `hash` with the eight bytes of `word` folded in, lowest byte first. */
uint64_t foldWord(uint64_t hash, uint64_t word) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (word >> (8 * byte)) & 0xffULL;
    hash *= kFnvPrime;
  }
  return hash;
}

}  // namespace

void ValueDump::addLine(absl::string_view path, absl::string_view value) {
  absl::StrAppend(&text_, path, " = ", value, "\n");
}

void ValueDump::addSection(absl::string_view name) {
  absl::StrAppend(&text_, "[", name, "]\n");
}

void ValueDump::addDouble(absl::string_view path, double value) {
  addLine(path, absl::StrFormat("%a (%.17g)", value, value));
}

void ValueDump::addInteger(absl::string_view path, int64_t value) {
  addLine(path, absl::StrCat(value));
}

void ValueDump::addBool(absl::string_view path, bool value) {
  addLine(path, value ? "true" : "false");
}

void ValueDump::addString(absl::string_view path, absl::string_view value) {
  addLine(path, absl::StrCat("\"", absl::CEscape(value), "\""));
}

void ValueDump::addStrings(absl::string_view path, const std::vector<std::string>& values) {
  addInteger(absl::StrCat(path, ".size"), static_cast<int64_t>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    addString(absl::StrCat(path, "[", i, "]"), values[i]);
  }
}

void ValueDump::addDoubles(absl::string_view path, const std::vector<double>& values) {
  addInteger(absl::StrCat(path, ".size"), static_cast<int64_t>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    addDouble(absl::StrCat(path, "[", i, "]"), values[i]);
  }
}

void ValueDump::addSizes(absl::string_view path, const std::vector<size_t>& values) {
  addInteger(absl::StrCat(path, ".size"), static_cast<int64_t>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    addInteger(absl::StrCat(path, "[", i, "]"), static_cast<int64_t>(values[i]));
  }
}

void ValueDump::addMatrix(absl::string_view path, const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  addLine(absl::StrCat(path, ".size"), absl::StrCat(matrix.rows(), "x", matrix.cols()));
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      const std::string element = matrix.cols() == 1 ? absl::StrCat(path, "[", row, "]") : absl::StrCat(path, "(", row, ",", col, ")");
      addDouble(element, matrix(row, col));
    }
  }
}

void ValueDump::addDiagonal(absl::string_view path, const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  addLine(absl::StrCat(path, ".size"), absl::StrCat(matrix.rows(), "x", matrix.cols()));
  const Eigen::Index size = std::min(matrix.rows(), matrix.cols());
  for (Eigen::Index i = 0; i < size; ++i) {
    addDouble(absl::StrCat(path, "[", i, "]"), matrix(i, i));
  }
}

void ValueDump::addFingerprint(absl::string_view path, const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  const Eigen::Index nonzeros = (matrix.array() != 0.0).count();
  addLine(path,
          absl::StrFormat("%dx%d nonzeros %d sum %a hash %016x", matrix.rows(), matrix.cols(), nonzeros, matrix.sum(), matrixHash(matrix)));
}

uint64_t matrixHash(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  uint64_t hash = kFnvOffsetBasis;
  hash = foldWord(hash, static_cast<uint64_t>(matrix.rows()));
  hash = foldWord(hash, static_cast<uint64_t>(matrix.cols()));
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      const double value = matrix(row, col);
      uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(value), "a double is 64 bits");
      std::memcpy(&bits, &value, sizeof(bits));
      hash = foldWord(hash, bits);
    }
  }
  return hash;
}

std::string joinPath(absl::string_view prefix, absl::string_view name) {
  return prefix.empty() ? std::string(name) : absl::StrCat(prefix, ".", name);
}

}  // namespace ocs2::humanoid::config_dump
