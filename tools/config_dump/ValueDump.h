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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::config_dump {

/**
 * The values of a loaded configuration as text, one line per value (tools/config_dump/README.md), so that two dumps
 * compare line by line with `diff`:
 *
 *   ValueDump dump;
 *   dump.addSection("model_settings");
 *   dump.addDouble("model_settings.phaseTransitionStanceTime", settings.phaseTransitionStanceTime);
 *   dump.addMatrix("initial_state", initialState);
 *   dump.addFingerprint("lq@0.3.cost.dfdxx", model.cost.dfdxx);
 *
 * Lines are `<path> = <value>`: a double as `%a (%.17g)`, so that equal lines are equal bits; an integer and a bool as
 * written; a string quoted and escaped; a vector or matrix one line per element, `<path>[<i>]` or `<path>(<row>,<col>)`,
 * after its size. A matrix too large to read element by element is dumped as its fingerprint: its size, its nonzeros,
 * its sum and a hash of its elements' bits, which a change of one bit of one element changes. Not thread-safe.
 */
class ValueDump {
 public:
  /** A `[name]` line, which starts a section of the dump. */
  void addSection(absl::string_view name);
  void addDouble(absl::string_view path, double value);
  void addInteger(absl::string_view path, int64_t value);
  void addBool(absl::string_view path, bool value);
  void addString(absl::string_view path, absl::string_view value);
  void addStrings(absl::string_view path, const std::vector<std::string>& values);
  void addDoubles(absl::string_view path, const std::vector<double>& values);
  void addSizes(absl::string_view path, const std::vector<size_t>& values);
  /** Every element of `matrix`, row by row; a vector (one column) as `<path>[<i>]`. Its size first. */
  void addMatrix(absl::string_view path, const Eigen::Ref<const Eigen::MatrixXd>& matrix);
  /** The diagonal of the square `matrix`, `<path>[<i>]`, after its size. */
  void addDiagonal(absl::string_view path, const Eigen::Ref<const Eigen::MatrixXd>& matrix);
  /** `<path> = <rows>x<cols> nonzeros <n> sum <%a> hash <16 hex digits>`: matrixHash() and its summary. */
  void addFingerprint(absl::string_view path, const Eigen::Ref<const Eigen::MatrixXd>& matrix);

  /** The lines so far, each ending in a newline. */
  const std::string& text() const { return text_; }

 private:
  void addLine(absl::string_view path, absl::string_view value);

  std::string text_;
};

/** The 64-bit FNV-1a hash of `matrix`'s size and of the IEEE-754 bits of its elements, row by row. */
uint64_t matrixHash(const Eigen::Ref<const Eigen::MatrixXd>& matrix);

/** `prefix.name`, or `name` for an empty prefix: the paths of nested values. */
std::string joinPath(absl::string_view prefix, absl::string_view name);

}  // namespace ocs2::humanoid::config_dump
