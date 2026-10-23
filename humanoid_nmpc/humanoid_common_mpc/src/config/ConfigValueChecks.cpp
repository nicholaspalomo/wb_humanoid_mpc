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

#include "humanoid_common_mpc/config/ConfigValueChecks.h"

#include <cmath>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

absl::Status requireFinite(absl::string_view path, double value) {
  if (std::isfinite(value)) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(path, ": ", value, " is not finite"));
}

absl::Status requirePositive(absl::string_view path, double value) {
  if (std::isfinite(value) && !(value <= 0.0)) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(path, ": ", value, " is not a finite positive number"));
}

absl::Status requireNonNegative(absl::string_view path, double value) {
  if (std::isfinite(value) && !(value < 0.0)) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(path, ": ", value, " is not a finite number of at least 0"));
}

absl::Status requireInOpenInterval(absl::string_view path, double value, double low, double high) {
  if (value > low && value < high) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(path, ": ", value, " is not in the open interval (", low, ", ", high, ")"));
}

absl::Status requireInHalfOpenInterval(absl::string_view path, double value, double low, double high) {
  if (value >= low && value < high) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(path, ": ", value, " is not in the interval [", low, ", ", high, ")"));
}

absl::Status requireInLeftOpenInterval(absl::string_view path, double value, double low, double high) {
  if (value > low && value <= high) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(path, ": ", value, " is not in the interval (", low, ", ", high, "]"));
}

absl::Status requireOrdered(absl::string_view lowerPath, double lower, absl::string_view upperPath, double upper) {
  if (lower <= upper) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(lowerPath, " (", lower, ") is greater than ", upperPath, " (", upper, ")"));
}

}  // namespace ocs2::humanoid
