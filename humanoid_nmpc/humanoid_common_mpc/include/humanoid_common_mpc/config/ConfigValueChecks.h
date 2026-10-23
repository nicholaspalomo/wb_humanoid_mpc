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

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

// The range checks of the conversions of the typed configuration (src/config/<area>/): each returns OK when `value`
// is in its range, and otherwise InvalidArgument naming the field by `path` (its path from the file message, e.g.
// "multiple_shooting.dt") and saying the range. Every check is written so that it passes only when its comparisons
// hold, so NaN, which compares false with everything, is refused by each of them.

/** OK when `value` is finite. */
absl::Status requireFinite(absl::string_view path, double value);

/** OK when `value` is finite and greater than 0. */
absl::Status requirePositive(absl::string_view path, double value);

/** OK when `value` is finite and at least 0. */
absl::Status requireNonNegative(absl::string_view path, double value);

/** OK when `value` lies in the open interval (low, high); `low` < `high` are finite. */
absl::Status requireInOpenInterval(absl::string_view path, double value, double low, double high);

/** OK when `value` lies in the half-open interval [low, high); `low` < `high` are finite. */
absl::Status requireInHalfOpenInterval(absl::string_view path, double value, double low, double high);

/** OK when `value` lies in the half-open interval (low, high]; `low` < `high` are finite. */
absl::Status requireInLeftOpenInterval(absl::string_view path, double value, double low, double high);

/**
 * OK when `lower` <= `upper`, both already checked finite: the bounds of an interval the file gives as two fields, named
 * by `lowerPath` and `upperPath`.
 */
absl::Status requireOrdered(absl::string_view lowerPath, double lower, absl::string_view upperPath, double upper);

}  // namespace ocs2::humanoid
