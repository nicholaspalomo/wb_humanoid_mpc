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

#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

// Test assertions on absl::Status and absl::StatusOr, the checked replacement
// of `.value()` in tests (ToTW #181). Each one is a fatal gtest assertion: it
// ends the current test function, so it is used in a TEST body or in a helper
// that returns void.

// Asserts that `expr`, an absl::Status, is OK, and prints the status when it is
// not. `expr` is evaluated once.
#define HUMANOID_ASSERT_OK(expr)                                            \
  HUMANOID_ASSERT_OK_IMPL(STATUS_MACROS_CONCAT_NAME(_status_, __COUNTER__), \
                          expr)

#define HUMANOID_ASSERT_OK_IMPL(status_var, expr)   \
  do {                                              \
    const absl::Status status_var = (expr);         \
    ASSERT_TRUE((status_var).ok()) << (status_var); \
  } while (0)

// Asserts that `rexpr`, an absl::StatusOr<T>, is OK and declares or assigns its
// value: `decl` is a declaration such as `const ModelSettings settings` or an
// lvalue. The value is moved out of the StatusOr.
#define HUMANOID_ASSERT_OK_AND_ASSIGN(decl, rexpr) \
  HUMANOID_ASSERT_OK_AND_ASSIGN_IMPL(              \
      STATUS_MACROS_CONCAT_NAME(_status_or_, __COUNTER__), decl, rexpr)

// `status_var` is a name and `decl` a declaration, which parentheses would
// break; they are used bare where declared.
#define HUMANOID_ASSERT_OK_AND_ASSIGN_IMPL(status_var, decl, rexpr)        \
  auto status_var =                                                        \
      (rexpr); /* NOLINT(bugprone-macro-parentheses): declares the name */ \
  ASSERT_TRUE((status_var).ok()) << (status_var).status();                 \
  decl = std::move(                                                        \
      *(status_var)) /* NOLINT(bugprone-macro-parentheses): a declaration */

namespace ocs2::humanoid {

/**
 * Returns the value of `statusOr`, moved out, and ends the test process with
 * the status when it is not OK: the checked replacement of `.value()` where a
 * fatal gtest assertion cannot stand, in a member initializer list, a
 * constructor, a helper that returns a value or the middle of an expression.
 * Prefer HUMANOID_ASSERT_OK_AND_ASSIGN in a TEST body.
 */
template <typename T>
T checkedValue(absl::StatusOr<T> statusOr) {
  ABSL_CHECK_OK(statusOr.status());
  return *std::move(statusOr);
}

}  // namespace ocs2::humanoid
