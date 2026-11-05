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

// The macros expand to absl::Status, absl::StatusOr and std::move where they
// are used, which include-cleaner does not see.
#include <utility>  // IWYU pragma: keep

#include "absl/status/status.h"    // IWYU pragma: keep
#include "absl/status/statusor.h"  // IWYU pragma: keep

#define STATUS_MACROS_CONCAT_NAME_INNER(x, y) x##y
#define STATUS_MACROS_CONCAT_NAME(x, y) STATUS_MACROS_CONCAT_NAME_INNER(x, y)

// Not const, so that the returned Status is moved rather than copied
// (performance-no-automatic-move).
#define RETURN_IF_ERROR(expr)      \
  do {                             \
    absl::Status _status = (expr); \
    if (!_status.ok()) {           \
      return _status;              \
    }                              \
  } while (0)

// `status_var` is a name and `decl` a declaration, which parentheses would
// break; they are used bare where declared.
#define ASSIGN_OR_RETURN_IMPL(status_var, decl, rexpr)                \
  /* NOLINTNEXTLINE(bugprone-macro-parentheses): declares the name */ \
  auto status_var = (rexpr);                                          \
  if (!(status_var).ok()) {                                           \
    return (status_var).status();                                     \
  }                                                                   \
  /* NOLINTNEXTLINE(bugprone-macro-parentheses): a declaration */     \
  decl = std::move(*(status_var))

#define ASSIGN_OR_RETURN(decl, rexpr)                                        \
  ASSIGN_OR_RETURN_IMPL(STATUS_MACROS_CONCAT_NAME(_status_or_, __COUNTER__), \
                        decl, rexpr)
