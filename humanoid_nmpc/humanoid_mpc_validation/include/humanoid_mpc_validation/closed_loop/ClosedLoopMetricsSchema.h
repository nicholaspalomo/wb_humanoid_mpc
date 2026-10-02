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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

/** The `schema` member of every closed-loop metrics document; a change of the fields below is a new version. */
inline constexpr absl::string_view kClosedLoopMetricsSchemaName = "humanoid_mpc_validation.closed_loop_metrics.v1";

/** What a schema field holds. */
enum class JsonFieldKind {
  kNumber,          ///< always a number
  kNullableNumber,  ///< a number, or null where there was nothing to measure (no samples, a formulation without it)
  kBool,
  kString,
  kOpenObject,  ///< an object whose members are free (the provenance, the settings)
};

/** One field of the schema, by its dotted path ("solve_time_ms.total.p99"). */
struct JsonSchemaField {
  std::string path;
  JsonFieldKind kind;
};

/** Every field of a closed-loop metrics document, as ClosedLoopMetrics::report() writes it. */
const std::vector<JsonSchemaField>& closedLoopMetricsSchema();

/**
 * OK when `document` is a closed-loop metrics document of this version: its `schema` is kClosedLoopMetricsSchemaName,
 * every field of closedLoopMetricsSchema() is present with its kind, and it holds no value the schema does not list
 * (the members of an open object are free). InvalidArgument naming every field that is missing, mistyped or unknown.
 */
absl::Status validateClosedLoopMetrics(const JsonValue& document);

}  // namespace ocs2::humanoid::validation
