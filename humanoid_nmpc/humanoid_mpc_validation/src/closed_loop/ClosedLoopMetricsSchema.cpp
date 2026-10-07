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

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetricsSchema.h"

#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace ocs2::humanoid::validation {
namespace {

std::vector<JsonSchemaField> makeSchema() {
  std::vector<JsonSchemaField> fields;
  const std::vector<std::string> phases = {"total", "lq_approximation", "solve_qp", "linesearch", "compute_controller"};
  // LINT.IfChange(metrics_schema)
  fields.push_back({"schema", JsonFieldKind::kString});
  fields.push_back({"label", JsonFieldKind::kString});
  fields.push_back({"robot", JsonFieldKind::kString});
  fields.push_back({"formulation", JsonFieldKind::kString});
  fields.push_back({"scenario", JsonFieldKind::kString});
  fields.push_back({"provenance", JsonFieldKind::kOpenObject});
  fields.push_back({"settings", JsonFieldKind::kOpenObject});
  fields.push_back({"evaluation.start_time_s", JsonFieldKind::kNullableNumber});
  fields.push_back({"evaluation.end_time_s", JsonFieldKind::kNullableNumber});
  fields.push_back({"evaluation.control_cycles", JsonFieldKind::kNumber});
  fields.push_back({"evaluation.solves", JsonFieldKind::kNumber});
  fields.push_back({"survival.survived", JsonFieldKind::kBool});
  fields.push_back({"survival.fall_time_s", JsonFieldKind::kNullableNumber});
  fields.push_back({"survival.fall_reason", JsonFieldKind::kString});
  fields.push_back({"base_height.mean_m", JsonFieldKind::kNullableNumber});
  fields.push_back({"base_height.std_m", JsonFieldKind::kNullableNumber});
  fields.push_back({"base_height.rms_error_m", JsonFieldKind::kNullableNumber});
  fields.push_back({"tilt.rms_rad", JsonFieldKind::kNullableNumber});
  fields.push_back({"tilt.max_rad", JsonFieldKind::kNullableNumber});
  fields.push_back({"velocity.rms_error_mps", JsonFieldKind::kNullableNumber});
  fields.push_back({"yaw_rate.rms_error_radps", JsonFieldKind::kNullableNumber});
  fields.push_back({"heading.cumulative_final_rad", JsonFieldKind::kNullableNumber});
  fields.push_back({"heading.cumulative_max_rad", JsonFieldKind::kNullableNumber});
  fields.push_back({"heading.cumulative_min_rad", JsonFieldKind::kNullableNumber});
  fields.push_back({"stance_foot_slip.max_m", JsonFieldKind::kNullableNumber});
  fields.push_back({"stance_foot_slip.rms_m", JsonFieldKind::kNullableNumber});
  fields.push_back({"stance_foot_slip.stance_phases", JsonFieldKind::kNumber});
  fields.push_back({"joint_torque.rms_nm", JsonFieldKind::kNullableNumber});
  fields.push_back({"initial_state_gap.max_rotation_rad", JsonFieldKind::kNullableNumber});
  fields.push_back({"initial_state_gap.max_rotation_time_s", JsonFieldKind::kNullableNumber});
  fields.push_back({"initial_state_gap.max_norm", JsonFieldKind::kNullableNumber});
  fields.push_back({"quaternion_norm.max_deviation", JsonFieldKind::kNullableNumber});
  fields.push_back({"non_finite_values", JsonFieldKind::kNumber});
  for (const std::string& phase : phases) {
    for (const std::string& statistic : {std::string("mean"), std::string("p50"), std::string("p99"), std::string("max")}) {
      fields.push_back({absl::StrCat("solve_time_ms.", phase, ".", statistic), JsonFieldKind::kNullableNumber});
    }
  }
  fields.push_back({"failures.failed_solves", JsonFieldKind::kNumber});
  fields.push_back({"failures.resets_served", JsonFieldKind::kNumber});
  fields.push_back({"failures.full_resets_served", JsonFieldKind::kNumber});
  fields.push_back({"failures.simulator_resets", JsonFieldKind::kNumber});
  fields.push_back({"failures.unhealthy_cycles", JsonFieldKind::kNumber});
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopMetrics.cpp:metrics_document, //humanoid_nmpc/humanoid_mpc_validation/README.md:metrics_schema)
  // clang-format on
  return fields;
}

bool hasKind(const JsonValue& value, JsonFieldKind kind) {
  switch (kind) {
    case JsonFieldKind::kNumber:
      return value.isNumber();
    case JsonFieldKind::kNullableNumber:
      return value.isNumber() || value.isNull();
    case JsonFieldKind::kBool:
      return value.isBool();
    case JsonFieldKind::kString:
      return value.isString();
    case JsonFieldKind::kOpenObject:
      return value.isObject();
  }
  return false;
}

/** Collects the paths of `value` the schema does not know: leaves, and objects that are neither listed nor open. */
void collectUnknown(const JsonValue& value,
                    const std::string& path,
                    const absl::flat_hash_map<std::string, JsonFieldKind>& known,
                    std::vector<std::string>& unknown) {
  if (!path.empty()) {
    const absl::flat_hash_map<std::string, JsonFieldKind>::const_iterator field = known.find(path);
    if (field != known.end()) return;  // a listed field, checked by kind; an open object's members are free
  }
  if (!value.isObject()) {
    unknown.push_back(path.empty() ? "<document>" : path);
    return;
  }
  for (size_t i = 0; i < value.size(); ++i) {
    collectUnknown(value.valueAt(i), path.empty() ? value.keyAt(i) : absl::StrCat(path, ".", value.keyAt(i)), known, unknown);
  }
}

}  // namespace

const std::vector<JsonSchemaField>& closedLoopMetricsSchema() {
  static const absl::NoDestructor<std::vector<JsonSchemaField>> kSchema(makeSchema());
  return *kSchema;
}

absl::Status validateClosedLoopMetrics(const JsonValue& document) {
  if (!document.isObject()) return absl::InvalidArgumentError("[validateClosedLoopMetrics] the document is not a JSON object");
  std::vector<std::string> problems;
  const JsonValue* absl_nullable schema = document.find("schema");
  if (schema == nullptr || !schema->isString() || schema->asString() != kClosedLoopMetricsSchemaName) {
    problems.push_back(absl::StrCat("schema is not '", kClosedLoopMetricsSchemaName, "'"));
  }
  absl::flat_hash_map<std::string, JsonFieldKind> known;
  for (const JsonSchemaField& field : closedLoopMetricsSchema()) {
    known.emplace(field.path, field.kind);
    const JsonValue* absl_nullable value = document.findPath(field.path);
    if (value == nullptr) {
      problems.push_back(absl::StrCat(field.path, " is missing"));
    } else if (!hasKind(*value, field.kind)) {
      problems.push_back(absl::StrCat(field.path, " has the wrong type"));
    }
  }
  std::vector<std::string> unknown;
  collectUnknown(document, /*path=*/"", known, unknown);
  for (const std::string& path : unknown) problems.push_back(absl::StrCat(path, " is not in the schema"));
  if (problems.empty()) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat("[validateClosedLoopMetrics] ", absl::StrJoin(problems, "; ")));
}

}  // namespace ocs2::humanoid::validation
