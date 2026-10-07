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

#include "humanoid_common_mpc/orientation/EulerBoundary.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_robotic_tools/common/RotationDerivativesTransforms.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"

namespace ocs2::humanoid {

namespace {

constexpr Eigen::Index kEulerDim = 3;
constexpr Eigen::Index kQuaternionDim = 4;
constexpr Eigen::Index kAngularVelocityDim = 3;

/** Rows of `segment` in the state. */
size_t stateDim(const TuningSegment& segment) {
  return segment.kind == TuningSegmentKind::kEulerZyxOrientation ? segment.tuningDim + 1 : segment.tuningDim;
}

/** "base_orientation 3 (yaw, pitch, roll)" in the tuning layout, "base_orientation 4 (quaternion x, y, z, w)" in the state. */
std::string describeSegment(const TuningSegment& segment, bool inState) {
  switch (segment.kind) {
    case TuningSegmentKind::kEuclidean:
      return absl::StrCat(segment.name, " ", segment.tuningDim);
    case TuningSegmentKind::kEulerZyxOrientation:
      return inState ? absl::StrCat(segment.name, " ", stateDim(segment), " (quaternion x, y, z, w)")
                     : absl::StrCat(segment.name, " ", segment.tuningDim, " (yaw, pitch, roll)");
    case TuningSegmentKind::kZyxOrderedAngularVelocity:
      return inState ? absl::StrCat(segment.name, " ", segment.tuningDim, " (body w_x, w_y, w_z)")
                     : absl::StrCat(segment.name, " ", segment.tuningDim, " (body w_z, w_y, w_x)");
  }
  return segment.name;
}

std::string describeLayout(const TuningLayout& layout, bool inState) {
  std::vector<std::string> segments;
  segments.reserve(layout.segments.size());
  for (const TuningSegment& segment : layout.segments) {
    segments.push_back(describeSegment(segment, inState));
  }
  const size_t rows = inState ? getTuningLayoutStateDim(layout) : getTuningLayoutDim(layout);
  return absl::StrCat(layout.name, layout.name.empty() ? "" : " ", inState ? "state layout" : "tuning layout", " (", rows,
                      " rows: ", absl::StrJoin(segments, ", "), ")");
}

/** A rotation block has exactly the 3 rows of its Euler angles or angular velocity. */
absl::Status checkRotationSegments(const TuningLayout& layout, absl::string_view caller) {
  for (const TuningSegment& segment : layout.segments) {
    if (segment.kind != TuningSegmentKind::kEuclidean && segment.tuningDim != static_cast<size_t>(kEulerDim)) {
      return absl::InvalidArgumentError(absl::StrCat("[", caller, "] the block '", segment.name, "' of the ",
                                                     describeLayout(layout, /*inState=*/false), " is a rotation with ", segment.tuningDim,
                                                     " rows; a rotation block has ", kEulerDim, "."));
    }
  }
  return absl::OkStatus();
}

}  // namespace

vector4_t quaternionFromEulerZyx(const vector3_t& eulerZyx) {
  return getQuaternionFromEulerAnglesZyx<scalar_t>(eulerZyx).coeffs();
}

vector3_t eulerZyxFromQuaternion(const vector4_t& xi) {
  const scalar_t x = xi(0);
  const scalar_t y = xi(1);
  const scalar_t z = xi(2);
  const scalar_t w = xi(3);
  // R(0, 0) and R(1, 0) = cos(theta) (cos(psi), sin(psi)). The expressions are those of the conversion of before the
  // switch, so that its output is reproduced bit for bit.
  const scalar_t r00 = 1.0 - 2.0 * (y * y + z * z);
  const scalar_t r10 = 2.0 * (w * z + x * y);
  const scalar_t yaw = std::atan2(r10, r00);
  if (r00 * r00 + r10 * r10 >= kGimbalLockCosPitch * kGimbalLockCosPitch) {
    const scalar_t pitch = std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
    const scalar_t roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    return vector3_t(yaw, pitch, roll);
  }
  // Gimbal lock: yaw is split from roll by round-off, but any yaw is exact once pitch and roll are read from
  // R_z(psi)^T R = R_y(theta) R_x(phi), whose (1, 0) entry -sin(psi) R(0, 0) + cos(psi) R(1, 0) is zero for this yaw.
  const scalar_t cosYaw = std::cos(yaw);
  const scalar_t sinYaw = std::sin(yaw);
  const scalar_t r01 = 2.0 * (x * y - w * z);
  const scalar_t r11 = 1.0 - 2.0 * (x * x + z * z);
  const scalar_t r02 = 2.0 * (x * z + w * y);
  const scalar_t r12 = 2.0 * (y * z - w * x);
  const scalar_t r20 = 2.0 * (x * z - w * y);
  const scalar_t pitch = std::clamp(std::atan2(-r20, cosYaw * r00 + sinYaw * r10), -M_PI_2, M_PI_2);
  const scalar_t roll = std::atan2(sinYaw * r02 - cosYaw * r12, cosYaw * r11 - sinYaw * r01);
  return vector3_t(yaw, pitch, roll);
}

matrix3_t eulerZyxRateToLocalAngularVelocityMatrix(const vector3_t& eulerZyx) {
  return getMappingFromEulerAnglesZyxDerivativeToLocalAngularVelocity<scalar_t>(eulerZyx);
}

vector3_t localAngularVelocityFromEulerZyxRates(const vector3_t& eulerZyx, const vector3_t& eulerZyxRates) {
  return getLocalAngularVelocityFromEulerAnglesZyxDerivatives<scalar_t>(eulerZyx, eulerZyxRates);
}

size_t getTuningLayoutDim(const TuningLayout& layout) {
  size_t rows = 0;
  for (const TuningSegment& segment : layout.segments) {
    rows += segment.tuningDim;
  }
  return rows;
}

size_t getTuningLayoutStateDim(const TuningLayout& layout) {
  size_t rows = 0;
  for (const TuningSegment& segment : layout.segments) {
    rows += stateDim(segment);
  }
  return rows;
}

std::string describeTuningLayout(const TuningLayout& layout) {
  return describeLayout(layout, /*inState=*/false);
}

absl::StatusOr<vector_t> stateFromTuningLayout(const vector_t& tuningVector, const TuningLayout& layout) {
  RETURN_IF_ERROR(checkRotationSegments(layout, "stateFromTuningLayout"));
  const size_t tuningDim = getTuningLayoutDim(layout);
  if (static_cast<size_t>(tuningVector.size()) != tuningDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[stateFromTuningLayout] the vector has ", tuningVector.size(), " entries, but the ", describeLayout(layout, /*inState=*/false),
        " has ", tuningDim,
        ". Each coordinate of the task file's initial_state, state_weights and final_state_weights is one row of this layout."));
  }
  vector_t state(static_cast<Eigen::Index>(getTuningLayoutStateDim(layout)));
  Eigen::Index tuningRow = 0;
  Eigen::Index stateRow = 0;
  for (const TuningSegment& segment : layout.segments) {
    const Eigen::Index rows = static_cast<Eigen::Index>(segment.tuningDim);
    switch (segment.kind) {
      case TuningSegmentKind::kEuclidean:
        state.segment(stateRow, rows) = tuningVector.segment(tuningRow, rows);
        stateRow += rows;
        break;
      case TuningSegmentKind::kEulerZyxOrientation:
        state.segment<kQuaternionDim>(stateRow) = quaternionFromEulerZyx(vector3_t(tuningVector.segment<kEulerDim>(tuningRow)));
        stateRow += kQuaternionDim;
        break;
      case TuningSegmentKind::kZyxOrderedAngularVelocity:
        state.segment<kAngularVelocityDim>(stateRow) = tuningVector.segment<kAngularVelocityDim>(tuningRow).reverse();
        stateRow += kAngularVelocityDim;
        break;
    }
    tuningRow += rows;
  }
  return state;
}

absl::StatusOr<vector_t> tuningLayoutFromState(const vector_t& state, const TuningLayout& layout) {
  RETURN_IF_ERROR(checkRotationSegments(layout, "tuningLayoutFromState"));
  const size_t expectedStateDim = getTuningLayoutStateDim(layout);
  if (static_cast<size_t>(state.size()) != expectedStateDim) {
    return absl::InvalidArgumentError(absl::StrCat("[tuningLayoutFromState] the state has ", state.size(), " entries, but the ",
                                                   describeLayout(layout, /*inState=*/false), " maps to the ",
                                                   describeLayout(layout, /*inState=*/true), "."));
  }
  vector_t tuningVector(static_cast<Eigen::Index>(getTuningLayoutDim(layout)));
  Eigen::Index tuningRow = 0;
  Eigen::Index stateRow = 0;
  for (const TuningSegment& segment : layout.segments) {
    const Eigen::Index rows = static_cast<Eigen::Index>(segment.tuningDim);
    switch (segment.kind) {
      case TuningSegmentKind::kEuclidean:
        tuningVector.segment(tuningRow, rows) = state.segment(stateRow, rows);
        stateRow += rows;
        break;
      case TuningSegmentKind::kEulerZyxOrientation:
        tuningVector.segment<kEulerDim>(tuningRow) =
            eulerZyxFromQuaternion(safelyNormalizedQuaternion(vector4_t(state.segment<kQuaternionDim>(stateRow))));
        stateRow += kQuaternionDim;
        break;
      case TuningSegmentKind::kZyxOrderedAngularVelocity:
        tuningVector.segment<kAngularVelocityDim>(tuningRow) = state.segment<kAngularVelocityDim>(stateRow).reverse();
        stateRow += kAngularVelocityDim;
        break;
    }
    tuningRow += rows;
  }
  return tuningVector;
}

}  // namespace ocs2::humanoid
