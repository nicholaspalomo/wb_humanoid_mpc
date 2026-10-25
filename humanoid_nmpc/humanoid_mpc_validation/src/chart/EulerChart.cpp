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

#include "humanoid_mpc_validation/chart/EulerChart.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid::validation {
namespace {

EulerStateSegment segment(std::string name,
                          EulerStateSegmentKind kind,
                          size_t dim) {
  return EulerStateSegment{.name = std::move(name), .kind = kind, .dim = dim};
}

/**
 * What the rows of a tuning block are in the Euler state: the rotation blocks
 * hold the ZYX Euler angles and, on the whole-body base_angular_velocity block
 * (body rates in the tuning, design decision D18), the ZYX Euler rates.
 */
EulerStateSegmentKind eulerSegmentKindOf(TuningSegmentKind kind) {
  switch (kind) {
    case TuningSegmentKind::kEuclidean:
      return EulerStateSegmentKind::kEuclidean;
    case TuningSegmentKind::kEulerZyxOrientation:
      return EulerStateSegmentKind::kEulerZyxAngles;
    case TuningSegmentKind::kZyxOrderedAngularVelocity:
      return EulerStateSegmentKind::kEulerZyxRates;
  }
  return EulerStateSegmentKind::kEuclidean;
}

/** Where each block starts in the Euler state and in the physical state, and
 * the angles each rate block belongs to. */
struct SegmentOffsets {
  std::vector<size_t> eulerStart;
  std::vector<size_t> physicalStart;
  /** For a kEulerZyxRates block, the index of the kEulerZyxAngles block before
   * it; the block's own index otherwise. */
  std::vector<size_t> anglesSegment;
};

/** The offsets of `layout`'s blocks; InvalidArgument for a malformed layout. */
absl::StatusOr<SegmentOffsets> layoutOffsets(const EulerStateLayout& layout) {
  SegmentOffsets offsets;
  size_t eulerRow = 0;
  size_t physicalRow = 0;
  bool hasAngles = false;
  size_t lastAngles = 0;
  for (size_t i = 0; i < layout.segments.size(); ++i) {
    const EulerStateSegment& block = layout.segments[i];
    if (block.kind != EulerStateSegmentKind::kEuclidean &&
        block.dim != kRotationTuningDim) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[EulerChart] the rotation block ", block.name, " of the ",
          layout.name, " layout has ", block.dim, " rows, not 3"));
    }
    if (block.kind == EulerStateSegmentKind::kEulerZyxRates && !hasAngles) {
      return absl::InvalidArgumentError(
          absl::StrCat("[EulerChart] the rate block ", block.name, " of the ",
                       layout.name, " layout has no angle block before it"));
    }
    if (block.kind == EulerStateSegmentKind::kEulerZyxAngles) {
      hasAngles = true;
      lastAngles = i;
    }
    offsets.eulerStart.push_back(eulerRow);
    offsets.physicalStart.push_back(physicalRow);
    offsets.anglesSegment.push_back(
        block.kind == EulerStateSegmentKind::kEulerZyxRates ? lastAngles : i);
    eulerRow += block.dim;
    physicalRow += block.kind == EulerStateSegmentKind::kEulerZyxAngles
                       ? block.dim + 1
                       : block.dim;
  }
  return offsets;
}

/** The offsets of `layout`'s blocks; InvalidArgument for a malformed layout or
 * a state that does not fit it. */
absl::StatusOr<SegmentOffsets> segmentOffsets(const vector_t& eulerState,
                                              const EulerStateLayout& layout) {
  if (static_cast<size_t>(eulerState.size()) != getEulerStateDim(layout)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[EulerChart] the ", layout.name, " Euler state has ",
        getEulerStateDim(layout), " rows, this one ", eulerState.size()));
  }
  return layoutOffsets(layout);
}

vector3_t anglesOf(const vector_t& eulerState,
                   const SegmentOffsets& offsets,
                   size_t segmentIndex) {
  return eulerState.segment<kEulerAnglesZyxDim>(static_cast<Eigen::Index>(
      offsets.eulerStart[offsets.anglesSegment[segmentIndex]]));
}

/**
 * The Jacobian of the chart in the coordinates `tangent` selects: the ambient
 * physical state (dPhi / dx_e, 1/2 G T_B on the angles) or its tangent (P_e,
 * T_B on the angles).
 */
enum class ChartCoordinates { kAmbient, kTangent };

absl::StatusOr<matrix_t> chartJacobian(const vector_t& eulerState,
                                       const EulerStateLayout& layout,
                                       ChartCoordinates coordinates) {
  absl::StatusOr<SegmentOffsets> offsets = segmentOffsets(eulerState, layout);
  if (!offsets.ok()) return offsets.status();
  const size_t eulerDim = getEulerStateDim(layout);
  const size_t rows = coordinates == ChartCoordinates::kAmbient
                          ? getPhysicalStateDim(layout)
                          : eulerDim;
  matrix_t jacobian = matrix_t::Zero(static_cast<Eigen::Index>(rows),
                                     static_cast<Eigen::Index>(eulerDim));
  size_t row = 0;
  for (size_t i = 0; i < layout.segments.size(); ++i) {
    const EulerStateSegment& block = layout.segments[i];
    const Eigen::Index column =
        static_cast<Eigen::Index>(offsets->eulerStart[i]);
    const Eigen::Index dim = static_cast<Eigen::Index>(block.dim);
    const Eigen::Index rowIndex = static_cast<Eigen::Index>(row);
    switch (block.kind) {
      case EulerStateSegmentKind::kEuclidean:
        jacobian.block(rowIndex, column, dim, dim).setIdentity();
        row += block.dim;
        break;
      case EulerStateSegmentKind::kEulerZyxAngles: {
        const vector3_t angles = eulerState.segment<kEulerAnglesZyxDim>(column);
        const matrix3_t rateMap =
            eulerZyxRateToLocalAngularVelocityMatrix(angles);
        if (coordinates == ChartCoordinates::kAmbient) {
          jacobian.block<kQuaternionCoefficients, kEulerAnglesZyxDim>(rowIndex,
                                                                      column) =
              quaternionTangentMap(quaternionFromEulerZyx(angles)) * rateMap;
          row += kQuaternionCoefficients;
        } else {
          jacobian.block<kRotationTangentDim, kEulerAnglesZyxDim>(
              rowIndex, column) = rateMap;
          row += kRotationTangentDim;
        }
        break;
      }
      case EulerStateSegmentKind::kEulerZyxRates: {
        const vector3_t angles = anglesOf(eulerState, *offsets, i);
        const Eigen::Index anglesColumn = static_cast<Eigen::Index>(
            offsets->eulerStart[offsets->anglesSegment[i]]);
        jacobian.block<kRotationTangentDim, kEulerAnglesZyxDim>(rowIndex,
                                                                anglesColumn) =
            localAngularVelocityEulerAngleDerivative(
                angles, eulerState.segment<kEulerAnglesZyxDim>(column));
        jacobian.block<kRotationTangentDim, kEulerAnglesZyxDim>(rowIndex,
                                                                column) =
            eulerZyxRateToLocalAngularVelocityMatrix(angles);
        row += kRotationTangentDim;
        break;
      }
    }
  }
  return jacobian;
}

}  // namespace

EulerStateLayout eulerStateLayoutOf(const StateInputLayout& layout) {
  // The Euler state is the tuning layout row for row; only the meaning of the
  // rotation blocks' rows is the Euler one.
  const TuningLayout tuning = tuningLayoutOf(layout);
  EulerStateLayout eulerLayout;
  eulerLayout.name = layout.mpc == StateInputLayout::Mpc::kCentroidal
                         ? "centroidal"
                         : "whole_body";
  for (const TuningSegment& block : tuning.segments) {
    eulerLayout.segments.push_back(
        segment(block.name, eulerSegmentKindOf(block.kind), block.tuningDim));
  }
  return eulerLayout;
}

size_t getEulerStateDim(const EulerStateLayout& layout) {
  size_t dim = 0;
  for (const EulerStateSegment& block : layout.segments) dim += block.dim;
  return dim;
}

size_t getPhysicalStateDim(const EulerStateLayout& layout) {
  size_t dim = 0;
  for (const EulerStateSegment& block : layout.segments) {
    dim += block.kind == EulerStateSegmentKind::kEulerZyxAngles ? block.dim + 1
                                                                : block.dim;
  }
  return dim;
}

absl::StatusOr<vector_t> physicalStateFromEulerState(
    const vector_t& eulerState, const EulerStateLayout& layout) {
  absl::StatusOr<SegmentOffsets> offsets = segmentOffsets(eulerState, layout);
  if (!offsets.ok()) return offsets.status();
  vector_t physical(static_cast<Eigen::Index>(getPhysicalStateDim(layout)));
  for (size_t i = 0; i < layout.segments.size(); ++i) {
    const EulerStateSegment& block = layout.segments[i];
    const Eigen::Index from = static_cast<Eigen::Index>(offsets->eulerStart[i]);
    const Eigen::Index to =
        static_cast<Eigen::Index>(offsets->physicalStart[i]);
    switch (block.kind) {
      case EulerStateSegmentKind::kEuclidean:
        physical.segment(to, static_cast<Eigen::Index>(block.dim)) =
            eulerState.segment(from, static_cast<Eigen::Index>(block.dim));
        break;
      case EulerStateSegmentKind::kEulerZyxAngles:
        physical.segment<kQuaternionCoefficients>(to) = quaternionFromEulerZyx(
            eulerState.segment<kEulerAnglesZyxDim>(from));
        break;
      case EulerStateSegmentKind::kEulerZyxRates:
        physical.segment<kRotationTangentDim>(to) =
            localAngularVelocityFromEulerZyxRates(
                anglesOf(eulerState, *offsets, i),
                eulerState.segment<kEulerAnglesZyxDim>(from));
        break;
    }
  }
  return physical;
}

absl::StatusOr<vector_t> eulerStateFromPhysicalState(
    const vector_t& physicalState, const EulerStateLayout& layout) {
  absl::StatusOr<SegmentOffsets> offsets = layoutOffsets(layout);
  if (!offsets.ok()) return offsets.status();
  if (static_cast<size_t>(physicalState.size()) !=
      getPhysicalStateDim(layout)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[EulerChart] the ", layout.name, " physical state has ",
        getPhysicalStateDim(layout), " rows, this one ", physicalState.size()));
  }
  vector_t euler(static_cast<Eigen::Index>(getEulerStateDim(layout)));
  // The angles come before the rates they belong to (layoutOffsets() checks
  // it), so a rate block reads its angles here.
  for (size_t i = 0; i < layout.segments.size(); ++i) {
    const EulerStateSegment& block = layout.segments[i];
    const Eigen::Index from =
        static_cast<Eigen::Index>(offsets->physicalStart[i]);
    const Eigen::Index to = static_cast<Eigen::Index>(offsets->eulerStart[i]);
    switch (block.kind) {
      case EulerStateSegmentKind::kEuclidean:
        euler.segment(to, static_cast<Eigen::Index>(block.dim)) =
            physicalState.segment(from, static_cast<Eigen::Index>(block.dim));
        break;
      case EulerStateSegmentKind::kEulerZyxAngles: {
        const vector4_t xi =
            physicalState.segment<kQuaternionCoefficients>(from);
        const scalar_t norm = xi.norm();
        if (!std::isfinite(norm) || norm == 0.0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "[EulerChart] the quaternion of the ", block.name,
              " block of the ", layout.name, " state is zero or not finite"));
        }
        euler.segment<kEulerAnglesZyxDim>(to) =
            eulerZyxFromQuaternion(vector4_t(xi / norm));
        break;
      }
      case EulerStateSegmentKind::kEulerZyxRates:
        euler.segment<kEulerAnglesZyxDim>(to) =
            eulerZyxRatesFromLocalAngularVelocity(
                anglesOf(euler, *offsets, i),
                physicalState.segment<kRotationTangentDim>(from));
        break;
    }
  }
  return euler;
}

absl::StatusOr<matrix_t> eulerChartJacobian(const vector_t& eulerState,
                                            const EulerStateLayout& layout) {
  return chartJacobian(eulerState, layout, ChartCoordinates::kAmbient);
}

absl::StatusOr<matrix_t> physicalTangentJacobian(
    const vector_t& eulerState, const EulerStateLayout& layout) {
  return chartJacobian(eulerState, layout, ChartCoordinates::kTangent);
}

matrix3_t localAngularVelocityEulerAngleDerivative(
    const vector3_t& eulerZyx, const vector3_t& eulerZyxRates) {
  // The chart is built through orientation/EulerBoundary.h only (design README
  // section 4.2).
  return localAngularVelocityEulerZyxJacobian(eulerZyx, eulerZyxRates);
}

}  // namespace ocs2::humanoid::validation
