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

#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"

#include <cstddef>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid {
namespace {

/** Tuning rows of a rotation block: (yaw, pitch, roll) of the orientation, (z, y, x) of the angular velocity. */
constexpr size_t kRotationTuningDim = 3;

/** The angular-velocity rows' residual e_w and the reference angular velocity a it subtracts, for its Jacobian. */
struct AngularVelocityError {
  /** e_w = P_3 (w_B - a), rows (z, y, x). */
  vector3_t residual;
  /** a = R(xi)^T R(xi_r) w_B,r: the reference's body angular velocity, in the body frame of xi. */
  vector3_t referenceInBody;
};

/**
 * e_w of design decision D8 for the unit base orientations `orientation` (xi) and `referenceOrientation` (xi_r) and the
 * body angular velocities `angularVelocity` (w_B) and `referenceAngularVelocity` (w_B,r). R(-xi) = R(xi) coefficient
 * for coefficient, so the result is invariant under the sign of either quaternion, bit for bit.
 */
AngularVelocityError angularVelocityError(const vector4_t& orientation,
                                          const vector4_t& referenceOrientation,
                                          const vector3_t& angularVelocity,
                                          const vector3_t& referenceAngularVelocity) {
  const vector3_t referenceInBody = quaternionRotationMatrix<scalar_t>(orientation).transpose() *
                                    (quaternionRotationMatrix<scalar_t>(referenceOrientation) * referenceAngularVelocity);
  const vector3_t difference = angularVelocity - referenceInBody;
  // P_3: the body axes (x, y, z) in the rows (z, y, x) of the tuning layout.
  return AngularVelocityError{.residual = vector3_t(difference(2), difference(1), difference(0)), .referenceInBody = referenceInBody};
}

/** P_3 m: the rows of `m` in reverse order, (x, y, z) -> (z, y, x). */
matrix_t zyxOrderedRows(const matrix_t& m) {
  return m.colwise().reverse();
}

}  // namespace

TuningLayoutDeviation::TuningLayoutDeviation(const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel)
    : mpcRobotModel_(ABSL_DIE_IF_NULL(mpcRobotModel)) {
  const TuningLayout& layout = mpcRobotModel_->getTuningLayout();
  tuningDim_ = getTuningLayoutDim(layout);
  stateDim_ = mpcRobotModel_->getStateDim();
  size_t tuningRow = 0;
  for (const TuningSegment& segment : layout.segments) {
    addSegment(segment, tuningRow);
    tuningRow += segment.tuningDim;
  }
  ABSL_CHECK(!angularVelocity_.has_value() || orientation_.has_value())
      << "TuningLayoutDeviation: the " << describeTuningLayout(layout)
      << " has a base angular velocity block but no base orientation block, whose rotation its rows are taken in";
}

void TuningLayoutDeviation::addSegment(const TuningSegment& segment, size_t tuningRow) {
  const TuningLayout& layout = mpcRobotModel_->getTuningLayout();
  switch (segment.kind) {
    case TuningSegmentKind::kEuclidean: {
      ABSL_CHECK_EQ(segment.stateDim, segment.tuningDim)
          << "TuningLayoutDeviation takes the difference of the block '" << segment.name << "' of the " << describeTuningLayout(layout)
          << ", which has " << segment.stateDim << " state rows";
      const bool continuesTheRun = !identityRuns_.empty() && identityRuns_.back().tuningRow + identityRuns_.back().size == tuningRow &&
                                   identityRuns_.back().stateColumn + identityRuns_.back().size == segment.stateStart;
      if (continuesTheRun) {
        identityRuns_.back().size += segment.tuningDim;
      } else if (segment.tuningDim > 0) {
        identityRuns_.push_back(
            TuningJacobianIdentityRun{.tuningRow = tuningRow, .stateColumn = segment.stateStart, .size = segment.tuningDim});
      }
      return;
    }
    case TuningSegmentKind::kEulerZyxOrientation:
      ABSL_CHECK(!orientation_.has_value()) << "TuningLayoutDeviation: the " << describeTuningLayout(layout)
                                            << " has a second base orientation block, '" << segment.name << "'";
      ABSL_CHECK_EQ(segment.tuningDim, kRotationTuningDim) << "TuningLayoutDeviation: the base orientation block '" << segment.name
                                                           << "' of the " << describeTuningLayout(layout) << " is not (yaw, pitch, roll)";
      orientation_ = RotationBlock{.tuningRow = tuningRow, .stateStart = segment.stateStart};
      return;
    case TuningSegmentKind::kZyxOrderedAngularVelocity:
      ABSL_CHECK(!angularVelocity_.has_value()) << "TuningLayoutDeviation: the " << describeTuningLayout(layout)
                                                << " has a second base angular velocity block, '" << segment.name << "'";
      ABSL_CHECK_EQ(segment.tuningDim, kRotationTuningDim) << "TuningLayoutDeviation: the base angular velocity block '" << segment.name
                                                           << "' of the " << describeTuningLayout(layout) << " is not (z, y, x)";
      ABSL_CHECK_EQ(segment.stateDim, kRotationTuningDim)
          << "TuningLayoutDeviation: the base angular velocity block '" << segment.name << "' of the " << describeTuningLayout(layout)
          << " has " << segment.stateDim << " state rows";
      angularVelocity_ = RotationBlock{.tuningRow = tuningRow, .stateStart = segment.stateStart};
      return;
  }
  LOG(FATAL) << "TuningLayoutDeviation: the block '" << segment.name << "' has no TuningSegmentKind";
}

vector_t TuningLayoutDeviation::differences(const vector_t& state, const vector_t& stateReference) const {
  vector_t residual(static_cast<Eigen::Index>(tuningDim_));
  for (const TuningJacobianIdentityRun& run : identityRuns_) {
    const Eigen::Index tuningRow = static_cast<Eigen::Index>(run.tuningRow);
    const Eigen::Index stateColumn = static_cast<Eigen::Index>(run.stateColumn);
    const Eigen::Index size = static_cast<Eigen::Index>(run.size);
    residual.segment(tuningRow, size) = state.segment(stateColumn, size) - stateReference.segment(stateColumn, size);
  }
  return residual;
}

vector_t TuningLayoutDeviation::evaluate(const vector_t& state, const vector_t& stateReference) const {
  vector_t residual = differences(state, stateReference);
  if (!orientation_.has_value()) return residual;
  const vector4_t orientation = mpcRobotModel_->getBaseOrientation(state);
  const vector4_t referenceOrientation = mpcRobotModel_->getBaseOrientation(stateReference);
  residual.segment<3>(static_cast<Eigen::Index>(orientation_->tuningRow)) = headingTiltResidual(orientation, referenceOrientation);
  if (!angularVelocity_.has_value()) return residual;
  residual.segment<3>(static_cast<Eigen::Index>(angularVelocity_->tuningRow)) =
      angularVelocityError(orientation, referenceOrientation, mpcRobotModel_->getBaseAngularVelocityLocal(state),
                           mpcRobotModel_->getBaseAngularVelocityLocal(stateReference))
          .residual;
  return residual;
}

TuningDeviationApproximation TuningLayoutDeviation::linearize(const vector_t& state, const vector_t& stateReference) const {
  TuningDeviationApproximation approximation{
      .residual = differences(state, stateReference), .identityRuns = identityRuns_, .denseBlocks = {}};
  if (!orientation_.has_value()) return approximation;
  const vector4_t orientation = mpcRobotModel_->getBaseOrientation(state);
  const vector4_t referenceOrientation = mpcRobotModel_->getBaseOrientation(stateReference);
  // d(body rotation tangent) / d(orientation slots): E+(xi) of the quaternion.
  const matrix_t rotationTangent = mpcRobotModel_->getBaseRotationTangentJacobian(state);
  const HeadingTiltError headingTilt = headingTiltError(orientation, referenceOrientation);
  approximation.residual.segment<3>(static_cast<Eigen::Index>(orientation_->tuningRow)) = headingTilt.residual;
  approximation.denseBlocks.reserve(angularVelocity_.has_value() ? 3 : 1);
  approximation.denseBlocks.push_back(TuningJacobianDenseBlock{.tuningRow = orientation_->tuningRow,
                                                               .stateColumn = orientation_->stateStart,
                                                               .value = headingTilt.jacobianBodyTangent * rotationTangent});
  if (!angularVelocity_.has_value()) return approximation;

  const AngularVelocityError angularVelocity =
      angularVelocityError(orientation, referenceOrientation, mpcRobotModel_->getBaseAngularVelocityLocal(state),
                           mpcRobotModel_->getBaseAngularVelocityLocal(stateReference));
  approximation.residual.segment<3>(static_cast<Eigen::Index>(angularVelocity_->tuningRow)) = angularVelocity.residual;
  const BaseAngularVelocityJacobian angularVelocityJacobian = mpcRobotModel_->getBaseAngularVelocityLocalJacobian(state);
  approximation.denseBlocks.push_back(TuningJacobianDenseBlock{.tuningRow = angularVelocity_->tuningRow,
                                                               .stateColumn = angularVelocity_->stateStart,
                                                               .value = zyxOrderedRows(angularVelocityJacobian.angularVelocitySlots)});
  // d a / d(body perturbation dphi) = [a]x, so d e_w / d(orientation slots) = P_3 (d w_B / d(slots) - [a]x rotationTangent),
  // column by column: d w_B / d(slot) - a x (the slot's column of rotationTangent).
  matrix_t orientationColumns = angularVelocityJacobian.orientationSlots;
  for (Eigen::Index column = 0; column < orientationColumns.cols(); ++column) {
    orientationColumns.col(column) -= angularVelocity.referenceInBody.cross(vector3_t(rotationTangent.col(column)));
  }
  approximation.denseBlocks.push_back(TuningJacobianDenseBlock{
      .tuningRow = angularVelocity_->tuningRow, .stateColumn = orientation_->stateStart, .value = zyxOrderedRows(orientationColumns)});
  return approximation;
}

}  // namespace ocs2::humanoid
