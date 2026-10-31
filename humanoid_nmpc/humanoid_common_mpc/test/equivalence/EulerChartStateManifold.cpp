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

#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerChartStateManifold.h"

#include <memory>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid {

using validation::EulerStateSegment;
using validation::EulerStateSegmentKind;

EulerChartStateManifold::EulerChartStateManifold(
    std::shared_ptr<const StateManifold> quaternionManifold,
    validation::EulerStateLayout layout)
    : quaternionManifold_(std::move(quaternionManifold)),
      layout_(std::move(layout)) {
  ABSL_CHECK(quaternionManifold_ != nullptr);
  ABSL_CHECK_EQ(quaternionManifold_->getAmbientDim(),
                validation::getPhysicalStateDim(layout_));
  ABSL_CHECK_EQ(quaternionManifold_->getTangentDim(),
                validation::getEulerStateDim(layout_));
}

vector_t EulerChartStateManifold::eulerStateOf(const vector_t& x) const {
  vector_t eulerState(
      static_cast<Eigen::Index>(validation::getEulerStateDim(layout_)));
  Eigen::Index euler = 0;
  Eigen::Index physical = 0;
  vector3_t angles = vector3_t::Zero();
  for (const EulerStateSegment& segment : layout_.segments) {
    const Eigen::Index dim = static_cast<Eigen::Index>(segment.dim);
    switch (segment.kind) {
      case EulerStateSegmentKind::kEuclidean:
        eulerState.segment(euler, dim) = x.segment(physical, dim);
        physical += dim;
        break;
      case EulerStateSegmentKind::kEulerZyxAngles:
        angles = eulerZyxFromQuaternion(
            safelyNormalizedQuaternion(vector4_t(x.segment<4>(physical))));
        eulerState.segment<3>(euler) = angles;
        physical += 4;
        break;
      case EulerStateSegmentKind::kEulerZyxRates:
        eulerState.segment<3>(euler) =
            eulerZyxRatesFromBodyRates(angles, x.segment<3>(physical));
        physical += 3;
        break;
    }
    euler += dim;
  }
  return eulerState;
}

vector_t EulerChartStateManifold::stateOf(const vector_t& eulerState) const {
  absl::StatusOr<vector_t> state =
      validation::physicalStateFromEulerState(eulerState, layout_);
  ABSL_CHECK(state.ok()) << state.status();
  return *std::move(state);
}

matrix_t EulerChartStateManifold::physicalTangentJacobian(
    const vector_t& eulerState) const {
  absl::StatusOr<matrix_t> jacobian =
      validation::physicalTangentJacobian(eulerState, layout_);
  ABSL_CHECK(jacobian.ok()) << jacobian.status();
  return *std::move(jacobian);
}

void EulerChartStateManifold::retract(const vector_t& x,
                                      const vector_t& dx,
                                      scalar_t alpha,
                                      vector_t& xNew) const {
  const vector_t eulerState = eulerStateOf(x);
  xNew = stateOf(
      eulerState +
      physicalTangentJacobian(eulerState).partialPivLu().solve(alpha * dx));
}

vector_t EulerChartStateManifold::difference(const vector_t& x0,
                                             const vector_t& x1) const {
  const vector_t eulerState = eulerStateOf(x0);
  return physicalTangentJacobian(eulerState) * (eulerStateOf(x1) - eulerState);
}

vector_t EulerChartStateManifold::interpolate(const vector_t& x0,
                                              const vector_t& x1,
                                              scalar_t alpha) const {
  return stateOf((1.0 - alpha) * eulerStateOf(x0) + alpha * eulerStateOf(x1));
}

void EulerChartStateManifold::differenceInto(const vector_t& x0,
                                             const vector_t& x1,
                                             vector_t& dx) const {
  dx = difference(x0, x1);
}

void EulerChartStateManifold::interpolateInto(const vector_t& x0,
                                              const vector_t& x1,
                                              scalar_t alpha,
                                              vector_t& x) const {
  x = interpolate(x0, x1, alpha);
}

void EulerChartStateManifold::pushForwardDynamics(
    const vector_t& x,
    const vector_t& xNext,
    VectorFunctionLinearApproximation& dynamics) const {
  vector_t end = dynamics.f;
  quaternionManifold_->project(end);
  // The quaternion manifold's sensitivities of a gap at its own end: E+(Pi(F))
  // Pi'(F) dF/dx E(x) (Jr^-1(0) = I).
  quaternionManifold_->pushForwardDynamics(x, end, dynamics);
  const vector_t endEuler = eulerStateOf(end);
  const vector_t nextEuler = eulerStateOf(xNext);
  const matrix_t nextMap = physicalTangentJacobian(nextEuler);
  const matrix_t endToNext =
      nextMap * physicalTangentJacobian(endEuler).inverse();
  dynamics.f = nextMap * (endEuler - nextEuler);
  dynamics.dfdx = endToNext * dynamics.dfdx;
  if (dynamics.dfdu.rows() > 0) dynamics.dfdu = endToNext * dynamics.dfdu;
}

}  // namespace ocs2::humanoid
