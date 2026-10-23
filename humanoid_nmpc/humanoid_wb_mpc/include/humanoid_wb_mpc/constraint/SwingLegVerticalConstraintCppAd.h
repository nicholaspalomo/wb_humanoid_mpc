/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_wb_mpc/constraint/EndEffectorDynamicsLinearAccConstraint.h"

namespace ocs2::humanoid {

/**
 * The equality constraint on the normal motion of a swing foot: one row on its position, velocity and acceleration along
 * the ground normal, with the coefficients the pre-computation (WBMpcPreComputation) derives from the swing trajectory
 * planner at each node. Active while the reference manager has the foot in swing. Not thread-safe; the solver clones one
 * per worker thread.
 */
class SwingLegVerticalConstraintCppAd final : public StateInputConstraint {
 public:
  /**
   * Makes the constraint of the contact `contactPointIndex`, active while the reference manager has it in swing.
   * @param [in] referenceManager : Switched model ReferenceManager; it must outlive the constraint and its clones.
   * @param [in] endEffectorDynamics: The kinematic interface to the target end-effector, which has exactly one; cloned.
   * @param [in] contactPointIndex : The 3 DoF contact index.
   * @return InvalidArgument when `endEffectorDynamics` has other than one end effector
   *         (EndEffectorDynamicsLinearAccConstraint::Create()).
   */
  static absl::StatusOr<std::unique_ptr<SwingLegVerticalConstraintCppAd>> Create(const SwitchedModelReferenceManager& referenceManager,
                                                                                 const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
                                                                                 size_t contactPointIndex);

  ~SwingLegVerticalConstraintCppAd() override = default;
  SwingLegVerticalConstraintCppAd& operator=(const SwingLegVerticalConstraintCppAd&) = delete;
  SwingLegVerticalConstraintCppAd(SwingLegVerticalConstraintCppAd&&) = delete;
  SwingLegVerticalConstraintCppAd& operator=(SwingLegVerticalConstraintCppAd&&) = delete;
  SwingLegVerticalConstraintCppAd* absl_nonnull clone() const override { return new SwingLegVerticalConstraintCppAd(*this); }

  bool isActive(scalar_t time) const override;
  size_t getNumConstraints(scalar_t /*time*/) const override { return 1; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

 private:
  SwingLegVerticalConstraintCppAd(const SwitchedModelReferenceManager& referenceManager,
                                  std::unique_ptr<EndEffectorDynamicsLinearAccConstraint> eeLinearConstraint,
                                  size_t contactPointIndex);
  SwingLegVerticalConstraintCppAd(const SwingLegVerticalConstraintCppAd& rhs);

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  std::unique_ptr<EndEffectorDynamicsLinearAccConstraint> eeLinearConstraintPtr_;
  const size_t contactPointIndex_;
};

}  // namespace ocs2::humanoid
