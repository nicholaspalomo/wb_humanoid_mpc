/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <memory>
#include <string>

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"

namespace ocs2::humanoid {

/**
 * Tracks the whole-body Center of Mass position and Angular Center of Mass
 * orientation against the reference trajectory, in place of the base pose.
 *
 * The aCOM orientation is theta_aCOM(q) = euler_zyx_base + P * Delta_theta(q_j),
 * where Delta_theta is the learned joint-induced orientation offset and P
 * reorders the network's native XYZ output into the centroidal state's ZYX Euler
 * convention. Because the offset depends only on the joints, the optimizer can
 * counter-rotate the arms and torso to hold whole-body orientation, which is what
 * produces emergent arm swing during walking.
 *
 * When this cost is active, HumanoidCostConstraintFactory zeroes the base pose
 * blocks of both the running and the terminal state cost so that orientation is
 * not regulated twice in two different coordinate systems.
 */
class ComAndAcomTrackingCost : public StateCost {
 public:
  /**
   * @param Q_com   3x3 weight on the CoM position error.
   * @param Q_acom  3x3 weight on the aCOM orientation error, in ZYX Euler order
   *     so that row 0 is yaw, matching the centroidal state.
   * @param pinocchioInterface  Reduced model matching the MPC's joint set.
   * @param info    Centroidal model info for that same model.
   * @param robotName  Selects the compiled-in aCOM weights.
   *
   * @throws std::runtime_error if the aCOM network was trained for a different
   *     number of joints than the model provides.
   */
  ComAndAcomTrackingCost(
      matrix_t Q_com, matrix_t Q_acom, PinocchioInterface pinocchioInterface, CentroidalModelInfo info, const std::string& robotName);

  ~ComAndAcomTrackingCost() override = default;
  ComAndAcomTrackingCost* clone() const override;

  bool isActive(scalar_t time) const override { return true; }

  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComp) const override;

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time,
                                                                 const vector_t& state,
                                                                 const TargetTrajectories& targetTrajectories,
                                                                 const PreComputation& preComp) const override;

  /**
   * @brief Update the CoM and ACoM tracking weight matrices in place.
   *
   * Called by the live parameter updater on every per-thread copy of the problem,
   * from the solver thread before the worker pool starts.
   *
   * @throws std::invalid_argument if either matrix is not 3x3.
   */
  void setWeights(matrix_t Q_com, matrix_t Q_acom);

  const matrix_t& getQCom() const { return Q_com_; }
  const matrix_t& getQAcom() const { return Q_acom_; }

 private:
  ComAndAcomTrackingCost(const ComAndAcomTrackingCost& rhs);

  /**
   * Computes the aCOM orientation error between a state and a reference state,
   * in ZYX Euler order with the yaw component wrapped to [-pi, pi].
   *
   * Shared by getValue and getQuadraticApproximation so the two cannot drift
   * apart in their Euler ordering or their angle wrapping.
   */
  vector3_t computeAcomError(const vector_t& state, const vector_t& stateRef) const;

  matrix_t Q_com_;
  matrix_t Q_acom_;
  mutable PinocchioInterface pinocchioInterface_;
  CentroidalModelInfo info_;
  std::string robotName_;
  std::unique_ptr<AngularCenterOfMass> acom_;
};

}  // namespace ocs2::humanoid
