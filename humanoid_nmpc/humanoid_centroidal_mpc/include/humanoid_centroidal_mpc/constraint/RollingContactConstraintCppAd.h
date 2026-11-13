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

#include <cstddef>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_robotic_tools/end_effector/EndEffectorKinematics.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The stance constraint of a rolling contact (ModelSettings::ContactKind::
 * kRolling) in the centroidal MPC: the wheel's material at its lowest point
 * does not slip, and that point stays on the ground. With p_wheel and
 * (v_wheel, w_wheel) the position and the world twist of the wheel-center
 * frame (the frame turns with the wheel, so w_wheel includes its spin), r the
 * wheel radius and n = e_z the flat ground's normal, the three rows are
 *
 *   g = Ax (p_contact - h_ref e_z) + Av v_contact,
 *   p_contact = p_wheel - r n,   v_contact = v_wheel + w_wheel x (-r n),
 *
 * in world axes, with Ax and Av the translation blocks of the stance-foot
 * gains (footTwistConfig(): Ax = diag(0, 0, position_error_gain_z), Av =
 * diag(linear_velocity_error_gain_xy, _xy, linear_velocity_error_gain_z)) and
 * h_ref the reference height of the contact point: the swing planner's, as
 * the point contacts' stance rows read it (HumanoidPreComputation), or the
 * height of the contact's world-aligned patch for a contact the swing planner
 * does not plan. g = 0 leaves the wheel free to roll along its heading
 * through its joint velocity and to steer through the leg; it holds the
 * contact point's lateral and longitudinal slip at zero. The Jacobian is
 * Av (J_v + r [n]x J_w) + Ax J_p; the derivation and the limits of the flat
 * ground are in humanoid_nmpc/docs/rolling_contact/README.md.
 *
 * Active while the reference manager's mode schedule has the contact in
 * contact. Use: CentroidalMpcInterface builds it for the zero_velocity entry
 * of the hard or soft constraint list on every rolling contact, under the
 * name contact_term::name(contact, contact_term::kRollingNoSlipName), and
 * ZeroVelocityGainsApplier applies hot-reloaded gains to it. Not thread-safe:
 * its kinematics evaluate on the Pinocchio data of one worker's
 * PreComputation, so the solver gives each worker its own clone().
 */
class RollingContactConstraintCppAd final : public StateInputConstraint {
 public:
  /**
   * Creates the constraint of the rolling contact `contactIndex` whose
   * wheel-center frame is the single end-effector of `wheelKinematics`
   * (copied), with the wheel radius `wheelRadius` [m] and the translation
   * blocks of `gains` (footTwistConfig() of the foot constraint).
   * `referenceManager` is retained and must outlive the constraint and every
   * clone of it.
   *
   * @return InvalidArgument when the radius is not finite and positive, when
   * the kinematics have other than one end-effector, or when an Ax or Av of
   * `gains` is neither empty nor at least 3 x 3.
   */
  static absl::StatusOr<std::unique_ptr<RollingContactConstraintCppAd>> Create(
      const SwitchedModelReferenceManager& referenceManager,
      const EndEffectorKinematics<scalar_t>& wheelKinematics,
      size_t contactIndex,
      scalar_t wheelRadius,
      const EndEffectorKinematicsTwistConstraint::Config& gains);

  ~RollingContactConstraintCppAd() override = default;
  RollingContactConstraintCppAd* absl_nonnull clone() const override {
    return new RollingContactConstraintCppAd(*this);
  }
  // Copied only by clone(), whose copy constructor is private; never assigned
  // or moved.
  RollingContactConstraintCppAd& operator=(
      const RollingContactConstraintCppAd&) = delete;
  RollingContactConstraintCppAd(RollingContactConstraintCppAd&&) = delete;
  RollingContactConstraintCppAd& operator=(RollingContactConstraintCppAd&&) =
      delete;

  /**
   * Replaces the gains with the translation blocks of `gains` (the top-left
   * 3 x 3 of its Ax and Av; an empty matrix is zero gains). Precondition:
   * each of Ax and Av is empty or at least 3 x 3, as footTwistConfig() makes
   * them.
   */
  void setGains(const EndEffectorKinematicsTwistConstraint::Config& gains);

  /** Returns the gains of the contact point's position error. */
  const matrix3_t& getPositionErrorGains() const { return posErrorGains_; }
  /** Returns the gains of the contact point's material velocity. */
  const matrix3_t& getVelocityErrorGains() const { return linvelErrorGains_; }

  /** Returns the wheel radius [m]. */
  scalar_t getWheelRadius() const { return wheelRadius_; }

  /** Returns whether the constraint applies at `time`: it is set active and
   * the contact is in contact then. */
  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t /*time*/) const override {
    return kPositionDim;
  }
  /** Returns g at (`time`, `state`, `input`); `preComp` is the problem's
   * HumanoidPreComputation, requested at that point. */
  vector_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const PreComputation& preComp) const override;
  /** Returns g and its exact first derivatives at (`time`, `state`, `input`);
   * `preComp` as for getValue(). */
  VectorFunctionLinearApproximation getLinearApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) const override;

 private:
  RollingContactConstraintCppAd(
      const SwitchedModelReferenceManager& referenceManager,
      const EndEffectorKinematics<scalar_t>& wheelKinematics,
      size_t contactIndex,
      scalar_t wheelRadius,
      const EndEffectorKinematicsTwistConstraint::Config& gains);
  RollingContactConstraintCppAd(const RollingContactConstraintCppAd& rhs);

  /** Returns h_ref [m] of the contact at `time` (the class comment). */
  scalar_t getDesPosContactRtWorldEwrtWorldZ(
      scalar_t time, const PreComputation& preComp) const;

  /** Returns [I, r [n]x], which maps the wheel-center twist [v; w] onto the
   * material velocity at the contact point. */
  Eigen::Matrix<scalar_t, kPositionDim, kTwistDim> getNoSlipMap() const;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  std::unique_ptr<EndEffectorKinematics<scalar_t>> wheelKinematicsPtr_;
  const size_t contactIndex_;
  // [m] Positive and finite (Create()).
  const scalar_t wheelRadius_;
  // Ax and Av of the class comment; zero when the gains are empty.
  matrix3_t posErrorGains_ = matrix3_t::Zero();
  matrix3_t linvelErrorGains_ = matrix3_t::Zero();
  bool isActive_ = true;
};

}  // namespace ocs2::humanoid
