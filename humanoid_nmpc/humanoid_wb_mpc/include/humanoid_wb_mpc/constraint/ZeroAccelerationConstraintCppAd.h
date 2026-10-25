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
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_wb_mpc/constraint/EndEffectorDynamicsAccelerationsConstraint.h"

namespace ocs2::humanoid {

/**
 * The whole-body MPC's stance-foot constraint (the zero_velocity term): the six
 * rows of an EndEffectorDynamicsAccelerationsConstraint on the foot's pose with
 * respect to the ground and its first two time derivatives
 * (stanceFootAccelerationConstraintConfig()), active while the reference
 * manager has the foot in contact. The tilt rows are the PD law on the foot's
 * orientation error e (the sine of half its tilt), Kp e + Kd de/dt + Ka
 * d2e/dt2, and the row along the ground normal holds the rotation rate about
 * the foot normal, Kd (v.w) / 2 + Ka (v.alpha) / 2, so that a stance foot can
 * neither tilt nor pivot; both rates are taped exactly, with the
 * state-dependent map from w and alpha, so the linearization is the derivative
 * (design README section 2.8.6). At a level foot the rates are w / 2 and alpha
 * / 2 + 1/2 (n.w)(n x w): angular gains twice those of a law on the raw w and
 * alpha reproduce it there, but for that last term. The height row servoes the
 * foot to the ground the swing trajectory planner stands it on (its stance
 * height, getZpositionConstraint(): terrain_height, moved by a hot reload), as
 * the centroidal MPC's stance constraint does, so that a stance foot is held
 * where a swing foot lands: g_z = Ax_zz (z - z_ground) + Av_zz v_z + Aa_zz a_z.
 *
 * Evaluated concurrently by the solver's workers, each on its own clone;
 * configure() is for the solver thread between two solves.
 */
class ZeroAccelerationConstraintCppAd final : public StateInputConstraint {
 public:
  /** The rows of the constraint: the foot's position and its orientation error
   * with respect to the ground. */
  static constexpr int kNumConstraints = kPoseTangentDim;

  /**
   * Makes the constraint of the contact `contactPointIndex`, active while the
   * reference manager has it in contact.
   * @param [in] referenceManager : Switched model ReferenceManager; it must
   * outlive the constraint and its clones.
   * @param [in] endEffectorDynamics: The dynamics interface to the target
   * end-effector, which has exactly one; cloned.
   * @param [in] contactPointIndex : The 3 DoF contact index.
   * @param [in] config: The constraint coefficients
   * @return InvalidArgument when `endEffectorDynamics` has other than one end
   * effector (EndEffectorDynamicsAccelerationsConstraint::Create()).
   */
  static absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>>
  Create(const SwitchedModelReferenceManager& referenceManager,
         const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
         size_t contactPointIndex,
         EndEffectorDynamicsAccelerationsConstraint::Config config =
             EndEffectorDynamicsAccelerationsConstraint::Config());

  ~ZeroAccelerationConstraintCppAd() override = default;
  ZeroAccelerationConstraintCppAd& operator=(
      const ZeroAccelerationConstraintCppAd&) = delete;
  ZeroAccelerationConstraintCppAd(ZeroAccelerationConstraintCppAd&&) = delete;
  ZeroAccelerationConstraintCppAd& operator=(
      ZeroAccelerationConstraintCppAd&&) = delete;
  ZeroAccelerationConstraintCppAd* absl_nonnull clone() const override {
    return new ZeroAccelerationConstraintCppAd(*this);
  }

  /**
   * Sets the coefficients of the stance foot's constraint from the next
   * evaluation on, as Create() with `config` would: the parameter updater's
   * retuning of the foot-constraint gains, between solves. Unchecked: `config`
   * has the six rows and 6 x 6 blocks of
   * stanceFootAccelerationConstraintConfig().
   */
  void configure(EndEffectorDynamicsAccelerationsConstraint::Config config) {
    eeAccelConstraintPtr_->configure(std::move(config));
  }

  bool isActive(scalar_t time) const override;
  size_t getNumConstraints(scalar_t /*time*/) const override {
    return kNumConstraints;
  }
  vector_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) const override;

 private:
  ZeroAccelerationConstraintCppAd(
      const SwitchedModelReferenceManager& referenceManager,
      std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>
          eeAccelConstraint,
      size_t contactPointIndex);
  ZeroAccelerationConstraintCppAd(const ZeroAccelerationConstraintCppAd& rhs);

  /** Ax_zz z_ground at `time`, which the height row subtracts; 0 without a
   * position gain. */
  scalar_t groundTerm(scalar_t time) const;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>
      eeAccelConstraintPtr_;
  const size_t contactPointIndex_;
};

/**
 * Returns the coefficients of the whole-body MPC's stance-foot constraint (the
 * zero_velocity term) from the task file's foot-constraint `gains`: g = Ax [p;
 * e] + Av d/dt [p; e] + Aa d2/dt2 [p; e]
 * (EndEffectorDynamicsAccelerationsConstraint), 6 rows, each matrix diagonal -
 * the position gain on z and the orientation gain on the orientation rows of Ax
 * (each left out when 0), the velocity and acceleration gains on Av and Aa: the
 * angular ones multiply the rate and the acceleration of the orientation error,
 * not the angular velocity and acceleration. ZeroAccelerationConstraintCppAd
 * subtracts the ground's height from the z row.
 */
EndEffectorDynamicsAccelerationsConstraint::Config
stanceFootAccelerationConstraintConfig(
    const ModelSettings::FootConstraintConfig& gains);

}  // namespace ocs2::humanoid
