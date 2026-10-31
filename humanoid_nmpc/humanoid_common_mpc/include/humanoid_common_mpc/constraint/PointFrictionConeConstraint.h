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

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The linearized friction cone of a 3-DoF (point) contact, written in the frame
 * of the patch it touches (humanoid_nmpc/docs/multi_contact_planning/README.md,
 * sections 2 and 7.2): h(t, u) = A_f p_R_w f + b >= 0, with f the contact's
 * world-frame force, p_R_w the rotation of the contact's patch
 * (SwitchedModelReferenceManager::getContactPatchFrame()) and A_f, b the
 * friction rows of the contact wrench cone (buildFrictionPyramidRows()):
 *  1. N facets of the friction pyramid inscribed in the Coulomb cone,
 *     mu cos(pi/N) (f_n + F_grip) - (cos(theta_k) f_t1 + sin(theta_k) f_t2)
 *     >= 0, theta_k = 2 pi k / N;
 *  2. the normal force bound f_n - minForceContactEwrtContactZ >= 0.
 * These are the first rows of ContactWrenchConeConstraint, of the same Config,
 * so a point contact and a surface contact on the same ground admit the same
 * forces; a point carries no moment, and the center-of-pressure and torsion
 * rows have nothing to bound.
 *
 * The rotation is a parameter of the contact phase, not of the state, so the
 * term's state Jacobian is zero and its input Jacobian A_f p_R_w dF/du, with
 * dF/du the model's own force Jacobian (inputJacobianForceContactEwrtParam()).
 *
 * Schedule gating follows ContactWrenchConeConstraint: gated, the term is
 * active while the mode schedule has the contact in contact; un-gated (the
 * contact-implicit formulation), it is active at every node and its affine
 * offsets are dropped, so that the zero force lies on its boundary.
 *
 * Not thread-safe; the solver gives each worker thread its own clone().
 */
class PointFrictionConeConstraint final : public StateInputConstraint {
 public:
  /**
   * The cone of a contact `contactIndex` whose input block is its three force
   * entries, after the checks of the constructor: InvalidArgument when the
   * model parameterizes the contact by anything but its force (a 6-DoF
   * contact, or basis-vector inputs), or when `config` fails
   * ContactWrenchConeConstraint::validateConfig(). The reference manager and
   * the model are borrowed and must outlive the term and its clones.
   */
  static absl::StatusOr<std::unique_ptr<PointFrictionConeConstraint>> Create(
      const SwitchedModelReferenceManager& referenceManager,
      size_t contactIndex,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      const ContactWrenchConeConstraint::Config& config,
      bool scheduleGated);

  /** OkStatus when the model's input block of `contactIndex` is its force:
   * kForceDim entries that getForceContactEwrtParam() reads one to one. */
  static absl::Status checkPointContactInput(
      const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactIndex);

  ~PointFrictionConeConstraint() override = default;
  PointFrictionConeConstraint& operator=(const PointFrictionConeConstraint&) =
      delete;
  PointFrictionConeConstraint(PointFrictionConeConstraint&&) = delete;
  PointFrictionConeConstraint& operator=(PointFrictionConeConstraint&&) =
      delete;
  PointFrictionConeConstraint* absl_nonnull clone() const override {
    return new PointFrictionConeConstraint(*this);
  }

  /** Whether this term is gated on the mode schedule's contact flag. */
  bool isScheduleGated() const { return scheduleGated_; }

  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t /*time*/) const override {
    return static_cast<size_t>(b_.size());
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
  PointFrictionConeConstraint(
      const SwitchedModelReferenceManager& referenceManager,
      size_t contactIndex,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      const ContactWrenchConeConstraint::Config& config,
      bool scheduleGated);
  PointFrictionConeConstraint(const PointFrictionConeConstraint& other) =
      default;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;
  const size_t contactIndex_;
  const bool scheduleGated_;
  bool isActive_ = true;
  /** (N + 1) x 3 and N + 1: the friction rows in the patch frame. */
  matrix_t A_f_;
  vector_t b_;
  /** 3 x getInputDim(): d(world-frame force) / d(input). */
  matrix_t inputJacobianForceContactEwrtParam_;
};

}  // namespace ocs2::humanoid
