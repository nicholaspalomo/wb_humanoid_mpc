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
#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_wb_mpc/constraint/EndEffectorDynamicsLinearAccConstraint.h"

namespace ocs2::humanoid {

/**
 * The pre-computation of the whole-body MPC: on a constraint request it updates the Pinocchio kinematics at the node and
 * derives, per foot, the coefficients of the swing foot's normal-motion constraint (SwingLegVerticalConstraintCppAd)
 * from the swing trajectory planner and its own gains, and the rotation of each contact frame. The gains start as the
 * model settings' and are retuned per copy (setSwingFootGains(), setNormalVelocityPositionErrorGain()), because the
 * model settings are shared by every worker thread's copy. Not thread-safe; the solver clones one per worker thread.
 */
class WBMpcPreComputation : public HumanoidPreComputation {
 public:
  /**
   * The gains of the swing foot's normal-motion constraint besides the position gain (the base's
   * getNormalVelocityPositionErrorGain()): model_settings.foot_constraint.linear_velocity_error_gain_z and
   * linear_acceleration_error_gain_z.
   */
  struct SwingFootGains {
    scalar_t linearVelocityErrorGainZ = 0.0;
    scalar_t linearAccelerationErrorGainZ = 0.0;
  };

  WBMpcPreComputation(PinocchioInterface pinocchioInterface,
                      const SwingTrajectoryPlanner& swingTrajectoryPlanner,
                      const MpcRobotModelBase<scalar_t>& mpcRobotModel);
  ~WBMpcPreComputation() override = default;
  WBMpcPreComputation& operator=(const WBMpcPreComputation&) = delete;
  WBMpcPreComputation(WBMpcPreComputation&&) = delete;
  WBMpcPreComputation& operator=(WBMpcPreComputation&&) = delete;

  WBMpcPreComputation* absl_nonnull clone() const override;

  void request(RequestSet request, scalar_t t, const vector_t& x, const vector_t& u) override;

  const std::vector<EndEffectorDynamicsLinearAccConstraint::Config>& getEeNormalAccelerationConstraintConfigs() const {
    return eeNormalAccConConfigs_;
  }

  /** Retunes the swing foot's velocity and acceleration gains from the next request on; for the parameter updater. */
  void setSwingFootGains(const SwingFootGains& gains) { swingFootGains_ = gains; }
  /** Returns the swing foot's velocity and acceleration gains in use. */
  const SwingFootGains& getSwingFootGains() const { return swingFootGains_; }

 private:
  WBMpcPreComputation(const WBMpcPreComputation& rhs);

  /** Returns the coefficients of the normal-motion constraint of the swing foot `footIndex` at time `t`. */
  EndEffectorDynamicsLinearAccConstraint::Config normalAccelerationConstraintConfig(size_t footIndex, scalar_t t) const;

  std::vector<EndEffectorDynamicsLinearAccConstraint::Config> eeNormalAccConConfigs_;
  // Seeded from the model settings at construction and carried by every copy; see setSwingFootGains().
  SwingFootGains swingFootGains_;
};

}  // namespace ocs2::humanoid
