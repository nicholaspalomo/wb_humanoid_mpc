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

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The equality constraint of a mimic joint, q_child = multiplier * q_parent (the task file's mimicJoints blocks, the
 * knees): one row, positionGain * (multiplier * q_parent - q_child) + (multiplier * v_parent - v_child) = 0, linear in
 * the joint angles of the state and the joint velocities of the input. Not thread-safe; the solver clones it per worker.
 */
class JointMimicKinematicConstraint final : public StateInputConstraint {
 public:
  /** The mimic pair of Create(), with both joints resolved to their index among the MPC joints. */
  struct Config {
    std::string parentJointName;
    std::string childJointName;
    size_t parentJointIndex = 0;
    size_t childJointIndex = 0;
    scalar_t multiplier = 0.0;  // q_child = multiplier* q_parent
    scalar_t positionGain = 0.0;
  };

  /**
   * The constraint q_child = multiplier * q_parent between two MPC joints of `mpcRobotModel`, which must outlive it.
   *
   * @return NotFound naming the joint when `parentJointName` or `childJointName` is not an active joint of the MPC
   *         model; InvalidArgument when `positionGain` is not positive.
   */
  static absl::StatusOr<std::unique_ptr<JointMimicKinematicConstraint>> Create(const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                               const std::string& parentJointName,
                                                                               const std::string& childJointName,
                                                                               scalar_t multiplier,
                                                                               scalar_t positionGain);

  ~JointMimicKinematicConstraint() override = default;
  JointMimicKinematicConstraint* absl_nonnull clone() const override { return new JointMimicKinematicConstraint(*this); }
  // Copied only by clone(), whose copy constructor is private; never assigned or moved.
  JointMimicKinematicConstraint& operator=(const JointMimicKinematicConstraint&) = delete;
  JointMimicKinematicConstraint(JointMimicKinematicConstraint&&) = delete;
  JointMimicKinematicConstraint& operator=(JointMimicKinematicConstraint&&) = delete;

  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t /*time*/) const override { return 1; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

  /** The mimic pair, as Create() resolved it. */
  const Config& getConfig() const { return config_; }

 private:
  JointMimicKinematicConstraint(const MpcRobotModelBase<scalar_t>& mpcRobotModel, Config config);
  JointMimicKinematicConstraint(const JointMimicKinematicConstraint& rhs);

  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;
  const Config config_;

  bool isActive_ = true;
};

}  // namespace ocs2::humanoid
