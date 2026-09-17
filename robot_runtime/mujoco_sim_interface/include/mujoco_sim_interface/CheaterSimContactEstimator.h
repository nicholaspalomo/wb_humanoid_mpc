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

#include <robot_model/ContactEstimatorRegistry.h>

#include <cstdint>
#include <string>
#include <vector>

namespace robot::mujoco_sim_interface {

class MujocoSimInterface;

/**
 * Contact estimator for simulation: the measured contact state is the ground truth of the physics engine.
 *
 * A contact point counts as touching when MuJoCo reports more than MujocoSimConfig::contactForceThreshold newtons of
 * normal force between its body and anything outside the robot (see MujocoSimInterface::updateGroundTruthContacts). The
 * flags are those of the latest simulation step and are read without blocking the simulation thread. A contact point
 * whose MuJoCo body could not be resolved, and every contact point of a simulator without contact detection, is
 * reported as touching, so that such a point keeps its planned contact wrenches as it always did.
 */
class CheaterSimContactEstimator final : public robot::model::ContactEstimator {
 public:
  explicit CheaterSimContactEstimator(const MujocoSimInterface& sim);

  std::vector<bool> estimateContactFlags(const robot::model::RobotState& robotState) override;

  std::string getName() const override { return "CheaterSimContactEstimator"; }

  /**
   * The flag vector for the given masks; exposed for tests.
   * @param groundTruthMask Bit i set: contact point i carries contact force.
   * @param unresolvedMask Bit i set: contact point i has no MuJoCo body (it is reported as touching).
   * @param numDetectedContacts Contact points the simulator resolves (0: no contact detection at all).
   * @param numContactPoints Contact points of the controller, the size of the result.
   */
  static std::vector<bool> flagsFromMasks(uint32_t groundTruthMask,
                                          uint32_t unresolvedMask,
                                          size_t numDetectedContacts,
                                          size_t numContactPoints);

 private:
  const MujocoSimInterface& sim_;
  bool warnedNoContactDetection_{false};
};

/** Name of the CheaterSimContactEstimator in the ContactEstimatorRegistry (task file `contactEstimator: cheater_sim`). */
constexpr const char* kCheaterSimContactEstimatorName = "cheater_sim";

/** Adds the CheaterSimContactEstimator of `sim` to `registry` under kCheaterSimContactEstimatorName. */
void registerCheaterSimContactEstimator(robot::model::ContactEstimatorRegistry& registry, const MujocoSimInterface& sim);

}  // namespace robot::mujoco_sim_interface
