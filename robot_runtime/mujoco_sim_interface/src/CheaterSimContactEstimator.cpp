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

#include "mujoco_sim_interface/CheaterSimContactEstimator.h"

#include <iostream>

#include "mujoco_sim_interface/MujocoSimInterface.h"

#include "absl/log/log.h"

namespace robot::mujoco_sim_interface {

CheaterSimContactEstimator::CheaterSimContactEstimator(const MujocoSimInterface& sim) : sim_(sim) {}

std::vector<bool> CheaterSimContactEstimator::flagsFromMasks(uint32_t groundTruthMask,
                                                             uint32_t unresolvedMask,
                                                             size_t numDetectedContacts,
                                                             size_t numContactPoints) {
  std::vector<bool> flags(numContactPoints, true);
  if (numDetectedContacts == 0) return flags;
  const uint32_t touching = groundTruthMask | unresolvedMask;
  for (size_t i = 0; i < numContactPoints && i < numDetectedContacts && i < 32; ++i) {
    flags[i] = ((touching >> i) & 1u) != 0u;
  }
  return flags;
}

std::vector<bool> CheaterSimContactEstimator::estimateContactFlags(const robot::model::RobotState& robotState) {
  const size_t numContactPoints = robotState.getContactFlags().size();
  const std::vector<bool> groundTruth = sim_.getGroundTruthContactFlags();
  if (groundTruth.empty() && !warnedNoContactDetection_) {
    warnedNoContactDetection_ = true;
    LOG(INFO) << "[CheaterSimContactEstimator] the simulator has no contact detection (no contact frame names were configured); every "
                 "contact point is reported as touching.";
  }
  uint32_t groundTruthMask = 0;
  for (size_t i = 0; i < groundTruth.size() && i < 32; ++i) {
    if (groundTruth[i]) groundTruthMask |= (1u << i);
  }
  return flagsFromMasks(groundTruthMask, sim_.getUnresolvedContactMask(), groundTruth.size(), numContactPoints);
}

// LINT.IfChange(cheater_sim_contact_estimator_name)
void registerCheaterSimContactEstimator(robot::model::ContactEstimatorRegistry& registry, const MujocoSimInterface& sim) {
  registry.add(kCheaterSimContactEstimatorName, "the ground truth of the MuJoCo physics (normal force above contactForceThreshold)",
               [&sim] { return std::make_shared<CheaterSimContactEstimator>(sim); });
}
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:contact_estimator, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:contact_estimator, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml:contact_estimator, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml:contact_estimator, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:contact_estimator)
// clang-format on

}  // namespace robot::mujoco_sim_interface
