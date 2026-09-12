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

#include <humanoid_centroidal_mpc_ros2/gains/GainsUpdaterUtils.h>
#include <humanoid_common_mpc/common/Types.h>
#include <iostream>

#include <humanoid_centroidal_mpc_ros2/gains/EndEffectorFootGainsUpdater.h>
#include <humanoid_centroidal_mpc_ros2/gains/EndEffectorKinematicsGainsUpdater.h>
#include <humanoid_centroidal_mpc_ros2/gains/FootCollisionGainsUpdater.h>
#include <humanoid_centroidal_mpc_ros2/gains/JointLimitsGainsUpdater.h>
#include <humanoid_centroidal_mpc_ros2/gains/QuadraticStateCostWeightsUpdater.h>
#include <humanoid_centroidal_mpc_ros2/gains/QuadraticStateInputGainsUpdater.h>
#include <humanoid_centroidal_mpc_ros2/gains/StateInputConstraintGainsUpdater.h>
#include <humanoid_centroidal_mpc_ros2/gains/StateInputSoftConstraintGainsUpdater.h>

namespace ocs2::humanoid::utils {

std::vector<std::string> getStateDescriptions(const ocs2::humanoid::ModelSettings& modelSettings) {
  std::vector<std::string> stateDescriptions = {"vcom_x",   "vcom_y",   "vcom_z",   "L_x / mass",   "L_y / mass",   "L_z / mass",
                                                "p_base_x", "p_base_y", "p_base_z", "theta_base_z", "theta_base_y", "theta_base_x"};
  stateDescriptions.insert(stateDescriptions.end(), modelSettings.mpcModelJointNames.begin(), modelSettings.mpcModelJointNames.end());
  // Centroidal state: 6 normalized momentum + 6 base pose entries followed by the MPC joints.
  const size_t expectedStateDim = 12 + modelSettings.mpc_joint_dim;
  if (stateDescriptions.size() != expectedStateDim) {
    std::cout << stateDescriptions.size() << " VS " << expectedStateDim << std::endl;
    throw std::runtime_error("[getStateDescriptions] Dimension mismatch!");
  }
  return stateDescriptions;
}

std::vector<std::string> getInputDescriptions(const ocs2::humanoid::ModelSettings& modelSettings, size_t inputDim) {
  const size_t wrenchInputDim = 6 * N_CONTACTS + modelSettings.mpc_joint_dim;
  std::vector<std::string> inputDescriptions;
  if (inputDim == wrenchInputDim) {
    inputDescriptions = {"W_l_x", "W_l_y", "W_l_z", "W_l_a", "W_l_b", "W_l_c", "W_r_x", "W_r_y", "W_r_z", "W_r_a", "W_r_b", "W_r_c"};
  } else {
    // Basis-vector layout: each contact contributes numBasisPerFoot scalings λ instead of a 6D wrench.
    if (inputDim < modelSettings.mpc_joint_dim || (inputDim - modelSettings.mpc_joint_dim) % N_CONTACTS != 0) {
      throw std::runtime_error("[getInputDescriptions] Input dimension " + std::to_string(inputDim) +
                               " is neither the wrench-space nor a basis-vector layout!");
    }
    const size_t numBasisPerFoot = (inputDim - modelSettings.mpc_joint_dim) / N_CONTACTS;
    static_assert(N_CONTACTS == 2, "Contact prefixes below assume a left and a right foot");
    for (const std::string& prefix : {std::string("lambda_l_"), std::string("lambda_r_")}) {
      for (size_t k = 0; k < numBasisPerFoot; ++k) {
        inputDescriptions.emplace_back(prefix + std::to_string(k));
      }
    }
  }
  for (const auto& jointName : modelSettings.mpcModelJointNames) {
    inputDescriptions.emplace_back("vel_" + jointName);
  }
  if (inputDescriptions.size() != inputDim) {
    std::cout << inputDescriptions.size() << " VS " << inputDim << std::endl;
    throw std::runtime_error("[getInputDescriptions] Dimension mismatch!");
  }
  return inputDescriptions;
}

std::vector<std::string> getInputDescriptions(const ocs2::humanoid::ModelSettings& modelSettings) {
  return getInputDescriptions(modelSettings, 6 * N_CONTACTS + modelSettings.mpc_joint_dim);
}

std::unordered_map<std::string, std::shared_ptr<GainsUpdaterInterface>> getGainsUpdaters(OptimalControlProblem& optimalControlProblem,
                                                                                         const CentroidalMpcInterface& centroidalInterface,
                                                                                         std::shared_ptr<GenericGuiInterface> gui) {
  // Initialize vector
  std::unordered_map<std::string, std::shared_ptr<GainsUpdaterInterface>> gainsUpdaters;

  // Find all relevant descriptions
  const auto costNames = centroidalInterface.getCostNames();
  const auto terminalCostNames = centroidalInterface.getTerminalCostNames();
  const auto stateSoftConstraintNames = centroidalInterface.getStateSoftConstraintNames();
  const auto softConstraintNames = centroidalInterface.getSoftConstraintNames();
  const auto equalityConstraintNames = centroidalInterface.getEqualityConstraintNames();

  const size_t totalSize = costNames.size() + terminalCostNames.size() + stateSoftConstraintNames.size() + softConstraintNames.size() +
                           equalityConstraintNames.size();
  std::vector<std::string> descriptions;
  descriptions.reserve(totalSize);

  descriptions.insert(descriptions.end(), costNames.begin(), costNames.end());
  descriptions.insert(descriptions.end(), terminalCostNames.begin(), terminalCostNames.end());
  descriptions.insert(descriptions.end(), stateSoftConstraintNames.begin(), stateSoftConstraintNames.end());
  descriptions.insert(descriptions.end(), softConstraintNames.begin(), softConstraintNames.end());
  descriptions.insert(descriptions.end(), equalityConstraintNames.begin(), equalityConstraintNames.end());

  // Initialize updaters
  for (const auto& description : descriptions) {
    bool found = false;

    auto checkAndAddCandidate = [&](std::shared_ptr<GainsUpdaterInterface> candidate) {
      if (candidate->initialize(optimalControlProblem, description)) {
        gainsUpdaters.insert({description, candidate});
        found = true;
      } else {
        candidate.reset();
      }
    };
    // The input cost lives in the OCP's input space, which is the basis-vector layout when that formulation is active.
    checkAndAddCandidate(std::make_shared<QuadraticStateInputGainsUpdater>(centroidalInterface.getEffectiveMpcRobotModel(),
                                                                           centroidalInterface.modelSettings(), gui));
    checkAndAddCandidate(std::make_shared<QuadraticStateCostWeightsUpdater>(centroidalInterface.getMpcRobotModel(),
                                                                            centroidalInterface.modelSettings(), gui));
    checkAndAddCandidate(std::make_shared<EndEffectorFootGainsUpdater>(gui));
    checkAndAddCandidate(std::make_shared<JointLimitsGainsUpdater>(gui));
    checkAndAddCandidate(std::make_shared<EndEffectorKinematicsGainsUpdater>(gui));
    checkAndAddCandidate(std::make_shared<FootCollisionGainsUpdater>(gui));
    checkAndAddCandidate(std::make_shared<StateInputConstraintGainsUpdater>(gui));
    checkAndAddCandidate(std::make_shared<StateInputSoftConstraintGainsUpdater>(gui));

    if (!found) std::cout << "[getGainsUpdaters] Could not find updater for `" << description << "`" << std::endl;
  }

  return gainsUpdaters;
}

}  // namespace ocs2::humanoid::utils
