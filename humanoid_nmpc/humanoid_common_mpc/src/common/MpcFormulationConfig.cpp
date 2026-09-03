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

#include "humanoid_common_mpc/common/MpcFormulationConfig.h"

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <stdexcept>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace ocs2::humanoid {

namespace {

std::string normalizeString(absl::string_view input) {
  std::string output;
  output.reserve(input.size());
  for (char c : input) {
    if (c != '_' && c != '-' && c != ' ') {
      output.push_back(absl::ascii_tolower(c));
    }
  }
  return output;
}

const absl::flat_hash_map<std::string, MpcCostType> kCostMap = {
    {"stateinputquadraticcost", MpcCostType::StateInputQuadraticCost},
    {"stateinputcost", MpcCostType::StateInputQuadraticCost},
    {"statequadraticcost", MpcCostType::StateQuadraticCost},
    {"statecost", MpcCostType::StateQuadraticCost},
    {"statetrackingcost", MpcCostType::StateQuadraticCost},
    {"inputquadraticcost", MpcCostType::InputQuadraticCost},
    {"inputcost", MpcCostType::InputQuadraticCost},
    {"inputeffortcost", MpcCostType::InputQuadraticCost},
    {"terminalcost", MpcCostType::TerminalCost},
    {"icpcost", MpcCostType::IcpCost},
    {"taskspacefootcost", MpcCostType::TaskSpaceFootCost},
    {"taskspacefoottrackingcost", MpcCostType::TaskSpaceFootCost},
    {"taskspacetorsocost", MpcCostType::TaskSpaceTorsoCost},
    {"taskspacekinematicscost", MpcCostType::TaskSpaceTorsoCost},
    {"externaltorquecost", MpcCostType::ExternalTorqueCost},
    {"legtorquecost", MpcCostType::ExternalTorqueCost},
    {"jointtorquecost", MpcCostType::JointTorqueCost},
};

const absl::flat_hash_map<std::string, MpcSoftConstraintType> kSoftConstraintMap = {
    {"jointlimits", MpcSoftConstraintType::JointLimits},
    {"jointlimitssoftconstraint", MpcSoftConstraintType::JointLimits},
    {"footcollision", MpcSoftConstraintType::FootCollision},
    {"footcollisionsoftconstraint", MpcSoftConstraintType::FootCollision},
    {"frictionforcecone", MpcSoftConstraintType::FrictionForceCone},
    {"frictionforceconesoftconstraint", MpcSoftConstraintType::FrictionForceCone},
    {"contactmomentxy", MpcSoftConstraintType::ContactMomentXY},
    {"contactmomentxyconstraint", MpcSoftConstraintType::ContactMomentXY},
    {"contactwrenchcone", MpcSoftConstraintType::ContactWrenchCone},
    {"contactwrenchconesoftconstraint", MpcSoftConstraintType::ContactWrenchCone},
};

const absl::flat_hash_map<std::string, MpcHardConstraintType> kHardConstraintMap = {
    {"zerowrench", MpcHardConstraintType::ZeroWrench},
    {"zerowrenchconstraint", MpcHardConstraintType::ZeroWrench},
    {"zerovelocity", MpcHardConstraintType::ZeroVelocity},
    {"zerovelocityconstraint", MpcHardConstraintType::ZeroVelocity},
    {"normalvelocity", MpcHardConstraintType::NormalVelocity},
    {"normalvelocityconstraint", MpcHardConstraintType::NormalVelocity},
    {"kneejointmimic", MpcHardConstraintType::KneeJointMimic},
    {"kneejointmimicconstraint", MpcHardConstraintType::KneeJointMimic},
    {"mimicjoints", MpcHardConstraintType::KneeJointMimic},
};

}  // namespace

absl::StatusOr<MpcCostType> stringToMpcCostType(absl::string_view name) {
  std::string normalized = normalizeString(name);
  auto it = kCostMap.find(normalized);
  if (it != kCostMap.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unknown MPC cost type: '", name, "'. Supported costs are: ",
      "state_quadratic_cost, input_quadratic_cost, state_input_quadratic_cost, terminal_cost, icp_cost, ",
      "task_space_foot_cost, task_space_torso_cost, external_torque_cost, joint_torque_cost."));
}

absl::StatusOr<std::string> mpcCostTypeToString(MpcCostType type) {
  switch (type) {
    case MpcCostType::StateInputQuadraticCost:
      return "state_input_quadratic_cost";
    case MpcCostType::StateQuadraticCost:
      return "state_quadratic_cost";
    case MpcCostType::InputQuadraticCost:
      return "input_quadratic_cost";
    case MpcCostType::TerminalCost:
      return "terminal_cost";
    case MpcCostType::IcpCost:
      return "icp_cost";
    case MpcCostType::TaskSpaceFootCost:
      return "task_space_foot_cost";
    case MpcCostType::TaskSpaceTorsoCost:
      return "task_space_torso_cost";
    case MpcCostType::ExternalTorqueCost:
      return "external_torque_cost";
    case MpcCostType::JointTorqueCost:
      return "joint_torque_cost";
    default:
      return absl::InvalidArgumentError(absl::StrCat("Unknown MpcCostType: ", static_cast<int>(type)));
  }
}

absl::StatusOr<MpcSoftConstraintType> stringToMpcSoftConstraintType(absl::string_view name) {
  std::string normalized = normalizeString(name);
  auto it = kSoftConstraintMap.find(normalized);
  if (it != kSoftConstraintMap.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unknown MPC soft constraint type: '", name, "'. Supported soft constraints are: ",
      "joint_limits, foot_collision, friction_force_cone, contact_moment_xy, contact_wrench_cone."));
}

absl::StatusOr<std::string> mpcSoftConstraintTypeToString(MpcSoftConstraintType type) {
  switch (type) {
    case MpcSoftConstraintType::JointLimits:
      return "joint_limits";
    case MpcSoftConstraintType::FootCollision:
      return "foot_collision";
    case MpcSoftConstraintType::FrictionForceCone:
      return "friction_force_cone";
    case MpcSoftConstraintType::ContactMomentXY:
      return "contact_moment_xy";
    case MpcSoftConstraintType::ContactWrenchCone:
      return "contact_wrench_cone";
    default:
      return absl::InvalidArgumentError(absl::StrCat("Unknown MpcSoftConstraintType: ", static_cast<int>(type)));
  }
}

absl::StatusOr<MpcHardConstraintType> stringToMpcHardConstraintType(absl::string_view name) {
  std::string normalized = normalizeString(name);
  auto it = kHardConstraintMap.find(normalized);
  if (it != kHardConstraintMap.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unknown MPC hard constraint type: '", name, "'. Supported hard constraints are: ",
      "zero_wrench, zero_velocity, normal_velocity, knee_joint_mimic."));
}

absl::StatusOr<std::string> mpcHardConstraintTypeToString(MpcHardConstraintType type) {
  switch (type) {
    case MpcHardConstraintType::ZeroWrench:
      return "zero_wrench";
    case MpcHardConstraintType::ZeroVelocity:
      return "zero_velocity";
    case MpcHardConstraintType::NormalVelocity:
      return "normal_velocity";
    case MpcHardConstraintType::KneeJointMimic:
      return "knee_joint_mimic";
    default:
      return absl::InvalidArgumentError(absl::StrCat("Unknown MpcHardConstraintType: ", static_cast<int>(type)));
  }
}

absl::StatusOr<MpcFormulationTasks> loadMpcFormulationTasks(absl::string_view taskFile, bool verbose) {
  MpcFormulationTasks formulationTasks;
  const std::string taskFilePath(taskFile);
  YAML::Node root;
  try {
    root = YAML::LoadFile(taskFilePath);
  } catch (const std::exception& e) {
    return absl::NotFoundError(
        absl::StrCat("[loadMpcFormulationTasks] Failed to load YAML file '", taskFilePath, "': ", e.what()));
  }

  // Find the node containing the task lists: check root, mpc_tasks, or tasks
  YAML::Node context = root;
  if (root["mpc_tasks"] && root["mpc_tasks"].IsMap()) {
    context = root["mpc_tasks"];
  } else if (root["tasks"] && root["tasks"].IsMap()) {
    context = root["tasks"];
  }

  // 1. Costs
  YAML::Node costsNode;
  if (context["costs"] && context["costs"].IsSequence()) {
    costsNode = context["costs"];
  } else if (root["costs"] && root["costs"].IsSequence()) {
    costsNode = root["costs"];
  }

  if (costsNode) {
    for (const YAML::Node& item : costsNode) {
      if (item.IsScalar()) {
        ASSIGN_OR_RETURN(MpcCostType cost, stringToMpcCostType(item.as<std::string>()));
        formulationTasks.costs.insert(cost);
      }
    }
  }

  // 2. Soft Constraints
  YAML::Node softNode;
  if (context["soft_constraints"] && context["soft_constraints"].IsSequence()) {
    softNode = context["soft_constraints"];
  } else if (root["soft_constraints"] && root["soft_constraints"].IsSequence()) {
    softNode = root["soft_constraints"];
  }

  if (softNode) {
    for (const YAML::Node& item : softNode) {
      if (item.IsScalar()) {
        ASSIGN_OR_RETURN(MpcSoftConstraintType sc, stringToMpcSoftConstraintType(item.as<std::string>()));
        formulationTasks.softConstraints.insert(sc);
      }
    }
  }

  // 3. Hard Constraints
  YAML::Node hardNode;
  if (context["hard_constraints"] && context["hard_constraints"].IsSequence()) {
    hardNode = context["hard_constraints"];
  } else if (root["hard_constraints"] && root["hard_constraints"].IsSequence()) {
    hardNode = root["hard_constraints"];
  }

  if (hardNode) {
    for (const YAML::Node& item : hardNode) {
      if (item.IsScalar()) {
        ASSIGN_OR_RETURN(MpcHardConstraintType hc, stringToMpcHardConstraintType(item.as<std::string>()));
        formulationTasks.hardConstraints.insert(hc);
      }
    }
  }

  if (verbose) {
    LOG(INFO) << "MPC Formulation Tasks Loaded from: " << taskFile;
    LOG(INFO) << "Costs (" << formulationTasks.costs.size() << "):";
    for (MpcCostType cost : formulationTasks.costs) {
      ASSIGN_OR_RETURN(const std::string costStr, mpcCostTypeToString(cost));
      LOG(INFO) << "   - " << costStr;
    }
    LOG(INFO) << "Soft Constraints (" << formulationTasks.softConstraints.size() << "):";
    for (MpcSoftConstraintType sc : formulationTasks.softConstraints) {
      ASSIGN_OR_RETURN(const std::string scStr, mpcSoftConstraintTypeToString(sc));
      LOG(INFO) << "   - " << scStr;
    }
    LOG(INFO) << "Hard Constraints (" << formulationTasks.hardConstraints.size() << "):";
    for (MpcHardConstraintType hc : formulationTasks.hardConstraints) {
      ASSIGN_OR_RETURN(const std::string hcStr, mpcHardConstraintTypeToString(hc));
      LOG(INFO) << "   - " << hcStr;
    }
  }

  return formulationTasks;
}

}  // namespace ocs2::humanoid
