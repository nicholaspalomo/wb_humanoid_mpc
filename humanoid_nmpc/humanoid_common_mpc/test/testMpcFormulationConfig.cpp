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

#include <fstream>
#include <stdexcept>
#include <string>
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"

using namespace ocs2::humanoid;

#define CHECK_TRUE(cond, msg)                                          \
  do {                                                                 \
    if (!(cond)) {                                                     \
      throw std::runtime_error(absl::StrCat("Check failed: ", (msg))); \
    }                                                                  \
  } while (0)

/**
 * Writes a task file with the contact-implicit formulation switched on - `zero_wrench` dropped, the three terms
 * listed - and `cone` (a cone term name, or empty for none) in soft_constraints, then loads it.
 */
absl::StatusOr<MpcFormulationTasks> loadContactImplicitTaskFile(const std::string& path, const std::string& cone) {
  {
    std::ofstream ofs(path);
    ofs << "hard_constraints: []\n\n"
        << "soft_constraints:\n"
        << "  - joint_limits\n";
    if (!cone.empty()) {
      ofs << "  - " << cone << "\n";
    }
    ofs << "  - contact_complementarity\n"
        << "  - force_weighted_slip\n"
        << "  - ground_penetration\n\n"
        << "costs:\n"
        << "  - state_quadratic_cost\n";
  }
  return loadMpcFormulationTasks(path, false);
}

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  LOG(INFO) << "Testing MPC Formulation Config Enum Conversions...";

  // Test cost string-to-enum
  CHECK_TRUE(stringToMpcCostType("state_input_quadratic_cost").ok(), "cost 1 ok");
  CHECK_TRUE(*stringToMpcCostType("state_input_quadratic_cost") == MpcCostType::StateInputQuadraticCost, "cost 1");
  CHECK_TRUE(*stringToMpcCostType("stateInputQuadraticCost") == MpcCostType::StateInputQuadraticCost, "cost 2");
  CHECK_TRUE(*stringToMpcCostType("state_quadratic_cost") == MpcCostType::StateQuadraticCost, "cost state_quadratic_cost");
  CHECK_TRUE(*stringToMpcCostType("stateQuadraticCost") == MpcCostType::StateQuadraticCost, "cost stateQuadraticCost");
  CHECK_TRUE(*stringToMpcCostType("input_quadratic_cost") == MpcCostType::InputQuadraticCost, "cost input_quadratic_cost");
  CHECK_TRUE(*stringToMpcCostType("inputQuadraticCost") == MpcCostType::InputQuadraticCost, "cost inputQuadraticCost");
  CHECK_TRUE(*stringToMpcCostType("terminal_cost") == MpcCostType::TerminalCost, "cost 3");
  CHECK_TRUE(*stringToMpcCostType("terminalCost") == MpcCostType::TerminalCost, "cost 4");
  CHECK_TRUE(*stringToMpcCostType("icp_cost") == MpcCostType::IcpCost, "cost 5");
  CHECK_TRUE(*stringToMpcCostType("task_space_foot_cost") == MpcCostType::TaskSpaceFootCost, "cost 6");
  CHECK_TRUE(*stringToMpcCostType("task_space_torso_cost") == MpcCostType::TaskSpaceTorsoCost, "cost 7");
  CHECK_TRUE(*stringToMpcCostType("external_torque_cost") == MpcCostType::ExternalTorqueCost, "cost 8");
  CHECK_TRUE(*stringToMpcCostType("joint_torque_cost") == MpcCostType::JointTorqueCost, "cost 9");

  // Test soft constraint string-to-enum
  CHECK_TRUE(stringToMpcSoftConstraintType("joint_limits").ok(), "soft 1 ok");
  CHECK_TRUE(*stringToMpcSoftConstraintType("joint_limits") == MpcSoftConstraintType::JointLimits, "soft 1");
  CHECK_TRUE(*stringToMpcSoftConstraintType("jointLimits") == MpcSoftConstraintType::JointLimits, "soft 2");
  CHECK_TRUE(*stringToMpcSoftConstraintType("foot_collision") == MpcSoftConstraintType::FootCollision, "soft 3");
  CHECK_TRUE(*stringToMpcSoftConstraintType("friction_force_cone") == MpcSoftConstraintType::FrictionForceCone, "soft 4");
  CHECK_TRUE(*stringToMpcSoftConstraintType("contact_moment_xy") == MpcSoftConstraintType::ContactMomentXY, "soft 5");
  CHECK_TRUE(*stringToMpcSoftConstraintType("contact_wrench_cone") == MpcSoftConstraintType::ContactWrenchCone, "soft 6");
  CHECK_TRUE(*stringToMpcSoftConstraintType("zero_velocity") == MpcSoftConstraintType::ZeroVelocity, "soft zero_velocity");
  CHECK_TRUE(*stringToMpcSoftConstraintType("zeroVelocity") == MpcSoftConstraintType::ZeroVelocity, "soft zeroVelocity");

  // Test hard constraint string-to-enum
  CHECK_TRUE(stringToMpcHardConstraintType("zero_wrench").ok(), "hard 1 ok");
  CHECK_TRUE(*stringToMpcHardConstraintType("zero_wrench") == MpcHardConstraintType::ZeroWrench, "hard 1");
  CHECK_TRUE(*stringToMpcHardConstraintType("zero_velocity") == MpcHardConstraintType::ZeroVelocity, "hard 2");
  CHECK_TRUE(*stringToMpcHardConstraintType("normal_velocity") == MpcHardConstraintType::NormalVelocity, "hard 3");
  CHECK_TRUE(*stringToMpcHardConstraintType("knee_joint_mimic") == MpcHardConstraintType::KneeJointMimic, "hard 4");

  // Test enum-to-string
  CHECK_TRUE(*mpcCostTypeToString(MpcCostType::StateQuadraticCost) == "state_quadratic_cost", "cost to string");
  CHECK_TRUE(*mpcSoftConstraintTypeToString(MpcSoftConstraintType::JointLimits) == "joint_limits", "soft to string");
  CHECK_TRUE(*mpcSoftConstraintTypeToString(MpcSoftConstraintType::ZeroVelocity) == "zero_velocity", "soft zero_velocity to string");
  CHECK_TRUE(*mpcHardConstraintTypeToString(MpcHardConstraintType::ZeroWrench) == "zero_wrench", "hard to string");

  // Test error handling: returns InvalidArgument status, does not throw
  absl::StatusOr<MpcCostType> invalidCost = stringToMpcCostType("invalid_cost_name");
  CHECK_TRUE(!invalidCost.ok(), "invalid cost should return error status");
  CHECK_TRUE(invalidCost.status().code() == absl::StatusCode::kInvalidArgument, "invalid cost code");

  absl::StatusOr<MpcSoftConstraintType> invalidSoft = stringToMpcSoftConstraintType("invalid_soft_name");
  CHECK_TRUE(!invalidSoft.ok(), "invalid soft constraint should return error status");
  CHECK_TRUE(invalidSoft.status().code() == absl::StatusCode::kInvalidArgument, "invalid soft code");

  absl::StatusOr<MpcHardConstraintType> invalidHard = stringToMpcHardConstraintType("invalid_hard_name");
  CHECK_TRUE(!invalidHard.ok(), "invalid hard constraint should return error status");
  CHECK_TRUE(invalidHard.status().code() == absl::StatusCode::kInvalidArgument, "invalid hard code");

  absl::StatusOr<std::string> invalidCostStr = mpcCostTypeToString(static_cast<MpcCostType>(999));
  CHECK_TRUE(!invalidCostStr.ok(), "invalid cost enum should return error status");
  CHECK_TRUE(invalidCostStr.status().code() == absl::StatusCode::kInvalidArgument, "invalid cost enum code");

  absl::StatusOr<std::string> invalidSoftStr = mpcSoftConstraintTypeToString(static_cast<MpcSoftConstraintType>(999));
  CHECK_TRUE(!invalidSoftStr.ok(), "invalid soft enum should return error status");
  CHECK_TRUE(invalidSoftStr.status().code() == absl::StatusCode::kInvalidArgument, "invalid soft enum code");

  absl::StatusOr<std::string> invalidHardStr = mpcHardConstraintTypeToString(static_cast<MpcHardConstraintType>(999));
  CHECK_TRUE(!invalidHardStr.ok(), "invalid hard enum should return error status");
  CHECK_TRUE(invalidHardStr.status().code() == absl::StatusCode::kInvalidArgument, "invalid hard enum code");

  // Test non-existent file: returns NotFound status, does not throw
  absl::StatusOr<MpcFormulationTasks> nonExistentFile = loadMpcFormulationTasks("/non/existent/path/task.yaml");
  CHECK_TRUE(!nonExistentFile.ok(), "non-existent file should return error status");
  CHECK_TRUE(nonExistentFile.status().code() == absl::StatusCode::kNotFound, "non-existent file code");

  // Test YAML loading
  const std::string testYamlPath = "/tmp/test_mpc_formulation.yaml";
  {
    std::ofstream ofs(testYamlPath);
    ofs << "hard_constraints:\n"
        << "  - zero_wrench\n"
        << "  - zero_velocity\n"
        << "  - normal_velocity\n\n"
        << "soft_constraints:\n"
        << "  - joint_limits\n"
        << "  - friction_force_cone\n"
        << "  - contact_moment_xy\n\n"
        << "costs:\n"
        << "  - state_quadratic_cost\n"
        << "  - input_quadratic_cost\n"
        << "  - terminal_cost\n";
  }

  absl::StatusOr<MpcFormulationTasks> tasks_result = loadMpcFormulationTasks(testYamlPath, true);
  CHECK_TRUE(tasks_result.ok(), "loadMpcFormulationTasks ok");
  MpcFormulationTasks tasks = *tasks_result;
  CHECK_TRUE(tasks.hardConstraints.size() == 3, "hard constraints count");
  CHECK_TRUE(tasks.softConstraints.size() == 3, "soft constraints count");
  CHECK_TRUE(tasks.costs.size() == 3, "costs count");
  CHECK_TRUE(tasks.hasCost(MpcCostType::StateQuadraticCost), "hasCost StateQuadraticCost");
  CHECK_TRUE(tasks.hasCost(MpcCostType::InputQuadraticCost), "hasCost InputQuadraticCost");
  CHECK_TRUE(tasks.hasCost(MpcCostType::TerminalCost), "hasCost TerminalCost");
  CHECK_TRUE(tasks.hasSoftConstraint(MpcSoftConstraintType::JointLimits), "hasSoftConstraint JointLimits");
  CHECK_TRUE(tasks.hasSoftConstraint(MpcSoftConstraintType::FrictionForceCone), "hasSoftConstraint FrictionForceCone");
  CHECK_TRUE(tasks.hasSoftConstraint(MpcSoftConstraintType::ContactMomentXY), "hasSoftConstraint ContactMomentXY");
  CHECK_TRUE(tasks.hasHardConstraint(MpcHardConstraintType::ZeroWrench), "hasHardConstraint ZeroWrench");
  CHECK_TRUE(tasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity), "hasHardConstraint ZeroVelocity");
  CHECK_TRUE(tasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity), "hasHardConstraint NormalVelocity");

  // Explicitly test absl::string_view caller
  absl::string_view svYamlPath = testYamlPath;
  absl::StatusOr<MpcFormulationTasks> sv_tasks_result = loadMpcFormulationTasks(svYamlPath, false);
  CHECK_TRUE(sv_tasks_result.ok(), "sv_tasks_result ok");
  MpcFormulationTasks tasksFromSv = *sv_tasks_result;
  CHECK_TRUE(tasksFromSv.costs.size() == 3, "sv load costs count");
  CHECK_TRUE(tasksFromSv.hasCost(MpcCostType::StateQuadraticCost), "sv load hasCost StateQuadraticCost");

  // Test YAML loading with soft zero_velocity
  const std::string testSoftZeroVelYamlPath = "/tmp/test_mpc_soft_zero_vel.yaml";
  {
    std::ofstream ofs(testSoftZeroVelYamlPath);
    ofs << "hard_constraints:\n"
        << "  - zero_wrench\n"
        << "  - normal_velocity\n\n"
        << "soft_constraints:\n"
        << "  - joint_limits\n"
        << "  - zero_velocity\n\n"
        << "costs:\n"
        << "  - state_quadratic_cost\n";
  }
  absl::StatusOr<MpcFormulationTasks> softZeroVelTasks = loadMpcFormulationTasks(testSoftZeroVelYamlPath, false);
  CHECK_TRUE(softZeroVelTasks.ok(), "soft zero velocity yaml ok");
  CHECK_TRUE(softZeroVelTasks->hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity), "hasSoftConstraint ZeroVelocity");
  CHECK_TRUE(!softZeroVelTasks->hasHardConstraint(MpcHardConstraintType::ZeroVelocity), "should not have hard ZeroVelocity");

  // Test mutual exclusion: zero_velocity as both hard and soft constraint must fail
  const std::string testConflictYamlPath = "/tmp/test_mpc_conflict.yaml";
  {
    std::ofstream ofs(testConflictYamlPath);
    ofs << "hard_constraints:\n"
        << "  - zero_velocity\n\n"
        << "soft_constraints:\n"
        << "  - zero_velocity\n\n"
        << "costs:\n"
        << "  - state_quadratic_cost\n";
  }
  absl::StatusOr<MpcFormulationTasks> conflictResult = loadMpcFormulationTasks(testConflictYamlPath, false);
  CHECK_TRUE(!conflictResult.ok(), "conflicting zero_velocity should fail");
  CHECK_TRUE(conflictResult.status().code() == absl::StatusCode::kInvalidArgument, "conflict code should be InvalidArgument");

  // ---------------------------------------------------------------------------------------------------------------
  // Dropping the hard `zero_wrench` constraint un-gates the contact cones, and a cone that is not listed enforces
  // nothing. f_n >= 0 is the first of the three conditions of rigid contact, and neither `contact_complementarity`
  // (f_n h = 0, which at h = 0 is satisfied by ANY f_n, adhesion included) nor `ground_penetration` (h >= 0) supplies
  // it. See humanoid_nmpc/docs/contact_implicit_mpc/README.md section 2a.
  // ---------------------------------------------------------------------------------------------------------------

  // The hole: the whole contact-implicit formulation listed, and nothing bounding any foot's wrench.
  const absl::StatusOr<MpcFormulationTasks> noCone = loadContactImplicitTaskFile("/tmp/test_mpc_ci_no_cone.yaml", "");
  CHECK_TRUE(!noCone.ok(), "contact-implicit without a cone must fail");
  CHECK_TRUE(noCone.status().code() == absl::StatusCode::kInvalidArgument, "no-cone code should be InvalidArgument");
  CHECK_TRUE(absl::StrContains(noCone.status().message(), "friction_force_cone"),
             "the no-cone message must name both cones, so the operator knows either will do");

  // Either cone closes it. `contact_wrench_cone` carries the friction, CoP and torsional rows together...
  const absl::StatusOr<MpcFormulationTasks> wrenchCone =
      loadContactImplicitTaskFile("/tmp/test_mpc_ci_wrench_cone.yaml", "contact_wrench_cone");
  CHECK_TRUE(wrenchCone.ok(), absl::StrCat("contact-implicit with contact_wrench_cone must load: ", wrenchCone.status().message()));
  CHECK_TRUE(!contactConstraintsAreScheduleGated(*wrenchCone), "without zero_wrench the cones must be un-gated");
  CHECK_TRUE(usesContactImplicitFormulation(*wrenchCone), "the three terms must be recognised as the contact-implicit formulation");

  // ...and `friction_force_cone` bounds the normal force below on its own, which is all this condition asks for.
  const absl::StatusOr<MpcFormulationTasks> frictionCone =
      loadContactImplicitTaskFile("/tmp/test_mpc_ci_friction_cone.yaml", "friction_force_cone");
  CHECK_TRUE(frictionCone.ok(), absl::StrCat("contact-implicit with friction_force_cone must load: ", frictionCone.status().message()));

  // The gate is keyed off `zero_wrench`, not off the contact-implicit terms, so dropping it alone is enough to
  // require a cone - a task file may legitimately drop it without listing the three terms.
  const std::string bareUngatedPath = "/tmp/test_mpc_bare_ungated.yaml";
  {
    std::ofstream ofs(bareUngatedPath);
    ofs << "hard_constraints:\n"
        << "  - zero_velocity\n\n"
        << "soft_constraints:\n"
        << "  - joint_limits\n\n"
        << "costs:\n"
        << "  - state_quadratic_cost\n";
  }
  const absl::StatusOr<MpcFormulationTasks> bareUngated = loadMpcFormulationTasks(bareUngatedPath, false);
  CHECK_TRUE(!bareUngated.ok(), "dropping zero_wrench without any cone must fail even with no contact-implicit term");
  CHECK_TRUE(bareUngated.status().code() == absl::StatusCode::kInvalidArgument, "bare un-gated code should be InvalidArgument");

  // And the shipped arrangement is untouched: while `zero_wrench` is listed the cones gate themselves on the mode
  // schedule, so a missing cone is merely redundant rather than dangerous.
  const std::string gatedNoConePath = "/tmp/test_mpc_gated_no_cone.yaml";
  {
    std::ofstream ofs(gatedNoConePath);
    ofs << "hard_constraints:\n"
        << "  - zero_wrench\n"
        << "  - zero_velocity\n\n"
        << "soft_constraints:\n"
        << "  - joint_limits\n\n"
        << "costs:\n"
        << "  - state_quadratic_cost\n";
  }
  const absl::StatusOr<MpcFormulationTasks> gatedNoCone = loadMpcFormulationTasks(gatedNoConePath, false);
  CHECK_TRUE(gatedNoCone.ok(), absl::StrCat("a gated task file without a cone must still load: ", gatedNoCone.status().message()));
  CHECK_TRUE(contactConstraintsAreScheduleGated(*gatedNoCone), "with zero_wrench the cones must be gated");

  LOG(INFO) << "All MPC Formulation Config tests passed successfully!";
  return 0;
}
