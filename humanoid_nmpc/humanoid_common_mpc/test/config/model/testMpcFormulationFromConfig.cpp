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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"

#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

/**
 * The formulation of the typed task file: the term lists resolved by their registries and the combinations they refuse,
 * the contact schedule source, the contact input parameterization and the centroidal model, by name.
 */
namespace ocs2::humanoid {
namespace {

using ::testing::HasSubstr;

/** The schedule-gated formulation every robot ships: the hard contact constraints and a cone. */
mpc_config::TaskFile scheduleGatedTask() {
  mpc_config::TaskFile task;
  task.hard_constraints = {"zero_wrench", "zero_velocity", "normal_velocity"};
  task.soft_constraints = {"joint_limits", "foot_collision", "contact_wrench_cone"};
  task.costs = {"state_quadratic_cost", "input_quadratic_cost", "terminal_cost", "task_space_foot_cost"};
  return task;
}

/** The contact-implicit formulation, complete: the three terms, the soft normal velocity and a cone. */
mpc_config::TaskFile contactImplicitTask() {
  mpc_config::TaskFile task;
  task.soft_constraints = {"contact_wrench_cone", "normal_velocity", "contact_complementarity", "force_weighted_slip",
                           "ground_penetration"};
  task.costs = {"state_quadratic_cost"};
  return task;
}

/** The message of the error `task` is refused with, or "accepted". */
std::string refusal(const mpc_config::TaskFile& task) {
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(task, FormulationLogging::kQuiet);
  if (tasks.ok()) return "accepted";
  EXPECT_EQ(tasks.status().code(), absl::StatusCode::kInvalidArgument);
  return std::string(tasks.status().message());
}

TEST(MpcFormulationFromConfig, TheListsResolveToTheirTerms) {
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(scheduleGatedTask(), FormulationLogging::kLogSummary);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_EQ(tasks->hardConstraints.size(), 3U);
  EXPECT_TRUE(tasks->hasHardConstraint(MpcHardConstraintType::kZeroWrench));
  EXPECT_TRUE(tasks->hasSoftConstraint(MpcSoftConstraintType::kContactWrenchCone));
  EXPECT_TRUE(tasks->hasCost(MpcCostType::kTerminalCost));
  EXPECT_EQ(tasks->costs.size(), 4U);
  EXPECT_TRUE(contactConstraintsAreScheduleGated(*tasks));
  EXPECT_FALSE(usesContactImplicitFormulation(*tasks));
}

TEST(MpcFormulationFromConfig, EverySpellingOfTheRegistriesIsAcceptedAndARepeatedNameIsOneTerm) {
  mpc_config::TaskFile task = scheduleGatedTask();
  task.costs = {"stateQuadraticCost", "state_quadratic_cost", "inputCost"};
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(task, FormulationLogging::kQuiet);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_EQ(tasks->costs.size(), 2U);
  EXPECT_TRUE(tasks->hasCost(MpcCostType::kStateQuadraticCost));
  EXPECT_TRUE(tasks->hasCost(MpcCostType::kInputQuadraticCost));
}

TEST(MpcFormulationFromConfig, AnUnknownNameIsRefusedNamingItsEntryAndTheRegisteredNames) {
  mpc_config::TaskFile task = scheduleGatedTask();
  task.costs = {"state_quadratic_cost", "state_quadratic_costs"};
  EXPECT_THAT(refusal(task), HasSubstr("costs[1]: Unknown MPC cost type: 'state_quadratic_costs'"));
  EXPECT_THAT(refusal(task), HasSubstr("dcm_terminal_cost"));

  task = scheduleGatedTask();
  task.soft_constraints.push_back("joint_limit");
  EXPECT_THAT(refusal(task), HasSubstr("soft_constraints[3]: Unknown MPC soft constraint type: 'joint_limit'"));

  task = scheduleGatedTask();
  task.hard_constraints = {"zero_wrenches"};
  EXPECT_THAT(refusal(task), HasSubstr("hard_constraints[0]: Unknown MPC hard constraint type"));
}

TEST(MpcFormulationFromConfig, TheCombinationsThatAreNoFormulationAreRefused) {
  mpc_config::TaskFile task = scheduleGatedTask();
  task.costs.push_back("dcm_terminal_cost");
  EXPECT_THAT(refusal(task), HasSubstr("both 'terminal_cost' and 'dcm_terminal_cost'"));

  task = scheduleGatedTask();
  task.soft_constraints.push_back("zero_velocity");
  EXPECT_THAT(refusal(task), HasSubstr("'zero_velocity' is listed in both"));

  task = scheduleGatedTask();
  task.soft_constraints.push_back("normal_velocity");
  EXPECT_THAT(refusal(task), HasSubstr("'normal_velocity' is listed in both"));

  task = contactImplicitTask();
  task.soft_constraints = {"contact_wrench_cone", "normal_velocity", "contact_complementarity"};
  EXPECT_THAT(refusal(task), HasSubstr("together or not at all"));
  EXPECT_THAT(refusal(task), HasSubstr("'force_weighted_slip' (without it"));

  task = contactImplicitTask();
  task.hard_constraints = {"zero_wrench"};
  EXPECT_THAT(refusal(task), HasSubstr("'contact_complementarity' and the hard 'zero_wrench' constraint are mutually exclusive"));

  task = contactImplicitTask();
  task.soft_constraints = {"normal_velocity", "contact_complementarity", "force_weighted_slip", "ground_penetration"};
  EXPECT_THAT(refusal(task), HasSubstr("'contact_wrench_cone' or 'friction_force_cone' must be listed"));

  task = contactImplicitTask();
  task.hard_constraints = {"zero_velocity"};
  EXPECT_THAT(refusal(task), HasSubstr("'force_weighted_slip' replaces 'zero_velocity'"));

  task = contactImplicitTask();
  task.soft_constraints = {"contact_wrench_cone", "contact_complementarity", "force_weighted_slip", "ground_penetration"};
  task.hard_constraints = {"normal_velocity"};
  EXPECT_THAT(refusal(task), HasSubstr("the hard 'normal_velocity' constraint are mutually exclusive"));

  task = contactImplicitTask();
  task.soft_constraints = {"contact_wrench_cone", "contact_complementarity", "force_weighted_slip", "ground_penetration"};
  EXPECT_THAT(refusal(task), HasSubstr("model_settings.foot_constraint.normal_velocity_soft_constraint_weight"));

  EXPECT_THAT(refusal(mpc_config::TaskFile{}), HasSubstr("must be listed in soft_constraints"))
      << "a file that lists nothing has neither zero_wrench nor a cone";
}

TEST(MpcFormulationFromConfig, TheContactImplicitFormulationIsAcceptedAsAWhole) {
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(contactImplicitTask(), FormulationLogging::kQuiet);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_TRUE(usesContactImplicitFormulation(*tasks));
  EXPECT_FALSE(contactConstraintsAreScheduleGated(*tasks));
}

TEST(MpcFormulationFromConfig, TheContactScheduleSourceIsResolvedByName) {
  mpc_config::TaskFile task;
  absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromConfig(task);
  ASSERT_TRUE(source.ok()) << source.status();
  EXPECT_EQ(*source, kDefaultContactScheduleSource) << "the schema's default is the loader's";
  for (const std::string& name : contactScheduleSourceNames()) {
    task.contact_schedule_source = name;
    source = contactScheduleSourceFromConfig(task);
    ASSERT_TRUE(source.ok()) << source.status();
    EXPECT_EQ(contactScheduleSourceName(*source), name);
  }
  task.contact_schedule_source = "planner";
  source = contactScheduleSourceFromConfig(task);
  EXPECT_EQ(source.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(source.status().message(), HasSubstr("contact_schedule_source is 'planner'"));
  EXPECT_THAT(source.status().message(), HasSubstr("gait_schedule, contact_planner"));
}

TEST(MpcFormulationFromConfig, TheContactInputParameterizationIsResolvedByName) {
  mpc_config::TaskFile task;
  absl::StatusOr<ContactInputParameterization> parameterization = contactInputParameterizationFromConfig(task);
  ASSERT_TRUE(parameterization.ok()) << parameterization.status();
  EXPECT_EQ(*parameterization, kDefaultContactInputParameterization) << "the schema's default is the loader's";
  for (const std::string& name : contactInputParameterizationNames()) {
    task.contact_input_parameterization = name;
    parameterization = contactInputParameterizationFromConfig(task);
    ASSERT_TRUE(parameterization.ok()) << parameterization.status();
    EXPECT_EQ(contactInputParameterizationName(*parameterization), name);
  }
  task.contact_input_parameterization = "basis";
  parameterization = contactInputParameterizationFromConfig(task);
  EXPECT_EQ(parameterization.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(parameterization.status().message(), HasSubstr("contact_input_parameterization is 'basis'"));
  EXPECT_THAT(parameterization.status().message(), HasSubstr("wrench, basis_vectors"));
}

TEST(MpcFormulationFromConfig, TheCentroidalModelIsRequiredAndOneOfItsNames) {
  mpc_config::TaskFile task;
  absl::StatusOr<CentroidalModelType> type = centroidalModelTypeFromConfig(task);
  EXPECT_EQ(type.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(type.status().message(), HasSubstr("sets no centroidal_model"));
  EXPECT_THAT(type.status().message(), HasSubstr("full_centroidal_dynamics, single_rigid_body_dynamics"));

  // Each name is its model, and the name of the model is the name: the registry and its inverse agree.
  EXPECT_EQ(centroidalModelNames(), (std::vector<std::string>{"full_centroidal_dynamics", "single_rigid_body_dynamics"}));
  for (const std::string& name : centroidalModelNames()) {
    task.centroidal_model = name;
    type = centroidalModelTypeFromConfig(task);
    ASSERT_TRUE(type.ok()) << type.status();
    EXPECT_EQ(centroidalModelName(*type), name);
  }

  for (const absl::string_view name : {"0", "FullCentroidalDynamics", ""}) {
    task.centroidal_model = std::string(name);
    type = centroidalModelTypeFromConfig(task);
    EXPECT_EQ(type.status().code(), absl::StatusCode::kInvalidArgument) << name;
    EXPECT_THAT(type.status().message(), HasSubstr("which is not a centroidal model; valid names are: full_centroidal_dynamics")) << name;
  }
}

TEST(MpcFormulationFromConfig, EachCentroidalModelNameIsTheModelOfTheIntCodeItReplaced) {
  // The parity of the retired centroidal_model_type: 0 was the full centroidal dynamics, 1 the single rigid body
  // dynamics, OCS2's CentroidalModelType in the order of its enumerators.
  mpc_config::TaskFile task;
  task.centroidal_model = "full_centroidal_dynamics";
  absl::StatusOr<CentroidalModelType> type = centroidalModelTypeFromConfig(task);
  ASSERT_TRUE(type.ok()) << type.status();
  EXPECT_EQ(*type, static_cast<CentroidalModelType>(0));
  task.centroidal_model = "single_rigid_body_dynamics";
  type = centroidalModelTypeFromConfig(task);
  ASSERT_TRUE(type.ok()) << type.status();
  EXPECT_EQ(*type, static_cast<CentroidalModelType>(1));
}

TEST(MpcFormulationFromConfig, TheRetiredFormulationKeysAreRefusedByTheParser) {
  // The booleans that became names and list entries, and the nesting of the lists, which the old loaders handled
  // themselves: the schema retires them, so a stale file never reaches the conversions.
  for (const absl::string_view retired :
       {"useComAndAcomTracking: true\n", "useDcmTerminalCost: false\n", "useContactPlanning: true\n", "useContactBasisVectorInputs: true\n",
        "mpc_tasks { }\n", "tasks { }\n", "centroidal_model_type: 0\n", "centroidalModelType: 1\n"}) {
    const absl::StatusOr<humanoid_mpc_config::TaskFile> parsed =
        nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(retired, "task.textproto");
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument) << retired;
    EXPECT_THAT(parsed.status().message(), HasSubstr("is retired: ")) << retired;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
