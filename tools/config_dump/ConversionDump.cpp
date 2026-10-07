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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "tools/config_dump/ConversionDump.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_centroidal_mpc/config/costs/DcmTerminalCostFromConfig.h"
#include "humanoid_centroidal_mpc/config/costs/IcpCostFromConfig.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/costs/JointLimitsFromConfig.h"
#include "humanoid_common_mpc/config/costs/JointMimicFromConfig.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/config/robot/ControllerSideSettingsFromConfig.h"
#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactPolygon.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"
#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_common_mpc_app/teleop/config/reference/KeyboardCommandLimitsFromConfig.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/config/robot/VisualizationConfigFromConfig.h"
#include "humanoid_wb_mpc/config/costs/EndEffectorDynamicsWeightsFromConfig.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"
#include "tools/config_dump/SettingsDump.h"
#include "tools/config_dump/ValueDump.h"

namespace ocs2::humanoid::config_dump {
namespace {

// The PD gain defaults of the conversion of the PD gains file: a controller that commands torque limits, so that the
// file's torque limits are read too.
constexpr scalar_t kPdGainsDefaultKp = 100.0;
constexpr scalar_t kPdGainsDefaultKd = 5.0;
constexpr scalar_t kPdGainsDefaultTorqueLimit = 150.0;

void addError(ValueDump& dump, absl::string_view path, const absl::Status& status) {
  dump.addString(joinPath(path, "error"), status.ToString());
}

void addBarrier(ValueDump& dump, absl::string_view path, const absl::StatusOr<RelaxedBarrierPenalty::Config>& barrier) {
  if (!barrier.ok()) {
    addError(dump, path, barrier.status());
    return;
  }
  dump.addDouble(joinPath(path, "mu"), barrier->mu);
  dump.addDouble(joinPath(path, "delta"), barrier->delta);
}

void addBarrier(ValueDump& dump, absl::string_view path, const absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config>& barrier) {
  if (!barrier.ok()) {
    addError(dump, path, barrier.status());
    return;
  }
  dump.addDouble(joinPath(path, "mu"), barrier->mu);
  dump.addDouble(joinPath(path, "delta"), barrier->delta);
}

void addMatrixDiagonal(ValueDump& dump, absl::string_view path, const absl::StatusOr<matrix_t>& matrix) {
  if (!matrix.ok()) {
    addError(dump, path, matrix.status());
    return;
  }
  dump.addDiagonal(path, *matrix);
}

void addVector(ValueDump& dump, absl::string_view path, const absl::StatusOr<vector_t>& vector) {
  if (!vector.ok()) {
    addError(dump, path, vector.status());
    return;
  }
  dump.addMatrix(path, *vector);
}

void addKinematicsWeights(ValueDump& dump, absl::string_view path, const EndEffectorKinematicsWeights& weights) {
  dump.addMatrix(path, weights.toVector());
}

void addDynamicsWeights(ValueDump& dump, absl::string_view path, const absl::StatusOr<EndEffectorDynamicsWeights>& weights) {
  if (!weights.ok()) {
    addError(dump, path, weights.status());
    return;
  }
  dump.addMatrix(path, weights->toVector());
}

/** Every double of a heuristic's parameter struct, which holds nothing else, in declaration order. */
template <typename Parameters>
void addParameters(ValueDump& dump, absl::string_view path, const Parameters& parameters) {
  static_assert(std::is_trivially_copyable_v<Parameters> && sizeof(Parameters) % sizeof(double) == 0);
  std::array<double, sizeof(Parameters) / sizeof(double)> members{};
  std::memcpy(members.data(), &parameters, sizeof(Parameters));
  dump.addDoubles(path, std::vector<double>(members.begin(), members.end()));
}

void dumpSolver(ValueDump& dump, const mpc_config::TaskFile& task) {
  dump.addSection("solver");
  const absl::StatusOr<SolverSettings> solver = solverSettingsFromConfig(task);
  if (!solver.ok()) {
    addError(dump, "solver", solver.status());
    return;
  }
  dumpSqpSettings(dump, "solver.sqp", solver->sqpSettings);
  dumpRolloutSettings(dump, "solver.rollout", solver->rolloutSettings);
  dumpMpcSettings(dump, "solver.mpc", solver->mpcSettings);
  dump.addBool("solver.verbose", solver->verbose);
}

/** The canonical names of the members of `set` that `toString` names, sorted: the set's own order is a hash order. */
template <typename Type>
std::vector<std::string> sortedNames(const absl::flat_hash_set<Type>& set, absl::StatusOr<std::string> (*absl_nonnull toString)(Type)) {
  std::vector<std::string> names;
  for (const Type type : set) {
    const absl::StatusOr<std::string> name = toString(type);
    names.push_back(name.ok() ? *name : name.status().ToString());
  }
  std::sort(names.begin(), names.end());
  return names;
}

void dumpFormulation(ValueDump& dump, const mpc_config::TaskFile& task, StateInputLayout::Mpc mpc) {
  dump.addSection("formulation");
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(task, FormulationLogging::kQuiet);
  if (tasks.ok()) {
    dump.addStrings("formulation.costs", sortedNames(tasks->costs, &mpcCostTypeToString));
    dump.addStrings("formulation.softConstraints", sortedNames(tasks->softConstraints, &mpcSoftConstraintTypeToString));
    dump.addStrings("formulation.hardConstraints", sortedNames(tasks->hardConstraints, &mpcHardConstraintTypeToString));
  } else {
    addError(dump, "formulation", tasks.status());
  }
  const absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromConfig(task);
  if (source.ok()) {
    dump.addString("formulation.contactScheduleSource", contactScheduleSourceName(*source));
  } else {
    addError(dump, "formulation.contactScheduleSource", source.status());
  }
  const absl::StatusOr<ContactInputParameterization> parameterization = contactInputParameterizationFromConfig(task);
  if (parameterization.ok()) {
    dump.addString("formulation.contactInputParameterization", contactInputParameterizationName(*parameterization));
  } else {
    addError(dump, "formulation.contactInputParameterization", parameterization.status());
  }
  if (mpc == StateInputLayout::Mpc::kCentroidal) {
    const absl::StatusOr<CentroidalModelType> modelType = centroidalModelTypeFromConfig(task);
    if (modelType.ok()) {
      dump.addInteger("formulation.centroidalModelType", static_cast<int64_t>(*modelType));
    } else {
      addError(dump, "formulation.centroidalModelType", modelType.status());
    }
  }
}

void dumpLegTorqueCost(ValueDump& dump, absl::string_view path, const mpc_config::JointWeights& weights, const StateInputLayout& layout) {
  const absl::StatusOr<ExternalTorqueQuadraticCostAD::Config> cost = legTorqueCostFromConfig(weights, layout, path);
  if (!cost.ok()) {
    addError(dump, path, cost.status());
    return;
  }
  dump.addStrings(joinPath(path, "activeJointNames"), cost->activeJointNames);
  dump.addMatrix(joinPath(path, "weights"), cost->weights);
}

void dumpWeights(ValueDump& dump, const mpc_config::TaskFile& task, const StateInputLayout& layout) {
  dump.addSection("weights");
  addMatrixDiagonal(dump, "state_weights", stateWeightsFromConfig(task.state_weights, layout, "state_weights"));
  addMatrixDiagonal(dump, "final_state_weights", stateWeightsFromConfig(task.final_state_weights, layout, "final_state_weights"));
  addMatrixDiagonal(dump, "input_weights", inputWeightsFromConfig(task.input_weights, layout, "input_weights"));
  addVector(dump, "initial_state", stateValuesFromConfig(task.initial_state, layout, "initial_state"));
  addMatrixDiagonal(dump, "com_weights", comWeightsFromConfig(task.com_weights, "com_weights"));
  addMatrixDiagonal(dump, "acom_weights", acomWeightsFromConfig(task.acom_weights, "acom_weights"));
  const absl::StatusOr<scalar_t> terminalCostScaling = terminalCostScalingFromConfig(task);
  if (terminalCostScaling.ok()) {
    dump.addDouble("terminal_cost_scaling", *terminalCostScaling);
  } else {
    addError(dump, "terminal_cost_scaling", terminalCostScaling.status());
  }
  if (layout.mpc == StateInputLayout::Mpc::kCentroidal) {
    dumpLegTorqueCost(dump, "left_leg_torque_cost", task.left_leg_torque_cost, layout);
    dumpLegTorqueCost(dump, "right_leg_torque_cost", task.right_leg_torque_cost, layout);
  } else {
    addVector(dump, "joint_torque_weights", jointTorqueWeightsFromConfig(task.joint_torque_weights, layout, "joint_torque_weights"));
  }
}

void dumpContacts(ValueDump& dump, const mpc_config::ContactsConfig& contacts, const ModelSettings& model) {
  dump.addSection("contacts");
  for (size_t i = 0; i < model.contactNames.size(); ++i) {
    const std::string path = absl::StrCat("contacts[", i, "]");
    const absl::StatusOr<ContactRectangle> rectangle = contactRectangleFromConfig(contacts, model, static_cast<int>(i));
    if (!rectangle.ok()) {
      addError(dump, path, rectangle.status());
      continue;
    }
    const ContactCenterPoint& center = rectangle->getContactCenterPoint();
    dump.addString(joinPath(path, "frameName"), center.frameName);
    dump.addString(joinPath(path, "parentJointName"), center.parentJointName);
    dump.addMatrix(joinPath(path, "translationFromParent"), center.translationFromParent);
    const PolygonBounds& bounds = rectangle->getBounds();
    dump.addDoubles(joinPath(path, "bounds"), {bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max});
  }
  const absl::StatusOr<ContactWrenchConeConstraint::Config> cone = contactWrenchConeConfigFromConfig(contacts);
  if (cone.ok()) {
    dump.addInteger("contact_wrench_cone.numBasisVectors", static_cast<int64_t>(cone->numBasisVectors));
    dump.addDouble("contact_wrench_cone.frictionCoefficient", cone->frictionCoefficient);
    dump.addDouble("contact_wrench_cone.torsionalFrictionCoefficient", cone->torsionalFrictionCoefficient);
    dump.addDouble("contact_wrench_cone.minNormalForce", cone->minNormalForce);
    dump.addDouble("contact_wrench_cone.gripperForce", cone->gripperForce);
    dump.addMatrix("contact_wrench_cone.patchOffset", cone->patchOffset);
  } else {
    addError(dump, "contact_wrench_cone", cone.status());
  }
  addBarrier(dump, "contact_wrench_cone.barrier", contactWrenchConeBarrierFromConfig(contacts));
  const absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> bases = contactWrenchConeBasesFromConfig(contacts, model);
  if (bases.ok()) {
    for (size_t i = 0; i < bases->size(); ++i) {
      const ContactWrenchConeBasisMatrix& basis = (*bases)[i];
      dump.addString(absl::StrCat("basis[", i, "].generatorSet"), basis.generatorSet());
      dump.addFingerprint(absl::StrCat("basis[", i, "].matrix"), basis.getBasisMatrix());
    }
  } else {
    addError(dump, "basis", bases.status());
  }
  addBarrier(dump, "basis.nonNegativityBarrier", basisNonNegativityBarrierFromConfig(contacts));
  const absl::StatusOr<BasisRegularizationSettings> regularization = basisRegularizationFromConfig(contacts);
  if (regularization.ok()) {
    dump.addDouble("basis.regularization.weight", regularization->basisScalingRegularization);
    dump.addString("basis.regularization.name", regularization->basisRegularization);
  } else {
    addError(dump, "basis.regularization", regularization.status());
  }
  const absl::StatusOr<FrictionForceConeConstraint::Config> friction = frictionForceConeConfigFromConfig(contacts);
  if (friction.ok()) {
    dump.addDouble("friction_force_cone.frictionCoefficient", friction->frictionCoefficient);
    dump.addDouble("friction_force_cone.regularization", friction->regularization);
    dump.addDouble("friction_force_cone.gripperForce", friction->gripperForce);
    dump.addDouble("friction_force_cone.hessianDiagonalShift", friction->hessianDiagonalShift);
  } else {
    addError(dump, "friction_force_cone", friction.status());
  }
  addBarrier(dump, "friction_force_cone.barrier", frictionForceConeBarrierFromConfig(contacts));
  addBarrier(dump, "contact_moment_xy.barrier", contactMomentXyBarrierFromConfig(contacts));
}

void dumpMimicJoints(ValueDump& dump, const absl::StatusOr<feet_array_t<JointMimicSettings>>& mimic) {
  if (!mimic.ok()) {
    addError(dump, "mimic_joints", mimic.status());
    return;
  }
  for (size_t i = 0; i < mimic->size(); ++i) {
    const JointMimicSettings& leg = (*mimic)[i];
    const std::string path = absl::StrCat("mimic_joints[", i, "]");
    dump.addString(joinPath(path, "parentJointName"), leg.parentJointName);
    dump.addString(joinPath(path, "childJointName"), leg.childJointName);
    dump.addDoubles(joinPath(path, "multiplierAndGains"), {leg.multiplier, leg.positionGain, leg.velocityGain});
  }
}

void dumpConstraints(ValueDump& dump, const mpc_config::TaskFile& task, StateInputLayout::Mpc mpc) {
  dump.addSection("constraints");
  addBarrier(dump, "joint_limits.barrier", jointLimitsBarrierFromConfig(task.joint_limits));
  const absl::StatusOr<FootCollisionConstraint::Config> collision = footCollisionConstraintConfigFromConfig(task.collision_constraint);
  if (collision.ok()) {
    dump.addStrings("collision_constraint.frames",
                    {collision->leftAnkleFrame, collision->rightAnkleFrame, collision->leftFootCenterFrame, collision->rightFootCenterFrame,
                     collision->leftFootFrame1, collision->rightFootFrame1, collision->leftFootFrame2, collision->rightFootFrame2,
                     collision->leftKneeFrame, collision->rightKneeFrame});
    dump.addDouble("collision_constraint.footCollisionSphereRadius", collision->footCollisionSphereRadius);
    dump.addDouble("collision_constraint.kneeCollisionSphereRadius", collision->kneeCollisionSphereRadius);
  } else {
    addError(dump, "collision_constraint", collision.status());
  }
  addBarrier(dump, "collision_constraint.barrier", footCollisionBarrierFromConfig(task.collision_constraint));
  dumpMimicJoints(dump, mpc == StateInputLayout::Mpc::kCentroidal ? kneeMimicKinematicJointsFromConfig(task.mimic_joints)
                                                                  : kneeMimicJointsFromConfig(task.mimic_joints));
}

void dumpTaskSpaceCosts(ValueDump& dump, const mpc_config::TaskFile& task, StateInputLayout::Mpc mpc) {
  dump.addSection("task_space");
  if (mpc == StateInputLayout::Mpc::kWholeBody) {
    addDynamicsWeights(dump, "task_space_foot_cost", wholeBodyFootCostWeightsFromConfig(task.task_space_foot_cost));
    return;
  }
  const absl::StatusOr<TaskSpaceFootCostSettings> foot = taskSpaceFootCostFromConfig(task.task_space_foot_cost);
  if (foot.ok()) {
    addKinematicsWeights(dump, "task_space_foot_cost.weights", foot->weights);
    dump.addBool("task_space_foot_cost.activeInStance", foot->activeInStance);
  } else {
    addError(dump, "task_space_foot_cost", foot.status());
  }
  const absl::StatusOr<std::vector<TaskSpaceLinkCostSettings>> links = taskSpaceLinkCostsFromConfig(task.task_space_costs);
  if (!links.ok()) {
    addError(dump, "task_space_costs", links.status());
    return;
  }
  for (size_t i = 0; i < links->size(); ++i) {
    const TaskSpaceLinkCostSettings& link = (*links)[i];
    const std::string path = absl::StrCat("task_space_costs[", i, "]");
    dump.addString(joinPath(path, "name"), link.name);
    dump.addString(joinPath(path, "linkName"), link.linkName);
    addKinematicsWeights(dump, joinPath(path, "weights"), link.weights);
  }
}

void dumpCentroidalCosts(ValueDump& dump, const mpc_config::TaskFile& task) {
  dump.addSection("centroidal_costs");
  const absl::StatusOr<DcmTerminalCost::Config> dcm = dcmTerminalCostConfigFromConfig(task.dcm_terminal_cost);
  if (dcm.ok()) {
    // An unset height (the model's pendulum) is dumped as the 0 that used to stand for it.
    dump.addDoubles("dcm_terminal_cost", {dcm->comHeight.value_or(0.0), dcm->gravity, dcm->weights.x(), dcm->weights.y(),
                                          dcm->velocityOffsetFactor, dcm->supportBlendTime});
  } else {
    addError(dump, "dcm_terminal_cost", dcm.status());
  }
  const absl::StatusOr<vector2_t> icp = icpCostWeightsFromConfig(task.icp_cost_weights);
  if (icp.ok()) {
    dump.addMatrix("icp_cost_weights", *icp);
  } else {
    addError(dump, "icp_cost_weights", icp.status());
  }
}

void dumpHeuristics(ValueDump& dump, const mpc_config::LocomotionHeuristicsConfig& config) {
  dump.addSection("locomotion_heuristics");
  const absl::StatusOr<LocomotionHeuristicConfig> heuristics = locomotionHeuristicConfigFromConfig(config);
  if (!heuristics.ok()) {
    addError(dump, "locomotion_heuristics", heuristics.status());
    return;
  }
  for (const HeuristicKind kind : allHeuristicKinds()) {
    dump.addStrings(joinPath("locomotion_heuristics", heuristicKindName(kind)), heuristics->formulation.list(kind));
  }
  addParameters(dump, "orientation_compensation", heuristics->orientationCompensation);
  addParameters(dump, "periodic_orientation", heuristics->periodicOrientation);
  addParameters(dump, "height_compensation", heuristics->heightCompensation);
  addParameters(dump, "hip_centered_stepping", heuristics->hipCenteredStepping);
  addParameters(dump, "capture_point", heuristics->capturePoint);
  addParameters(dump, "translational_stepping", heuristics->translationalStepping);
  addParameters(dump, "in_place_turning", heuristics->inPlaceTurning);
  addParameters(dump, "high_speed_turning", heuristics->highSpeedTurning);
  addParameters(dump, "impulse_scaling", heuristics->impulseScaling);
  addParameters(dump, "centripetal_acceleration", heuristics->centripetalAcceleration);
}

void dumpReference(ValueDump& dump, const ConversionInputs& inputs, const ModelSettings* absl_nullable model) {
  dump.addSection("reference");
  const absl::StatusOr<ReferenceSettings> reference = referenceSettingsFromConfig(inputs.reference);
  if (reference.ok()) {
    dump.addDoubles("reference.settings",
                    {reference->targetDisplacementVelocity, reference->targetRotationVelocity, reference->maxDisplacementVelocityX,
                     reference->maxDisplacementVelocityY, reference->maxDeltaPelvisHeight, reference->maxRotationVelocity,
                     reference->maxLinearAcceleration, reference->maxAngularAcceleration, reference->velocityCommandFilterBreakFrequency,
                     reference->defaultBaseHeight});
    dump.addBool("reference.targetJointStateInterpolationTimeConstant.has_value",
                 reference->targetJointStateInterpolationTimeConstant.has_value());
    if (reference->targetJointStateInterpolationTimeConstant.has_value()) {
      dump.addDouble("reference.targetJointStateInterpolationTimeConstant", *reference->targetJointStateInterpolationTimeConstant);
    }
  } else {
    addError(dump, "reference.settings", reference.status());
  }
  if (model != nullptr) {
    addVector(dump, "reference.default_joint_state",
              defaultJointStateFromConfig(inputs.reference, model->mpcModelJointNames, model->fixedJointNames));
  }
  const absl::StatusOr<ModeSchedule> schedule = initialModeScheduleFromConfig(inputs.reference);
  if (schedule.ok()) {
    dumpModeSchedule(dump, "reference.initial_mode_schedule", *schedule);
  } else {
    addError(dump, "reference.initial_mode_schedule", schedule.status());
  }
  const absl::StatusOr<ModeSequenceTemplate> pattern = defaultModeSequenceTemplateFromConfig(inputs.reference);
  if (pattern.ok()) {
    dump.addDoubles("reference.default_mode_sequence_template.switchingTimes", pattern->switchingTimes);
    dump.addSizes("reference.default_mode_sequence_template.modeSequence", pattern->modeSequence);
  } else {
    addError(dump, "reference.default_mode_sequence_template", pattern.status());
  }
  const absl::StatusOr<teleop::KeyboardCommandLimits> keyboard = teleop::keyboardCommandLimitsFromConfig(inputs.reference);
  if (keyboard.ok()) {
    dumpKeyboardCommandLimits(dump, "reference.keyboard", *keyboard);
  } else {
    addError(dump, "reference.keyboard", keyboard.status());
  }
  dump.addSection("gaits");
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits = gaitMapFromConfig(inputs.gait);
  if (gaits.ok()) {
    dumpGaitMap(dump, "gaits", *gaits);
  } else {
    addError(dump, "gaits", gaits.status());
  }
}

void dumpRobot(ValueDump& dump, const ConversionInputs& inputs, const ModelSettings* absl_nullable model) {
  dump.addSection("robot");
  if (model != nullptr) {
    const JointPdGainsDefaults defaults{.kp = kPdGainsDefaultKp, .kd = kPdGainsDefaultKd, .torqueLimit = kPdGainsDefaultTorqueLimit};
    const absl::StatusOr<JointPdGains> gains =
        jointPdGainsFromConfig(inputs.pdGains, defaults, model->mpcModelJointNames, model->fixedJointNames);
    if (gains.ok()) {
      dumpJointPdGains(dump, "pd_gains", *gains);
    } else {
      addError(dump, "pd_gains", gains.status());
    }
    const absl::StatusOr<visualization::VisualizationConfig> visualization =
        visualization::visualizationConfigFromConfig(inputs.task, model->contactNames);
    if (visualization.ok()) {
      dumpVisualizationConfig(dump, "visualization", *visualization);
    } else {
      addError(dump, "visualization", visualization.status());
    }
  }
  const absl::StatusOr<RobotProcessSettings> robot = robotProcessSettingsFromConfig(inputs.task);
  if (robot.ok()) {
    dumpRobotProcessSettings(dump, "robot_process", *robot);
  } else {
    addError(dump, "robot_process", robot.status());
  }
  const ControllerSideConfig controllerSide = controllerSideSettingsFromConfig(inputs.task);
  dump.addString("controller_side.contactEstimator", controllerSide.contactEstimator);
  dump.addBool("controller_side.contactWrenchGate.has_value", controllerSide.contactWrenchGate.has_value());
  if (controllerSide.contactWrenchGate.has_value()) {
    dump.addDoubles("controller_side.contactWrenchGate",
                    {controllerSide.contactWrenchGate->debounceTime, controllerSide.contactWrenchGate->rampTime});
  }
  dump.addStrings("controller_side.problems", controllerSide.problems);
}

}  // namespace

std::string dumpConversions(const ConversionInputs& inputs, const std::string& urdfFile, StateInputLayout::Mpc mpc) {
  ValueDump dump;
  dump.addSection("model_settings");
  const char* absl_nonnull mpcName = mpc == StateInputLayout::Mpc::kCentroidal ? "centroidal_mpc_" : "wb_mpc_";
  const absl::StatusOr<ModelSettings> model = ModelSettings::Create(inputs.task, urdfFile, mpcName, /*verbose=*/false);
  if (model.ok()) {
    dumpModelSettings(dump, "model_settings", *model);
  } else {
    addError(dump, "model_settings", model.status());
  }
  dumpSolver(dump, inputs.task);
  dumpFormulation(dump, inputs.task, mpc);
  if (model.ok()) {
    dumpWeights(dump, inputs.task, stateInputLayout(*model, mpc));
    dumpContacts(dump, inputs.task.contacts, *model);
  }
  dumpConstraints(dump, inputs.task, mpc);
  dumpTaskSpaceCosts(dump, inputs.task, mpc);
  if (mpc == StateInputLayout::Mpc::kCentroidal) {
    dumpCentroidalCosts(dump, inputs.task);
    dumpHeuristics(dump, inputs.task.locomotion_heuristics);
  }
  dump.addSection("swing");
  const absl::StatusOr<SwingTrajectoryPlanner::Config> swing = swingTrajectorySettingsFromConfig(inputs.task.swing_trajectory_config);
  if (swing.ok()) {
    dumpSwingSettings(dump, "swing", *swing);
  } else {
    addError(dump, "swing", swing.status());
  }
  const ModelSettings* absl_nullable modelOrNull = model.ok() ? &*model : nullptr;
  dumpReference(dump, inputs, modelOrNull);
  dumpRobot(dump, inputs, modelOrNull);
  return dump.text();
}

}  // namespace ocs2::humanoid::config_dump
