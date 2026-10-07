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

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"
#include "ocs2_mpc/MPC_MRT_Interface.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_sqp/SqpMpc.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcResetTarget.h"
#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/config/ConfigReload.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/parameter_update/CommandLimitsReloaders.h"
#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/options.pb.h"
#include "robot_core/ResourcePaths.h"
#include "support/TypedConfigFiles.h"

/**
 * A live update of the centroidal MPC is a fresh start (test L4 of the live-tuning design): every RELOAD_HOT numeric
 * field of the task file that the centroidal formulation applies (centroidalHotFieldNames()) is perturbed, by a
 * deterministic step that keeps the file valid (a nonzero value times 1.05 to 1.10, an integer plus one), into the file E;
 * an MPC started from the shipped file and handed E through its parameter updater, before any solve, then solves exactly
 * as an MPC started from E does: the same policies, solve for solve, standing and then walking (expected bit for bit;
 * the check allows 1e-12 relative). On the DRC Atlas as it ships (basis-vector contact inputs, the CoM + ACoM cost, the
 * DCM terminal cost) and on the variants whose problems carry the terms it does not: the contact-implicit formulation, and
 * the quadratic terminal cost with a soft zero_velocity, the state-input quadratic cost and listed locomotion heuristics.
 *
 * terrain_height is left out of that equality on purpose (a live ground change lifts the target in use once, a fresh
 * start on the new ground does not); its live semantics are pinned on their own (L4t). Every perturbed field is also
 * shown to change what it writes, one field at a time, on the variant whose problem carries its term (the liveness of the
 * design); and an updater fed nothing changes no solve at all (L5).
 *
 * Generates the Atlas CppAD libraries on a cold cache.
 */
namespace ocs2::humanoid {
namespace {

using google::protobuf::Descriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::Message;
using google::protobuf::Reflection;

constexpr scalar_t kSolvePeriod = 0.02;  // [s] of solver time per solve
constexpr size_t kStandingSolves = 4;
constexpr size_t kWalkingSolves = 12;
constexpr scalar_t kWalkingCommand = 0.5;  // of the command limit along x
constexpr scalar_t kRelativeTolerance = 1.0e-12;

// --------------------------------------------------------------------------------------------------------------------
// The configurations
// --------------------------------------------------------------------------------------------------------------------

/** The shipped DRC Atlas configuration; fails the test and is the default configuration when it does not load. */
CentroidalMpcConfig shippedAtlas() {
  const absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(atlasFiles());
  EXPECT_TRUE(config.ok()) << config.status();
  return config.ok() ? *config : CentroidalMpcConfig{};
}

/** `list` with `from` replaced by `to`; fails the test when it does not list `from`. */
std::vector<std::string> replaced(std::vector<std::string> list, absl::string_view from, absl::string_view to) {
  const std::vector<std::string>::iterator found = std::find(list.begin(), list.end(), from);
  EXPECT_NE(found, list.end()) << from << " is not listed";
  if (found != list.end()) *found = std::string(to);
  return list;
}

/** `list` without `entry`; fails the test when it does not list it. */
std::vector<std::string> without(std::vector<std::string> list, absl::string_view entry) {
  const std::vector<std::string>::iterator found = std::find(list.begin(), list.end(), entry);
  EXPECT_NE(found, list.end()) << entry << " is not listed";
  if (found != list.end()) list.erase(found);
  return list;
}

/** The contact-implicit formulation of `config`: no hard constraint, its three terms and the soft normal velocity. */
CentroidalMpcConfig contactImplicitAtlas(CentroidalMpcConfig config) {
  config.task.hard_constraints.clear();
  config.task.soft_constraints = {"joint_limits",      "foot_collision",          "contact_wrench_cone",
                                  "normal_velocity",   "contact_complementarity", "force_weighted_slip",
                                  "ground_penetration"};
  return config;
}

/**
 * `config` ending on the quadratic terminal cost, with one state-input quadratic cost in place of the state and the
 * input ones, a soft zero_velocity, and the locomotion heuristics `basePose`, `foothold` and `wrench` listed.
 */
CentroidalMpcConfig quadraticCostsAtlas(CentroidalMpcConfig config,
                                        std::vector<std::string> basePose,
                                        std::vector<std::string> foothold,
                                        std::vector<std::string> wrench) {
  config.task.costs = replaced(config.task.costs, "dcm_terminal_cost", "terminal_cost");
  config.task.costs = replaced(config.task.costs, "state_quadratic_cost", "state_input_quadratic_cost");
  config.task.costs = without(config.task.costs, "input_quadratic_cost");
  config.task.hard_constraints = without(config.task.hard_constraints, "zero_velocity");
  config.task.soft_constraints.emplace_back("zero_velocity");
  config.task.locomotion_heuristics.base_pose = std::move(basePose);
  config.task.locomotion_heuristics.foothold = std::move(foothold);
  config.task.locomotion_heuristics.wrench = std::move(wrench);
  return config;
}

/** The variant of quadraticCostsAtlas() the policy test runs: the heuristics the integration test enables together. */
CentroidalMpcConfig quadraticCostsWithHeuristicsAtlas(CentroidalMpcConfig config) {
  return quadraticCostsAtlas(std::move(config), {"orientation_compensation", "height_compensation"},
                             {"hip_centered_stepping", "translational_stepping"}, {"impulse_scaling"});
}

/**
 * The variant of the liveness check whose heuristics are the ones quadraticCostsWithHeuristicsAtlas() does not list, and
 * without the CoM + ACoM cost, so that the base-pose blocks of Q and Q_final are weighed rather than zeroed.
 */
CentroidalMpcConfig otherHeuristicsWithoutAcomAtlas(CentroidalMpcConfig config) {
  config = quadraticCostsAtlas(std::move(config), {"periodic_orientation"}, {"capture_point", "in_place_turning", "high_speed_turning"},
                               {"centripetal_acceleration"});
  config.task.costs = without(config.task.costs, "com_and_acom_tracking_cost");
  return config;
}

// --------------------------------------------------------------------------------------------------------------------
// The perturbed file
// --------------------------------------------------------------------------------------------------------------------

/** Whether the declared path `declared` stands for `path`: it is the path, or a block above it. */
bool covers(absl::string_view declared, absl::string_view path) {
  return path == declared || absl::StartsWith(path, absl::StrCat(declared, ".")) || absl::StartsWith(path, absl::StrCat(declared, "["));
}

/**
 * The paths of the task file's leaves a centroidal reload applies (configLeaves() paths): RELOAD_HOT, rendered by the
 * GUI, read by the MPC of the centroidal formulation and covered by centroidalHotFieldNames(). terrain_height is left out
 * (see the file comment).
 */
std::set<std::string> appliedLeafPaths() {
  const std::vector<std::string> names = centroidalHotFieldNames();
  std::set<std::string> paths;
  for (const ConfigLeaf& leaf : configLeaves(*humanoid_mpc_config::TaskFile::descriptor())) {
    if (leaf.reload != ConfigReload::kHot || !leaf.tunable) continue;
    if (!leaf.consumer.empty() && leaf.consumer != "mpc") continue;
    if (!leaf.formulations.empty() &&
        std::find(leaf.formulations.begin(), leaf.formulations.end(), kCentroidalFormulation) == leaf.formulations.end()) {
      continue;
    }
    if (leaf.path == "terrain_height") continue;
    const bool applied = std::any_of(names.begin(), names.end(), [&leaf](const std::string& name) { return covers(name, leaf.path); });
    if (applied) paths.insert(leaf.path);
  }
  return paths;
}

/** How a leaf is perturbed. */
enum class Step {
  // Every nonzero number times its factor, every integer plus one; a zero stays zero, so that every value keeps its sign
  // and its meaning (the policy test).
  kScale,
  // As kScale, and a zero becomes 0.05, so that the leaf changes whatever its value (the liveness check).
  kScaleOrSet,
};

/** The deterministic factor of the leaf with `ordinal` among the perturbed ones, in [1.05, 1.10]. */
scalar_t factorOf(size_t ordinal) {
  return 1.05 + 0.01 * static_cast<scalar_t>(ordinal % 6);
}

/** The perturbation of a number field of a message: where it is, and how. */
struct NumberField {
  Message* absl_nonnull message;
  const FieldDescriptor* absl_nonnull field;
  // The index of a repeated field's element; -1 for a singular field.
  int index = -1;
};

/** Perturbs the number `number` with `factor` (Step). */
void perturbNumber(const NumberField& number, scalar_t factor, Step step) {
  Message& message = *number.message;
  const FieldDescriptor& field = *number.field;
  const Reflection& reflection = *message.GetReflection();
  const bool repeated = number.index >= 0;
  if (field.cpp_type() == FieldDescriptor::CPPTYPE_DOUBLE) {
    const double value = repeated ? reflection.GetRepeatedDouble(message, &field, number.index) : reflection.GetDouble(message, &field);
    const double perturbed = value != 0.0 ? value * factor : (step == Step::kScaleOrSet ? 0.05 : 0.0);
    if (repeated) {
      reflection.SetRepeatedDouble(&message, &field, number.index, perturbed);
    } else {
      reflection.SetDouble(&message, &field, perturbed);
    }
  } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_FLOAT) {
    const float value = repeated ? reflection.GetRepeatedFloat(message, &field, number.index) : reflection.GetFloat(message, &field);
    const float perturbed = value != 0.0F ? static_cast<float>(value * factor) : (step == Step::kScaleOrSet ? 0.05F : 0.0F);
    if (repeated) {
      reflection.SetRepeatedFloat(&message, &field, number.index, perturbed);
    } else {
      reflection.SetFloat(&message, &field, perturbed);
    }
  } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_INT32) {
    const int32_t value = repeated ? reflection.GetRepeatedInt32(message, &field, number.index) : reflection.GetInt32(message, &field);
    if (repeated) {
      reflection.SetRepeatedInt32(&message, &field, number.index, value + 1);
    } else {
      reflection.SetInt32(&message, &field, value + 1);
    }
  } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_INT64) {
    const int64_t value = repeated ? reflection.GetRepeatedInt64(message, &field, number.index) : reflection.GetInt64(message, &field);
    if (repeated) {
      reflection.SetRepeatedInt64(&message, &field, number.index, value + 1);
    } else {
      reflection.SetInt64(&message, &field, value + 1);
    }
  }
}

/** Whether `field` holds a number this test perturbs. */
bool isNumber(const FieldDescriptor& field) {
  return field.cpp_type() == FieldDescriptor::CPPTYPE_DOUBLE || field.cpp_type() == FieldDescriptor::CPPTYPE_FLOAT ||
         field.cpp_type() == FieldDescriptor::CPPTYPE_INT32 || field.cpp_type() == FieldDescriptor::CPPTYPE_INT64;
}

/**
 * Calls `visit` with every number field of `message` whose configLeaves() path is in `paths`, with that path. A scalar
 * that is absent and has no default, which means "derive it" (an optional value), and an absent (nproto.optional_message)
 * block, which means "off", are not visited: a value there would change what the file means, not tune it.
 */
void forEachNumber(Message& message,
                   const std::string& prefix,
                   const std::set<std::string>& paths,
                   absl::FunctionRef<void(const NumberField&, const std::string&)> visit) {
  const Descriptor& descriptor = *message.GetDescriptor();
  const Reflection& reflection = *message.GetReflection();
  for (int i = 0; i < descriptor.field_count(); ++i) {
    const FieldDescriptor& field = *descriptor.field(i);
    if (field.options().deprecated() || field.is_map()) continue;
    const std::string path = prefix.empty() ? std::string(field.name()) : absl::StrCat(prefix, ".", field.name());
    if (field.cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      if (field.is_repeated()) {
        for (int index = 0; index < reflection.FieldSize(message, &field); ++index) {
          forEachNumber(*reflection.MutableRepeatedMessage(&message, &field, index), absl::StrCat(path, "[*]"), paths, visit);
        }
        continue;
      }
      if (!reflection.HasField(message, &field) && field.options().GetExtension(nproto::optional_message)) continue;
      forEachNumber(*reflection.MutableMessage(&message, &field), path, paths, visit);
      continue;
    }
    if (!isNumber(field) || !paths.contains(path)) continue;
    if (field.is_repeated()) {
      for (int index = 0; index < reflection.FieldSize(message, &field); ++index) {
        visit(NumberField{.message = &message, .field = &field, .index = index}, path);
      }
      continue;
    }
    if (field.has_presence() && !reflection.HasField(message, &field) && !field.has_default_value()) continue;
    visit(NumberField{.message = &message, .field = &field, .index = -1}, path);
  }
}

/** `task` with every number of a leaf in `paths` perturbed by `step`; returns the paths of the numbers it perturbed. */
std::set<std::string> perturb(mpc_config::TaskFile& task, const std::set<std::string>& paths, Step step) {
  humanoid_mpc_config::TaskFile message;
  mpc_config::ToProto(task, &message);
  std::set<std::string> perturbed;
  std::map<std::string, size_t> ordinals;
  forEachNumber(message, /*prefix=*/"", paths, [&](const NumberField& number, const std::string& path) {
    const size_t ordinal = ordinals.try_emplace(path, ordinals.size()).first->second;
    perturbNumber(number, factorOf(ordinal), step);
    perturbed.insert(path);
  });
  const absl::Status converted = mpc_config::FromProto(message, &task);
  EXPECT_TRUE(converted.ok()) << converted;
  return perturbed;
}

// --------------------------------------------------------------------------------------------------------------------
// The MPC, wired as the MPC node wires it
// --------------------------------------------------------------------------------------------------------------------

/**
 * The centroidal MPC of a configuration as CentroidalMpcNode::Create() wires it: the interface, the SQP MPC, the target
 * calculator and the procedural motion manager, and the parameter updater (makeCentroidalMpcParameterUpdater(), watching
 * no file), registered after the motion manager; driven by an MRT interface whose robot follows the predicted state.
 */
class Stack {
 public:
  /** Whether the updater is registered with the solver. */
  enum class Updater { kRegistered, kNone };

  Stack(const CentroidalMpcConfig& config, Updater updaterUse) {
    const CentroidalRobotFiles files = atlasFiles();
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(config, files.urdfFile);
    EXPECT_TRUE(created.ok()) << created.status();
    if (!created.ok()) return;
    interface_ = *std::move(created);
    mpc_ = std::make_unique<SqpMpc>(interface_->mpcSettings(), interface_->sqpSettings(), interface_->getOptimalControlProblem(),
                                    interface_->getInitializer());
    const MpcRobotModelBase<scalar_t>& effectiveModel = interface_->getEffectiveMpcRobotModel();
    absl::StatusOr<std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>> calculator =
        CentroidalMpcTargetTrajectoriesCalculator::Create(config.reference, effectiveModel, interface_->getPinocchioInterface(),
                                                          interface_->getCentroidalModelInfo(), interface_->mpcSettings().timeHorizon_);
    EXPECT_TRUE(calculator.ok()) << calculator.status();
    if (!calculator.ok()) return;
    calculator_ = *std::move(calculator);
    calculator_->setTerrainHeightSource(
        [referenceManager = interface_->getSwitchedModelReferenceManagerPtr()]() { return referenceManager->getAppliedTerrainHeight(); });
    CentroidalMpcTargetTrajectoriesCalculator* absl_nonnull calculatorPtr = calculator_.get();
    absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> motionManager = ProceduralMpcMotionManager::Create(
        robot::resolveResourcePath("humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto").value(), files.referenceFile,
        interface_->getSwitchedModelReferenceManagerPtr(), effectiveModel,
        [calculatorPtr](const vector4_t& velocity, scalar_t initTime, scalar_t /*finalTime*/, const vector_t& initState) {
          return calculatorPtr->commandedVelocityToTargetTrajectories(velocity, initTime, initState);
        });
    EXPECT_TRUE(motionManager.ok()) << motionManager.status();
    if (!motionManager.ok()) return;
    motionManager_ = *std::move(motionManager);
    motionManager_->setResetHook([calculatorPtr]() { calculatorPtr->reset(); });
    mpc_->getSolverPtr()->setReferenceManager(interface_->getReferenceManagerPtr());
    mpc_->getSolverPtr()->addSynchronizedModule(motionManager_);
    absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> parameterUpdater = makeCentroidalMpcParameterUpdater(
        mpc_.get(), *interface_, /*taskFile=*/"", /*referenceFile=*/"", makeCommandLimitsReloaders(calculatorPtr, motionManager_));
    EXPECT_TRUE(parameterUpdater.ok()) << parameterUpdater.status();
    if (!parameterUpdater.ok()) return;
    updater_ = *std::move(parameterUpdater);
    if (updaterUse == Updater::kRegistered) mpc_->getSolverPtr()->addSynchronizedModule(updater_);
    mrt_ = std::make_unique<MPC_MRT_Interface>(*mpc_);
    observation_.time = 0.0;
    observation_.state = interface_->getInitialState();
    observation_.input = vector_t::Zero(effectiveModel.getInputDim());
    observation_.mode = ModeNumber::kStance;
    mrt_->setCurrentObservation(observation_);
    mrt_->resetMpcNode(centroidalMpcResetTargetTrajectories(observation_, interface_->getCentroidalModelInfo(), effectiveModel,
                                                            interface_->getPinocchioInterface()));
  }

  ~Stack() = default;
  Stack(const Stack&) = delete;
  Stack& operator=(const Stack&) = delete;

  bool ok() const { return mrt_ != nullptr; }
  CentroidalMpcInterface& interface() { return *interface_; }
  SqpSolver& solver() { return dynamic_cast<SqpSolver&>(*mpc_->getSolverPtr()); }
  MpcParameterUpdaterModule& updater() { return *updater_; }
  SwitchedModelReferenceManager& referenceManager() { return *interface_->getSwitchedModelReferenceManagerPtr(); }
  scalar_t time() const { return observation_.time; }

  /** Hands the updater `task` as the tuning GUI does, and runs its pre-solve hook once, before the next solve. */
  void applyNow(const mpc_config::TaskFile& task) {
    mpc_config::MpcParameterUpdate update;
    update.task = task;
    updater_->enqueueParameterUpdate(update);
    updater_->preSolverRun(observation_.time, observation_.time + interface_->mpcSettings().timeHorizon_, observation_.state,
                           *interface_->getReferenceManagerPtr());
  }

  /** Hands the updater `task` as the tuning GUI does; the next solve applies it. */
  void enqueue(const mpc_config::TaskFile& task) {
    mpc_config::MpcParameterUpdate update;
    update.task = task;
    updater_->enqueueParameterUpdate(update);
  }

  /** The operator's walking command: `forward` of the command limit along x. */
  void command(scalar_t forward) {
    motionManager_->setAndScaleVelocityCommand(WalkingVelocityCommand(forward, /*v_y=*/0.0, /*desired_pelvis_h=*/0.0, /*v_yaw=*/0.0));
  }

  /** One solve at the current observation; the robot then moves as the policy predicts. Returns the solve's status. */
  absl::Status solve() {
    mrt_->setCurrentObservation(observation_);
    const absl::Status status = mrt_->advanceMpc();
    mrt_->updatePolicy();
    if (status.ok() && mrt_->initialPolicyReceived()) {
      vector_t state;
      vector_t input;
      size_t mode = 0;
      mrt_->evaluatePolicy(observation_.time + kSolvePeriod, observation_.state, state, input, mode);
      observation_.state = state;
      observation_.mode = mode;
    }
    observation_.time += kSolvePeriod;
    return status;
  }

  /** The policy of the last solve. */
  const PrimalSolution& policy() const { return mrt_->getPolicy(); }

 private:
  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> calculator_;
  std::shared_ptr<ProceduralMpcMotionManager> motionManager_;
  std::shared_ptr<MpcParameterUpdaterModule> updater_;
  std::unique_ptr<MPC_MRT_Interface> mrt_;
  SystemObservation observation_;
};

/** `value` at 17 significant digits, so that two dumps differ exactly when the values do. */
std::string exact(scalar_t value);
std::string exact(const matrix_t& value);

/** The largest relative difference of `a` and `b` (relative to max(1, |a|)); infinity when their sizes differ. */
scalar_t relativeDifference(const vector_t& a, const vector_t& b) {
  if (a.size() != b.size()) return std::numeric_limits<scalar_t>::infinity();
  scalar_t largest = 0.0;
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    largest = std::max(largest, std::abs(a(i) - b(i)) / std::max(1.0, std::abs(a(i))));
  }
  return largest;
}

/**
 * The largest relative difference of the state and input trajectories of the policies `a` and `b`; infinity when their
 * time trajectories differ.
 */
scalar_t policyDifference(const PrimalSolution& a, const PrimalSolution& b) {
  if (a.timeTrajectory_ != b.timeTrajectory_ || a.stateTrajectory_.size() != b.stateTrajectory_.size() ||
      a.inputTrajectory_.size() != b.inputTrajectory_.size()) {
    return std::numeric_limits<scalar_t>::infinity();
  }
  scalar_t largest = 0.0;
  for (size_t k = 0; k < a.stateTrajectory_.size(); ++k) {
    largest = std::max(largest, relativeDifference(a.stateTrajectory_[k], b.stateTrajectory_[k]));
  }
  for (size_t k = 0; k < a.inputTrajectory_.size(); ++k) {
    largest = std::max(largest, relativeDifference(a.inputTrajectory_[k], b.inputTrajectory_[k]));
  }
  return largest;
}

/**
 * Runs the standing and then the walking solves on `a` and `b` together and compares every policy; records the largest
 * difference of all of them as the test property largest_relative_difference (0: bit for bit).
 */
void expectTheSameSolves(Stack& a, Stack& b) {
  scalar_t largest = 0.0;
  for (size_t k = 0; k < kStandingSolves + kWalkingSolves; ++k) {
    SCOPED_TRACE(absl::StrCat("solve ", k, " at t = ", a.time(), k < kStandingSolves ? " (standing)" : " (walking)"));
    if (k == kStandingSolves) {
      a.command(kWalkingCommand);
      b.command(kWalkingCommand);
    }
    ASSERT_TRUE(a.solve().ok());
    ASSERT_TRUE(b.solve().ok());
    const scalar_t difference = policyDifference(a.policy(), b.policy());
    EXPECT_LE(difference, kRelativeTolerance) << "the policies differ by up to " << difference << " relative";
    largest = std::max(largest, difference);
  }
  ::testing::Test::RecordProperty("largest_relative_difference", exact(largest));
}

/** L4 on `config`: the shipped file plus a live update of E solves as a fresh start from E. */
void expectALiveUpdateIsAFreshStart(const CentroidalMpcConfig& config) {
  CentroidalMpcConfig edited = config;
  const std::set<std::string> perturbed = perturb(edited.task, appliedLeafPaths(), Step::kScale);
  ASSERT_GT(perturbed.size(), 50U) << "the perturbation reached almost no field: " << absl::StrJoin(perturbed, ", ");
  Stack live(config, Stack::Updater::kRegistered);
  ASSERT_TRUE(live.ok());
  live.applyNow(edited.task);
  Stack fresh(edited, Stack::Updater::kRegistered);
  ASSERT_TRUE(fresh.ok());
  expectTheSameSolves(live, fresh);
}

TEST(LiveUpdateEqualsFreshStart, OnTheShippedAtlas) {
  expectALiveUpdateIsAFreshStart(shippedAtlas());
}

TEST(LiveUpdateEqualsFreshStart, OnTheContactImplicitAtlas) {
  expectALiveUpdateIsAFreshStart(contactImplicitAtlas(shippedAtlas()));
}

TEST(LiveUpdateEqualsFreshStart, OnTheAtlasWithQuadraticCostsASoftZeroVelocityAndHeuristics) {
  expectALiveUpdateIsAFreshStart(quadraticCostsWithHeuristicsAtlas(shippedAtlas()));
}

TEST(LiveUpdateEqualsFreshStart, AnUpdaterFedNothingChangesNoSolve) {
  // L5: the updater registered, fed nothing and watching no changed file, against no updater at all.
  const CentroidalMpcConfig config = shippedAtlas();
  Stack withUpdater(config, Stack::Updater::kRegistered);
  Stack withoutUpdater(config, Stack::Updater::kNone);
  ASSERT_TRUE(withUpdater.ok() && withoutUpdater.ok());
  for (size_t k = 0; k < kStandingSolves + kWalkingSolves; ++k) {
    SCOPED_TRACE(absl::StrCat("solve ", k));
    if (k == kStandingSolves) {
      withUpdater.command(kWalkingCommand);
      withoutUpdater.command(kWalkingCommand);
    }
    ASSERT_TRUE(withUpdater.solve().ok());
    ASSERT_TRUE(withoutUpdater.solve().ok());
    const PrimalSolution& a = withUpdater.policy();
    const PrimalSolution& b = withoutUpdater.policy();
    ASSERT_EQ(a.stateTrajectory_.size(), b.stateTrajectory_.size());
    for (size_t i = 0; i < a.stateTrajectory_.size(); ++i) {
      ASSERT_TRUE(a.stateTrajectory_[i] == b.stateTrajectory_[i]) << "node " << i << ": not bit for bit";
      ASSERT_TRUE(a.inputTrajectory_[i] == b.inputTrajectory_[i]) << "node " << i << ": not bit for bit";
    }
  }
}

// --------------------------------------------------------------------------------------------------------------------
// L4t: terrain_height
// --------------------------------------------------------------------------------------------------------------------

/** The one ground every contact-implicit term of every worker is on, or NaN when they disagree or there is none. */
scalar_t contactImplicitTermsTerrainHeight(SqpSolver& solver, const std::vector<std::string>& contactNames) {
  std::vector<scalar_t> heights;
  for (OptimalControlProblem& ocp : solver.getOcpDefinitions()) {
    for (const std::string& footName : contactNames) {
      heights.push_back(
          ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kContactComplementarity))
              .get<ContactComplementarityConstraint>()
              .getTerrainHeight());
      heights.push_back(ocp.stateSoftConstraintPtr->get<StateSoftConstraint>(contact_term::name(footName, contact_term::kGroundPenetration))
                            .get<GroundPenetrationConstraint>()
                            .getTerrainHeight());
    }
  }
  if (heights.empty()) return std::numeric_limits<scalar_t>::quiet_NaN();
  for (const scalar_t height : heights) {
    if (height != heights.front()) return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  return heights.front();
}

/** The height reference of the stance feet of `stack` at its current time (the ground), or NaN with no foot in stance. */
scalar_t stanceFootHeightReference(Stack& stack) {
  SwitchedModelReferenceManager& referenceManager = stack.referenceManager();
  for (size_t foot = 0; foot < stack.interface().modelSettings().contactNames.size(); ++foot) {
    if (referenceManager.isInContact(stack.time(), foot)) {
      return referenceManager.getSwingTrajectoryPlanner()->getZpositionConstraint(foot, stack.time());
    }
  }
  return std::numeric_limits<scalar_t>::quiet_NaN();
}

TEST(LiveTerrainHeight, TakesEffectAtTheNextSolveMovesTheGroundOnceAndTheContactImplicitTermsFollow) {
  // Delivered as in production: the updater runs after the reference manager's pre-solve hook, so a ground enqueued
  // before solve 1 reaches the reference manager in solve 1 and the references of solve 2 stand on it.
  const CentroidalMpcConfig config = contactImplicitAtlas(shippedAtlas());
  Stack stack(config, Stack::Updater::kRegistered);
  ASSERT_TRUE(stack.ok());
  const std::vector<std::string>& contactNames = stack.interface().modelSettings().contactNames;
  const scalar_t launched = stack.interface().modelSettings().terrainHeight;
  constexpr scalar_t kStep = 0.02;
  ASSERT_TRUE(stack.solve().ok());
  EXPECT_EQ(stack.referenceManager().getAppliedTerrainHeight(), launched);
  EXPECT_EQ(contactImplicitTermsTerrainHeight(stack.solver(), contactNames), launched);

  mpc_config::TaskFile task = config.task;
  task.terrain_height = launched + kStep;
  stack.enqueue(task);
  ASSERT_TRUE(stack.solve().ok());
  // Solve 1 ran on the references of the old ground: the update arrived after they were built.
  EXPECT_EQ(stack.referenceManager().getTerrainHeight(), launched + kStep);
  EXPECT_EQ(stack.referenceManager().getAppliedTerrainHeight(), launched);
  EXPECT_EQ(contactImplicitTermsTerrainHeight(stack.solver(), contactNames), launched) << "the terms moved before the references";

  // Solve 2 stands on the new ground, and the terms with it; solve 3 does not move it again.
  ASSERT_TRUE(stack.solve().ok());
  EXPECT_EQ(stack.referenceManager().getAppliedTerrainHeight(), launched + kStep);
  EXPECT_EQ(contactImplicitTermsTerrainHeight(stack.solver(), contactNames), launched + kStep);
  const scalar_t afterTheChange = stanceFootHeightReference(stack);
  EXPECT_NEAR(afterTheChange, launched + kStep, 1.0e-12) << "the stance feet do not stand on the new ground";
  ASSERT_TRUE(stack.solve().ok());
  EXPECT_EQ(stack.referenceManager().getAppliedTerrainHeight(), launched + kStep);
  EXPECT_EQ(stanceFootHeightReference(stack), afterTheChange) << "the ground moved a second time";
}

// --------------------------------------------------------------------------------------------------------------------
// Liveness: every perturbed leaf changes what it writes
// --------------------------------------------------------------------------------------------------------------------

std::string exact(scalar_t value) {
  return absl::StrFormat("%.17g", value);
}

std::string exact(const matrix_t& value) {
  std::string out = absl::StrCat(value.rows(), "x", value.cols(), ":");
  for (Eigen::Index i = 0; i < value.size(); ++i) absl::StrAppend(&out, " ", exact(value.data()[i]));
  return out;
}

using Dump = std::map<std::string, std::string>;

/** The parameters of every penalty of `softConstraint`. */
template <typename SoftConstraint>
std::string penaltyParameters(SoftConstraint& softConstraint) {
  std::string out;
  for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softConstraint.getPenalty().getPenaltyPtrArray()) {
    vector_t parameters;
    penalty->getParameters(parameters);
    absl::StrAppend(&out, exact(parameters), ";");
  }
  return out;
}

/** The twist constraint `twist` of a zero_velocity term. */
std::string twistParameters(EndEffectorKinematicsTwistConstraint& twist) {
  return absl::StrCat(exact(twist.getConfig().Ax), " | ", exact(twist.getConfig().Av), " | ", twist.getNumConstraints(/*time=*/0.0), " | ",
                      twist.getConstrainYawRateAboutNormal());
}

/** Appends the parameters of every running term the parameter updater writes, by collection and term, to `dump`. */
void dumpTerms(OptimalControlProblem& ocp, Dump& dump) {
  for (const std::pair<const std::string, size_t>& term : ocp.costPtr->getTermNameMap()) {
    StateInputCost& cost = ocp.costPtr->get<StateInputCost>(term.first);
    const std::string key = absl::StrCat("cost/", term.first);
    if (QuadraticStateInputCost* absl_nullable quadratic = dynamic_cast<QuadraticStateInputCost*>(&cost)) {
      matrix_t Q;
      matrix_t R;
      matrix_t P;
      quadratic->getGains(Q, R, P);
      dump[key] = absl::StrCat(exact(Q), " | ", exact(R), " | ", exact(P));
    } else if (EndEffectorKinematicsQuadraticCost* absl_nullable link = dynamic_cast<EndEffectorKinematicsQuadraticCost*>(&cost)) {
      vector12_t weights;
      link->getWeights(weights);
      dump[key] = exact(weights);
    } else if (ExternalTorqueQuadraticCostAD* absl_nullable torque = dynamic_cast<ExternalTorqueQuadraticCostAD*>(&cost)) {
      vector_t weights;
      torque->getWeights(weights);
      dump[key] = exact(weights);
    } else if (CentroidalMpcEndEffectorFootCost* absl_nullable foot = dynamic_cast<CentroidalMpcEndEffectorFootCost*>(&cost)) {
      vector12_t weights;
      foot->getWeights(weights);
      dump[key] = absl::StrCat(exact(weights), " | ", foot->getActiveInStance());
    } else if (ICPCost* absl_nullable icp = dynamic_cast<ICPCost*>(&cost)) {
      vector2_t weights;
      icp->getWeights(weights);
      dump[key] = exact(weights);
    } else if (BasisScalingNonNegativityConstraint* absl_nullable barrier = dynamic_cast<BasisScalingNonNegativityConstraint*>(&cost)) {
      dump[key] = absl::StrCat(exact(barrier->getBarrierConfig().mu), " ", exact(barrier->getBarrierConfig().delta));
    }
  }
  for (const std::pair<const std::string, size_t>& term : ocp.stateCostPtr->getTermNameMap()) {
    if (ComAndAcomTrackingCost* absl_nullable comAndAcom =
            dynamic_cast<ComAndAcomTrackingCost*>(&ocp.stateCostPtr->get<StateCost>(term.first))) {
      dump[absl::StrCat("stateCost/", term.first)] = absl::StrCat(exact(comAndAcom->getQCom()), " | ", exact(comAndAcom->getQAcom()));
    }
  }
  for (const std::pair<const std::string, size_t>& term : ocp.finalCostPtr->getTermNameMap()) {
    StateCost& cost = ocp.finalCostPtr->get<StateCost>(term.first);
    const std::string key = absl::StrCat("finalCost/", term.first);
    if (QuadraticStateCost* absl_nullable quadratic = dynamic_cast<QuadraticStateCost*>(&cost)) {
      matrix_t Q;
      quadratic->getGains(Q);
      dump[key] = exact(Q);
    } else if (DcmTerminalCost* absl_nullable dcm = dynamic_cast<DcmTerminalCost*>(&cost)) {
      const DcmTerminalCost::Config& config = dcm->getConfig();
      dump[key] = absl::StrCat(exact(config.comHeight.value_or(-1.0)), " ", exact(config.gravity), " ", exact(config.weights), " ",
                               exact(config.velocityOffsetFactor), " ", exact(config.supportBlendTime));
    } else if (ComAndAcomTrackingCost* absl_nullable comAndAcom = dynamic_cast<ComAndAcomTrackingCost*>(&cost)) {
      dump[key] = absl::StrCat(exact(comAndAcom->getQCom()), " | ", exact(comAndAcom->getQAcom()));
    }
  }
  for (const std::pair<const std::string, size_t>& term : ocp.softConstraintPtr->getTermNameMap()) {
    StateInputSoftConstraint* absl_nullable softCon =
        dynamic_cast<StateInputSoftConstraint*>(&ocp.softConstraintPtr->get<StateInputCost>(term.first));
    if (softCon == nullptr) continue;
    std::string value = penaltyParameters(*softCon);
    StateInputConstraint& constraint = softCon->get<StateInputConstraint>();
    if (ContactComplementarityConstraint* absl_nullable complementarity = dynamic_cast<ContactComplementarityConstraint*>(&constraint)) {
      absl::StrAppend(&value, " | ", exact(complementarity->getHeightReference()), " ", exact(complementarity->getGapSmoothing()), " ",
                      exact(complementarity->getTerrainHeight()));
    } else if (ForceWeightedSlipConstraint* absl_nullable slip = dynamic_cast<ForceWeightedSlipConstraint*>(&constraint)) {
      absl::StrAppend(&value, " | ", exact(slip->getInverseTwistReference()));
    } else if (ZeroVelocityConstraintCppAd* absl_nullable zeroVelocity = dynamic_cast<ZeroVelocityConstraintCppAd*>(&constraint)) {
      absl::StrAppend(&value, " | ", twistParameters(zeroVelocity->getTwistConstraint()));
    }
    dump[absl::StrCat("softConstraint/", term.first)] = value;
  }
  for (const std::pair<const std::string, size_t>& term : ocp.stateSoftConstraintPtr->getTermNameMap()) {
    StateCost& cost = ocp.stateSoftConstraintPtr->get<StateCost>(term.first);
    const std::string key = absl::StrCat("stateSoftConstraint/", term.first);
    if (JointLimitsSoftConstraint* absl_nullable jointLimits = dynamic_cast<JointLimitsSoftConstraint*>(&cost)) {
      scalar_t mu = 0.0;
      scalar_t delta = 0.0;
      jointLimits->getGains(mu, delta);
      dump[key] = absl::StrCat(exact(mu), " ", exact(delta));
    } else if (StateSoftConstraint* absl_nullable softCon = dynamic_cast<StateSoftConstraint*>(&cost)) {
      std::string value = penaltyParameters(*softCon);
      if (GroundPenetrationConstraint* absl_nullable penetration =
              dynamic_cast<GroundPenetrationConstraint*>(&softCon->get<StateConstraint>())) {
        absl::StrAppend(&value, " | ", exact(penetration->getTerrainHeight()));
      }
      dump[key] = value;
    }
  }
  for (const std::pair<const std::string, size_t>& term : ocp.equalityConstraintPtr->getTermNameMap()) {
    if (ZeroVelocityConstraintCppAd* absl_nullable zeroVelocity =
            dynamic_cast<ZeroVelocityConstraintCppAd*>(&ocp.equalityConstraintPtr->get<StateInputConstraint>(term.first))) {
      dump[absl::StrCat("equalityConstraint/", term.first)] = twistParameters(zeroVelocity->getTwistConstraint());
    }
  }
  if (const HumanoidPreComputation* absl_nullable preComputation =
          dynamic_cast<const HumanoidPreComputation*>(ocp.preComputationPtr.get())) {
    dump["preComputation/normalVelocityPositionErrorGain"] = exact(preComputation->getNormalVelocityPositionErrorGain());
  }
}

/** Everything a reload of `stack` writes: the first worker's terms, the solver's settings, the references. */
Dump dumpOf(Stack& stack) {
  Dump dump;
  dumpTerms(stack.solver().getOcpDefinitions().front(), dump);
  const sqp::Settings& settings = stack.solver().getSettings();
  dump["sqp"] = absl::StrCat(settings.sqpIteration, " ", exact(settings.deltaTol), " ", exact(settings.g_max), " ", exact(settings.g_min));
  const SwingTrajectoryPlanner::Config& swing = stack.referenceManager().getSwingTrajectoryPlanner()->getConfig();
  dump["swingTrajectoryPlanner"] =
      absl::StrJoin({exact(swing.liftOffVelocity), exact(swing.touchDownVelocity), exact(swing.swingHeight), exact(swing.swingTimeScale),
                     exact(swing.touchDownHeightOffset), exact(swing.impactProximityFactorLiftOffVelocity),
                     exact(swing.impactProximityFactorTouchDownVelocity), exact(swing.impactProximityFactorMidPointValue),
                     exact(swing.swingPitchAngle), exact(swing.swingPitchRiseFraction), exact(swing.swingPitchFallFraction)},
                    " ");
  dump["referenceManager/terrainHeight"] = exact(stack.referenceManager().getTerrainHeight());
  if (const std::shared_ptr<LocomotionHeuristicLayer>& layer = stack.interface().getLocomotionHeuristicLayerPtr()) {
    dump["locomotionHeuristics"] = layer->summary();
  }
  return dump;
}

/** The keys whose values differ between `a` and `b`. */
std::vector<std::string> changedKeys(const Dump& a, const Dump& b) {
  std::vector<std::string> changed;
  for (const std::pair<const std::string, std::string>& entry : a) {
    const Dump::const_iterator other = b.find(entry.first);
    if (other == b.end() || other->second != entry.second) changed.push_back(entry.first);
  }
  return changed;
}

// The applied leaves that change nothing on any variant below, and why. Every other applied number must change what it
// writes on at least one of them.
constexpr std::array<absl::string_view, 18> kNotExercisedHere = {
    // The cones of the wrench parameterization, which basis-vector contact inputs do not build;
    // testContactConstraintScheduleGating reloads them on the factory's own terms.
    "contacts.contact_moment_xy_soft_constraint.delta",
    "contacts.contact_moment_xy_soft_constraint.mu",
    "contacts.contact_wrench_cone_soft_constraint.delta",
    "contacts.contact_wrench_cone_soft_constraint.mu",
    "contacts.friction_force_cone_soft_constraint.delta",
    "contacts.friction_force_cone_soft_constraint.mu",
    // The acceleration weights of the task-space costs, which the centroidal MPC does not weigh: the conversion refuses
    // a nonzero one by its field (taskSpaceFootCostFromConfig(), taskSpaceLinkCostsFromConfig()), and only the
    // whole-body MPC applies them.
    "task_space_costs[*].weights.ang_acceleration_x",
    "task_space_costs[*].weights.ang_acceleration_y",
    "task_space_costs[*].weights.ang_acceleration_z",
    "task_space_costs[*].weights.lin_acceleration_x",
    "task_space_costs[*].weights.lin_acceleration_y",
    "task_space_costs[*].weights.lin_acceleration_z",
    "task_space_foot_cost.weights.ang_acceleration_x",
    "task_space_foot_cost.weights.ang_acceleration_y",
    "task_space_foot_cost.weights.ang_acceleration_z",
    "task_space_foot_cost.weights.lin_acceleration_x",
    "task_space_foot_cost.weights.lin_acceleration_y",
    "task_space_foot_cost.weights.lin_acceleration_z",
};

TEST(LiveUpdateEqualsFreshStart, EveryAppliedNumberChangesWhatItWritesAndAReloadOfTheFileRestoresIt) {
  const std::set<std::string> leaves = appliedLeafPaths();
  std::set<std::string> live;
  std::set<std::string> visited;
  const CentroidalMpcConfig shipped = shippedAtlas();
  const std::vector<CentroidalMpcConfig> variants = {shipped, contactImplicitAtlas(shipped), quadraticCostsWithHeuristicsAtlas(shipped),
                                                     otherHeuristicsWithoutAcomAtlas(shipped)};
  for (size_t v = 0; v < variants.size(); ++v) {
    SCOPED_TRACE(absl::StrCat("variant ", v));
    Stack stack(variants[v], Stack::Updater::kNone);
    ASSERT_TRUE(stack.ok());
    stack.applyNow(variants[v].task);
    const Dump running = dumpOf(stack);
    for (const std::string& leaf : leaves) {
      mpc_config::TaskFile task = variants[v].task;
      if (perturb(task, {leaf}, Step::kScaleOrSet).empty()) continue;
      visited.insert(leaf);
      stack.applyNow(task);
      if (!changedKeys(running, dumpOf(stack)).empty()) live.insert(leaf);
      stack.applyNow(variants[v].task);
      EXPECT_THAT(changedKeys(running, dumpOf(stack)), testing::IsEmpty())
          << "a reload of the file after " << leaf << " did not restore it";
    }
  }
  std::vector<std::string> notLive;
  for (const std::string& leaf : visited) {
    const bool listed = std::find(kNotExercisedHere.begin(), kNotExercisedHere.end(), leaf) != kNotExercisedHere.end();
    if (live.contains(leaf) == listed) notLive.push_back(leaf);
  }
  EXPECT_THAT(notLive, testing::IsEmpty()) << "a leaf a reload applies changes nothing it writes on any variant, or a leaf of "
                                              "kNotExercisedHere does: "
                                           << absl::StrJoin(notLive, ", ");
  EXPECT_GT(visited.size(), 100U);
}

}  // namespace
}  // namespace ocs2::humanoid
