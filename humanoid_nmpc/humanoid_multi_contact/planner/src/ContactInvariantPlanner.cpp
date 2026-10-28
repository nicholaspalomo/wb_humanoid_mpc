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

#include "humanoid_multi_contact/planner/ContactInvariantPlanner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/manifold/ProductStateManifold.h"
#include "ocs2_core/manifold/StateManifoldSegment.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/multi_contact/MultiContactPlanSource.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/planner/contact_invariant/LibraryName.h"
#include "humanoid_multi_contact/planner/contact_invariant/PlanSeeding.h"
#include "humanoid_multi_contact/planner/contact_invariant/PlanTrajectory.h"
#include "humanoid_multi_contact/planner/contact_invariant/TerrainClearanceConstraint.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The default stages of the continuation (design section 4.2). Forces are in
// units of the robot's weight, distances in meters, velocities in m/s.
constexpr double kSlipWeight = 1.0;
constexpr double kTiltWeight = 1.0;
constexpr double kGateStiffness = 1.0e-2;  // k1
constexpr double kGateMomentArm = 0.1;     // [m]
constexpr double kActivationRateWeight = 0.1;
constexpr double kMomentumRateWeight = 1.0e-2;
// The robot always claims nearly one full contact: no flight phase.
constexpr double kMinimumSupport = 0.9;
// [s] The shortest stance of the thresholded sequence.
constexpr double kMinimumStance = 0.25;
// [s] The stances change at multiples of this: of the tracking MPC's period
// (0.02 s on the Atlas), the plan's knots (0.025 s) and solves 1 and 2's nodes
// (0.05 s).
constexpr double kChangeQuantum = 0.1;
constexpr double kFreeForceGate = 1.0e-3;     // k0 of solve 1
constexpr double kFreeForceInvariance = 1.0;  // the CI weight of solve 1
constexpr int kFreeForceIterations = 40;
// Solve 2 raises k0 and the CI weight by these factors kContactSteps times.
constexpr int kContactSteps = 3;
constexpr double kGateGrowth = 10.0;
constexpr double kInvarianceGrowth = 10.0;
constexpr int kContactIterations = 60;
// The penalties by default: the squared hinge of c and 1 - c, the barriers of
// the cones (in weights) and of the pads' clearance (in meters).
constexpr double kActivationPenaltyMu = 1.0e3;
constexpr double kConeBarrierMu = 0.1;
constexpr double kConeBarrierDelta = 1.0e-2;
constexpr double kPadBarrierMu = 1.0e-2;
constexpr double kPadBarrierDelta = 1.0e-3;
// The certifier's friction pyramids are inscribed in the cone: eight edges at
// mu cos(pi / 8).
constexpr double kInscribedFriction = 0.92388;  // cos(pi / 8)
// The relaxed barrier of the MPC's wrench cones in solves 1 and 2 (planner
// A's own, so that the tracking MPC's tuning in the task file does not change
// the plans; delta in newtons).
constexpr double kWrenchConeBarrierMu = 0.2;
constexpr double kWrenchConeBarrierDelta = 5.0;
// The relaxed barrier of the collision spheres in solves 1 and 2.
constexpr double kSphereBarrierMu = 1.0e-3;
constexpr double kSphereBarrierDelta = 2.0e-3;
// The names of the terms planner A adds to the MPC's problem.
constexpr char kTerrainClearanceTerm[] = "terrainClearance";
// The hard and soft constraints and the costs of solves 1 and 2, of the task
// file's names (MpcFormulationConfig.h): every allowed candidate is in
// contact, so the stance and swing terms are not built.
constexpr char kZeroWrench[] = "zero_wrench";
constexpr char kJointLimits[] = "joint_limits";
constexpr char kContactWrenchCone[] = "contact_wrench_cone";
constexpr char kWeightShareReference[] = "weight_share";

/** The default stages: solve 1 with the forces almost free, then the force
 * gate and the contact-invariant weight raised in steps (solve 2 of the
 * design, split so that each step starts near its solution). */
std::vector<ContinuationStage> defaultStages() {
  ContinuationStage free;
  free.name = "solve 1: free forces";
  free.weights.contactInvariant =
      ContactInvariantWeights{.weight = kFreeForceInvariance,
                              .slipWeight = kSlipWeight,
                              .tiltWeight = kTiltWeight};
  free.weights.forceGate = ForceGateWeights{.k0 = kFreeForceGate,
                                            .k1 = kGateStiffness,
                                            .forceScale = 1.0,
                                            .momentArm = kGateMomentArm};
  free.weights.activationRateWeight = kActivationRateWeight;
  free.weights.linearMomentumRateWeight = kMomentumRateWeight;
  free.weights.angularMomentumRateWeight = kMomentumRateWeight;
  free.weights.minimumSupport = kMinimumSupport;
  free.sqpIterations = kFreeForceIterations;
  std::vector<ContinuationStage> stages = {free};
  double invariance = kFreeForceInvariance;
  double gate = kFreeForceGate;
  for (int step = 1; step <= kContactSteps; ++step) {
    invariance *= kInvarianceGrowth;
    gate *= kGateGrowth;
    ContinuationStage contact = free;
    contact.name = absl::StrFormat("solve 2.%d: contact", step);
    contact.weights.contactInvariant.weight = invariance;
    contact.weights.forceGate.k0 = gate;
    contact.sqpIterations = kContactIterations;
    stages.push_back(contact);
  }
  return stages;
}

/** The scene's least friction coefficient. */
double leastFriction(const TerrainModel& terrain) {
  double friction = terrain.primitive(0).friction;
  for (int p = 1; p < terrain.numPrimitives(); ++p) {
    friction = std::min(friction, terrain.primitive(p).friction);
  }
  return friction;
}

/** What a run of planner A shares. */
struct PlannerRobot {
  CentroidalMpcConfig config;
  // The MPC's models, without a problem.
  std::unique_ptr<CentroidalMpcInterface> models;
  std::unique_ptr<RobotStanceModel> stanceModel;
  // The allowed candidates in the contact set's order, and their MPC
  // contacts.
  std::vector<int> allowed;
  std::vector<size_t> mpcContacts;
  ContactMask allowedMask = 0;
  ContactMask initialMask = 0;
  ContactMask goalMask = 0;
  double mass = 0.0;
};

absl::StatusOr<PlannerRobot> robotOf(const LoadedScenario& loaded) {
  PlannerRobot robot;
  ASSIGN_OR_RETURN(robot.config, plannerMpcConfigOf(loaded));
  ASSIGN_OR_RETURN(robot.models, CentroidalMpcInterface::CreateControllerModels(
                                     robot.config, loaded.urdfPath));
  ASSIGN_OR_RETURN(robot.stanceModel, RobotStanceModel::Create(
                                          loaded.urdfPath, loaded.contactSet));
  robot.mass = robot.models->getCentroidalModelInfo().robotMass;
  const std::vector<std::string>& contactNames =
      robot.models->modelSettings().contactNames;
  for (const std::string& name : loaded.scenario.allowed_contacts) {
    ASSIGN_OR_RETURN(const int index, loaded.contactSet.indexOf(name));
    robot.allowed.push_back(index);
  }
  std::sort(robot.allowed.begin(), robot.allowed.end());
  for (const int index : robot.allowed) {
    const std::string& name = loaded.contactSet.candidate(index).name;
    const std::vector<std::string>::const_iterator found =
        std::find(contactNames.begin(), contactNames.end(), name);
    if (found == contactNames.end()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "scenario '", loaded.scenario.name, "': the allowed contact '", name,
          "' is not a contact of the MPC of ", loaded.scenario.mpc_task_file));
    }
    robot.mpcContacts.push_back(
        static_cast<size_t>(found - contactNames.begin()));
    robot.allowedMask |= contactBit(index);
  }
  ASSIGN_OR_RETURN(robot.initialMask,
                   contactsOf(loaded.scenario.initial_pose, loaded.contactSet));
  std::vector<std::string> goalNames;
  for (const config::Scenario::GoalContact& contact :
       loaded.scenario.goal.contacts) {
    goalNames.push_back(contact.contact);
  }
  ASSIGN_OR_RETURN(robot.goalMask, loaded.contactSet.maskOf(goalNames));
  return robot;
}

/** The layout of the allowed candidates on the MPC's input. */
ContactInvariantLayout layoutOf(const PlannerRobot& robot,
                                const LoadedScenario& loaded) {
  const MpcRobotModelBase<scalar_t>& model = robot.models->getMpcRobotModel();
  ContactInvariantLayout layout;
  layout.physicalStateDim = static_cast<int>(model.getStateDim());
  layout.physicalInputDim = static_cast<int>(model.getInputDim());
  for (size_t k = 0; k < robot.allowed.size(); ++k) {
    const ContactCandidate& candidate =
        loaded.contactSet.candidate(robot.allowed[k]);
    layout.candidates.push_back(ActivationCandidate{
        .name = candidate.name,
        .forceOffset = static_cast<int>(
            model.getContactForceStartIndices(robot.mpcContacts[k])),
        .wrenchDim =
            candidate.kind == ContactKind::kSurface ? kWrenchDim : kForceDim});
  }
  return layout;
}

/** The goal configuration: the goal pose, or the initial one moved to the
 * goal region's center. */
absl::StatusOr<RobotConfiguration> goalConfigurationOf(
    const LoadedScenario& loaded) {
  const config::Scenario::Goal& goal = loaded.scenario.goal;
  if (goal.pose.has_value()) return configurationOf(*goal.pose);
  ASSIGN_OR_RETURN(RobotConfiguration configuration,
                   configurationOf(loaded.scenario.initial_pose));
  if (goal.base_region.has_value()) {
    configuration.basePosition = Eigen::Vector3d(goal.base_region->center);
  }
  return configuration;
}

/**
 * The guess of solves 1 and 2 on the MPC's layout: the base and the joints
 * interpolated from the initial to the goal configuration, the momentum of
 * that path (zero at the first node), the initial stance's weight on its
 * allowed contacts, and the joints' constant velocity; every allowed
 * candidate active.
 */
absl::StatusOr<MpcTrajectory> guessOf(const PlannerRobot& robot,
                                      const LoadedScenario& loaded,
                                      const std::vector<double>& times) {
  const MpcRobotModelBase<scalar_t>& model = robot.models->getMpcRobotModel();
  ASSIGN_OR_RETURN(const RobotConfiguration initial,
                   configurationOf(loaded.scenario.initial_pose));
  ASSIGN_OR_RETURN(const RobotConfiguration goal, goalConfigurationOf(loaded));
  const vector_t start = mpcStateOf(model, initial);
  const vector_t end = mpcStateOf(model, goal);
  const double duration = times.back() - times.front();
  const vector_t jointVelocity =
      (model.getJointAngles(end) - model.getJointAngles(start)) / duration;
  const Eigen::Vector3d baseVelocity =
      (model.getBasePosition(end) - model.getBasePosition(start)) / duration;
  const Eigen::Quaterniond startOrientation(
      Eigen::Vector4d(model.getBaseOrientation(start)));
  const Eigen::Quaterniond endOrientation(
      Eigen::Vector4d(model.getBaseOrientation(end)));
  // The weight shared by the initial stance's allowed contacts.
  std::vector<size_t> supports;
  for (size_t k = 0; k < robot.allowed.size(); ++k) {
    if (isContactActive(robot.initialMask, robot.allowed[k])) {
      supports.push_back(robot.mpcContacts[k]);
    }
  }
  const double share =
      supports.empty()
          ? 0.0
          : robot.mass * kGravitationalAcceleration / supports.size();
  MpcTrajectory guess;
  guess.times = times;
  for (size_t k = 0; k < times.size(); ++k) {
    const double alpha = (times[k] - times.front()) / duration;
    vector_t state = start;
    model.setBasePosition(state, (1.0 - alpha) * model.getBasePosition(start) +
                                     alpha * model.getBasePosition(end));
    model.setBaseOrientation(
        state, startOrientation.slerp(alpha, endOrientation).coeffs());
    model.setJointAngles(state, (1.0 - alpha) * model.getJointAngles(start) +
                                    alpha * model.getJointAngles(end));
    if (k > 0) model.setBaseComLinearVelocity(state, baseVelocity);
    vector_t input =
        vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
    for (const size_t contact : supports) {
      model.setContactForce(input, Eigen::Vector3d(0.0, 0.0, share), contact);
    }
    model.setJointVelocities(state, input, jointVelocity);
    guess.states.push_back(std::move(state));
    guess.inputs.push_back(std::move(input));
    guess.masks.push_back(robot.allowedMask);
  }
  return guess;
}

/** The task file of solves 1 and 2: every allowed candidate in contact, its
 * wrench free in its cone, the posture weights scaled. */
CentroidalMpcConfig continuationConfigOf(
    CentroidalMpcConfig config, const ContactInvariantPlannerOptions& options) {
  config.task.hard_constraints = {kZeroWrench};
  config.task.soft_constraints = {kJointLimits, kContactWrenchCone};
  config.task.contacts.contact_wrench_cone_soft_constraint.mu =
      kWrenchConeBarrierMu;
  config.task.contacts.contact_wrench_cone_soft_constraint.delta =
      kWrenchConeBarrierDelta;
  config.task.state_weights.scaling *= options.guessTrackingScale;
  // The forces are the force gate's: no reference pulls them to the guess's.
  config.task.multi_contact_plan.force_reference = kWeightShareReference;
  config.task.interface.verbose = false;
  return config;
}

/** The SQP settings of an offline solve at `dt`. */
sqp::Settings offlineSettings(sqp::Settings settings,
                              const RefinementOptions& options,
                              double dt) {
  settings.dt = dt;
  settings.nThreads = static_cast<size_t>(options.numThreads);
  settings.useFeedbackPolicy = false;
  settings.enableLogging = false;
  settings.printSolverStatus = false;
  settings.printSolverStatistics = false;
  settings.printLinesearch = false;
  return settings;
}

/** The manifold of the MPC's state with `numCandidates` contact variables. */
absl::StatusOr<std::shared_ptr<const StateManifold>> augmentedManifoldOf(
    const MpcRobotModelBase<scalar_t>& model, int numCandidates) {
  const size_t orientation = model.getBaseOrientationStartIndex();
  const size_t tail = model.getStateDim() - orientation -
                      static_cast<size_t>(kQuaternionCoefficients) +
                      static_cast<size_t>(numCandidates);
  ASSIGN_OR_RETURN(std::shared_ptr<const ProductStateManifold> manifold,
                   ProductStateManifold::create(
                       {StateManifoldSegment::euclidean(orientation),
                        StateManifoldSegment::unitQuaternion(),
                        StateManifoldSegment::euclidean(tail)}));
  return std::shared_ptr<const StateManifold>(std::move(manifold));
}

/** The features of the allowed candidates on the MPC of `interface`. */
absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> featuresOf(
    CentroidalMpcInterface& interface,
    const PlannerRobot& robot,
    const LoadedScenario& loaded,
    const ContactInvariantPlannerOptions& options) {
  std::vector<RobotFeatureCandidate> candidates;
  std::vector<std::string> names;
  for (size_t k = 0; k < robot.allowed.size(); ++k) {
    const ContactCandidate& candidate =
        loaded.contactSet.candidate(robot.allowed[k]);
    candidates.push_back(
        RobotFeatureCandidate{.frameName = candidate.name,
                              .mpcContactIndex = robot.mpcContacts[k],
                              .kind = candidate.kind,
                              .padRotation = candidate.padRotation,
                              .geomRadius = candidate.geomRadius});
    names.push_back(candidate.name);
  }
  const RobotFeatureSettings settings{.friction = options.frictionShare *
                                                  kInscribedFriction *
                                                  leastFriction(loaded.terrain),
                                      .coneSmoothing = options.coneSmoothing,
                                      .maxPointTilt = options.maxPointTilt};
  const ModelSettings& modelSettings = interface.modelSettings();
  const FeatureLibrarySettings library{
      .name = hashedLibraryName(
          "cio_contact_features",
          absl::StrCat("terrain ",
                       terrainSceneHash(loaded.scenario.terrain_scene),
                       " contact set ", contactSetHash(loaded.contactSetConfig),
                       " candidates ", absl::StrJoin(names, " "), " friction ",
                       settings.friction, " smoothing ", settings.coneSmoothing,
                       " tilt ", settings.maxPointTilt)),
      .folder = modelSettings.modelFolderCppAd,
      .recompile = modelSettings.recompileLibrariesCppAd,
      .verbose = modelSettings.verboseCppAd};
  return robotContactFeatures(interface.getPinocchioInterface(),
                              interface.getCentroidalModelInfo(), candidates,
                              loaded.terrain, settings, library);
}

/** The physical problem of solves 1 and 2: the MPC's, without the point
 * candidates' cones (the blended cones replace them), with the collision
 * spheres' clearance. */
absl::StatusOr<OptimalControlProblem> physicalProblemOf(
    CentroidalMpcInterface& interface,
    const PlannerRobot& robot,
    const LoadedScenario& loaded) {
  OptimalControlProblem physical(interface.getOptimalControlProblem());
  for (const std::string& name : interface.modelSettings().contactNames) {
    physical.softConstraintPtr->erase(
        contact_term::name(name, contact_term::kPointFrictionCone));
  }
  ASSIGN_OR_RETURN(
      CollisionModel collision,
      CollisionModel::Create(*loaded.collisionModelConfig, *robot.stanceModel));
  ASSIGN_OR_RETURN(std::vector<ClearanceSphere> spheres,
                   clearanceSpheresOf(collision, *robot.stanceModel,
                                      interface.getPinocchioInterface()));
  const ModelSettings& settings = interface.modelSettings();
  const size_t numSpheres = spheres.size();
  physical.stateSoftConstraintPtr->add(
      kTerrainClearanceTerm,
      std::make_unique<StateSoftConstraint>(
          std::make_unique<TerrainClearanceConstraint>(
              interface.getPinocchioInterface(),
              interface.getCentroidalModelInfo(), std::move(spheres),
              loaded.terrain, /*margin=*/0.0,
              hashedLibraryName(
                  "cio_terrain_clearance",
                  absl::StrCat("terrain ",
                               terrainSceneHash(loaded.scenario.terrain_scene),
                               " spheres ", numSpheres)),
              settings.modelFolderCppAd, settings.recompileLibrariesCppAd,
              settings.verboseCppAd),
          std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(
              kSphereBarrierMu, kSphereBarrierDelta))));
  return physical;
}

/** The augmented guess: `guess` with c = the initial stance at the first
 * node and `activation` after it, and u_c = 0. */
GuessTrajectory augmentedGuessOf(const MpcTrajectory& guess,
                                 const PlannerRobot& robot,
                                 double activation) {
  const int k = static_cast<int>(robot.allowed.size());
  GuessTrajectory augmented;
  augmented.times = guess.times;
  for (size_t n = 0; n < guess.times.size(); ++n) {
    vector_t state(guess.states[n].size() + k);
    state.head(guess.states[n].size()) = guess.states[n];
    for (int i = 0; i < k; ++i) {
      state(guess.states[n].size() + i) =
          n == 0 ? (isContactActive(robot.initialMask, robot.allowed[i]) ? 1.0
                                                                         : 0.0)
                 : activation;
    }
    vector_t input = vector_t::Zero(guess.inputs[n].size() + k);
    input.head(guess.inputs[n].size()) = guess.inputs[n];
    augmented.states.push_back(std::move(state));
    augmented.inputs.push_back(std::move(input));
  }
  return augmented;
}

/** Solves 1 and 2: the contact variables and the motion they found. */
absl::StatusOr<std::vector<StageSolution>> continuationOf(
    const PlannerRobot& robot,
    const LoadedScenario& loaded,
    const MpcTrajectory& guess,
    const ContactInvariantPlannerOptions& options) {
  const MpcRobotModelBase<scalar_t>& model = robot.models->getMpcRobotModel();
  const std::vector<std::string>& contactNames =
      robot.models->modelSettings().contactNames;
  ASSIGN_OR_RETURN(
      config::MultiContactPlan seed,
      planOfTrajectory(guess,
                       PlanConversion{.scenario = &loaded,
                                      .model = &model,
                                      .contactNames = contactNames,
                                      .mass = robot.mass,
                                      .stanceModel = robot.stanceModel.get(),
                                      .planner = "contact_invariant_guess"}));
  ASSIGN_OR_RETURN(
      std::unique_ptr<CentroidalMpcInterface> interface,
      CentroidalMpcInterface::Create(
          continuationConfigOf(robot.config, options), loaded.urdfPath,
          MultiContactPlanSource{.scenario = loaded, .plan = std::move(seed)}));
  ASSIGN_OR_RETURN(const OptimalControlProblem physical,
                   physicalProblemOf(*interface, robot, loaded));
  ASSIGN_OR_RETURN(std::unique_ptr<ContactFeatureFunction> features,
                   featuresOf(*interface, robot, loaded, options));
  const ContactInvariantLayout layout = layoutOf(robot, loaded);
  ASSIGN_OR_RETURN(std::shared_ptr<const StateManifold> manifold,
                   augmentedManifoldOf(model, layout.numCandidates()));
  const ContinuationProblem problem{
      .physical = &physical,
      .physicalReferenceManager = interface->getReferenceManagerPtr(),
      .features = features.get(),
      .layout = layout,
      .augmentedManifold = std::move(manifold),
      .barriers = options.barriers,
      .settings = offlineSettings(interface->sqpSettings(), options.refinement,
                                  options.continuationDt),
      .initTime = guess.times.front(),
      .finalTime = guess.times.back(),
      .initMode = interface->getPlannedMotionReferenceManagerPtr()
                      ->plannedMotion()
                      .modeSequence()
                      .front()};
  std::vector<ContinuationStage> stages = options.stages;
  for (ContinuationStage& stage : stages) {
    stage.weights.forceGate.forceScale =
        robot.mass * kGravitationalAcceleration;
  }
  return solveContinuation(
      problem, augmentedGuessOf(guess, robot, options.guessActivation), stages);
}

/** The stances of `solution` and the motion as the guess of solve 3: the
 * forces of the inactive candidates zeroed. */
absl::StatusOr<MpcTrajectory> thresholdedGuessOf(
    const StageSolution& solution,
    const PlannerRobot& robot,
    const LoadedScenario& loaded,
    const ContactInvariantPlannerOptions& options) {
  const MpcRobotModelBase<scalar_t>& model = robot.models->getMpcRobotModel();
  const size_t n = model.getStateDim();
  const size_t m = model.getInputDim();
  const size_t k = robot.allowed.size();
  const GuessTrajectory& trajectory = solution.trajectory;
  std::vector<std::vector<double>> activations;
  for (const vector_t& augmented : trajectory.states) {
    const vector_t state = augmented.head(n);
    RobotConfiguration configuration;
    configuration.basePosition = model.getBasePosition(state);
    configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
    configuration.baseOrientation.normalize();
    configuration.jointNames = model.modelSettings.mpcModelJointNames;
    configuration.jointPositions = model.getJointAngles(state);
    RETURN_IF_ERROR(robot.stanceModel->setConfiguration(configuration));
    std::vector<double> node;
    for (size_t i = 0; i < k; ++i) {
      const double distance = loaded.terrain.signedDistance<double>(
          robot.stanceModel->padPosition(robot.allowed[i]));
      node.push_back(std::abs(distance) <= options.contactDistance
                         ? augmented(n + i)
                         : 0.0);
    }
    activations.push_back(std::move(node));
  }
  std::vector<bool> initial;
  std::vector<bool> final;
  for (const int candidate : robot.allowed) {
    initial.push_back(isContactActive(robot.initialMask, candidate));
    final.push_back(isContactActive(robot.goalMask, candidate));
  }
  ASSIGN_OR_RETURN(const StanceSchedule schedule,
                   thresholdActivations(trajectory.times, activations, initial,
                                        final, options.thresholding));
  MpcTrajectory guess;
  guess.times = trajectory.times;
  for (size_t node = 0; node < trajectory.times.size(); ++node) {
    guess.states.push_back(trajectory.states[node].head(n));
    vector_t input = trajectory.inputs[node].head(m);
    ContactMask mask = 0;
    for (size_t i = 0; i < k; ++i) {
      if (schedule.active[node][i]) {
        mask |= contactBit(robot.allowed[i]);
      } else {
        const int dim = model.getContactInputDim(robot.mpcContacts[i]);
        input
            .segment(model.getContactForceStartIndices(robot.mpcContacts[i]),
                     dim)
            .setZero();
      }
    }
    guess.inputs.push_back(std::move(input));
    guess.masks.push_back(mask);
  }
  return guess;
}

/** Logs every node of `solution` at --v=1: the base, and per allowed
 * candidate its pad, its distance to the terrain, c and its force over the
 * weight. */
void logStage(const StageSolution& solution,
              const PlannerRobot& robot,
              const LoadedScenario& loaded) {
  if (!VLOG_IS_ON(1)) return;
  const MpcRobotModelBase<scalar_t>& model = robot.models->getMpcRobotModel();
  const size_t n = model.getStateDim();
  const double weight = robot.mass * kGravitationalAcceleration;
  for (size_t node = 0; node < solution.trajectory.times.size(); ++node) {
    // The robot model reads the joints at the end of its state: the physical
    // part only.
    const vector_t& augmented = solution.trajectory.states[node];
    const vector_t state = augmented.head(n);
    const vector_t input =
        solution.trajectory.inputs[node].head(model.getInputDim());
    RobotConfiguration configuration;
    configuration.basePosition = model.getBasePosition(state);
    configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
    configuration.baseOrientation.normalize();
    configuration.jointNames = model.modelSettings.mpcModelJointNames;
    configuration.jointPositions = model.getJointAngles(state);
    if (!robot.stanceModel->setConfiguration(configuration).ok()) continue;
    std::string line = absl::StrFormat(
        "%s t=%.2f base=(%.3f %.3f %.3f) com=(%.3f %.3f %.3f)", solution.stage,
        solution.trajectory.times[node], configuration.basePosition.x(),
        configuration.basePosition.y(), configuration.basePosition.z(),
        robot.stanceModel->centerOfMass().x(),
        robot.stanceModel->centerOfMass().y(),
        robot.stanceModel->centerOfMass().z());
    for (size_t k = 0; k < robot.allowed.size(); ++k) {
      const Eigen::Vector3d pad =
          robot.stanceModel->padPosition(robot.allowed[k]);
      absl::StrAppendFormat(
          &line, " | %s p=(%.3f %.3f %.3f) d=%.3f c=%.2f f=%.2f",
          loaded.contactSet.candidate(robot.allowed[k]).name, pad.x(), pad.y(),
          pad.z(), loaded.terrain.signedDistance<double>(pad), augmented(n + k),
          model.getContactForce(input, robot.mpcContacts[k]).norm() / weight);
    }
    VLOG(1) << line;
  }
}

/** The stance sequence of `plan`, for the log: "foot_l_contact,
 * foot_r_contact -> ...". */
std::string stancesOf(const config::MultiContactPlan& plan,
                      const ContactSet& contactSet) {
  std::vector<std::string> stances;
  ContactMask last = 0;
  for (size_t k = 0; k < plan.knots.size(); ++k) {
    if (k > 0 && plan.knots[k].active_contacts == last) continue;
    last = plan.knots[k].active_contacts;
    const absl::StatusOr<std::vector<std::string>> names =
        contactSet.namesOf(last);
    stances.push_back(
        absl::StrFormat("%.2f s [%s]", plan.knots[k].time,
                        names.ok() ? absl::StrJoin(*names, ", ") : "?"));
  }
  return absl::StrJoin(stances, " -> ");
}

}  // namespace

ContactInvariantPlannerOptions::ContactInvariantPlannerOptions()
    : stages(defaultStages()) {
  // No flight phase (design section 2: every instant is statically stable or
  // at least supported).
  thresholding.minimumContacts = 1;
  // A stance lasts long enough to shift the weight before its next lift-off
  // (the unloading window of solve 3, and as much again).
  thresholding.minimumChangeSeparation = kMinimumStance;
  thresholding.changeQuantum = kChangeQuantum;
  barriers.activationBounds =
      SquaredHingePenalty::Config(kActivationPenaltyMu, /*deltaParam=*/0.0);
  barriers.frictionCone =
      RelaxedBarrierPenalty::Config(kConeBarrierMu, kConeBarrierDelta);
  barriers.padClearance =
      RelaxedBarrierPenalty::Config(kPadBarrierMu, kPadBarrierDelta);
}

ContactInvariantPlanner::ContactInvariantPlanner(
    const ContactInvariantPlannerOptions& options)
    : options_(options) {}

absl::StatusOr<std::unique_ptr<ContactInvariantPlanner>>
ContactInvariantPlanner::Create(const ContactInvariantPlannerOptions& options) {
  if (options.stages.empty() || !(options.frictionShare > 0.0) ||
      !(options.frictionShare <= 1.0) || !(options.coneSmoothing > 0.0) ||
      !(options.maxPointTilt >= 0.0) || !(options.guessActivation >= 0.0) ||
      !(options.guessActivation <= 1.0) ||
      !(options.guessTrackingScale >= 0.0) || !(options.continuationDt > 0.0) ||
      !(options.refinement.dt > 0.0)) {
    return absl::InvalidArgumentError(
        "contact_invariant: the stages, the friction share in (0, 1], the "
        "smoothing, the tilt, the guess and the time step must be set");
  }
  for (const ContinuationStage& stage : options.stages) {
    if (stage.sqpIterations < 1 || !(stage.weights.forceGate.k1 > 0.0)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "contact_invariant: stage '", stage.name,
          "' needs iterations and a positive force gate stiffness k1"));
    }
  }
  return absl::WrapUnique(new ContactInvariantPlanner(options));
}

absl::StatusOr<config::MultiContactPlan> refinedPlan(
    HardContactRefinement& refinement,
    const config::MultiContactPlan& seed,
    const MpcTrajectory& guess,
    const LoadedScenario& loaded,
    absl::string_view planner) {
  ASSIGN_OR_RETURN(RefinementResult result, refinement.refine(seed, guess));
  LOG(INFO) << absl::StrFormat(
      "%s: '%s' solve 3: %d SQP iterations in %.1f s, %d polygon updates; "
      "the stances %s",
      planner, loaded.scenario.name, result.sqpIterations, result.seconds,
      result.polygonUpdates, stancesOf(result.plan, loaded.contactSet));
  if (VLOG_IS_ON(1)) {
    for (const config::CertifierReport::Knot& knot :
         result.plan.report->knots) {
      VLOG(1) << absl::StrFormat(
          "solve 3 t=%.2f [%s] margin=%s torque=%s residual=%.4f pads=%.4f "
          "spheres=%s %s",
          knot.time, absl::StrJoin(knot.active_contacts, ","),
          knot.static_margin.has_value()
              ? absl::StrFormat("%.3f", *knot.static_margin)
              : "-",
          knot.torque_utilization.has_value()
              ? absl::StrFormat("%.2f(%s)", *knot.torque_utilization,
                                knot.max_torque_joint)
              : "-",
          knot.momentum_rate_residual, knot.max_active_contact_distance,
          knot.min_sphere_clearance.has_value()
              ? absl::StrFormat("%.4f(%s)", *knot.min_sphere_clearance,
                                knot.min_sphere_clearance_link)
              : "-",
          absl::StrJoin(knot.failures, "; "));
    }
  }
  if (!result.plan.report->passed) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "%s: the plan for '%s' fails the certifier at %d of %d knots after "
        "solve 3; the first: %s",
        planner, loaded.scenario.name, result.plan.report->failed_knots,
        static_cast<int>(result.plan.knots.size()),
        firstFailingKnots(*result.plan.report)));
  }
  if (const absl::Status reached = checkGoalReached(result.plan, loaded);
      !reached.ok()) {
    return absl::NotFoundError(
        absl::StrCat(planner, ": the plan for '", loaded.scenario.name,
                     "' does not reach the goal: ", reached.message()));
  }
  return std::move(result.plan);
}

absl::StatusOr<config::MultiContactPlan> ContactInvariantPlanner::plan(
    const LoadedScenario& loaded) {
  const absl::Time started = absl::Now();
  if (!loaded.collisionModelConfig.has_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "scenario '", loaded.scenario.name,
        "': the contact-invariant planner needs a collision_model_file"));
  }
  ASSIGN_OR_RETURN(const PlannerRobot robot, robotOf(loaded));
  const std::vector<double> times =
      knotGrid(loaded.scenario.duration, options_.continuationDt);
  ASSIGN_OR_RETURN(const MpcTrajectory guess, guessOf(robot, loaded, times));
  ASSIGN_OR_RETURN(const std::vector<StageSolution> stages,
                   continuationOf(robot, loaded, guess, options_));
  for (const StageSolution& stage : stages) logStage(stage, robot, loaded);
  ASSIGN_OR_RETURN(const MpcTrajectory thresholded,
                   thresholdedGuessOf(stages.back(), robot, loaded, options_));
  const MpcTrajectory refinementGuess = resampledTrajectory(
      thresholded,
      knotGrid(loaded.scenario.duration, options_.refinement.knotSpacing),
      robot.models->getStateManifold().get());
  ASSIGN_OR_RETURN(
      const config::MultiContactPlan seed,
      planOfTrajectory(
          refinementGuess,
          PlanConversion{
              .scenario = &loaded,
              .model = &robot.models->getMpcRobotModel(),
              .contactNames = robot.models->modelSettings().contactNames,
              .mass = robot.mass,
              .stanceModel = robot.stanceModel.get(),
              .planner = std::string(kContactInvariantPlannerName)}));
  for (const config::MultiContactPlan::ContactPhase& phase : seed.phases) {
    const Eigen::Vector3d target = phase.target_pad_pose.head<kPositionDim>();
    const Eigen::Vector3d normal = phase.target_pad_pose.tail<kPositionDim>();
    VLOG(1) << absl::StrFormat(
        "seed phase %s [%.3f, %.3f] on %s face %d at (%.3f %.3f %.3f) n (%.2f "
        "%.2f %.2f)",
        phase.contact, phase.start_time, phase.end_time, phase.primitive,
        phase.face, target.x(), target.y(), target.z(), normal.x(), normal.y(),
        normal.z());
  }
  LOG(INFO) << "contact_invariant: '" << loaded.scenario.name
            << "' thresholded stances: " << stancesOf(seed, loaded.contactSet);
  ASSIGN_OR_RETURN(
      std::unique_ptr<HardContactRefinement> refinement,
      HardContactRefinement::Create(loaded, robot.config, loaded.urdfPath,
                                    options_.refinement));
  ASSIGN_OR_RETURN(config::MultiContactPlan plan,
                   refinedPlan(*refinement, seed, refinementGuess, loaded,
                               kContactInvariantPlannerName));
  LOG(INFO) << absl::StrFormat(
      "contact_invariant: '%s' planned in %.1f s: %d knots, %.1f s of motion",
      loaded.scenario.name, absl::ToDoubleSeconds(absl::Now() - started),
      static_cast<int>(plan.knots.size()), plan.knots.back().time);
  return plan;
}

}  // namespace ocs2::humanoid::multi_contact
