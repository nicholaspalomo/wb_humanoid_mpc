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

#include "humanoid_multi_contact/planner/contact_invariant/HardContactRefinement.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_data/StateTrajectoryInterpolation.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/multi_contact/MultiContactPlanSource.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/planner/contact_invariant/LandingAlignmentCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/LibraryName.h"
#include "humanoid_multi_contact/planner/contact_invariant/LiftOffUnloadingCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/QuasiStaticTorqueConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/RobotContactFeatures.h"
#include "humanoid_multi_contact/planner/contact_invariant/TerrainClearanceConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/TrajectoryInitializer.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The barriers by default: the torque rows in fractions of the effort limit,
// the sphere and polygon rows in meters.
constexpr double kTorqueBarrierMu = 0.05;
constexpr double kTorqueBarrierDelta = 0.02;
constexpr double kClearanceBarrierMu = 0.1;
constexpr double kClearanceBarrierDelta = 5.0e-4;
constexpr double kPolygonBarrierMu = 0.05;
constexpr double kPolygonBarrierDelta = 5.0e-3;
constexpr double kWrenchConeBarrierMu = 1.0;
constexpr double kWrenchConeBarrierDelta = 1.0;  // [N]
// [s] Node times closer than this are one time (an event's two nodes).
constexpr double kTimeTolerance = 1.0e-9;
// How many failing knots a message names.
constexpr int kNamedFailures = 3;

// The names of the terms solve 3 adds to the MPC's problem.
constexpr char kQuasiStaticTorqueTerm[] = "quasiStaticTorque";
constexpr char kTerrainClearanceTerm[] = "terrainClearance";
constexpr char kSupportPolygonTerm[] = "supportPolygon";
constexpr char kLiftOffUnloadingTerm[] = "liftOffUnloading";
constexpr char kLandingAlignmentTerm[] = "landingAlignment";
// The cone constants of the landing features' tape, which their tilt rows do
// not read.
constexpr double kLandingFeatureFriction = 0.5;
constexpr double kLandingFeatureSmoothing = 0.01;

/** The landing alignment of the surface candidates of `trajectory`'s
 * stances: their features (in contact-set order of the candidates ever in
 * contact) and the windows before their touch-downs; std::nullopt when no
 * surface candidate touches down. */
absl::StatusOr<std::unique_ptr<LandingAlignmentCost>> landingAlignmentOf(
    const MpcTrajectory& trajectory,
    CentroidalMpcInterface& interface,
    const LoadedScenario& loaded,
    const RefinementOptions& options) {
  const std::vector<std::string>& contactNames =
      interface.modelSettings().contactNames;
  const std::vector<int> candidates =
      candidateIndicesOf(contactNames, loaded.contactSet);
  std::vector<RobotFeatureCandidate> featured;
  std::vector<std::string> names;
  std::vector<LandingWindow> windows;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i] < 0) continue;
    const ContactCandidate& candidate =
        loaded.contactSet.candidate(candidates[i]);
    if (candidate.kind != ContactKind::kSurface) continue;
    std::vector<LandingWindow> touchDowns;
    for (size_t k = 1; k < trajectory.times.size(); ++k) {
      if (!isContactActive(trajectory.masks[k - 1], candidates[i]) &&
          isContactActive(trajectory.masks[k], candidates[i])) {
        touchDowns.push_back(LandingWindow{
            .candidate = static_cast<int>(featured.size()),
            .startTime = trajectory.times[k] - options.landingWindow,
            .endTime = trajectory.times[k]});
      }
    }
    if (touchDowns.empty()) continue;
    windows.insert(windows.end(), touchDowns.begin(), touchDowns.end());
    featured.push_back(
        RobotFeatureCandidate{.frameName = candidate.name,
                              .mpcContactIndex = i,
                              .kind = candidate.kind,
                              .padRotation = candidate.padRotation,
                              .geomRadius = candidate.geomRadius});
    names.push_back(candidate.name);
  }
  if (featured.empty()) return nullptr;
  const ModelSettings& settings = interface.modelSettings();
  const RobotFeatureSettings featureSettings{
      .friction = kLandingFeatureFriction,
      .coneSmoothing = kLandingFeatureSmoothing,
      .maxPointTilt = 0.0};
  ASSIGN_OR_RETURN(
      std::unique_ptr<ContactFeatureFunction> features,
      robotContactFeatures(
          interface.getPinocchioInterface(), interface.getCentroidalModelInfo(),
          featured, loaded.terrain, featureSettings,
          FeatureLibrarySettings{
              .name = hashedLibraryName(
                  "cio_landing_features",
                  absl::StrCat("terrain ",
                               terrainSceneHash(loaded.scenario.terrain_scene),
                               " candidates ", absl::StrJoin(names, " "))),
              .folder = settings.modelFolderCppAd,
              .recompile = settings.recompileLibrariesCppAd,
              .verbose = settings.verboseCppAd}));
  return std::make_unique<LandingAlignmentCost>(*features, std::move(windows),
                                                options.landingWeight);
}

/** The unloading windows of the lift-offs of `trajectory`'s stances, on the
 * MPC's inputs. */
std::vector<UnloadingWindow> unloadingWindowsOf(
    const MpcTrajectory& trajectory,
    const MpcRobotModelBase<scalar_t>& model,
    const std::vector<std::string>& contactNames,
    const ContactSet& contactSet,
    double window) {
  std::vector<UnloadingWindow> windows;
  const std::vector<int> candidates =
      candidateIndicesOf(contactNames, contactSet);
  for (size_t k = 1; k < trajectory.times.size(); ++k) {
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i] < 0) continue;
      if (isContactActive(trajectory.masks[k - 1], candidates[i]) &&
          !isContactActive(trajectory.masks[k], candidates[i])) {
        windows.push_back(UnloadingWindow{
            .forceOffset =
                static_cast<int>(model.getContactWrenchStartIndices(i)),
            .wrenchDim = static_cast<int>(model.getContactInputDim(i)),
            .startTime = trajectory.times[k] - window,
            .endTime = trajectory.times[k]});
      }
    }
  }
  return windows;
}

/** The configuration of the MPC state `state` by joint name. */
RobotConfiguration configurationOf(const MpcRobotModelBase<scalar_t>& model,
                                   const vector_t& state) {
  RobotConfiguration configuration;
  configuration.basePosition = model.getBasePosition(state);
  configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
  configuration.baseOrientation.normalize();
  configuration.jointNames = model.modelSettings.mpcModelJointNames;
  configuration.jointPositions = model.getJointAngles(state);
  return configuration;
}

/** The solution at `times`: its state interpolated on the manifold, its input
 * the one of the interval that starts at or before the time (the solver's
 * zero-order hold; at an event, the interval after it). */
void resample(const PrimalSolution& solution,
              const StateManifold* absl_nullable manifold,
              MpcTrajectory& trajectory) {
  const std::vector<scalar_t>& nodes = solution.timeTrajectory_;
  trajectory.states.clear();
  trajectory.inputs.clear();
  for (const double time : trajectory.times) {
    trajectory.states.push_back(interpolateStateTrajectory(
        manifold, time, nodes, solution.stateTrajectory_));
    const size_t after =
        std::upper_bound(nodes.begin(), nodes.end(), time + kTimeTolerance) -
        nodes.begin();
    const size_t index = after == 0 ? 0 : after - 1;
    trajectory.inputs.push_back(solution.inputTrajectory_[std::min(
        index, solution.inputTrajectory_.size() - 1)]);
  }
}

}  // namespace

std::vector<JointEffortLimit> constrainedEffortLimits(
    JointLimits jointLimits,
    const pinocchio::Model& model,
    const ModelSettings& settings) {
  std::vector<JointEffortLimit> limits;
  if (!enforcesEffortLimits(jointLimits)) return limits;
  for (const std::string& name : settings.mpcModelJointNames) {
    const pinocchio::JointIndex joint = model.getJointId(name);
    const int index = model.joints[joint].idx_v();
    const double effort = model.effortLimit(index);
    if (effort > 0.0) {
      limits.push_back(JointEffortLimit{
          .velocityIndex = static_cast<size_t>(index), .effort = effort});
    }
  }
  return limits;
}

RefinementOptions::RefinementOptions()
    : torqueBarrier(kTorqueBarrierMu, kTorqueBarrierDelta),
      clearanceBarrier(kClearanceBarrierMu, kClearanceBarrierDelta),
      polygonBarrier(kPolygonBarrierMu, kPolygonBarrierDelta),
      wrenchConeBarrier(kWrenchConeBarrierMu, kWrenchConeBarrierDelta) {}

std::string firstFailingKnots(const config::CertifierReport& report) {
  std::vector<std::string> failures;
  for (size_t k = 0; k < report.knots.size(); ++k) {
    const config::CertifierReport::Knot& knot = report.knots[k];
    if (knot.passed) continue;
    failures.push_back(
        absl::StrFormat("knot %d, t = %.3f s [%s]: %s", static_cast<int>(k),
                        knot.time, absl::StrJoin(knot.active_contacts, ", "),
                        absl::StrJoin(knot.failures, "; ")));
    if (static_cast<int>(failures.size()) >= kNamedFailures) break;
  }
  return failures.empty() ? "no knot fails" : absl::StrJoin(failures, " | ");
}

bool failsStaticCondition(const config::CertifierReport& report) {
  for (const config::CertifierReport::Knot& knot : report.knots) {
    if (knot.static_condition.value_or(true) == false ||
        knot.transition_condition.value_or(true) == false) {
      return true;
    }
  }
  return false;
}

HardContactRefinement::HardContactRefinement(
    LoadedScenario loaded,
    CentroidalMpcConfig config,
    std::string urdfFile,
    const RefinementOptions& options,
    std::unique_ptr<RobotStanceModel> stanceModel,
    CollisionModel collision,
    std::unique_ptr<StabilityCertifier> certifier)
    : loaded_(std::move(loaded)),
      config_(std::move(config)),
      urdfFile_(std::move(urdfFile)),
      options_(options),
      stanceModel_(std::move(stanceModel)),
      collision_(std::move(collision)),
      certifier_(std::move(certifier)) {}

absl::StatusOr<std::unique_ptr<HardContactRefinement>>
HardContactRefinement::Create(const LoadedScenario& loaded,
                              CentroidalMpcConfig config,
                              const std::string& urdfFile,
                              const RefinementOptions& options) {
  if (!(options.dt > 0.0) || !(options.knotSpacing >= options.dt) ||
      options.sqpIterations < 1 || options.numThreads < 1 ||
      options.maxPolygonUpdates < 0 || !(options.staticMarginBuffer >= 0.0) ||
      !(options.torqueMarginBuffer >= 0.0)) {
    return absl::InvalidArgumentError(
        "solve 3: the time step, the iterations, the threads, the buffers and "
        "the polygon updates must be positive");
  }
  if (!loaded.collisionModelConfig.has_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "scenario '", loaded.scenario.name,
        "': the contact-invariant planner needs a collision_model_file"));
  }
  ASSIGN_OR_RETURN(std::unique_ptr<RobotStanceModel> stanceModel,
                   RobotStanceModel::Create(urdfFile, loaded.contactSet));
  ASSIGN_OR_RETURN(
      CollisionModel collision,
      CollisionModel::Create(*loaded.collisionModelConfig, *stanceModel));
  ASSIGN_OR_RETURN(std::unique_ptr<StabilityCertifier> certifier,
                   makeCertifier(loaded));
  return absl::WrapUnique(new HardContactRefinement(
      loaded, std::move(config), urdfFile, options, std::move(stanceModel),
      std::move(collision), std::move(certifier)));
}

absl::StatusOr<std::vector<TimedPolygon>> HardContactRefinement::polygonsOf(
    const MpcTrajectory& trajectory, const MpcRobotModelBase<scalar_t>& model) {
  std::vector<TimedPolygon> polygons;
  const size_t n = trajectory.times.size();
  for (size_t k = 0; k < n; ++k) {
    // The smallest stance around the knot: condition 3 holds the CoM in the
    // common stance of a change at both of its knots.
    ContactMask mask = trajectory.masks[k];
    if (k > 0) mask &= trajectory.masks[k - 1];
    if (k + 1 < n) mask &= trajectory.masks[k + 1];
    if (mask == 0) continue;
    RETURN_IF_ERROR(stanceModel_->setConfiguration(
        configurationOf(model, trajectory.states[k])));
    ASSIGN_OR_RETURN(const StanceOnTerrain stance,
                     stanceOnTerrain(*stanceModel_, loaded_.terrain, mask));
    ASSIGN_OR_RETURN(const StaticEquilibrium equilibrium,
                     StaticEquilibrium::Create(
                         stance.stance, certifier_->options().equilibrium));
    // An LP HPIPM cannot solve gives the knot no polygon, as the stance
    // graph treats it: the certifier names the knot if it fails.
    const absl::StatusOr<std::vector<Eigen::Vector2d>> vertices =
        equilibrium.supportPolygon();
    if (!vertices.ok()) {
      VLOG(1) << "solve 3: no support polygon at t = " << trajectory.times[k]
              << ": " << vertices.status();
      continue;
    }
    const double halfStep =
        0.5 * (k + 1 < n ? trajectory.times[k + 1] - trajectory.times[k]
                         : trajectory.times[k] - trajectory.times[k - 1]);
    const absl::StatusOr<TimedPolygon> polygon =
        polygonOf(*vertices, trajectory.times[k] - halfStep,
                  trajectory.times[k] + halfStep - kTimeTolerance);
    // A stance without a polygon (no equilibrium, or a single point) has no
    // constraint: the certifier names the knot.
    if (polygon.ok()) polygons.push_back(*polygon);
  }
  return polygons;
}

absl::StatusOr<RefinementResult> HardContactRefinement::solve(
    const config::MultiContactPlan& seed,
    const MpcTrajectory& guess,
    const std::vector<TimedPolygon>& polygons) {
  CentroidalMpcConfig config = config_;
  config.task.interface.verbose = false;
  config.task.contacts.contact_wrench_cone_soft_constraint.mu =
      options_.wrenchConeBarrier.mu;
  config.task.contacts.contact_wrench_cone_soft_constraint.delta =
      options_.wrenchConeBarrier.delta;
  config.task.contacts.contact_rectangle.x_min += options_.copMargin;
  config.task.contacts.contact_rectangle.x_max -= options_.copMargin;
  config.task.contacts.contact_rectangle.y_min += options_.copMargin;
  config.task.contacts.contact_rectangle.y_max -= options_.copMargin;
  config.task.model_settings.foot_constraint.position_error_gain_z =
      options_.stanceNormalGain;
  config.task.multi_contact_plan.contact_swing.weight = options_.swingWeight;
  // The solve's own landing alignment (LandingAlignmentCost, on the
  // features' tilt over its landing window) and swing clearance, not the
  // tracking MPC's of the task file: with those the solve converged to plans
  // that failed the sphere clearance.
  config.task.multi_contact_plan.contact_swing.landing_alignment_weight = 0.0;
  config.task.multi_contact_plan.swing.clearance = options_.swingClearance;
  ASSIGN_OR_RETURN(
      std::unique_ptr<CentroidalMpcInterface> interface,
      CentroidalMpcInterface::Create(
          std::move(config), urdfFile_,
          MultiContactPlanSource{.scenario = loaded_, .plan = seed}));
  OptimalControlProblem& problem = interface->getOptimalControlProblemRef();
  const ModelSettings& settings = interface->modelSettings();
  const PinocchioInterface& pinocchio = interface->getPinocchioInterface();
  const CentroidalModelInfo& info = interface->getCentroidalModelInfo();
  const std::string terrain = absl::StrCat(
      "terrain ", terrainSceneHash(loaded_.scenario.terrain_scene));

  // Under "position_only" no effort limit is constrained, and solve 3 has no
  // torque term.
  ASSIGN_OR_RETURN(const JointLimits jointLimits,
                   jointLimitsOf(loaded_.scenario.joint_limits));
  std::vector<JointEffortLimit> effortLimits =
      constrainedEffortLimits(jointLimits, pinocchio.getModel(), settings);
  if (!effortLimits.empty()) {
    const double torqueMargin =
        loaded_.scenario.torque_margin - options_.torqueMarginBuffer;
    problem.softConstraintPtr->add(
        kQuasiStaticTorqueTerm,
        std::make_unique<StateInputSoftConstraint>(
            std::make_unique<QuasiStaticTorqueConstraint>(
                pinocchio, info, settings.contactNames, std::move(effortLimits),
                torqueMargin,
                hashedLibraryName(
                    "cio_quasi_static_torque",
                    absl::StrCat("margin ", torqueMargin, " contacts ",
                                 absl::StrJoin(settings.contactNames, " "))),
                settings.modelFolderCppAd, settings.recompileLibrariesCppAd,
                settings.verboseCppAd),
            std::make_unique<RelaxedBarrierPenalty>(options_.torqueBarrier)));
  }
  ASSIGN_OR_RETURN(std::vector<ClearanceSphere> spheres,
                   clearanceSpheresOf(collision_, *stanceModel_, pinocchio));
  const size_t numSpheres = spheres.size();
  problem.stateSoftConstraintPtr->add(
      kTerrainClearanceTerm,
      std::make_unique<StateSoftConstraint>(
          std::make_unique<TerrainClearanceConstraint>(
              pinocchio, info, std::move(spheres), loaded_.terrain,
              /*margin=*/0.0,
              hashedLibraryName("cio_terrain_clearance",
                                absl::StrCat(terrain, " spheres ", numSpheres)),
              settings.modelFolderCppAd, settings.recompileLibrariesCppAd,
              settings.verboseCppAd),
          std::make_unique<RelaxedBarrierPenalty>(options_.clearanceBarrier)));
  if (!polygons.empty() && loaded_.scenario.static_margin.has_value()) {
    const double margin =
        *loaded_.scenario.static_margin + options_.staticMarginBuffer;
    problem.stateSoftConstraintPtr->add(
        kSupportPolygonTerm,
        std::make_unique<StateSoftConstraint>(
            std::make_unique<SupportPolygonConstraint>(
                pinocchio, info, polygons, margin,
                hashedLibraryName("cio_support_polygon",
                                  absl::StrCat("margin ", margin)),
                settings.modelFolderCppAd, settings.recompileLibrariesCppAd,
                settings.verboseCppAd),
            std::make_unique<RelaxedBarrierPenalty>(options_.polygonBarrier)));
  }

  problem.costPtr->add(
      kLiftOffUnloadingTerm,
      std::make_unique<LiftOffUnloadingCost>(
          unloadingWindowsOf(guess, interface->getMpcRobotModel(),
                             settings.contactNames, loaded_.contactSet,
                             options_.unloadingWindow),
          options_.unloadingWeight,
          info.robotMass * kGravitationalAcceleration));

  ASSIGN_OR_RETURN(std::unique_ptr<LandingAlignmentCost> landing,
                   landingAlignmentOf(guess, *interface, loaded_, options_));
  if (landing != nullptr) {
    problem.costPtr->add(kLandingAlignmentTerm, std::move(landing));
  }

  sqp::Settings sqp = interface->sqpSettings();
  sqp.dt = options_.dt;
  sqp.sqpIteration = static_cast<size_t>(options_.sqpIterations);
  sqp.nThreads = static_cast<size_t>(options_.numThreads);
  sqp.useFeedbackPolicy = false;
  sqp.enableLogging = false;
  sqp.printSolverStatus = false;
  sqp.printSolverStatistics = false;
  sqp.printLinesearch = false;
  const GuessTrajectory warm{
      .times = guess.times, .states = guess.states, .inputs = guess.inputs};
  const TrajectoryInitializer initializer(warm, interface->getStateManifold());
  PrimalSolution start;
  start.timeTrajectory_ = warm.times;
  start.stateTrajectory_ = warm.states;
  start.inputTrajectory_ = warm.inputs;
  const size_t initMode = interface->getPlannedMotionReferenceManagerPtr()
                              ->plannedMotion()
                              .modeSequence()
                              .front();
  RefinementResult result;
  result.trajectory.times = guess.times;
  result.trajectory.masks = guess.masks;
  const absl::Time started = absl::Now();
  // OCS2's SQP throws when a QP has no solution; converted at once.
  try {  // NOLINT(exceptions): OCS2's solver throws.
    SqpSolver solver(sqp, problem, initializer);
    solver.setReferenceManager(interface->getReferenceManagerPtr());
    solver.run(guess.times.front(), guess.states.front(), initMode,
               guess.times.back(), start);
    resample(solver.primalSolution(guess.times.back()),
             interface->getStateManifold().get(), result.trajectory);
    result.sqpIterations = static_cast<int>(solver.getNumIterations());
    const PerformanceIndex& performance = solver.getPerformanceIndeces();
    LOG(INFO) << absl::StrFormat(
        "solve 3: cost %.4g, dynamics defect %.3g, equality defect %.3g",
        performance.cost, performance.dynamicsViolationSSE,
        performance.equalityConstraintsSSE);
  } catch (const std::exception& error) {  // NOLINT(exceptions): see above.
    return absl::InternalError(absl::StrCat(
        "solve 3 of '", loaded_.scenario.name, "': ", error.what()));
  }
  result.seconds = absl::ToDoubleSeconds(absl::Now() - started);
  ASSIGN_OR_RETURN(
      result.plan,
      planOfTrajectory(result.trajectory,
                       PlanConversion{.scenario = &loaded_,
                                      .model = &interface->getMpcRobotModel(),
                                      .contactNames = settings.contactNames,
                                      .mass = info.robotMass,
                                      .stanceModel = stanceModel_.get(),
                                      .planner = seed.planner}));
  ASSIGN_OR_RETURN(result.plan.report, certifier_->certify(result.plan));
  return result;
}

absl::StatusOr<RefinementResult> HardContactRefinement::refine(
    const config::MultiContactPlan& seed, const MpcTrajectory& guess) {
  std::vector<TimedPolygon> polygons;
  // The MPC's model, for the polygons of the seed: built once without a
  // problem.
  ASSIGN_OR_RETURN(
      std::unique_ptr<CentroidalMpcInterface> models,
      CentroidalMpcInterface::CreateControllerModels(config_, urdfFile_));
  if (loaded_.scenario.static_margin.has_value()) {
    ASSIGN_OR_RETURN(polygons, polygonsOf(guess, models->getMpcRobotModel()));
  }
  MpcTrajectory start = guess;
  RefinementResult result;
  double seconds = 0.0;
  int iterations = 0;
  for (int update = 0;; ++update) {
    ASSIGN_OR_RETURN(result, solve(seed, start, polygons));
    seconds += result.seconds;
    iterations += result.sqpIterations;
    result.polygonUpdates = update;
    LOG(INFO) << absl::StrFormat(
        "contact_invariant: solve 3 of '%s' (polygons %d): %d iterations in "
        "%.1f s, %d of %d knots fail the certifier",
        loaded_.scenario.name, update, result.sqpIterations, result.seconds,
        result.plan.report->failed_knots,
        static_cast<int>(result.plan.knots.size()));
    if (result.plan.report->passed ||
        !failsStaticCondition(*result.plan.report) ||
        !loaded_.scenario.static_margin.has_value() ||
        update >= options_.maxPolygonUpdates) {
      break;
    }
    ASSIGN_OR_RETURN(polygons,
                     polygonsOf(result.trajectory, models->getMpcRobotModel()));
    start = result.trajectory;
  }
  result.seconds = seconds;
  result.sqpIterations = iterations;
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
