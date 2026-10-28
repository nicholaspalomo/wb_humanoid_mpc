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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/planner/contact_invariant/PlanTrajectory.h"
#include "humanoid_multi_contact/planner/contact_invariant/QuasiStaticTorqueConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/SupportPolygonConstraint.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"

namespace ocs2::humanoid::multi_contact {

/** The defaults of RefinementOptions; its fields say what each is. */
inline constexpr double kDefaultRefinementDt = 0.0125;
inline constexpr double kDefaultKnotSpacing = 0.025;
inline constexpr double kDefaultUnloadingWindow = 0.1;
inline constexpr double kDefaultUnloadingWeight = 1.0e5;
inline constexpr double kDefaultCopMargin = 0.01;
inline constexpr double kDefaultStanceNormalGain = 20.0;
inline constexpr double kDefaultSwingWeight = 1000.0;
inline constexpr double kDefaultRefinementSwingClearance = 0.12;
inline constexpr double kDefaultLandingWindow = 0.2;
inline constexpr double kDefaultLandingWeight = 1.0e3;
inline constexpr int kDefaultRefinementIterations = 60;
inline constexpr int kDefaultPlannerThreads = 4;
inline constexpr double kDefaultRefinementMarginBuffer = 0.005;
inline constexpr double kDefaultRefinementTorqueBuffer = 0.02;
inline constexpr int kDefaultMaxPolygonUpdates = 3;

/** The settings of solve 3 (design sections 4.2 and 4.3). */
struct RefinementOptions {
  RefinementOptions();

  // [s] The solver's node spacing: fine, because the stance pads are held
  // at the nodes only (zero_velocity); and the plan's knot spacing.
  double dt = kDefaultRefinementDt;
  double knotSpacing = kDefaultKnotSpacing;
  int sqpIterations = kDefaultRefinementIterations;
  int numThreads = kDefaultPlannerThreads;
  // The relaxed barriers of the torques (rows in fractions of the effort
  // limit), the collision spheres (rows in meters) and the support polygons
  // (rows in meters).
  RelaxedBarrierPenalty::Config torqueBarrier;
  RelaxedBarrierPenalty::Config clearanceBarrier;
  RelaxedBarrierPenalty::Config polygonBarrier;
  // [m] Added to the scenario's static margin in the polygon constraints;
  // subtracted from the torque margin in the torque constraints: the
  // certifier must find the scenario's margins.
  double staticMarginBuffer = kDefaultRefinementMarginBuffer;
  double torqueMarginBuffer = kDefaultRefinementTorqueBuffer;
  // How often the polygons are recomputed at the solution when the certifier
  // fails a static condition (design section 4.3).
  int maxPolygonUpdates = kDefaultMaxPolygonUpdates;
  // The relaxed barrier of the MPC's contact wrench cones in solve 3 (rows in
  // newtons), stiffer than the task file's online tuning: the certifier
  // judges the planned momentum rates exactly.
  RelaxedBarrierPenalty::Config wrenchConeBarrier;
  // [s] How long before a lift-off the contact is unloaded, and the weight of
  // its wrench there (LiftOffUnloadingCost).
  double unloadingWindow = kDefaultUnloadingWindow;
  double unloadingWeight = kDefaultUnloadingWeight;
  // [m] How much solve 3 shrinks the soles' rectangle of the MPC's wrench
  // cones on every side: the centers of pressure keep this far inside the
  // certifier's rectangle.
  double copMargin = kDefaultCopMargin;
  // [1/s] The stance pads' normal position gain (the zero_velocity
  // constraint's) and the weight of the swing servo (contact_swing) of solve
  // 3: an offline plan lands its pads on their patches.
  double stanceNormalGain = kDefaultStanceNormalGain;
  double swingWeight = kDefaultSwingWeight;
  // [m] The clearance of the swings of solve 3 (multi_contact_plan.swing), in
  // place of the task file's: a sole's pad is its center, so its toe and heel
  // keep this less the sole's half length.
  double swingClearance = kDefaultRefinementSwingClearance;
  // [s] How long before a touch-down a sole aligns with its surface, and the
  // weight of its tilt there (LandingAlignmentCost).
  double landingWindow = kDefaultLandingWindow;
  double landingWeight = kDefaultLandingWeight;
};

/** What solve 3 found. Passive data. */
struct RefinementResult {
  // The plan at the seed's knots, stamped, with the certifier's report.
  config::MultiContactPlan plan;
  MpcTrajectory trajectory;
  int polygonUpdates = 0;
  int sqpIterations = 0;
  double seconds = 0.0;
};

/**
 * Solve 3 of planner A (design section 4.2): the hard-contact problem of a
 * known stance sequence, the centroidal MPC of the scenario's multi-contact
 * task file tracking a seed plan over its whole duration in one offline SQP
 * solve (CentroidalMpcInterface with the plan in memory): the plan's stances
 * are the mode schedule, its states the targets, the swings rebuilt between
 * its contact phases, every inactive contact's force zero. On top of the
 * MPC's terms it adds
 *  - under the scenario's joint limits "position_and_effort", the
 *    quasi-static joint torques within the scenario's torque margin
 *    (QuasiStaticTorqueConstraint, constrainedEffortLimits()),
 *  - the collision spheres out of the terrain (TerrainClearanceConstraint),
 *  - with a static margin, the CoM in the static equilibrium polygon of the
 *    smallest stance around every knot (SupportPolygonConstraint, design
 *    section 4.3), computed at the seed and, when the certifier fails a
 *    static condition, again at the solution, at most maxPolygonUpdates
 *    times.
 *
 * Not thread-safe.
 */
class HardContactRefinement {
 public:
  /**
   * The refinement of `loaded` (copied) on the MPC of `config` and the URDF
   * at `urdfFile`. InvalidArgument for inconsistent options; the errors of
   * the stance model, the collision model and the certifier.
   */
  static absl::StatusOr<std::unique_ptr<HardContactRefinement>> Create(
      const LoadedScenario& loaded,
      CentroidalMpcConfig config,
      const std::string& urdfFile,
      const RefinementOptions& options);

  HardContactRefinement(const HardContactRefinement&) = delete;
  HardContactRefinement& operator=(const HardContactRefinement&) = delete;
  ~HardContactRefinement() = default;

  /**
   * Solves the hard-contact problem of `seed` (a plan of the scenario: its
   * knots on the option's grid, its stances changing one contact at a time)
   * from the warm start `guess` (the MPC's states and inputs at the seed's
   * knots), and certifies the result. The result's report says whether it
   * passed; Internal when the SQP fails, the MPC's errors for a seed it
   * refuses.
   */
  absl::StatusOr<RefinementResult> refine(const config::MultiContactPlan& seed,
                                          const MpcTrajectory& guess);

  const RefinementOptions& options() const { return options_; }
  RobotStanceModel& stanceModel() { return *stanceModel_; }

 private:
  HardContactRefinement(LoadedScenario loaded,
                        CentroidalMpcConfig config,
                        std::string urdfFile,
                        const RefinementOptions& options,
                        std::unique_ptr<RobotStanceModel> stanceModel,
                        CollisionModel collision,
                        std::unique_ptr<StabilityCertifier> certifier);

  /** The support polygons of the smallest stance around every knot of
   * `trajectory`, at its configurations (states of `model`'s layout). */
  absl::StatusOr<std::vector<TimedPolygon>> polygonsOf(
      const MpcTrajectory& trajectory,
      const MpcRobotModelBase<scalar_t>& model);
  /** One solve of `seed` from `guess` with `polygons`; the trajectory at the
   * seed's knots and the SQP's iterations. */
  absl::StatusOr<RefinementResult> solve(
      const config::MultiContactPlan& seed,
      const MpcTrajectory& guess,
      const std::vector<TimedPolygon>& polygons);

  LoadedScenario loaded_;
  CentroidalMpcConfig config_;
  std::string urdfFile_;
  RefinementOptions options_;
  std::unique_ptr<RobotStanceModel> stanceModel_;
  CollisionModel collision_;
  std::unique_ptr<StabilityCertifier> certifier_;
};

/**
 * The effort limits solve 3 holds the quasi-static torques within: every joint
 * of the MPC (`settings`' joints of `model`) that has one under
 * JointLimits::kPositionAndEffort; none under kPositionOnly, where solve 3 adds
 * no torque constraint.
 */
std::vector<JointEffortLimit> constrainedEffortLimits(
    JointLimits jointLimits,
    const pinocchio::Model& model,
    const ModelSettings& settings);

/** The first failing knots of `report`, for messages. */
std::string firstFailingKnots(const config::CertifierReport& report);

/** Whether a knot of `report` fails a static condition (1 or 3). */
bool failsStaticCondition(const config::CertifierReport& report);

}  // namespace ocs2::humanoid::multi_contact
