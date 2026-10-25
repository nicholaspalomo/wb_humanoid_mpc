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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerInterface.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipModel.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipStandingBlend.h"

namespace ocs2::humanoid {

/**
 * The reduced-order layer of "Reduced-Order Model Guided Contact-Implicit Model
 * Predictive Control for Humanoid Locomotion" (Esteban, Kurtz, Ghansah, Ames,
 * arXiv:2502.15630): a closed-form contact planner on the Hybrid Linear
 * Inverted Pendulum. See humanoid_nmpc/docs/hlip_contact_planner/README.md.
 *
 * There is no optimization in this planner and nothing is searched. A plan is
 * three steps:
 *
 *  1. The cadence is fixed. Single support phases of `hlip.sspDuration`
 * alternate between the feet, separated by double supports of
 * `hlip.dspDuration`, continuing the phase the robot is executing at the
 * planning instant. The contact sequence is therefore decided before the first
 * number is computed, which is exactly what the paper asks of the reduced-order
 * model: propose a nominal gait, and leave the whole-body controller free to
 * depart from it.
 *  2. Each foothold is the H-LIP deadbeat step. The reduced model is rolled
 * forward to the pre-impact state of the step, and the step length follows from
 * u = u* + K (x - x*), with the nominal orbit of the commanded velocity (a
 * period-one orbit along the heading, a period-two orbit laterally) and the
 * closed-form gain K of HlipModel. No gain and no weight is tuned; the clip to
 * the reachable step is the only bound applied to the result.
 *  3. Standing and walking are blended by the paper's alpha(phi)
 * (HlipStandingBlend): below a half the planner emits a standing plan and the
 * feet stay where they are, above it the commanded velocity is scaled by alpha,
 * so the stride grows continuously out of standing.
 *
 * What this replaces: the mixed-integer program of LipContactPlanner and
 * everything that is part of it - the branch and bound, the search stages, the
 * cost terms, the reachability, support region and separation constraints, and
 * the combinatorial phase-duration and alternation rules (README section 5
 * counts them against the term registry). None of them are needed once the
 * whole-body MPC, and not the planner, owns the contact decision. The execution
 * rules are NOT part of what is replaced: ContactPlanningReferenceManager
 * applies `formulation.execution` under either planner, and
 * planned_com_override, which hands this planner's center-of-mass roll-out to
 * the whole-body MPC, is mandatory.
 *
 * The plan is a ContactPlan on the node grid of `planner.dt`, like the
 * mixed-integer planner's, so everything downstream reads it the same way. The
 * cadence and the reduced model are continuous, and so is the mode schedule:
 * the plan carries its gait in continuous time as well
 * (ContactPlan::phaseContacts / phaseStartTimes), which
 * ContactPlan::toModeSchedule() takes its events from, so every executed single
 * support lasts `hlip.sspDuration` and every executed double support
 * `hlip.dspDuration` - the durations the deadbeat gain is computed for. The
 * per-node quantities are the gait sampled onto the grid: every phase boundary
 * is assigned to a node once (GaitPhase::endNode)
 * - the committed window on the nodes the executed schedule was sampled on,
 * every later boundary to the nearest node - and the contacts, footholds and
 * ZMP follow those nodes, while the deadbeat step and the center of mass are
 * evaluated at the continuous times. `planner.dt` should be small (0.025 s is a
 * millisecond-scale plan, since nothing is solved).
 *
 * Not thread-safe: plan() and setConfig() must not run concurrently with each
 * other or with the const methods. The const methods may be called concurrently
 * with each other. Built with Create(); neither copyable nor movable
 * (ContactPlannerInterface).
 */
class HlipContactPlanner final : public ContactPlannerInterface {
 public:
  /** One phase of the nominal gait. `swingFoot` is -1 in double support and
   * while standing. */
  struct GaitPhase {
    // [s] the phase in continuous time. Inside the committed window the
    // boundaries are the executed schedule's own event times
    // (ContactPlannerInput::committedPhaseStartTimes), so a swing in flight
    // ends at the touch-down the robot executes, not at the node that
    // touch-down falls in.
    scalar_t startTime = 0.0;
    scalar_t endTime = 0.0;
    // The plan node the phase ends at: intervals [previous endNode, endNode)
    // carry `contacts`. Inside the committed window this is the node where the
    // committed contacts change; after it, endTime rounded to the nearest node.
    // A phase shorter than half a node can end on the node it starts on and
    // then covers no interval at all.
    int endNode = 0;
    contact_flag_t contacts = makeFeetArray(true);
    int swingFoot = -1;

    scalar_t duration() const { return endTime - startTime; }
    bool isSingleSupport() const { return swingFoot >= 0; }
  };

  /** What buildGait() continues the committed window with. */
  enum class GaitKind {
    kStanding,  // double support over the rest of the horizon
    kWalking,   // the nominal cadence
  };

  /**
   * Builds the planner of a configuration. Returns the Status of
   * ContactPlanningConfig::validateStatus() when the configuration does not
   * validate, and the H-LIP model's when its cadence or pendulum is out of
   * bounds.
   */
  static absl::StatusOr<std::unique_ptr<HlipContactPlanner>> Create(
      ContactPlanningConfig config);

  ContactPlan plan(const ContactPlannerInput& input) override;
  /** Refuses a configuration that does not validate (or whose H-LIP model
   * cannot be built) and keeps the running one. */
  absl::Status setConfig(const ContactPlanningConfig& config) override;
  /** The planner keeps nothing between plans, so this does nothing; it exists
   * for the interface. */
  void reset() override {}
  std::string getFormulationSummary() const override {
    return summaryOf(config_, model_);
  }

  /**
   * The same summary for a configuration, without building a planner. It ends
   * with the start-up check below: whether the first step out of a standstill
   * fits `hlip.maxStepWidth` and, when it does not, whether the reduced model
   * recovers from the clipped step or locks (startUpLateralWidths).
   * InvalidArgument when the configuration's H-LIP model cannot be built
   * (HlipModel::Create()).
   */
  static absl::StatusOr<std::string> formulationSummary(
      const ContactPlanningConfig& config);

  /**
   * The lateral step the deadbeat law demands of the first step out of a
   * standstill, in meters, before any clip.
   *
   * This is the quantity that decides whether the gait can start cleanly, and
   * it is not obvious from the cadence. A robot standing still has its center
   * of mass half a step width from either foot with no lateral velocity, so
   * once the first foot leaves the ground it falls sideways fast: at the
   * shipped height it reaches the pre-impact instant far outside the orbit, and
   * catching it takes a step much wider than the nominal one. If that step does
   * not fit inside `hlip.maxStepWidth` it is clipped, and a clipped step is no
   * longer the deadbeat step. What happens next depends on how much was cut:
   * the lateral pendulum amplifies the residual by cosh(omega * sspDuration)
   * every step, and the gait either recovers within a few steps or locks into
   * an alternating wide and narrow limit cycle it can never leave (the feet
   * come together, the robot walks itself sideways and falls).
   * startUpLateralWidths() rolls the reduced model forward to tell the two
   * apart.
   *
   * Shortening the single support is what shrinks the demand: less time falling
   * before the foot lands. At the shipped `hlip.dspDuration` of 0.05 s and the
   * Atlas's 1.0805 m pendulum (its model's, which an unset shared.comHeight
   * resolves to) it drops from 0.47 m at a 0.35 s single support to 0.39 m at
   * 0.25 s, which against a 0.45 m reach is the difference between not fitting
   * and fitting. The double support and the pendulum have to be named with the
   * figures because both enter the deadbeat gain: at 0.35 s the demand is 0.45
   * m with dspDuration 0 and 0.49 m with 0.1, and on the 0.85 m pendulum the
   * Atlas was once hand-set to it was 0.52 m at 0.35 s / 0.05 s.
   */
  static absl::StatusOr<scalar_t> startUpLateralStep(
      const ContactPlanningConfig& config);

  /**
   * [m] the lateral distance between the feet after each of the first
   * `numSteps` steps out of a standstill, in the reduced model under the
   * planner's own step law and step clip: the start-up startUpLateralStep()
   * describes the first step of, rolled forward. The width settles at
   * `hlip.stepWidth` when the gait recovers, and alternates between two values
   * when a clipped first step has locked it. Both return InvalidArgument when
   * the configuration's H-LIP model cannot be built (HlipModel::Create()).
   */
  static absl::StatusOr<std::vector<scalar_t>> startUpLateralWidths(
      const ContactPlanningConfig& config, int numSteps);

  const ContactPlanningConfig& getConfig() const { return config_; }
  const HlipModel& getModel() const { return model_; }
  const HlipStandingBlend& getBlend() const { return blend_; }

  // The following are public for the tests.

  /** Whether the blend asks for a stepping gait at this input. */
  bool isWalking(const ContactPlannerInput& input) const;

  /** alpha, the blend's weight of the walking gait at this input: the stride is
   * planned for alpha times the command. */
  scalar_t blendWeight(const ContactPlannerInput& input) const;

  /**
   * [m/s] the velocity the blend measures, in the heading frame: the measured
   * center-of-mass velocity, with its lateral component taken relative to the
   * lateral sway of stepping in place.
   *
   * Laterally a biped cannot walk without swaying: the deadbeat law regulates a
   * zero command onto the period-two orbit at +-hlip.stepWidth, which crosses
   * every touch-down at 0.132 m/s at the shipped Atlas cadence on its
   * model's 1.0805 m pendulum (0.165 m/s on the 0.85 m pendulum of the
   * planner's unit tests) - several times the blend's half point. Measured raw,
   * that sway alone held alpha at one through every double support, the next
   * lift-off was committed every step, and a robot that had walked never came
   * back to standing at a zero command. So the lateral velocity counts only by
   * how far it lies outside the range between rest and the zero-command orbit
   * at the phase the robot is in (zeroCommandOrbitLateralVelocity): stepping in
   * place and slowing down from it read as zero, a push beyond the orbit's own
   * sway or against it reads in full. Along the heading the zero-command orbit
   * is rest, so the sagittal velocity is taken as measured.
   */
  vector2_t blendVelocity(const ContactPlannerInput& input) const;

  /**
   * [m/s] the lateral velocity (heading frame) of the zero-command period-two
   * orbit at the phase the robot is in at `input.time`: in single support the
   * orbit flowed for the time the swinging foot has been in flight, in double
   * support the orbit's drift after the touch-down of `input.lastSwungFoot`.
   * Empty when the robot has not stepped yet (double support with no known last
   * swing), and in flight.
   */
  std::optional<scalar_t> zeroCommandOrbitLateralVelocity(
      const ContactPlannerInput& input) const;

  /**
   * The nominal gait over the planning horizon, continuing the phase the robot
   * is executing at `input.time`. The committed window is replayed first, at
   * the executed schedule's event times; a standing input then ends in one
   * double support phase covering the rest of the horizon (after finishing a
   * swing still in flight at the end of the window), a walking input continues
   * the cadence.
   */
  std::vector<GaitPhase> buildGait(const ContactPlannerInput& input,
                                   GaitKind kind) const;

  /** The foot that swings next, from the last swung foot when it is known and
   * from the phase timing otherwise. */
  static size_t nextSwingFoot(const ContactPlannerInput& input);

 private:
  /** `model` must be the H-LIP of `config` (makeModel()). */
  HlipContactPlanner(ContactPlanningConfig config, const HlipModel& model);

  /** The H-LIP of a configuration; InvalidArgument when its cadence or pendulum
   * is out of bounds. */
  static absl::StatusOr<HlipModel> makeModel(
      const ContactPlanningConfig& config);

  /** formulationSummary(), startUpLateralStep() and startUpLateralWidths() on
   * `model`, the H-LIP of `config`. */
  static std::string summaryOf(const ContactPlanningConfig& config,
                               const HlipModel& model);
  static scalar_t startUpLateralStepOf(const ContactPlanningConfig& config,
                                       const HlipModel& model);
  static std::vector<scalar_t> startUpLateralWidthsOf(
      const ContactPlanningConfig& config,
      const HlipModel& model,
      int numSteps);

  /** The lateral step clipped to the reachable side of the stance foot:
   * [minStepWidth, maxStepWidth] for the left. */
  static scalar_t clipLateralStep(const HlipParameters& params,
                                  scalar_t lateralStep,
                                  bool placesLeftFoot);

  /** The heading at `time`: the commanded yaw rate, scaled by the blend weight
   * `alpha`, integrated from the planning instant. */
  scalar_t headingAt(const ContactPlannerInput& input,
                     scalar_t time,
                     scalar_t alpha) const;

  /** A deadbeat step and whether the reachable region cut it, which costs the
   * deadbeat property. */
  struct DeadbeatStep {
    vector2_t step = vector2_t::Zero();  // [m] in the heading frame
    bool clipped = false;
  };

  /**
   * The deadbeat step that places `swingFoot`, in the heading frame: the
   * pre-impact state of the step per axis, the commanded velocity in that
   * frame, and the foot being placed. A clipped step is counted onto the plan.
   */
  DeadbeatStep deadbeatStep(const HlipModel::State& preImpactX,
                            const HlipModel::State& preImpactY,
                            const vector2_t& commandedVelocity,
                            size_t swingFoot) const;

  /** The reduced model at the start of one phase of the gait, and the feet
   * during it (plan()). */
  struct PhaseRollOut {
    vector2_t comPosition =
        vector2_t::Zero();  // world, at the start of the phase
    vector2_t comVelocity = vector2_t::Zero();
    size_t stanceFoot = kContactLeftIndex;  // single support only
    Eigen::Matrix<scalar_t, kHorizontalPositionDim, kHorizontalPositionDim>
        world_R_phase = Eigen::Matrix<scalar_t,
                                      kHorizontalPositionDim,
                                      kHorizontalPositionDim>::Identity();
    HlipModel::State stateX =
        HlipModel::State::Zero();  // relative to the stance foot, in the
                                   // phase's heading frame
    HlipModel::State stateY = HlipModel::State::Zero();
    feet_array_t<vector2_t>
        feet{};  // during the phase: a swinging foot at its landing spot
    feet_array_t<scalar_t> footYaws = makeFeetArray(0.0);
  };

  /** The roll-out of a whole gait, one entry per phase, and how many of its
   * steps were clipped. */
  struct GaitRollOut {
    std::vector<PhaseRollOut> phases;
    int numClippedSteps = 0;
  };

  /** The center of mass of the reduced model at one instant, in the world
   * frame. */
  struct ComState {
    vector2_t position = vector2_t::Zero();
    vector2_t velocity = vector2_t::Zero();
  };

  // The steps of plan().

  /** A valid plan on the node grid of the configuration, every node and
   * interval at the input's measured values. */
  ContactPlan initialPlan(const ContactPlannerInput& input) const;
  /** Rolls the reduced model forward through `gait`, placing every swing at its
   * deadbeat step. */
  GaitRollOut rollOutGait(const ContactPlannerInput& input,
                          const std::vector<GaitPhase>& gait,
                          scalar_t alpha,
                          const vector2_t& plannedCommand) const;
  /** The center of mass of the roll-out `entry` of `phase` at `time`. */
  ComState comAt(const GaitPhase& phase,
                 const PhaseRollOut& entry,
                 scalar_t time) const;
  /** Fills the per-node center of mass, footholds and (with the heading model)
   * heading of `plan`. */
  void fillNodes(const ContactPlannerInput& input,
                 const std::vector<GaitPhase>& gait,
                 const std::vector<PhaseRollOut>& rollOut,
                 scalar_t alpha,
                 ContactPlan& plan) const;
  /** Fills `plan.phaseZmp`: the stance foot of every single support, none in
   * double support. */
  static void fillPhaseZmps(const std::vector<GaitPhase>& gait,
                            const std::vector<PhaseRollOut>& rollOut,
                            ContactPlan& plan);
  /** Blends the center of mass and the phase ZMPs of a standing plan towards
   * the center of `standingFeet` by `alpha`. */
  static void blendTowardsStanding(const feet_array_t<vector2_t>& standingFeet,
                                   scalar_t alpha,
                                   ContactPlan& plan);
  /** Fills the per-interval contacts and ZMP and the continuous-time phases of
   * `plan` from `gait`. */
  void fillContactsAndPhases(const ContactPlannerInput& input,
                             const std::vector<GaitPhase>& gait,
                             ContactPlan& plan) const;

  ContactPlanningConfig config_;
  HlipModel model_;
  HlipStandingBlend blend_;
};

}  // namespace ocs2::humanoid
