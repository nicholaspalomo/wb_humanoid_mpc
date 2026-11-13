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

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

/**
 * How the stance-foot slip of a contact is measured. A point fixed on a link
 * stays where it touched down while it does not slip, so its slip is how far it
 * drifted from there. The contact point of a wheel moves with the wheel
 * (humanoid_nmpc/docs/rolling_contact/README.md), so its slip is the distance
 * the wheel material at the contact point slid over the ground: the integral of
 * the horizontal speed of that material, zero while the wheel rolls.
 */
enum class ContactSlipMeasure {
  kDrift,         ///< a contact point fixed on its link
  kMaterialSlip,  ///< a rolling contact
};

/** What the lockstep loop measures at one control cycle of a scenario's
 * commands. */
struct ControlCycleSample {
  double time = 0.0;  ///< [s] simulation time
  Eigen::Vector3d posBaseRtWorldEwrtWorld =
      Eigen::Vector3d::Zero();  ///< [m] world frame
  Eigen::Vector4d quatBaseRtWorldXyzw =
      Eigen::Vector4d::UnitW();  ///< base to world, coefficients (x, y, z, w)
  Eigen::Vector3d linvelBaseRtWorldEwrtWorld =
      Eigen::Vector3d::Zero();  ///< [m/s]
  Eigen::Vector3d angvelBaseRtWorldEwrtBase =
      Eigen::Vector3d::Zero();  ///< [rad/s] base frame
  /// The velocity the MPC is asked to track: (forward, lateral) in the heading
  /// frame [m/s] and the yaw rate [rad/s], i.e. the operator's command after
  /// the motion manager's filter and acceleration ramps.
  Eigen::Vector3d refVelocityCommand = Eigen::Vector3d::Zero();
  double referenceBaseHeight =
      0.0;  ///< [m] world height of the base the command asks for
  /// [m] world, per contact of the robot's set in contact order; as many as
  /// the ClosedLoopMetrics was made for.
  ContactArray<Eigen::Vector3d> posContactsRtWorldEwrtWorld;
  /// [m/s] world, per contact of the set in contact order: the velocity of
  /// the material at a rolling contact's point, v_wheel + w_wheel x (-r e_z),
  /// the simulator's ground truth. Read only for the contacts measured by
  /// ContactSlipMeasure::kMaterialSlip, and empty when there is none.
  ContactArray<Eigen::Vector3d> linvelContactsRtWorldEwrtWorld;
  /// Measured: the simulator's ground truth, per contact of the set.
  ContactMask contactFlags;
  Eigen::VectorXd
      jointTorques;  ///< [N m] commanded total torque of each MPC joint
  size_t nonFiniteValues = 0;  ///< in the observation, the policy input and the
                               ///< joint action of this cycle
  bool mpcHealthy = true;      ///< MpcResetSupervisor::isHealthy()
};

/**
 * The acceptance rule of the SQP's filter line search that judged a step
 * (FilterLinesearch::StepType), with the codes the time series writes
 * (TimeSeriesLabels.h, time_series::kSolveSqpStep).
 */
enum class SqpStepKind {
  kUnknown = 0,  ///< judged by no rule
  kConstraint =
      1,  ///< the step's violation above g_max: only its decrease is accepted
  /// Every other step not above g_max - between g_min and g_max, or below g_min
  /// after an iterate at or above it or without a descent direction: a decrease
  /// of the merit or of the violation is accepted.
  kDual = 2,
  kCost = 3,  ///< the violation below g_min before and after the step, on a
              ///< descent direction: the Armijo condition on the merit
  kZero = 4,  ///< no step was accepted
};

/** The step the SQP's line search took in the last iteration of a solve
 * (sqp_iteration 1: its only one). Passive data. */
struct SqpStepSample {
  double stepSize = 0.0;  ///< alpha: 1 the full step, 0 none
  SqpStepKind stepKind =
      SqpStepKind::kUnknown;  ///< the rule that accepted the step taken
  SqpStepKind fullStepKind =
      SqpStepKind::kUnknown;  ///< the rule that judged the full step, taken or
                              ///< not
  double fullStepConstraintViolation =
      0.0;  ///< the full step's constraint violation, taken or not
  /// The line search's constraint violation after the step
  /// (FilterLinesearch::totalConstraintViolation()): the square root of the
  /// squared shooting gaps and equality constraints, each weighted by its
  /// node's interval, plus the squared initial-state gap.
  double constraintViolation = 0.0;
  /// The largest |residual| of the hard swing-foot normal-velocity rows over
  /// the nodes of the solution; empty when no such row is active in the horizon
  /// (no swing). The rows' own units: [m/s] on the centroidal MPCs, whose row
  /// is v_z - zdot_ref + k_p (z - z_ref), and [m/s^2] on the whole-body MPC,
  /// whose row is k_v (v_z - zdot_ref) + k_a (a_z - zddot_ref) + k_p (z -
  /// z_ref) (WBMpcPreComputation::normalAccelerationConstraintConfig()).
  std::optional<double> swingNormalVelocityConstraintResidual;
};

/** What the lockstep loop measures at one solve during a scenario's commands.
 */
struct SolveSample {
  double time = 0.0;        ///< [s] observation time of the solve
  bool succeeded = true;    ///< MPC_MRT_Interface::advanceMpc() returned OK
  double wallTimeMs = 0.0;  ///< [ms] the solver iteration: reset served,
                            ///< synchronized modules and SQP
  double lqApproximationMs = 0.0;  ///< SqpSolver::getBenchmarks()
  double solveQpMs = 0.0;
  double linesearchMs = 0.0;
  double computeControllerMs = 0.0;
  /// The rotation part [rad] and the norm of the initial-state gap delta_x0 of
  /// the solve's first SQP iteration (SqpSolver::getInitialStateGap()): the
  /// observation against the node 0 the SQP warm-started from, zero after a
  /// reset the solve served first. Empty when the solve failed.
  std::optional<double> initialStateGapAngle;
  std::optional<double> initialStateGapNorm;
  /// max | |xi_k| - 1 | over the nodes of the new solution; empty for a
  /// formulation without a quaternion in its state.
  std::optional<double> unitQuaternionNormDeviation;
  /// The SQP's step in the solve (SqpSolver::getIterationRecords()); empty when
  /// the solve failed.
  std::optional<SqpStepSample> sqpStep;
};

/** Counters of the whole run, from the controller's reset supervisor and the
 * simulator. */
struct ClosedLoopCounters {
  size_t resetsServed =
      0;  ///< MpcResetSupervisor::numResetsServed() during the commands
  size_t fullResetsServed =
      0;  ///< MpcResetSupervisor::numFullResetsServed() during the commands
  size_t simulatorResets =
      0;  ///< MujocoSimInterface::resetEpoch() moves during the whole run
};

/** What a metrics file says about itself, beside the metrics. */
struct ClosedLoopRunInfo {
  std::string label;        ///< the label the run is recorded under (make
                            ///< closed-loop-metrics LABEL=)
  std::string robot;        ///< RobotConfiguration::name
  std::string formulation;  ///< centroidal or whole_body
  std::string scenario;     ///< ClosedLoopScenario::name
  JsonValue provenance =
      JsonValue::object();  ///< commit, worktree, configuration hashes, machine
  JsonValue settings =
      JsonValue::object();  ///< rates, threads, timeline of the run
};

/**
 * The closed-loop metrics of the quaternion design's section 4.5, accumulated
 * over the control cycles and the solves of a scenario's commands, and written
 * as the JSON document ClosedLoopMetricsSchema describes:
 *  - survival (the fall flag, its time and reason);
 *  - base height: mean, standard deviation and RMS error against the commanded
 * height;
 *  - tilt |tau| (the swing angle of the heading-tilt split): RMS and maximum;
 *  - velocity: RMS of the error of the base velocity in the heading frame
 * against the reference;
 *  - yaw rate: RMS of the error of the world yaw rate against the reference;
 *  - heading: the twist heading unwrapped from the first command on, at the end
 * and at its extremes;
 *  - stance-foot slip: the horizontal drift of a contact point while it stays
 * in measured contact, or, for a rolling contact, the distance its wheel slid
 * meanwhile (ContactSlipMeasure);
 *  - joint torque RMS;
 *  - the largest rotation part and norm of the initial-state gap, and when the
 * rotation part peaked;
 *  - max | |xi| - 1 | of the solutions (null without a solve; the Euler
 * formulation of the labels recorded before the switch had none);
 *  - the count of non-finite values;
 *  - solve time [ms]: mean, p50, p99 and max of the solver iteration and of
 * each SQP phase;
 *  - the SQP's steps: how many were full, reduced or none, how many full steps
 * were refused above g_max, the rule that accepted each, the longest run of
 * solves whose constraint violation grew, and the statistics of the constraint
 *    violation and of the swing-foot normal-velocity residual after each step
 * (a Gauss-Newton SQP cycling from one solve to the next shows as growth runs
 * ended by refusals above g_max);
 *  - failed solves, resets, simulator resets and the cycles with an unhealthy
 * MPC. Pure: no simulator and no MPC, so the definitions are unit-tested on
 * synthetic samples.
 *
 * Not thread-safe: one thread collects a run.
 */
class ClosedLoopMetrics {
 public:
  /**
   * Creates the metrics of a robot whose contacts' slip is measured as
   * `slipMeasures` says, one ContactSlipMeasure per contact in contact order
   * (at most kMaxNumContacts); every control cycle's sample carries that many
   * contact positions and flags, and the samples of a set with a
   * kMaterialSlip contact carry its material velocity as well. The
   * `numContacts` overload (ModelSettings::numContacts()) measures every
   * contact as kDrift, for a set of points fixed on their links.
   */
  explicit ClosedLoopMetrics(size_t numContacts);
  explicit ClosedLoopMetrics(std::vector<ContactSlipMeasure> slipMeasures);

  /** Adds one control cycle. Aborts on a sample of another contact count than
   * the metrics', a bug of the loop that collects them. */
  void addControlCycle(const ControlCycleSample& sample);
  void addSolve(const SolveSample& sample);
  /** The robot fell at `time` [s] (tilt, height or a simulator reset, said in
   * `reason`); the run ends there. */
  void setFall(double time, std::string reason);
  void setCounters(const ClosedLoopCounters& counters) { counters_ = counters; }

  size_t numSolves() const { return wallTimesMs_.size(); }
  bool survived() const { return !fallTime_.has_value(); }

  /** The metrics document, with `info` around it. */
  JsonValue report(const ClosedLoopRunInfo& info) const;

 private:
  /** One contact point's stance phase in progress. */
  struct StancePhase {
    bool active = false;
    Eigen::Vector2d touchdownPosContactRtWorldEwrtWorldXy =
        Eigen::Vector2d::Zero();
    /// [m] The stance phase's slip so far: for kDrift the largest distance
    /// from touchdownPosContactRtWorldEwrtWorldXy, for kMaterialSlip
    /// slidDistance.
    double maxSlip = 0.0;
    /// [m] kMaterialSlip: the distance slid since touch-down; 0 for kDrift.
    double slidDistance = 0.0;
  };

  void closeStancePhase(size_t contact,
                        const Eigen::Vector2d& posContactRtWorldEwrtWorldXy);
  /** Returns the slip a stance phase of `contact` ended with: its drift from
   * touch-down to `posContactRtWorldEwrtWorldXy`, or the distance it slid. */
  double finalSlipOf(size_t contact,
                     const Eigen::Vector2d& posContactRtWorldEwrtWorldXy) const;
  /** Counts the SQP step of a solve and extends or ends the run of growing
   * constraint violations. */
  void addSqpStep(const SqpStepSample& step);

  size_t numCycles_ = 0;
  double firstTime_ = 0.0;
  double lastTime_ = 0.0;

  double heightSum_ = 0.0;
  /// Welford's running mean of the height and sum of squared deviations from
  /// it, whose spread does not cancel as E[h^2] - E[h]^2 does: a constant
  /// height has a deviation sum of exactly zero.
  double heightRunningMean_ = 0.0;
  double heightSquaredDeviationSum_ = 0.0;
  double heightErrorSquaredSum_ = 0.0;
  double tiltSquaredSum_ = 0.0;
  double tiltMax_ = 0.0;
  double squaredSumErrLinvelBaseRtWorldEwrtHeadingXy_ = 0.0;
  double yawRateErrorSquaredSum_ = 0.0;

  bool haveHeading_ = false;
  double lastHeading_ = 0.0;
  double cumulativeHeading_ = 0.0;
  double cumulativeHeadingMax_ = 0.0;
  double cumulativeHeadingMin_ = 0.0;

  // Per contact of the set, sized at construction.
  std::vector<ContactSlipMeasure> slipMeasures_;
  bool hasMaterialSlip_ = false;  ///< a contact is measured by kMaterialSlip
  ContactArray<StancePhase> stance_;
  ContactArray<Eigen::Vector2d> prevPosContactsRtWorldEwrtWorldXy_;
  // [m] Per ended stance phase, finalSlipOf(): drift for kDrift, slid distance
  // for kMaterialSlip.
  std::vector<double> stanceFinalSlips_;
  double slipMax_ = 0.0;

  double jointTorqueSquaredSum_ = 0.0;
  size_t jointTorqueCount_ = 0;
  size_t nonFiniteValues_ = 0;
  size_t unhealthyCycles_ = 0;

  std::vector<double> wallTimesMs_;
  std::vector<double> lqApproximationMs_;
  std::vector<double> solveQpMs_;
  std::vector<double> linesearchMs_;
  std::vector<double> computeControllerMs_;
  size_t failedSolves_ = 0;
  // The SQP's steps (SolveSample::sqpStep).
  size_t sqpSteps_ = 0;
  size_t fullSteps_ = 0;
  size_t reducedSteps_ = 0;
  size_t zeroSteps_ = 0;
  size_t fullStepsRefusedAboveGMax_ = 0;
  size_t acceptedByConstraint_ = 0;
  size_t acceptedByDual_ = 0;
  size_t acceptedByCost_ = 0;
  /// The constraint violation of the last solve that recorded a step; empty
  /// after a solve without one, which ends a run.
  std::optional<double> lastConstraintViolation_;
  size_t violationGrowthRun_ = 0;
  size_t longestViolationGrowthRun_ = 0;
  std::vector<double> constraintViolations_;
  std::vector<double> swingNormalVelocityConstraintResiduals_;
  std::optional<double> maxGapAngle_;
  double maxGapAngleTime_ = 0.0;
  std::optional<double> maxGapNorm_;
  std::optional<double> maxUnitQuaternionNormDeviation_;
  size_t nonFiniteUnitQuaternionSolves_ = 0;

  std::optional<double> fallTime_;
  std::string fallReason_;
  ClosedLoopCounters counters_;
};

/** Statistics of a sample: mean, the nearest-rank p50 and p99, and the maximum;
 * all null when it is empty. */
JsonValue summarizeSample(std::vector<double> values);

/** `angle` wrapped into (-pi, pi]. */
double wrapAngle(double angle);

}  // namespace ocs2::humanoid::validation
