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

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

/** What the lockstep loop measures at one control cycle of a scenario's commands. */
struct ControlCycleSample {
  double time = 0.0;                                                   ///< [s] simulation time
  Eigen::Vector3d basePosition = Eigen::Vector3d::Zero();              ///< [m] world frame
  Eigen::Vector4d baseQuaternion = Eigen::Vector4d::UnitW();           ///< base to world, coefficients (x, y, z, w)
  Eigen::Vector3d baseLinearVelocityWorld = Eigen::Vector3d::Zero();   ///< [m/s]
  Eigen::Vector3d baseAngularVelocityLocal = Eigen::Vector3d::Zero();  ///< [rad/s] base frame
  /// The velocity the MPC is asked to track: (forward, lateral) in the heading frame [m/s] and the yaw rate [rad/s],
  /// i.e. the operator's command after the motion manager's filter and acceleration ramps.
  Eigen::Vector3d referenceVelocity = Eigen::Vector3d::Zero();
  double referenceBaseHeight = 0.0;  ///< [m] world height of the base the command asks for
  std::array<Eigen::Vector3d, 2> contactPositions{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};  ///< [m] world
  std::array<bool, 2> contactFlags{false, false};  ///< measured: the simulator's ground truth
  Eigen::VectorXd jointTorques;                    ///< [N m] commanded total torque of each MPC joint
  size_t nonFiniteValues = 0;                      ///< in the observation, the policy input and the joint action of this cycle
  bool mpcHealthy = true;                          ///< MpcResetSupervisor::isHealthy()
};

/** What the lockstep loop measures at one solve during a scenario's commands. */
struct SolveSample {
  double time = 0.0;               ///< [s] observation time of the solve
  bool succeeded = true;           ///< MPC_MRT_Interface::advanceMpc() returned OK
  double wallTimeMs = 0.0;         ///< [ms] the solver iteration: reset served, synchronized modules and SQP
  double lqApproximationMs = 0.0;  ///< SqpSolver::getBenchmarks()
  double solveQpMs = 0.0;
  double linesearchMs = 0.0;
  double computeControllerMs = 0.0;
  /// The rotation part [rad] and the norm of the initial-state gap delta_x0 of the solve's first SQP iteration
  /// (SqpSolver::getInitialStateGap()): the observation against the node 0 the SQP warm-started from, zero after a reset
  /// the solve served first. Empty when the solve failed.
  std::optional<double> initialStateRotationGap;
  std::optional<double> initialStateGapNorm;
  /// max | |xi_k| - 1 | over the nodes of the new solution; empty for a formulation without a quaternion in its state.
  std::optional<double> quaternionNormDeviation;
};

/** Counters of the whole run, from the controller's reset supervisor and the simulator. */
struct ClosedLoopCounters {
  size_t resetsServed = 0;      ///< MpcResetSupervisor::numResetsServed() during the commands
  size_t fullResetsServed = 0;  ///< MpcResetSupervisor::numFullResetsServed() during the commands
  size_t simulatorResets = 0;   ///< MujocoSimInterface::resetEpoch() moves during the whole run
};

/** What a metrics file says about itself, beside the metrics. */
struct ClosedLoopRunInfo {
  std::string label;                           ///< the baseline label, e.g. M0
  std::string robot;                           ///< RobotConfiguration::name
  std::string formulation;                     ///< centroidal or whole_body
  std::string scenario;                        ///< ClosedLoopScenario::name
  JsonValue provenance = JsonValue::object();  ///< commit, worktree, configuration hashes, machine
  JsonValue settings = JsonValue::object();    ///< rates, threads, timeline of the run
};

/**
 * The closed-loop metrics of the quaternion design's section 4.5, accumulated over the control cycles and the solves of
 * a scenario's commands, and written as the JSON document ClosedLoopMetricsSchema describes:
 *  - survival (the fall flag, its time and reason);
 *  - base height: mean, standard deviation and RMS error against the commanded height;
 *  - tilt |tau| (the swing angle of the heading-tilt split): RMS and maximum;
 *  - velocity: RMS of the error of the base velocity in the heading frame against the reference;
 *  - yaw rate: RMS of the error of the world yaw rate against the reference;
 *  - heading: the twist heading unwrapped from the first command on, at the end and at its extremes;
 *  - stance-foot slip: the horizontal drift of a contact point while it stays in measured contact;
 *  - joint torque RMS;
 *  - the largest rotation part and norm of the initial-state gap, and when the rotation part peaked;
 *  - max | |xi| - 1 | of the solutions (null for the Euler formulation);
 *  - the count of non-finite values;
 *  - solve time [ms]: mean, p50, p99 and max of the solver iteration and of each SQP phase;
 *  - failed solves, resets, simulator resets and the cycles with an unhealthy MPC.
 * Pure: no simulator and no MPC, so the definitions are unit-tested on synthetic samples.
 */
class ClosedLoopMetrics {
 public:
  void addControlCycle(const ControlCycleSample& sample);
  void addSolve(const SolveSample& sample);
  /** The robot fell at `time` [s] (tilt, height or a simulator reset, said in `reason`); the run ends there. */
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
    Eigen::Vector2d touchDownPosition = Eigen::Vector2d::Zero();
    double maxDrift = 0.0;
  };

  void closeStancePhase(size_t contact, const Eigen::Vector2d& position);

  size_t numCycles_ = 0;
  double firstTime_ = 0.0;
  double lastTime_ = 0.0;

  double heightSum_ = 0.0;
  double heightSquaredSum_ = 0.0;
  double heightErrorSquaredSum_ = 0.0;
  double tiltSquaredSum_ = 0.0;
  double tiltMax_ = 0.0;
  double velocityErrorSquaredSum_ = 0.0;
  double yawRateErrorSquaredSum_ = 0.0;

  bool haveHeading_ = false;
  double lastHeading_ = 0.0;
  double cumulativeHeading_ = 0.0;
  double cumulativeHeadingMax_ = 0.0;
  double cumulativeHeadingMin_ = 0.0;

  std::array<StancePhase, 2> stance_{};
  std::array<Eigen::Vector2d, 2> lastContactPosition_{Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero()};
  std::vector<double> stanceFinalDrifts_;
  double slipMax_ = 0.0;

  double torqueSquaredSum_ = 0.0;
  size_t torqueCount_ = 0;
  size_t nonFiniteValues_ = 0;
  size_t unhealthyCycles_ = 0;

  std::vector<double> wallTimesMs_;
  std::vector<double> lqApproximationMs_;
  std::vector<double> solveQpMs_;
  std::vector<double> linesearchMs_;
  std::vector<double> computeControllerMs_;
  size_t failedSolves_ = 0;
  std::optional<double> maxRotationGap_;
  double maxRotationGapTime_ = 0.0;
  std::optional<double> maxGapNorm_;
  std::optional<double> maxQuaternionNormDeviation_;

  std::optional<double> fallTime_;
  std::string fallReason_;
  ClosedLoopCounters counters_;
};

/** Statistics of a sample [ms]: mean, the nearest-rank p50 and p99, and the maximum; all null when it is empty. */
JsonValue summarizeTimes(std::vector<double> values);

/** `angle` wrapped into (-pi, pi]. */
double wrapAngle(double angle);

}  // namespace ocs2::humanoid::validation
