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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_mpc_validation/closed_loop/LockstepClosedLoop.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <mujoco_sim_interface/MujocoSimInterface.h>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/mrt/ControlMode.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_mpc_validation/closed_loop/LockstepSolveSchedule.h"
#include "humanoid_mpc_validation/closed_loop/TimeSeriesLabels.h"

namespace ocs2::humanoid::validation {
namespace {

/** Where the runner is in the operator's sequence. */
enum class Phase { kJointPd, kEnteringMpc, kStanding, kCommands };

/** [rad] The fall threshold without a simMaxBaseTiltAngle. */
constexpr double kDefaultMaxBaseTilt = 1.0;
/** Times within this of each other are the same [s]: the clock is a sum of simulation steps. */
constexpr double kTimeTolerance = 1e-9;

size_t countNonFinite(const vector_t& values) {
  size_t count = 0;
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values(i))) ++count;
  }
  return count;
}

/** Rows appended one at a time, as one matrix of a golden file. */
class RowCollector {
 public:
  explicit RowCollector(Eigen::Index cols) : cols_(cols) {}
  void append(const Eigen::VectorXd& row) { rows_.push_back(row); }
  golden_matrix_t matrix() const {
    golden_matrix_t matrix(static_cast<Eigen::Index>(rows_.size()), cols_);
    for (size_t row = 0; row < rows_.size(); ++row) matrix.row(static_cast<Eigen::Index>(row)) = rows_[row].transpose();
    return matrix;
  }

 private:
  Eigen::Index cols_;
  std::vector<Eigen::VectorXd> rows_;
};

}  // namespace

LockstepClosedLoop::LockstepClosedLoop(RobotConfiguration configuration, LockstepOptions options)
    : configuration_(std::move(configuration)), options_(std::move(options)) {}

absl::StatusOr<LockstepResult> LockstepClosedLoop::run(const ClosedLoopScenario& scenario, ClosedLoopRunInfo info) const {
  ASSIGN_OR_RETURN(std::unique_ptr<ClosedLoopDriver> driver, createClosedLoopDriver(configuration_, options_.driver));
  ASSIGN_OR_RETURN(const robot::mujoco_sim_interface::MujocoSimConfig simulatorConfig, driver->simulatorConfig());
  robot::mujoco_sim_interface::MujocoSimInterface sim(simulatorConfig, configuration_.urdfFile);
  RETURN_IF_ERROR(driver->connectSimulator(sim));
  RobotController& controller = driver->robotController();

  // As RobotProcess::start() starts: the backend initialized (one step), the first observation and the MPC's start-up
  // reset from it; then, standing in for the MPC node, the first policy before the simulation runs.
  sim.initSim();
  sim.updateInterfaceStateFromRobot();
  const robot::model::RobotState initialState = sim.getRobotState();
  driver->startMpc(initialState);
  const DriverSolveOutcome firstSolve = driver->solve();
  if (!firstSolve.status.ok()) return firstSolve.status;
  // The JOINT_PD posture is the posture the robot is spawned in, as the robot process's FSM bridge takes it from
  // createInitialSimState(): the simulator's initial state before any physics step (initSim() has already run one,
  // without torques), by joint name, which the two robot descriptions share.
  const robot::model::RobotDescription& description = sim.getRobotDescription();
  const robot::model::RobotState& spawnState = driver->initialRobotState();
  const robot::model::RobotDescription& spawnDescription = driver->robotDescription();
  std::vector<scalar_t> nominalPosture(description.getNumJoints(), 0.0);
  for (const std::string& jointName : description.getJointNames()) {
    nominalPosture[description.getJointIndex(jointName)] = spawnState.getJointPosition(spawnDescription.getJointIndex(jointName));
  }

  const double timeStep = sim.getModel()->opt.timestep;
  const int stepsPerCycle = std::max(1, static_cast<int>(std::lround(1.0 / (driver->mrtFrequency() * timeStep))));
  const double controlPeriod = stepsPerCycle * timeStep;
  const double mpcPeriod = 1.0 / driver->mpcFrequency();
  const double startTime = initialState.getTime();
  const double initialHeight = initialState.getRootPositionInWorldFrame().z();
  const double configuredMaxTilt = driver->settings().fallRecovery.maxBaseTiltAngle;
  const double maxTilt = configuredMaxTilt > 0.0 ? configuredMaxTilt : kDefaultMaxBaseTilt;
  const uint64_t initialResetEpoch = sim.resetEpoch();
  const double commandDuration = getCommandDuration(scenario);

  // JOINT_PD with the torques on, on the locked gantry.
  sim.enableTorques();
  std::string mode(control_mode::kJointPd);
  Phase phase = Phase::kJointPd;
  double phaseStart = startTime;
  std::optional<double> entryEndTime;
  std::optional<double> releaseTime;
  std::optional<double> commandStartTime;
  LockstepSolveSchedule solves(/*firstSolveTime=*/startTime + mpcPeriod, mpcPeriod, kTimeTolerance);
  bool commandSaturated = false;
  uint64_t resetsAtCommandStart = 0;
  uint64_t fullResetsAtCommandStart = 0;

  ClosedLoopMetrics metrics;
  LockstepResult result;
  result.recordedStates.jointNames = description.getJointNames();
  RowCollector seriesTime(/*cols=*/1);
  RowCollector seriesPosition(/*cols=*/3);
  RowCollector seriesQuaternion(/*cols=*/4);
  RowCollector seriesVelocity(/*cols=*/3);
  RowCollector seriesReference(/*cols=*/3);
  RowCollector seriesSolveRotationGap(/*cols=*/2);
  double nextSeriesTime = -std::numeric_limits<double>::infinity();
  const double seriesPeriod = options_.timeSeriesRate > 0.0 ? 1.0 / options_.timeSeriesRate : std::numeric_limits<double>::infinity();

  while (true) {
    const robot::model::RobotState& state = sim.getRobotState();
    const double time = state.getTime();
    const vector4_t baseQuaternion = state.getRootRotationLocalToWorldFrame().coeffs();

    // A fall ends the run. The gantry holds the robot until it is released; a reset by the simulator counts at any time.
    std::string fallReason;
    if (sim.resetEpoch() != initialResetEpoch) {
      fallReason = "the simulator reset the robot";
    } else if (phase == Phase::kStanding || phase == Phase::kCommands) {
      const double tilt = tiltVector(baseQuaternion).norm();
      const double height = state.getRootPositionInWorldFrame().z();
      if (tilt > maxTilt) {
        fallReason = absl::StrCat("the base tilted ", tilt, " rad, past ", maxTilt, " rad");
      } else if (height < 0.5 * initialHeight) {
        fallReason = absl::StrCat("the base dropped to ", height, " m, below half its initial ", initialHeight, " m");
      }
    }
    if (!fallReason.empty()) {
      metrics.setFall(time, fallReason);
      break;
    }
    // The scenario's commands from the cycle the standing time is over in. The GUI's message is no FSM command (the robot
    // process never sees it; the MPC node takes it in whenever it arrives), so the operator's sequence below does not
    // delay it: the commands start standingTime after the gantry let go, as in M0.
    if (phase == Phase::kStanding && time - phaseStart >= scenario.standingTime - kTimeTolerance) phase = Phase::kCommands;
    if (phase == Phase::kCommands && !commandStartTime.has_value()) {
      // The first cycle of the commands.
      commandStartTime = time;
      resetsAtCommandStart = driver->resetSupervisor().numResetsServed();
      fullResetsAtCommandStart = driver->resetSupervisor().numFullResetsServed();
    }
    if (phase == Phase::kCommands && time - *commandStartTime >= commandDuration - kTimeTolerance) break;
    const bool inCommands = phase == Phase::kCommands;

    // The GUI's velocity message, which the MPC node takes in whenever it arrives and the next solve reads.
    const Eigen::Vector3d command = inCommands ? commandAt(scenario, time - *commandStartTime) : Eigen::Vector3d::Zero();
    const GuiVelocityCommand gui = toGuiVelocityCommand(command, driver->commandLimits(), driver->defaultPelvisHeight());
    if (inCommands) commandSaturated = commandSaturated || gui.saturated;
    driver->setGuiVelocityCommand(gui.message);

    // The control cycle in the order of RobotProcess::cycle(): the mode and the posture the operator's commands of the
    // previous cycle set, the action, the controller-side settings, then this cycle's commands, and the action last.
    // LINT.IfChange(robot_process_cycle)
    robot::model::RobotJointAction& action = sim.getRobotJointAction();
    controller.prepareCycle(mode, nominalPosture);
    controller.computeJointControlAction(state, action);

    // The solve due in this cycle, on its observation: what the MPC node's solver thread does meanwhile, paced and backed
    // off as that thread is (LockstepSolveSchedule).
    if (solves.isDue(time, driver->resetSupervisor().resetRequestedSinceLastFailure())) {
      const DriverSolveOutcome outcome = driver->solve();
      const std::optional<InitialStateGap> gap = outcome.status.ok() ? driver->lastSolveInitialStateGap() : std::nullopt;
      solves.onSolve(time, outcome.retryDelay);
      if (inCommands) {
        SolveSample solve;
        solve.time = time;
        solve.succeeded = outcome.status.ok();
        solve.wallTimeMs = outcome.wallTimeMs;
        solve.lqApproximationMs = outcome.lqApproximationMs;
        solve.solveQpMs = outcome.solveQpMs;
        solve.linesearchMs = outcome.linesearchMs;
        solve.computeControllerMs = outcome.computeControllerMs;
        if (gap.has_value()) {
          solve.initialStateRotationGap = gap->rotation;
          solve.initialStateGapNorm = gap->norm;
          seriesSolveRotationGap.append(Eigen::Vector2d(time, gap->rotation));
        }
        solve.quaternionNormDeviation = driver->maxQuaternionNormDeviation();
        metrics.addSolve(solve);
        if (options_.recordRobotStates && time - *commandStartTime >= options_.recordingDelay - kTimeTolerance &&
            result.recordedStates.records.size() < options_.maxRecordedStates) {
          result.recordedStates.records.push_back(recordRobotState(state, description, gui.message));
        }
      }
    }

    driver->applyControllerSideSettings(time);

    // The operator's sequence: the FSM commands of this cycle, which the robot process serves after the action and
    // before it applies the action (SimFsmBridge::processCommands(), then the fall recovery). The commands of the
    // scenario start at the top of the cycle, above.
    if (phase == Phase::kJointPd && time - phaseStart >= options_.timeline.jointPdSettleTime - kTimeTolerance) {
      // WB_MPC from the next cycle on.
      mode = std::string(control_mode::kWbMpc);
      phase = Phase::kEnteringMpc;
      phaseStart = time;
    } else if (phase == Phase::kEnteringMpc) {
      if (time - phaseStart > options_.timeline.maxEntryTime) {
        return absl::DeadlineExceededError(absl::StrCat("[LockstepClosedLoop] ", configuration_.name,
                                                        ": the entry into WB_MPC did not end within ", options_.timeline.maxEntryTime,
                                                        " s"));
      }
      // The controller arms its hold in the first cycle in WB_MPC, after the one that commanded it; from that cycle on it
      // says when the entry is over.
      if (!entryEndTime.has_value() && time > phaseStart + kTimeTolerance && !driver->isEnteringMpc()) entryEndTime = time;
      if (entryEndTime.has_value() && time - *entryEndTime >= options_.timeline.entryMargin - kTimeTolerance) {
        // UNLOCK_GANTRY, as the robot process serves it: the gantry lets go, and the fall recovery, seeing it unlocked,
        // resets the MPC (SimFallRecovery::Cycle::gantryUnlocked).
        sim.unlockGantry();
        controller.requestMpcReset();
        releaseTime = time;
        phase = Phase::kStanding;
        phaseStart = time;
      }
    }
    // clang-format off
    // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotProcess.cpp:robot_process_cycle)
    // clang-format on

    if (inCommands) {
      Eigen::Quaterniond rotation;
      rotation.coeffs() = baseQuaternion;
      ControlCycleSample sample;
      sample.time = time;
      sample.basePosition = state.getRootPositionInWorldFrame();
      sample.baseQuaternion = baseQuaternion;
      sample.baseLinearVelocityWorld = rotation.normalized() * state.getRootLinearVelocityInLocalFrame();
      sample.baseAngularVelocityLocal = state.getRootAngularVelocityInLocalFrame();
      sample.referenceVelocity = driver->referenceVelocity();
      sample.referenceBaseHeight = driver->referenceBaseHeight(clampGuiVelocityMessage(gui.message)(2));
      sample.contactPositions = driver->contactPositions(driver->currentObservation());
      const std::vector<bool> contacts = sim.getGroundTruthContactFlags();
      for (size_t contact = 0; contact < 2 && contact < contacts.size(); ++contact) sample.contactFlags[contact] = contacts[contact];
      const std::vector<robot::joint_index_t>& joints = driver->mpcJointIndices();
      sample.jointTorques.resize(static_cast<Eigen::Index>(joints.size()));
      size_t nonFinite = countNonFinite(driver->currentObservation().state) + countNonFinite(driver->latestPolicyInput());
      for (size_t i = 0; i < joints.size(); ++i) {
        const robot::model::JointAction& jointAction = action.at(joints[i]).value();
        const double torque = jointAction.getTotalFeedbackTorque(state.getJointPosition(joints[i]), state.getJointVelocity(joints[i]));
        sample.jointTorques(static_cast<Eigen::Index>(i)) = torque;
        if (!std::isfinite(torque)) ++nonFinite;
      }
      sample.nonFiniteValues = nonFinite;
      sample.mpcHealthy = driver->resetSupervisor().isHealthy();
      metrics.addControlCycle(sample);

      if (time + kTimeTolerance >= nextSeriesTime) {
        nextSeriesTime = time + seriesPeriod - kTimeTolerance;
        seriesTime.append(Eigen::VectorXd::Constant(/*size=*/1, time));
        seriesPosition.append(sample.basePosition);
        seriesQuaternion.append(sample.baseQuaternion);
        seriesVelocity.append(Eigen::Vector3d(sample.baseLinearVelocityWorld.x(), sample.baseLinearVelocityWorld.y(),
                                              (rotation.normalized() * sample.baseAngularVelocityLocal).z()));
        seriesReference.append(sample.referenceVelocity);
      }
    }

    // The action reaches the simulator, which runs the control period.
    sim.applyJointAction();
    for (int step = 0; step < stepsPerCycle; ++step) sim.simulationStep();
    sim.updateInterfaceStateFromRobot();
  }

  ClosedLoopCounters counters;
  counters.resetsServed = static_cast<size_t>(driver->resetSupervisor().numResetsServed() - resetsAtCommandStart);
  counters.fullResetsServed = static_cast<size_t>(driver->resetSupervisor().numFullResetsServed() - fullResetsAtCommandStart);
  if (!commandStartTime.has_value()) counters.resetsServed = counters.fullResetsServed = 0;
  counters.simulatorResets = static_cast<size_t>(sim.resetEpoch() - initialResetEpoch);
  metrics.setCounters(counters);

  info.robot = configuration_.name;
  info.formulation = formulationName(configuration_.formulation);
  info.scenario = scenario.name;
  JsonValue& settings = info.settings;
  settings.set("mpc_frequency_hz", JsonValue::number(driver->mpcFrequency()));
  settings.set("mrt_frequency_hz", JsonValue::number(driver->mrtFrequency()));
  settings.set("simulation_time_step_s", JsonValue::number(timeStep));
  settings.set("control_period_s", JsonValue::number(controlPeriod));
  settings.set("solver_threads", JsonValue::number(static_cast<double>(driver->solverThreads())));
  settings.set("solve_latency", JsonValue::string("none: a solve's policy is in use from the next control cycle"));
  settings.set("joint_pd_settle_time_s", JsonValue::number(options_.timeline.jointPdSettleTime));
  settings.set("entry_end_time_s",
               JsonValue::optionalNumber(entryEndTime.has_value() ? std::optional<double>(*entryEndTime - startTime) : std::nullopt));
  settings.set("gantry_release_time_s",
               JsonValue::optionalNumber(releaseTime.has_value() ? std::optional<double>(*releaseTime - startTime) : std::nullopt));
  settings.set("standing_time_s", JsonValue::number(scenario.standingTime));
  settings.set(
      "command_start_time_s",
      JsonValue::optionalNumber(commandStartTime.has_value() ? std::optional<double>(*commandStartTime - startTime) : std::nullopt));
  settings.set("command_duration_s", JsonValue::number(commandDuration));
  JsonValue limits = JsonValue::array();
  for (Eigen::Index i = 0; i < 3; ++i) limits.append(JsonValue::number(driver->commandLimits()(i)));
  settings.set("command_limits", std::move(limits));
  settings.set("command_saturated", JsonValue::boolean(commandSaturated));
  settings.set("scenario_description", JsonValue::string(scenario.description));
  result.metrics = metrics.report(info);

  result.timeSeries.provenance.notes.emplace_back("robot", configuration_.name);
  result.timeSeries.provenance.notes.emplace_back("scenario", scenario.name);
  result.timeSeries.entries = {{time_series::kTime, seriesTime.matrix()},
                               {time_series::kBasePosition, seriesPosition.matrix()},
                               {time_series::kBaseQuaternion, seriesQuaternion.matrix()},
                               {time_series::kBaseVelocity, seriesVelocity.matrix()},
                               {time_series::kReferenceVelocity, seriesReference.matrix()},
                               {time_series::kSolveRotationGap, seriesSolveRotationGap.matrix()}};
  result.finalObservationState = driver->currentObservation().state;
  return result;
}

}  // namespace ocs2::humanoid::validation
