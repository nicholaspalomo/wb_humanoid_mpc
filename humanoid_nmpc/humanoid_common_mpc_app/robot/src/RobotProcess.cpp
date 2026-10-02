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

#include "humanoid_common_mpc_app/robot/RobotProcess.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/mrt/ControlMode.h"
#include "humanoid_common_mpc_app/robot/ControllerSideSettings.h"
#include "humanoid_common_mpc_app/robot/JointNamesByIndex.h"
#include "humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/fsm_state.nproto.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"

namespace ocs2::humanoid {
namespace {

/** How often the communication thread empties the realtime thread's mailboxes. */
constexpr absl::Duration kPumpPeriod = absl::Milliseconds(5);
/** robot/fsm_state is published on every change and again at this period. */
constexpr std::chrono::milliseconds kFsmStateRepublishPeriod{500};
/** The task file's controller-side keys are checked this often, as the in-process parameter updater checks them. */
constexpr absl::Duration kTaskFileCheckPeriod = absl::Seconds(1);
/** The ROS sims' default telemetry rate, when the task file names none [Hz]. */
constexpr double kDefaultTelemetryFrequency = 100.0;

std::chrono::nanoseconds periodOf(scalar_t frequency) {
  return std::chrono::nanoseconds(static_cast<std::int64_t>(std::llround(1e9 / frequency)));
}

RealtimeLoopConfig loopConfig(const RobotProcess::Config& config) {
  RealtimeLoopConfig loop;
  loop.period = periodOf(config.controlFrequency);
  loop.thread = defaultRealtimeThreadConfig();
  loop.thread.priority = config.realtimePriority;
  loop.thread.cores = config.realtimeCores;
  loop.reportingWindow = config.loopTimingWindow;
  return loop;
}

}  // namespace

absl::StatusOr<std::unique_ptr<RobotProcess>> RobotProcess::Create(robot::ipc::Bus& bus,
                                                                   RobotBackend& backend,
                                                                   RobotController& controller,
                                                                   const robot::model::ContactEstimatorRegistry& contactEstimators,
                                                                   Config config,
                                                                   Hooks hooks) {
  if (backend.simulator() == nullptr) {
    return absl::UnimplementedError(
        absl::StrCat("the robot backend '", backend.name(),
                     "' is not a simulator: the robot process's FSM bridge and fall recovery drive the "
                     "simulator's gantry and torque switch, and a hardware backend needs an FSM bridge of its own."));
  }
  if (!(config.controlFrequency > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("the control frequency (mpc.mrtDesiredFrequency) must be positive, got ", config.controlFrequency, " Hz"));
  }
  if (!config.initialState.has_value()) {
    return absl::InvalidArgumentError("the robot process needs the state the robot starts in");
  }
  if (bus.isRunning()) {
    return absl::FailedPreconditionError("RobotProcess: the bus is running; create the process before Bus::start()");
  }
  std::unique_ptr<RobotProcess> process(new RobotProcess(bus, backend, controller, std::move(config), std::move(hooks)));
  const robot::model::RobotDescription& description = backend.hardware().getRobotDescription();
  const Config& processConfig = process->config_;

  // The operator's commands, and the controller-side settings, with the task file's contact estimator installed.
  OperatorCommandMailbox::Config mailboxConfig;
  mailboxConfig.jointNames = jointNamesByIndex(description);
  mailboxConfig.initialNominalPositions.resize(description.getNumJoints(), 0.0);
  for (size_t joint = 0; joint < description.getNumJoints(); ++joint) {
    mailboxConfig.initialNominalPositions[joint] = processConfig.initialState->getJointPosition(joint);
  }
  OperatorCommandMailbox::Hooks mailboxHooks;
  mailboxHooks.pdGainsYaml = [&controller](absl::string_view yamlText) { return controller.setPdGainsYaml(yamlText); };
  ASSIGN_OR_RETURN(process->mailbox_, OperatorCommandMailbox::Create(std::move(mailboxConfig), contactEstimators, std::move(mailboxHooks)));
  absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> estimator =
      process->mailbox_->contactEstimator(processConfig.settings.contactEstimator);
  if (!estimator.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("contactEstimator of the task file: ", estimator.status().message()));
  }
  // The controller reports through the event log from now on: the realtime thread writes no log line.
  controller.setEventSink(&process->eventLog_);
  controller.setContactEstimator(*estimator);
  process->contactEstimatorName_ = robot::model::ContactEstimatorRegistry::canonicalName(processConfig.settings.contactEstimator);

  process->fsmBridge_ = std::make_unique<SimFsmBridge>(description, *processConfig.initialState, *process->mailbox_, process->fsmStates_,
                                                       &process->eventLog_);
  process->fallRecovery_ = std::make_unique<SimFallRecovery>(processConfig.settings.fallRecovery, *process->simulator_,
                                                             processConfig.restJointIndices, &process->eventLog_);

  // Telemetry: the sinks the task file names; none, no sampling at all.
  if (!processConfig.settings.telemetrySinks.empty()) {
    const TelemetrySinkRegistry sinkRegistry;
    const TelemetrySinkContext sinkContext{.bus = &bus};
    for (const std::string& name : processConfig.settings.telemetrySinks) {
      ASSIGN_OR_RETURN(std::unique_ptr<TelemetrySink> sink, sinkRegistry.create(name, sinkContext));
      process->sinks_.push_back(std::move(sink));
    }
    const double telemetryFrequency = processConfig.settings.telemetryFrequency.value_or(
        std::min(kDefaultTelemetryFrequency, static_cast<double>(processConfig.controlFrequency)));
    TelemetrySampler::Config samplerConfig;
    samplerConfig.jointNames = jointNamesByIndex(description);
    samplerConfig.decimation = std::max<size_t>(1, static_cast<size_t>(std::round(processConfig.controlFrequency / telemetryFrequency)));
    process->sampler_ = std::make_unique<TelemetrySampler>(std::move(samplerConfig));
    LOG(INFO) << "[RobotProcess] Telemetry at " << processConfig.controlFrequency / static_cast<scalar_t>(process->sampler_->decimation())
              << " Hz (decimation " << process->sampler_->decimation() << ") to " << processConfig.settings.telemetrySinks.size()
              << " sink(s).";
  } else {
    LOG(INFO) << "[RobotProcess] No telemetry: the task file lists no telemetrySinks.";
  }

  if (process->hooks_.takeViewerAnnotations) {
    process->annotator_ = std::make_unique<MujocoViewerAnnotator>(*process->simulator_);
  }
  if (!processConfig.taskFile.empty()) {
    RobotProcess* const processPtr = process.get();
    process->taskFileWatcher_ = std::make_unique<TaskFileWatcher>(
        processConfig.taskFile, [processPtr](const std::string& file) { processPtr->onTaskFileChanged(file); });
  }
  RETURN_IF_ERROR(process->registerOnBus());
  return process;
}

RobotProcess::RobotProcess(robot::ipc::Bus& bus, RobotBackend& backend, RobotController& controller, Config config, Hooks hooks)
    : bus_(bus),
      backend_(backend),
      simulator_(backend.simulator()),
      controller_(controller),
      config_(std::move(config)),
      hooks_(std::move(hooks)),
      loop_(loopConfig(config_)),
      plannedContactFlags_(N_CONTACTS, /*value=*/false),
      noContactFlags_() {
  // Longer than any mode or estimator name, so that the realtime thread's assignments never grow them.
  currentMode_.reserve(32);
  currentMode_.assign(control_mode::kZeroTorque.data(), control_mode::kZeroTorque.size());
  contactEstimatorName_.reserve(64);
  measuredContactForces_.fill(vector3_t::Zero());
}

RobotProcess::~RobotProcess() {
  stop();
}

absl::Status RobotProcess::registerOnBus() {
  RETURN_IF_ERROR(mailbox_->registerOnBus(bus_));
  RETURN_IF_ERROR(bus_.addPeriodicCallback(kPumpPeriod, [this]() { pumpRealtimeMailboxes(); }));
  // The gains file watcher, every pdGainsFileCheckInterval control periods (~1 Hz), as the controller used to run it
  // inside computeJointControlAction().
  const absl::Duration pdGainsPeriod =
      absl::Seconds(static_cast<double>(std::max<size_t>(1, config_.pdGainsFileCheckInterval)) / config_.controlFrequency);
  RETURN_IF_ERROR(bus_.addPeriodicCallback(pdGainsPeriod, [this]() { controller_.pollPdGainsFile(); }));
  if (taskFileWatcher_ != nullptr) {
    RETURN_IF_ERROR(bus_.addPeriodicCallback(kTaskFileCheckPeriod, [this]() { taskFileWatcher_->poll(); }));
  }
  return absl::OkStatus();
}

absl::Status RobotProcess::start() {
  if (started_) {
    return absl::FailedPreconditionError("RobotProcess::start(): the process was started before");
  }
  if (!bus_.isRunning()) {
    return absl::FailedPreconditionError("RobotProcess::start(): start the bus first, so that the MPC link and the mailboxes are served");
  }
  started_ = true;
  robot::model::RobotHWInterfaceBase& hardware = backend_.hardware();
  RETURN_IF_ERROR(backend_.initialize());
  hardware.updateInterfaceStateFromRobot();
  controller_.startMpc(hardware.getRobotState());

  // The robot spawns in zero-torque mode, passively held by the gantry; the MPC keeps receiving the state.
  const absl::Status backendStarted = backend_.start(config_.backendCores);
  if (!backendStarted.ok()) {
    LOG(WARNING) << "[RobotProcess] The robot backend '" << backend_.name() << "' started with a problem: " << backendStarted.message();
  }
  const absl::Status loopConfigured = loop_.start([this]() { cycle(); }, [this]() { onCycleFault(); });
  if (!loopConfigured.ok()) {
    LOG(WARNING) << "[RobotProcess] The realtime thread runs without part of its set-up: " << loopConfigured.message();
  }
  LOG(INFO) << "[RobotProcess] Control loop at " << config_.controlFrequency << " Hz on backend '" << backend_.name()
            << "' (realtime priority " << config_.realtimePriority << "). Zero-torque mode: robot spawned. Waiting for FSM command to "
            << "enable torques...";
  return absl::OkStatus();
}

void RobotProcess::stop() {
  if (stopped_) {
    return;
  }
  stopped_ = true;
  loop_.stop();
  // No action of a controller that no longer runs stays in force.
  if (started_) backend_.enterSafeState();
  // The bus's callbacks reach this process: they end here, before anything they use is destroyed.
  bus_.stop();
  // What the realtime thread reported in its last cycles.
  eventLog_.drainToLog();
}

absl::Status RobotProcess::runUntilShutdown(const std::function<bool()>& shutdownRequested) {
  while (!shutdownRequested() && !faulted()) {
    absl::SleepFor(absl::Milliseconds(50));
  }
  const bool cycleFailed = faulted();
  stop();
  if (cycleFailed) {
    return absl::InternalError(absl::StrCat("the realtime loop stopped: its cycle threw (", faultMessage(), "). The robot backend '",
                                            backend_.name(), "' was put in its safe state (zero torque)."));
  }
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------------------------------------------------
// The realtime thread
// ---------------------------------------------------------------------------------------------------------------------

void RobotProcess::cycle() {
  // LINT.IfChange(robot_process_cycle)
  robot::model::RobotHWInterfaceBase& hardware = backend_.hardware();
  hardware.updateInterfaceStateFromRobot();
  const robot::model::RobotState& robotState = hardware.getRobotState();

  // The FSM mode and the JOINT_PD posture reach the controller, which computes the action of every mode itself.
  fsmBridge_->applyJointTargetUpdates();
  controller_.prepareCycle(currentMode_, fsmBridge_->getNominalJointPositions());
  // In every mode, so that the MPC keeps planning from the robot as it is: in zero torque the action is computed but
  // not applied, which keeps the solver warm for the transition back into an active mode.
  controller_.computeJointControlAction(robotState, hardware.getRobotJointAction());

  // Contact timeline in the MuJoCo viewer: the contact state the executed policy plans for now, against the physics.
  const std::optional<contact_flag_t> planned = controller_.plannedContactFlags();
  if (planned.has_value()) {
    for (size_t contact = 0; contact < N_CONTACTS; ++contact) plannedContactFlags_[contact] = (*planned)[contact];
    simulator_->setTargetContactFlags(plannedContactFlags_);
  } else {
    simulator_->setTargetContactFlags(noContactFlags_);
  }

  if (sampler_ != nullptr) {
    backend_.readMeasuredContactForces(measuredContactForces_);
    sampler_->sample(robotState, hardware.getRobotJointAction(), currentMode_, controller_.measuredContactFlags(), measuredContactForces_);
  }

  applyControllerSettings();

  // The operator's commands first, then the fall recovery, which compares the gantry and the simulator's reset epoch
  // with the previous cycle: a catch, a reset the simulator made on its own thread and a LOCK_GANTRY are each one
  // discontinuity, after which the controller starts again from where the robot is (and holds it until then), in
  // JOINT_PD. While a caught robot settles on the gantry the recovery keeps it in JOINT_PD whatever was commanded.
  fsmBridge_->processCommands(currentMode_, *simulator_);
  const SimFallRecovery::Cycle recovery =
      fallRecovery_->update(robotState, fsmBridge_->getNominalJointPositions(), *simulator_, currentMode_);
  if (recovery.gantryUnlocked) {
    // Releasing the base invalidates the warm start; the policy in use carries the robot until the new one is in use.
    controller_.requestMpcReset();
    eventLog_.post(RealtimeEventCode::kGantryUnlockedMpcReset);
  }
  if (recovery.discontinuity) {
    controller_.requestMpcResetAndHold();
    // The remote control follows this state and re-centers its joysticks on every controller reset it counts, also
    // on one that changes neither the mode nor the gantry (a simulator reset while locked in JOINT_PD).
    fsmBridge_->publishControllerReset(currentMode_, simulator_->isGantryLocked());
  } else if (recovery.modeChanged) {
    // The remote control follows this state, and re-centers its joysticks on a transition into a passive mode.
    fsmBridge_->publishFsmState(currentMode_, simulator_->isGantryLocked());
  }

  // The action of this cycle, after its FSM command and fall recovery: when they switched the torques on, the backend
  // gets the action this cycle computed (ZERO_TORQUE's: no gain, no torque), not the one latched before they went off.
  if (backend_.acceptsJointAction()) {
    hardware.applyJointAction();
  }
  fsmBridge_->setMpcHealthy(controller_.isMpcHealthy(), currentMode_, simulator_->isGantryLocked());
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/LockstepClosedLoop.cpp:robot_process_cycle)
}

void RobotProcess::onCycleFault() {
  backend_.enterSafeState();
}

void RobotProcess::applyControllerSettings() {
  // LINT.IfChange(controller_side_updates)
  mailbox_->takeControllerSettings([this](const ControllerSettingsUpdate& update) {
    // Touch-down shaping of the contact wrenches, hot-reloaded through the GUI or the task file.
    if (update.hasContactWrenchGate) {
      const ContactWrenchGate::Config& current = controller_.contactWrenchGateConfig();
      if (update.contactWrenchGate.debounceTime != current.debounceTime || update.contactWrenchGate.rampTime != current.rampTime) {
        controller_.setContactWrenchGateConfig(update.contactWrenchGate);
      }
    }
    // The contact estimator, built by the communication thread and kept alive by the mailbox.
    if (update.hasContactEstimator && update.contactEstimatorName != contactEstimatorName_) {
      controller_.setContactEstimator(update.contactEstimator);
      contactEstimatorName_.assign(update.contactEstimatorName);
      eventLog_.post(RealtimeEventCode::kContactEstimatorSwapped, /*detail=*/0, contactEstimatorName_);
    }
  });
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:controller_side_updates)
}

// ---------------------------------------------------------------------------------------------------------------------
// The communication thread
// ---------------------------------------------------------------------------------------------------------------------

void RobotProcess::pumpRealtimeMailboxes() {
  eventLog_.drainToLog();
  if (sampler_ != nullptr) {
    sampler_->drain([this](const msgs::RobotStateSample& sample) {
      ToProto(sample, &telemetryMessage_);
      for (const std::unique_ptr<TelemetrySink>& sink : sinks_) sink->write(telemetryMessage_);
    });
  }
  publishFsmState(/*republish=*/false);
  if (loop_.takeTimingSnapshot(timingSnapshot_)) {
    publishLoopTiming(timingSnapshot_);
  }
  if (annotator_ != nullptr && hooks_.takeViewerAnnotations(annotations_)) {
    annotator_->apply(annotations_);
  }
}

void RobotProcess::publishFsmState(bool republish) {
  const bool fresh = fsmStates_.take(fsmState_);
  haveFsmState_ = haveFsmState_ || fresh;
  if (!haveFsmState_) {
    return;
  }
  const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  if (!fresh && !republish && now - lastFsmStatePublish_ < kFsmStateRepublishPeriod) {
    return;
  }
  ToProto(fsmState_, &fsmStateMessage_);
  const absl::Status published = bus_.publishFromIoThread(ipc::topics::kRobotFsmState, fsmStateMessage_);
  if (!published.ok()) {
    LOG_EVERY_N_SEC(WARNING, 5.0) << "[RobotProcess] Publishing robot/fsm_state failed: " << published.message();
  }
  lastFsmStatePublish_ = now;
}

void RobotProcess::publishLoopTiming(const robot::realtime::LoopTimingSnapshot& snapshot) {
  // LINT.IfChange(loop_timing_fields)
  loopTimingMessage_.set_target_period_s(snapshot.targetPeriodS);
  loopTimingMessage_.set_cycles(snapshot.totalCycles);
  loopTimingMessage_.set_mean_period_s(snapshot.meanPeriodS);
  loopTimingMessage_.set_max_period_s(snapshot.maxPeriodS);
  loopTimingMessage_.set_max_compute_time_s(snapshot.maxComputeTimeS);
  loopTimingMessage_.set_overruns(snapshot.totalOverruns);
  loopTimingMessage_.set_missed_periods(snapshot.totalMissedPeriods);
  loopTimingMessage_.set_max_lateness_s(snapshot.maxLatenessS);
  loopTimingMessage_.set_telemetry_samples_dropped(sampler_ != nullptr ? sampler_->dropped() : 0);
  loopTimingMessage_.set_stale_policies_dropped(0);
  loopTimingMessage_.set_policy_age_s(-1.0);
  if (hooks_.fillLinkStatistics) {
    hooks_.fillLinkStatistics(loopTimingMessage_);
  }
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_msgs/loop_timing.proto, //robot_runtime/robot_realtime/include/robot_realtime/LoopTimingSnapshot.h)
  // clang-format on
  const absl::Status published = bus_.publishFromIoThread(ipc::topics::kRobotLoopTiming, loopTimingMessage_);
  if (!published.ok()) {
    LOG_EVERY_N_SEC(WARNING, 5.0) << "[RobotProcess] Publishing robot/loop_timing failed: " << published.message();
  }
  if (snapshot.windowOverruns > 0 || snapshot.windowMissedPeriods > 0) {
    LOG_EVERY_N_SEC(WARNING, 10.0) << "[RobotProcess] The realtime loop overran its period of " << 1e3 * snapshot.targetPeriodS << " ms in "
                                   << snapshot.windowOverruns << " of " << snapshot.windowCycles << " cycles of the last "
                                   << snapshot.windowDurationS << " s and skipped " << snapshot.windowMissedPeriods
                                   << " periods (longest compute " << 1e3 * snapshot.maxComputeTimeS << " ms, latest wake-up "
                                   << 1e3 * snapshot.maxLatenessS << " ms late).";
  }
  if (eventLog_.dropped() > 0) {
    LOG_EVERY_N_SEC(WARNING, 10.0) << "[RobotProcess] " << eventLog_.dropped() << " reports of the realtime thread were dropped.";
  }
}

void RobotProcess::onTaskFileChanged(const std::string& file) {
  const absl::StatusOr<ControllerSideSettings> settings = loadControllerSideSettings(file);
  if (!settings.ok()) {
    LOG(WARNING) << "[RobotProcess] Not applying the controller-side keys of " << file << ": " << settings.status().message();
    return;
  }
  LOG(INFO) << "[RobotProcess] " << file << " changed; applying its controller-side keys (contactEstimator, contact_wrench_gate).";
  mailbox_->postControllerSettings(*settings, file);
}

}  // namespace ocs2::humanoid
