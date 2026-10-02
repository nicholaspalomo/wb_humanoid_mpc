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

#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc_app/robot/DodgeballThrowParser.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "robot_ipc/Delivery.h"

namespace ocs2::humanoid {
namespace {

using DodgeballThrow = robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow;

constexpr double kLogPeriodSeconds = 5.0;
// LINT.IfChange(gantry_height_limits)
/// [m] The gantry heights the walking command's desired_pelvis_height is clamped to, as the ROS FSM bridge did.
constexpr double kMinGantryHeight = 0.2;
constexpr double kMaxGantryHeight = 1.5;
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:operator_mailbox)

ControllerSettingsUpdate settingsPrototype() {
  ControllerSettingsUpdate prototype;
  prototype.contactEstimatorName.reserve(64);
  return prototype;
}

}  // namespace

absl::StatusOr<std::unique_ptr<OperatorCommandMailbox>> OperatorCommandMailbox::Create(
    Config config, const robot::model::ContactEstimatorRegistry& contactEstimators, Hooks hooks) {
  if (config.jointNames.size() != config.initialNominalPositions.size()) {
    return absl::InvalidArgumentError(absl::StrCat("OperatorCommandMailbox: ", config.initialNominalPositions.size(),
                                                   " initial joint positions for ", config.jointNames.size(), " joints"));
  }
  if (config.commandQueueCapacity == 0 || config.dodgeballQueueCapacity == 0) {
    return absl::InvalidArgumentError("OperatorCommandMailbox: the queue capacities must be positive");
  }
  return std::unique_ptr<OperatorCommandMailbox>(new OperatorCommandMailbox(std::move(config), contactEstimators, std::move(hooks)));
}

OperatorCommandMailbox::OperatorCommandMailbox(Config config, const robot::model::ContactEstimatorRegistry& contactEstimators, Hooks hooks)
    : config_(std::move(config)),
      contactEstimatorRegistry_(contactEstimators),
      hooks_(std::move(hooks)),
      fsmCommands_(config_.commandQueueCapacity),
      dodgeballs_(config_.dodgeballQueueCapacity),
      controllerSettings_(config_.commandQueueCapacity, settingsPrototype()),
      nominalPosture_(config_.initialNominalPositions),
      desiredGantryHeight_(std::numeric_limits<double>::quiet_NaN()),
      ioNominalPositions_(config_.initialNominalPositions) {
  for (std::size_t joint = 0; joint < config_.jointNames.size(); ++joint) {
    jointIndexByName_.emplace(config_.jointNames[joint], joint);
  }
}

absl::Status OperatorCommandMailbox::registerOnBus(robot::ipc::Bus& bus) {
  absl::Status status =
      bus.subscribe<humanoid_mpc_msgs::FsmCommand>(ipc::topics::kOperatorFsmCommand, robot::ipc::Delivery::kAll,
                                                   [this](const humanoid_mpc_msgs::FsmCommand& message) { onFsmCommand(message); });
  if (status.ok()) {
    status =
        bus.subscribe<humanoid_mpc_msgs::JointTargets>(ipc::topics::kOperatorJointTargets, robot::ipc::Delivery::kLatest,
                                                       [this](const humanoid_mpc_msgs::JointTargets& message) { onJointTargets(message); });
  }
  if (status.ok()) {
    status = bus.subscribe<humanoid_mpc_msgs::YamlDocument>(
        ipc::topics::kOperatorDodgeballThrow, robot::ipc::Delivery::kAll,
        [this](const humanoid_mpc_msgs::YamlDocument& message) { onDodgeballThrow(message); });
  }
  if (status.ok()) {
    status = bus.subscribe<humanoid_mpc_msgs::WalkingVelocityCommand>(
        ipc::topics::kOperatorWalkingVelocityCommand, robot::ipc::Delivery::kLatest,
        [this](const humanoid_mpc_msgs::WalkingVelocityCommand& message) { onWalkingVelocityCommand(message); });
  }
  if (status.ok()) {
    status = bus.subscribe<humanoid_mpc_msgs::YamlDocument>(
        ipc::topics::kOperatorMpcParameters, robot::ipc::Delivery::kLatest,
        [this](const humanoid_mpc_msgs::YamlDocument& message) { onMpcParameters(message); });
  }
  if (status.ok()) {
    status = bus.subscribe<humanoid_mpc_msgs::YamlDocument>(ipc::topics::kOperatorPdGains, robot::ipc::Delivery::kLatest,
                                                            [this](const humanoid_mpc_msgs::YamlDocument& message) { onPdGains(message); });
  }
  return status;
}

// ---------------------------------------------------------------------------------------------------------------------
// The realtime thread
// ---------------------------------------------------------------------------------------------------------------------

bool OperatorCommandMailbox::takeFsmCommand(FsmCommandEvent& command) {
  return fsmCommands_.tryPop(command);
}

bool OperatorCommandMailbox::takeDodgeballThrow(DodgeballThrow& throwCommand) {
  bool took = false;
  while (dodgeballs_.tryPop(throwCommand)) {
    took = true;
  }
  return took;
}

bool OperatorCommandMailbox::takeNominalPosture(std::vector<double>& nominalPositions) {
  if (!nominalPosture_.acquireRead()) {
    return false;
  }
  // Both have one entry per joint, so the assignment copies into the storage nominalPositions already has.
  nominalPositions = nominalPosture_.readSlot();
  return true;
}

std::optional<double> OperatorCommandMailbox::desiredGantryHeight() const {
  const double height = desiredGantryHeight_.load(std::memory_order_acquire);
  if (std::isnan(height)) {
    return std::nullopt;
  }
  return height;
}

// ---------------------------------------------------------------------------------------------------------------------
// The IO thread
// ---------------------------------------------------------------------------------------------------------------------

void OperatorCommandMailbox::onFsmCommand(const humanoid_mpc_msgs::FsmCommand& message) {
  if (message.command().empty()) {
    return;  // no command, as the ROS FSM bridge treated an empty string
  }
  if (message.sequence() != 0 && haveFsmSequence_ && message.sequence() == lastFsmSequence_) {
    fsmCommandsRepeated_.fetch_add(1);
    return;
  }
  const std::optional<FsmCommandKind> kind = parseFsmCommand(message.command());
  if (!kind.has_value()) {
    fsmCommandsUnknown_.fetch_add(1);
    LOG(WARNING) << "[OperatorCommandMailbox] Ignoring the unknown FSM command '" << message.command()
                 << "' (a control mode name, LOCK_GANTRY or UNLOCK_GANTRY).";
    return;
  }
  if (message.sequence() != 0) {
    lastFsmSequence_ = message.sequence();
    haveFsmSequence_ = true;
  }
  if (fsmCommands_.tryPush(makeFsmCommandEvent(*kind, message.command()))) {
    fsmCommandsQueued_.fetch_add(1);
  } else {
    LOG(ERROR) << "[OperatorCommandMailbox] The realtime thread has " << fsmCommands_.capacity()
               << " FSM commands waiting already; dropping '" << message.command() << "'.";
  }
}

void OperatorCommandMailbox::onJointTargets(const humanoid_mpc_msgs::JointTargets& message) {
  bool changed = false;
  for (const google::protobuf::Map<std::string, double>::value_type& entry : message.positions()) {
    const absl::flat_hash_map<std::string, std::size_t>::const_iterator joint = jointIndexByName_.find(entry.first);
    if (joint == jointIndexByName_.end()) {
      continue;  // a joint of another robot, as the ROS subscriber skipped it
    }
    if (!std::isfinite(entry.second)) {
      jointTargetsRejected_.fetch_add(1);
      LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
          << "[OperatorCommandMailbox] Ignoring the joint target " << entry.second << " of " << entry.first << ": not a finite number.";
      continue;
    }
    ioNominalPositions_[joint->second] = entry.second;
    changed = true;
  }
  if (!changed) {
    return;
  }
  nominalPosture_.writeSlot() = ioNominalPositions_;
  nominalPosture_.publishWrite();
  jointTargetsApplied_.fetch_add(1);
}

void OperatorCommandMailbox::onDodgeballThrow(const humanoid_mpc_msgs::YamlDocument& message) {
  // All of the reading and the validation is in parseDodgeballThrow, which is unit-tested; see it for what is read and
  // what is rejected.
  const absl::StatusOr<DodgeballThrow> command = parseDodgeballThrow(message.yaml());
  if (!command.ok()) {
    dodgeballsRejected_.fetch_add(1);
    LOG(WARNING) << command.status().message();
    return;
  }
  if (dodgeballs_.tryPush(*command)) {
    dodgeballsQueued_.fetch_add(1);
  }
}

void OperatorCommandMailbox::onWalkingVelocityCommand(const humanoid_mpc_msgs::WalkingVelocityCommand& message) {
  if (std::isfinite(message.desired_pelvis_height())) {
    desiredGantryHeight_.store(std::clamp(message.desired_pelvis_height(), kMinGantryHeight, kMaxGantryHeight), std::memory_order_release);
  }
  if (hooks_.walkingVelocityCommand) {
    hooks_.walkingVelocityCommand(message);
  }
}

void OperatorCommandMailbox::onMpcParameters(const humanoid_mpc_msgs::YamlDocument& message) {
  const absl::StatusOr<ControllerSideSettings> settings = parseControllerSideSettings(message.yaml());
  if (!settings.ok()) {
    controllerSettingsRejected_.fetch_add(1);
    LOG(WARNING) << "[OperatorCommandMailbox] Ignoring the controller-side keys of operator/mpc_parameters: "
                 << settings.status().message();
  } else {
    postControllerSettings(*settings, "operator/mpc_parameters");
  }
  if (hooks_.mpcParameters) {
    hooks_.mpcParameters(message);
  }
}

void OperatorCommandMailbox::onPdGains(const humanoid_mpc_msgs::YamlDocument& message) {
  pdGainsDocuments_.fetch_add(1);
  if (hooks_.pdGainsYaml) {
    // The controller logs a document it refuses and keeps its gains.
    hooks_.pdGainsYaml(message.yaml()).IgnoreError();
  }
}

void OperatorCommandMailbox::postControllerSettings(const ControllerSideSettings& settings, absl::string_view source) {
  for (const std::string& problem : settings.problems) {
    LOG(WARNING) << "[OperatorCommandMailbox] " << source << ": " << problem;
  }
  if (settings.empty()) {
    return;
  }
  std::shared_ptr<robot::model::ContactEstimator> estimator;
  std::string estimatorName;
  if (settings.contactEstimator.has_value()) {
    estimatorName = robot::model::ContactEstimatorRegistry::canonicalName(*settings.contactEstimator);
    absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> resolved = contactEstimator(estimatorName);
    if (resolved.ok()) {
      estimator = *std::move(resolved);
    } else {
      controllerSettingsRejected_.fetch_add(1);
      LOG(ERROR) << "[OperatorCommandMailbox] Unknown contactEstimator '" << *settings.contactEstimator << "' in " << source
                 << "; keeping the estimator in use. " << resolved.status().message();
    }
  }
  if (estimator == nullptr && !settings.contactWrenchGate.has_value()) {
    return;
  }
  const bool pushed = controllerSettings_.tryPushInPlace([&](ControllerSettingsUpdate& slot) {
    slot.hasContactEstimator = estimator != nullptr;
    slot.contactEstimatorName = estimatorName;
    slot.contactEstimator = estimator;
    slot.hasContactWrenchGate = settings.contactWrenchGate.has_value();
    if (settings.contactWrenchGate.has_value()) slot.contactWrenchGate = *settings.contactWrenchGate;
  });
  if (pushed) {
    controllerSettingsQueued_.fetch_add(1);
  } else {
    LOG(ERROR) << "[OperatorCommandMailbox] The realtime thread has " << controllerSettings_.capacity()
               << " controller settings waiting already; dropping those of " << source << ".";
  }
}

absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> OperatorCommandMailbox::contactEstimator(const std::string& name) {
  const std::string canonical = robot::model::ContactEstimatorRegistry::canonicalName(name);
  absl::MutexLock lock(estimatorsMutex_);
  const absl::flat_hash_map<std::string, std::shared_ptr<robot::model::ContactEstimator>>::const_iterator cached =
      estimators_.find(canonical);
  if (cached != estimators_.end()) {
    return cached->second;
  }
  if (!contactEstimatorRegistry_.has(canonical)) {
    return absl::NotFoundError(
        absl::StrCat("There is no contact estimator '", name, "'. Available: ", contactEstimatorRegistry_.availableNames(), "."));
  }
  std::shared_ptr<robot::model::ContactEstimator> estimator = contactEstimatorRegistry_.create(canonical);
  estimators_.emplace(canonical, estimator);
  return estimator;
}

OperatorCommandMailbox::Statistics OperatorCommandMailbox::statistics() const {
  Statistics statistics;
  statistics.fsmCommandsQueued = fsmCommandsQueued_.load();
  statistics.fsmCommandsRepeated = fsmCommandsRepeated_.load();
  statistics.fsmCommandsUnknown = fsmCommandsUnknown_.load();
  statistics.jointTargetsApplied = jointTargetsApplied_.load();
  statistics.jointTargetsRejected = jointTargetsRejected_.load();
  statistics.dodgeballsQueued = dodgeballsQueued_.load();
  statistics.dodgeballsRejected = dodgeballsRejected_.load();
  statistics.controllerSettingsQueued = controllerSettingsQueued_.load();
  statistics.controllerSettingsRejected = controllerSettingsRejected_.load();
  statistics.pdGainsDocuments = pdGainsDocuments_.load();
  statistics.queueDrops = fsmCommands_.droppedCount() + dodgeballs_.droppedCount() + controllerSettings_.droppedCount();
  return statistics;
}

}  // namespace ocs2::humanoid
