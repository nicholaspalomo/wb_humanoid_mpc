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

#include "legged_common_mpc/gait/RollingStepScheduler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/gait/MotionPhaseDefinition.h"
#include "legged_mpc_config/gait_file.nproto.h"
#include "legged_mpc_config/rolling_step_scheduler_config.nproto.h"

namespace ocs2::legged {
namespace {

// The index of the yaw rate in a [v_x, v_y, yaw rate] command.
constexpr Eigen::Index kYawRateIndex = 2;
// [s] Two groups due within this time of each other are due together, and the
// one leading the motion swings first.
constexpr scalar_t kSimultaneousDueTime = 1.0e-9;

/** InvalidArgument for `field` of rolling_lateral_debt unless `value` is finite
 * and positive. */
absl::Status requirePositive(absl::string_view field, scalar_t value) {
  if (std::isfinite(value) && value > 0.0) return absl::OkStatus();
  return absl::InvalidArgumentError(
      absl::StrCat("rolling_lateral_debt.", field, " is ", value,
                   "; it must be finite and positive"));
}

/** The contact index of every contact name of `modelSettings`. */
absl::flat_hash_map<std::string, size_t> contactIndicesOf(
    const ModelSettings& modelSettings) {
  absl::flat_hash_map<std::string, size_t> indices;
  for (size_t contact = 0; contact < modelSettings.numContacts(); ++contact) {
    indices.try_emplace(modelSettings.contactNames[contact], contact);
  }
  return indices;
}

/** The step groups of `config` as contact indices of `modelSettings`, every
 * contact of the set rolling and in exactly one group. */
absl::StatusOr<std::vector<std::vector<size_t>>> stepGroupsFromConfig(
    const mpc_config::RollingStepSchedulerConfig& config,
    const ModelSettings& modelSettings) {
  for (size_t contact = 0; contact < modelSettings.numContacts(); ++contact) {
    if (!modelSettings.isRollingContact(contact)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "gait_scheduler '", kRollingLateralDebtSchedulerName,
          "' steps the rolling contacts of model_settings.contacts_rolling, "
          "and the contact '",
          modelSettings.contactNames[contact], "' is not one"));
    }
  }
  const absl::flat_hash_map<std::string, size_t> indices =
      contactIndicesOf(modelSettings);
  std::vector<std::optional<size_t>> groupOfContact(
      modelSettings.numContacts());
  std::vector<std::vector<size_t>> groups;
  for (size_t group = 0; group < config.step_groups.size(); ++group) {
    const std::vector<std::string>& names = config.step_groups[group].contacts;
    const std::string path =
        absl::StrCat("rolling_lateral_debt.step_groups[", group, "]");
    if (names.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat(path, ".contacts is empty"));
    }
    std::vector<size_t> contacts;
    for (const std::string& name : names) {
      const absl::flat_hash_map<std::string, size_t>::const_iterator index =
          indices.find(name);
      if (index == indices.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat(path, ".contacts names '", name,
                         "', which is not a contact of the set (",
                         absl::StrJoin(modelSettings.contactNames, ", "), ")"));
      }
      if (groupOfContact[index->second].has_value()) {
        return absl::InvalidArgumentError(absl::StrCat(
            path, ".contacts names '", name, "', which step_groups[",
            *groupOfContact[index->second], "] already has"));
      }
      groupOfContact[index->second] = group;
      contacts.push_back(index->second);
    }
    groups.push_back(std::move(contacts));
  }
  for (size_t contact = 0; contact < modelSettings.numContacts(); ++contact) {
    if (!groupOfContact[contact].has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "rolling_lateral_debt.step_groups has no group with the contact '",
          modelSettings.contactNames[contact],
          "'; every rolling contact is in one group"));
    }
  }
  return groups;
}

/** The mean of the nominal places of the contacts of `geometry` below
 * `numContacts`, from the base in the heading frame; zero without one. */
vector2_t computeNomPosFeetmeanRtBaseEwrtHeadingXy(
    absl::Span<const RollingContactGeometry> geometry, size_t numContacts) {
  vector2_t sum = vector2_t::Zero();
  size_t count = 0;
  for (const RollingContactGeometry& wheel : geometry) {
    if (wheel.contactIndex >= numContacts) continue;
    sum += wheel.nomPosContactRtBaseEwrtHeadingXy;
    ++count;
  }
  if (count == 0) return sum;
  return sum / static_cast<scalar_t>(count);
}

}  // namespace

std::vector<std::string> gaitSchedulerNames() {
  return {kGaitSwitchingTableSchedulerName, kRollingLateralDebtSchedulerName};
}

absl::StatusOr<RollingStepSchedulerSettings> rollingStepTimingFromConfig(
    const mpc_config::RollingStepSchedulerConfig& config) {
  RollingStepSchedulerSettings settings;
  settings.maxLateralDebt = config.max_lateral_debt;
  settings.swingDuration = config.swing_duration;
  settings.minStanceDuration = config.min_stance_duration;
  settings.minLiftOffLead = config.min_lift_off_lead;
  settings.commitWindow = config.commit_window;
  RETURN_IF_ERROR(requirePositive("max_lateral_debt", settings.maxLateralDebt));
  RETURN_IF_ERROR(requirePositive("swing_duration", settings.swingDuration));
  RETURN_IF_ERROR(
      requirePositive("min_stance_duration", settings.minStanceDuration));
  RETURN_IF_ERROR(
      requirePositive("min_lift_off_lead", settings.minLiftOffLead));
  // A swing is planned at least min_lift_off_lead ahead, so a shorter window
  // never commits a swing that is already due, and the swing slides ahead of
  // every solve.
  if (!std::isfinite(settings.commitWindow) ||
      !(settings.commitWindow >= settings.minLiftOffLead)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "rolling_lateral_debt.commit_window is ", settings.commitWindow,
        "; it must be finite and at least min_lift_off_lead (",
        settings.minLiftOffLead, ")"));
  }
  return settings;
}

absl::StatusOr<GaitScheduler> gaitSchedulerFromName(absl::string_view name) {
  if (name == kGaitSwitchingTableSchedulerName) {
    return GaitScheduler::kGaitSwitchingTable;
  }
  if (name == kRollingLateralDebtSchedulerName) {
    return GaitScheduler::kRollingLateralDebt;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("unknown gait_scheduler '", name, "'; the names are ",
                   absl::StrJoin(gaitSchedulerNames(), ", ")));
}

absl::StatusOr<std::optional<RollingStepSchedulerSettings>>
rollingStepSchedulerSettingsFromConfig(const mpc_config::GaitFile& gaitFile,
                                       const ModelSettings& modelSettings) {
  ASSIGN_OR_RETURN(const GaitScheduler scheduler,
                   gaitSchedulerFromName(gaitFile.gait_scheduler));
  switch (scheduler) {
    case GaitScheduler::kGaitSwitchingTable:
      if (gaitFile.rolling_lateral_debt.has_value()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "the gait file has a rolling_lateral_debt block, which only "
            "gait_scheduler: \"",
            kRollingLateralDebtSchedulerName, "\" reads; gait_scheduler is '",
            gaitFile.gait_scheduler, "'"));
      }
      return std::nullopt;
    case GaitScheduler::kRollingLateralDebt:
      break;
  }
  if (scheduler != GaitScheduler::kRollingLateralDebt) {
    return absl::InternalError("unhandled gait scheduler");
  }
  if (!gaitFile.rolling_lateral_debt.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("gait_scheduler '", kRollingLateralDebtSchedulerName,
                     "' needs the gait file's rolling_lateral_debt block"));
  }
  ASSIGN_OR_RETURN(RollingStepSchedulerSettings settings,
                   rollingStepTimingFromConfig(*gaitFile.rolling_lateral_debt));
  ASSIGN_OR_RETURN(
      settings.stepGroups,
      stepGroupsFromConfig(*gaitFile.rolling_lateral_debt, modelSettings));
  return settings;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

RollingStepScheduler::RollingStepScheduler(
    RollingStepSchedulerSettings settings,
    size_t numContacts,
    size_t stanceMode)
    : settings_(std::move(settings)),
      numContacts_(numContacts),
      stanceMode_(stanceMode),
      debts_(numContacts, /*value=*/0.0),
      rates_(numContacts, /*value=*/0.0),
      leads_(numContacts, /*value=*/0.0) {
  for (const std::vector<size_t>& group : settings_.stepGroups) {
    size_t bits = 0;
    for (const size_t contact : group) {
      bits |= contactModeBit(contact, numContacts_);
    }
    groupModeBits_.push_back(bits);
  }
}

void RollingStepScheduler::reset() {
  swings_.clear();
  std::fill(debts_.begin(), debts_.end(), 0.0);
  std::fill(rates_.begin(), rates_.end(), 0.0);
  std::fill(leads_.begin(), leads_.end(), 0.0);
  hasUpdated_ = false;
  lastUpdateTime_ = 0.0;
}

scalar_t RollingStepScheduler::lateralDebtRate(
    const vector3_t& cmdTwistBaseRtWorldEwrtHeading,
    const RollingContactGeometry& geometry,
    const vector2_t& nomPosFeetmeanRtBaseEwrtHeadingXy) {
  const vector2_t nomPosContactRtFeetmeanEwrtHeadingXy =
      geometry.nomPosContactRtBaseEwrtHeadingXy -
      nomPosFeetmeanRtBaseEwrtHeadingXy;
  const scalar_t yawRate = cmdTwistBaseRtWorldEwrtHeading(kYawRateIndex);
  // The nominal place's velocity: v + omega e_z x p, the turn about the mean
  // of the contacts.
  const vector2_t nomLinvelContactRtWorldEwrtHeadingXy(
      cmdTwistBaseRtWorldEwrtHeading(0) -
          yawRate * nomPosContactRtFeetmeanEwrtHeadingXy.y(),
      cmdTwistBaseRtWorldEwrtHeading(1) +
          yawRate * nomPosContactRtFeetmeanEwrtHeadingXy.x());
  // The axle in the ground plane, e_z x t.
  const vector2_t& dir = geometry.dirWheelEwrtHeadingXy;
  const vector2_t axle(-dir.y(), dir.x());
  return nomLinvelContactRtWorldEwrtHeadingXy.dot(axle);
}

void RollingStepScheduler::advanceDebts(scalar_t initTime) {
  for (size_t group = 0; group < settings_.stepGroups.size(); ++group) {
    // The last touchdown of the group since the last update.
    std::optional<scalar_t> touchDown;
    for (const RollingStepSwing& swing : swings_) {
      if (swing.group == group && swing.touchDownTime > lastUpdateTime_ &&
          swing.touchDownTime <= initTime) {
        touchDown = swing.touchDownTime;
      }
    }
    for (const size_t contact : settings_.stepGroups[group]) {
      debts_[contact] =
          touchDown.has_value()
              ? rates_[contact] * (initTime - *touchDown)
              : debts_[contact] +
                    rates_[contact] * (initTime - lastUpdateTime_);
    }
  }
}

std::optional<scalar_t> RollingStepScheduler::dueTime(
    size_t group,
    scalar_t cursor,
    absl::Span<const PredictedDebt> debts) const {
  std::optional<scalar_t> due;
  for (const size_t contact : settings_.stepGroups[group]) {
    const scalar_t rate = rates_[contact];
    const scalar_t debtAtCursor =
        debts[contact].debt + rate * (cursor - debts[contact].time);
    std::optional<scalar_t> contactDue;
    if (std::abs(debtAtCursor) >= settings_.maxLateralDebt) {
      contactDue = cursor;
    } else if (rate != 0.0) {
      // The debt reaches the bound on the side it grows toward.
      const scalar_t bound = std::copysign(settings_.maxLateralDebt, rate);
      contactDue = cursor + (bound - debtAtCursor) / rate;
    }
    if (contactDue.has_value() && (!due.has_value() || *contactDue < *due)) {
      due = contactDue;
    }
  }
  return due;
}

scalar_t RollingStepScheduler::leadOf(size_t group) const {
  scalar_t lead = 0.0;
  for (const size_t contact : settings_.stepGroups[group]) {
    lead += leads_[contact];
  }
  return lead;
}

void RollingStepScheduler::planSwings(scalar_t initTime, scalar_t finalTime) {
  std::vector<PredictedDebt> debts(numContacts_);
  for (size_t contact = 0; contact < numContacts_; ++contact) {
    debts[contact] = PredictedDebt{.debt = debts_[contact], .time = initTime};
  }
  // The earliest lift-off: the lead after the solve, and the stance after
  // every committed swing, whose touchdowns zero their contacts' debts.
  scalar_t cursor = initTime + settings_.minLiftOffLead;
  for (const RollingStepSwing& swing : swings_) {
    cursor =
        std::max(cursor, swing.touchDownTime + settings_.minStanceDuration);
    if (swing.touchDownTime <= initTime) continue;
    for (const size_t contact : settings_.stepGroups[swing.group]) {
      debts[contact] = PredictedDebt{.debt = 0.0, .time = swing.touchDownTime};
    }
  }
  // Each swing moves the cursor on by at least the swing and the stance after
  // it, so the loop ends past the horizon.
  while (cursor <= finalTime) {
    std::optional<size_t> next;
    scalar_t nextDue = 0.0;
    for (size_t group = 0; group < settings_.stepGroups.size(); ++group) {
      const std::optional<scalar_t> due = dueTime(group, cursor, debts);
      if (!due.has_value()) continue;
      const bool earlier =
          !next.has_value() || *due < nextDue - kSimultaneousDueTime;
      const bool together =
          next.has_value() && std::abs(*due - nextDue) <= kSimultaneousDueTime;
      if (earlier || (together && leadOf(group) > leadOf(*next))) {
        next = group;
        nextDue = *due;
      }
    }
    if (!next.has_value() || nextDue > finalTime) break;
    const scalar_t touchDown = nextDue + settings_.swingDuration;
    swings_.push_back(RollingStepSwing{
        .group = *next,
        .liftOffTime = nextDue,
        .touchDownTime = touchDown,
        .committed = nextDue <= initTime + settings_.commitWindow});
    for (const size_t contact : settings_.stepGroups[*next]) {
      debts[contact] = PredictedDebt{.debt = 0.0, .time = touchDown};
    }
    cursor = touchDown + settings_.minStanceDuration;
  }
}

ModeSchedule RollingStepScheduler::modeSchedule(scalar_t finalTime) const {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence = {stanceMode_};
  scalar_t lastEvent = finalTime;
  for (const RollingStepSwing& swing : swings_) {
    eventTimes.push_back(swing.liftOffTime);
    modeSequence.push_back(stanceMode_ & ~groupModeBits_[swing.group]);
    eventTimes.push_back(swing.touchDownTime);
    modeSequence.push_back(stanceMode_);
    lastEvent = std::max(lastEvent, swing.touchDownTime);
  }
  // A last stance event after the horizon and every touchdown, from which
  // GaitSchedule::getModeSchedule() tiles its template.
  eventTimes.push_back(lastEvent + settings_.minStanceDuration);
  modeSequence.push_back(stanceMode_);
  return ModeSchedule(std::move(eventTimes), std::move(modeSequence));
}

ModeSchedule RollingStepScheduler::update(
    scalar_t initTime,
    scalar_t finalTime,
    const vector3_t& cmdTwistBaseRtWorldEwrtHeading,
    absl::Span<const RollingContactGeometry> geometry) {
  if (!hasUpdated_ || initTime < lastUpdateTime_) {
    reset();
    hasUpdated_ = true;
    lastUpdateTime_ = initTime;
  }
  advanceDebts(initTime);
  // The uncommitted swings are planned again, and the committed ones are kept
  // for a horizon after their touchdown, as long as the schedule reaches back.
  const scalar_t keepAfter = initTime - (finalTime - initTime);
  swings_.erase(std::remove_if(swings_.begin(), swings_.end(),
                               [keepAfter](const RollingStepSwing& swing) {
                                 return !swing.committed ||
                                        swing.touchDownTime < keepAfter;
                               }),
                swings_.end());
  std::fill(rates_.begin(), rates_.end(), 0.0);
  std::fill(leads_.begin(), leads_.end(), 0.0);
  const vector2_t nomPosFeetmeanRtBaseEwrtHeadingXy =
      computeNomPosFeetmeanRtBaseEwrtHeadingXy(geometry, numContacts_);
  for (const RollingContactGeometry& wheel : geometry) {
    if (wheel.contactIndex >= numContacts_) continue;
    const scalar_t rate = lateralDebtRate(cmdTwistBaseRtWorldEwrtHeading, wheel,
                                          nomPosFeetmeanRtBaseEwrtHeadingXy);
    rates_[wheel.contactIndex] = rate;
    if (rate == 0.0) continue;
    // The offset of the nominal place from the mean of the contacts along the
    // direction it moves, which is the axle's direction or its opposite.
    const vector2_t& dir = wheel.dirWheelEwrtHeadingXy;
    const vector2_t axle(-dir.y(), dir.x());
    leads_[wheel.contactIndex] =
        std::copysign(1.0, rate) * (wheel.nomPosContactRtBaseEwrtHeadingXy -
                                    nomPosFeetmeanRtBaseEwrtHeadingXy)
                                       .dot(axle);
  }
  planSwings(initTime, finalTime);
  lastUpdateTime_ = initTime;
  return modeSchedule(finalTime);
}

}  // namespace ocs2::legged
