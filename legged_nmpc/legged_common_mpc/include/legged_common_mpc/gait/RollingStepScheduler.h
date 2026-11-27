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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_mpc_config/gait_file.nproto.h"
#include "legged_mpc_config/rolling_step_scheduler_config.nproto.h"

namespace ocs2::legged {

// The schedulers of the gait file's gait_scheduler (gait_file.proto).
// LINT.IfChange(gait_scheduler_names)
inline constexpr char kGaitSwitchingTableSchedulerName[] =
    "gait_switching_table";
inline constexpr char kRollingLateralDebtSchedulerName[] =
    "rolling_lateral_debt";
// LINT.ThenChange(//legged_nmpc/legged_mpc_config/gait_file.proto:gait_scheduler_names)

/** What writes the mode schedule of the MPC (the gait file's gait_scheduler).
 */
enum class GaitScheduler {
  // ProceduralMpcMotionManager's gait switching table inserts gaits.
  kGaitSwitchingTable,
  // RollingStepScheduler plans the steps of the rolling contacts.
  kRollingLateralDebt,
};

/** Returns the names gait_scheduler accepts, the default first. Allocates. */
std::vector<std::string> gaitSchedulerNames();

/** Returns the scheduler named `name`, or InvalidArgument listing the names
 * (gaitSchedulerNames()). */
absl::StatusOr<GaitScheduler> gaitSchedulerFromName(absl::string_view name);

/**
 * Where a rolling contact stands relative to the base at the nominal posture
 * and which way its wheel rolls, both in the heading frame
 * (SwitchedModelReferenceManager::configureNominalStance()). Passive data.
 */
struct RollingContactGeometry {
  /** The contact, an index of ModelSettings::contactNames. */
  size_t contactIndex = 0;
  /** [m] The contact point relative to the base. */
  vector2_t nomPosContactRtBaseEwrtHeadingXy = vector2_t::Zero();
  /** The unit direction the wheel center moves for a positive wheel joint
   * velocity; zero for an axle that is close to vertical. */
  vector2_t dirWheelEwrtHeadingXy = vector2_t::Zero();
};

/**
 * The settings of RollingStepScheduler, read from the gait file's
 * rolling_lateral_debt block (rolling_step_scheduler_config.proto, which says
 * what each one means). Passive data.
 */
struct RollingStepSchedulerSettings {
  /** The contact indices of each step group, in the order of the block. */
  std::vector<std::vector<size_t>> stepGroups;
  scalar_t maxLateralDebt =
      mpc_config::RollingStepSchedulerConfig::kDefaultMaxLateralDebt;  // [m]
  scalar_t swingDuration =
      mpc_config::RollingStepSchedulerConfig::kDefaultSwingDuration;  // [s]
  scalar_t minStanceDuration =
      mpc_config::RollingStepSchedulerConfig::kDefaultMinStanceDuration;  // [s]
  scalar_t minLiftOffLead =
      mpc_config::RollingStepSchedulerConfig::kDefaultMinLiftOffLead;  // [s]
  scalar_t commitWindow =
      mpc_config::RollingStepSchedulerConfig::kDefaultCommitWindow;  // [s]
};

/**
 * Returns the durations and the debt bound of the rolling_lateral_debt block
 * `config`, without its step groups, which need the contact set
 * (rollingStepSchedulerSettingsFromConfig()).
 *
 * @return InvalidArgument for a duration or debt that is not finite or not
 * positive, and a commit_window that is negative or shorter than
 * min_lift_off_lead, which would keep every swing uncommitted.
 */
absl::StatusOr<RollingStepSchedulerSettings> rollingStepTimingFromConfig(
    const mpc_config::RollingStepSchedulerConfig& config);

/**
 * Returns the settings of the event-driven steps of `gaitFile` on the contact
 * set of `modelSettings`, or nullopt when its gait_scheduler is the gait
 * switching table.
 *
 * @return InvalidArgument for an unknown gait_scheduler (listing the names), a
 * rolling_lateral_debt block that the scheduler does not read or that
 * "rolling_lateral_debt" lacks, a contact set with a contact that is not
 * rolling, a group without contacts, a contact that is not of the set or is in
 * a second group, a rolling contact in no group, and the errors of
 * rollingStepTimingFromConfig().
 */
absl::StatusOr<std::optional<RollingStepSchedulerSettings>>
rollingStepSchedulerSettingsFromConfig(const mpc_config::GaitFile& gaitFile,
                                       const ModelSettings& modelSettings);

/** A swing of a step group of RollingStepScheduler. Passive data. */
struct RollingStepSwing {
  /** The step group, an index of RollingStepSchedulerSettings::stepGroups. */
  size_t group = 0;
  scalar_t liftOffTime = 0.0;    // [s]
  scalar_t touchDownTime = 0.0;  // [s]
  /** Whether later updates keep it; an uncommitted swing is planned again. */
  bool committed = false;
};

/**
 * Plans the steps of a robot on rolling contacts from the lateral debt of its
 * wheels, and writes them as the mode schedule of the MPC.
 *
 * A wheel's contact point may move only along the wheel's heading. The
 * commanded base twist moves the nominal place of each contact (where it stands
 * relative to the base at the nominal posture) at v + omega e_z x p_contact,
 * with p_contact from the mean of the contacts;
 * the part of that velocity along the wheel's axle is the rate of the wheel's
 * lateral debt. The debt is integrated from the commanded twist since the wheel
 * last landed, open loop. Each update predicts the debts over the horizon with
 * the current command, and where the debt of a contact reaches
 * maxLateralDebt its group swings for swingDuration, after which it lands at
 * its nominal foothold and its debt is zero again. The groups swing one at a
 * time, with minStanceDuration on every contact between two swings; of two
 * groups due at the same time the one further over the bound swings first, and
 * of two with the same debt the one leading the motion. A swing
 * that lifts off within commitWindow of an update is committed and never moves
 * again; the others are planned again at every update.
 *
 * The turn is about the mean of the contacts' nominal places, so a straight
 * drive, and on a robot whose axles share one line a turn in place, builds no
 * debt and never steps. Not thread-safe:
 * ProceduralMpcMotionManager calls it on the solver thread.
 */
class RollingStepScheduler {
 public:
  /**
   * The scheduler of `settings` (rollingStepSchedulerSettingsFromConfig()) on
   * a set of `numContacts` contacts whose stance mode is `stanceMode`
   * (GaitModeLayout).
   */
  RollingStepScheduler(RollingStepSchedulerSettings settings,
                       size_t numContacts,
                       size_t stanceMode);

  RollingStepScheduler(const RollingStepScheduler&) = default;
  RollingStepScheduler& operator=(const RollingStepScheduler&) = default;
  RollingStepScheduler(RollingStepScheduler&&) noexcept = default;
  RollingStepScheduler& operator=(RollingStepScheduler&&) noexcept = default;
  ~RollingStepScheduler() = default;

  /**
   * Advances the debts to `initTime` with the command of the last update, plans
   * the swings of the horizon [initTime, finalTime] with the command
   * `cmdTwistBaseRtWorldEwrtHeading` ([v_x, v_y, yaw rate] in the heading
   * frame) on the nominal geometry `geometry` (a contact without an entry
   * builds no debt), and returns the mode schedule of the kept, committed and
   * planned swings, every other mode the stance mode. A time before the last
   * update's starts afresh, as after reset().
   */
  ModeSchedule update(scalar_t initTime,
                      scalar_t finalTime,
                      const vector3_t& cmdTwistBaseRtWorldEwrtHeading,
                      absl::Span<const RollingContactGeometry> geometry);

  /** Forgets every swing and debt; the next update() starts at rest. */
  void reset();

  /** The swings of the last update(), in increasing lift-off time. */
  absl::Span<const RollingStepSwing> swings() const { return swings_; }

  /** [m] The lateral debt of each contact at the last update's initial time, in
   * contact order. */
  absl::Span<const scalar_t> lateralDebts() const { return debts_; }

  /**
   * Returns [m/s] the rate of the lateral debt of the contact of `geometry`
   * under the command `cmdTwistBaseRtWorldEwrtHeading` ([v_x, v_y, yaw rate]):
   * the velocity of its nominal place along the wheel's axle, signed along e_z
   * x dirWheelEwrtHeadingXy, with the yaw about
   * `nomPosFeetmeanRtBaseEwrtHeadingXy`, the mean of the contacts' nominal
   * places. update() turns about that mean rather than the base: a biped whose
   * axle passes a little in front of the base turns about its axle on its
   * wheels, and the base circles the axle by the offset.
   */
  static scalar_t lateralDebtRate(
      const vector3_t& cmdTwistBaseRtWorldEwrtHeading,
      const RollingContactGeometry& geometry,
      const vector2_t& nomPosFeetmeanRtBaseEwrtHeadingXy);

 private:
  /** A contact's predicted debt: `debt` at `time`, growing at its rate. */
  struct PredictedDebt {
    scalar_t debt = 0.0;  // [m]
    scalar_t time = 0.0;  // [s]
  };

  /** Moves the debts from the last update's time to `initTime` at the last
   * update's rates, resetting each at a touchdown in between. */
  void advanceDebts(scalar_t initTime);

  /** Plans the swings of the horizon after the committed ones. */
  void planSwings(scalar_t initTime, scalar_t finalTime);

  /** Returns the first time at or after `cursor` at which a contact of `group`
   * has a debt of maxLateralDebt; nullopt if none ever does. */
  std::optional<scalar_t> dueTime(size_t group,
                                  scalar_t cursor,
                                  absl::Span<const PredictedDebt> debts) const;

  /** Returns [m] how far the largest debt of `group`'s contacts is above
   * maxLateralDebt at `cursor`; 0 for a group that is not overdue. */
  scalar_t overdueDebt(size_t group,
                       scalar_t cursor,
                       absl::Span<const PredictedDebt> debts) const;

  /** Returns how far the contacts of `group` stand ahead in the direction their
   * nominal places move: the sum of each contact's offset from the base along
   * that direction. */
  scalar_t leadOf(size_t group) const;

  /** Returns the mode schedule of swings_ after `finalTime`'s horizon. */
  ModeSchedule modeSchedule(scalar_t finalTime) const;

  RollingStepSchedulerSettings settings_;
  size_t numContacts_ = 0;
  size_t stanceMode_ = 0;
  // The mode bits of each step group (contactModeBit()).
  std::vector<size_t> groupModeBits_;
  // In increasing lift-off time, none overlapping.
  std::vector<RollingStepSwing> swings_;
  // [m] Per contact, at lastUpdateTime_.
  std::vector<scalar_t> debts_;
  // [m/s] Per contact, of the last update's command.
  std::vector<scalar_t> rates_;
  // [m] Per contact, the nominal place relative to the base along the
  // direction it moves under the last update's command (leadOf()).
  std::vector<scalar_t> leads_;
  bool hasUpdated_ = false;
  scalar_t lastUpdateTime_ = 0.0;
};

}  // namespace ocs2::legged
