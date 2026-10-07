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

#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"

#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/mode_schedule_config.nproto.h"
#include "humanoid_mpc_config/mode_sequence_template_config.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {
namespace {

/** The mode numbers of the mode names `names`, at least one, the field `path`.mode_sequence. */
absl::StatusOr<std::vector<size_t>> modeNumbers(const std::vector<std::string>& names, absl::string_view path) {
  if (names.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(path, ".mode_sequence is empty; it needs at least one mode"));
  }
  std::vector<size_t> modes;
  modes.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    const absl::StatusOr<size_t> mode = parseModeNumber(names[i]);
    if (!mode.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(path, ".mode_sequence[", i, "]: ", mode.status().message()));
    }
    modes.push_back(*mode);
  }
  return modes;
}

/** The times `times`, the field `path`.`field`, each a finite number. */
absl::StatusOr<std::vector<scalar_t>> finiteTimes(const Eigen::VectorXd& times, absl::string_view path, absl::string_view field) {
  std::vector<scalar_t> values(times.data(), times.data() + times.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      return absl::InvalidArgumentError(absl::StrCat(path, ".", field, "[", i, "] is ", values[i], "; it must be a finite number"));
    }
  }
  return values;
}

}  // namespace

absl::StatusOr<ModeSequenceTemplate> modeSequenceTemplateFromConfig(const mpc_config::ModeSequenceTemplateConfig& config,
                                                                    absl::string_view path) {
  ASSIGN_OR_RETURN(std::vector<size_t> modes, modeNumbers(config.mode_sequence, path));
  ASSIGN_OR_RETURN(std::vector<scalar_t> switchingTimes, finiteTimes(config.switching_times, path, "switching_times"));
  // A time out of order would give a phase of negative duration, which nothing downstream checks for.
  if (switchingTimes.size() != modes.size() + 1) {
    return absl::InvalidArgumentError(absl::StrCat(path, ".switching_times has ", switchingTimes.size(), " times for the ", modes.size(),
                                                   " modes of ", path, ".mode_sequence; it needs one more than there are modes"));
  }
  for (size_t i = 1; i < switchingTimes.size(); ++i) {
    if (!(switchingTimes[i] > switchingTimes[i - 1])) {
      return absl::InvalidArgumentError(absl::StrCat(path, ".switching_times [", absl::StrJoin(switchingTimes, ", "),
                                                     "] is not strictly increasing at ", i, " (", switchingTimes[i], " after ",
                                                     switchingTimes[i - 1], "), which gives mode ", i - 1, " a duration of zero or less"));
    }
  }
  return ModeSequenceTemplate(std::move(switchingTimes), std::move(modes));
}

absl::StatusOr<ModeSchedule> modeScheduleFromConfig(const mpc_config::ModeScheduleConfig& config, absl::string_view path) {
  ASSIGN_OR_RETURN(std::vector<size_t> modes, modeNumbers(config.mode_sequence, path));
  ASSIGN_OR_RETURN(std::vector<scalar_t> eventTimes, finiteTimes(config.event_times, path, "event_times"));
  if (eventTimes.size() + 1 != modes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(path, ".event_times has ", eventTimes.size(), " times for the ", modes.size(),
                                                   " modes of ", path, ".mode_sequence; it needs one fewer than there are modes"));
  }
  return ModeSchedule(std::move(eventTimes), std::move(modes));
}

absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaitMapFromConfig(const mpc_config::GaitFile& file) {
  absl::flat_hash_set<std::string> listed;
  for (const std::string& name : file.gait_list) {
    if (!listed.insert(name).second) {
      return absl::InvalidArgumentError(absl::StrCat("gait_list names the gait '", name, "' more than once"));
    }
  }
  std::map<std::string, ModeSequenceTemplate> gaits;
  for (size_t i = 0; i < file.gaits.size(); ++i) {
    const mpc_config::ModeSequenceTemplateConfig& gait = file.gaits[i];
    if (gait.name.empty()) {
      return absl::InvalidArgumentError(absl::StrCat("gaits[", i, "] has no name"));
    }
    const std::string path = absl::StrCat("gaits[", gait.name, "]");
    if (!listed.contains(gait.name)) {
      return absl::InvalidArgumentError(absl::StrCat(path, " is not named in gait_list; list it there, or delete it"));
    }
    if (gaits.contains(gait.name)) {
      return absl::InvalidArgumentError(absl::StrCat("gaits defines the gait '", gait.name, "' more than once"));
    }
    ASSIGN_OR_RETURN(ModeSequenceTemplate pattern, modeSequenceTemplateFromConfig(gait, path));
    gaits.emplace(gait.name, std::move(pattern));
  }
  for (const std::string& name : file.gait_list) {
    if (!gaits.contains(name)) {
      return absl::InvalidArgumentError(absl::StrCat("gait_list names the gait '", name, "', which gaits does not define"));
    }
  }
  return gaits;
}

absl::StatusOr<ModeSchedule> initialModeScheduleFromConfig(const mpc_config::ReferenceFile& file) {
  ASSIGN_OR_RETURN(ModeSchedule schedule, modeScheduleFromConfig(file.initial_mode_schedule, "initial_mode_schedule"));
  // GaitSchedule::getModeSchedule() drops the schedule's last mode, the stance it ends in, and tiles the gait from the
  // last event time on: a schedule of one mode, which has none, is undefined behavior there, so it is refused here
  // rather than at the first solve.
  if (schedule.eventTimes.empty()) {
    return absl::InvalidArgumentError(
        "initial_mode_schedule has one mode and no event time; the gait schedule needs at least two modes and the event time "
        "between them, e.g. mode_sequence: \"STANCE\" mode_sequence: \"STANCE\" event_times: 0.5");
  }
  return schedule;
}

absl::StatusOr<ModeSequenceTemplate> defaultModeSequenceTemplateFromConfig(const mpc_config::ReferenceFile& file) {
  if (!file.default_mode_sequence_template.name.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("default_mode_sequence_template.name is '", file.default_mode_sequence_template.name,
                                                   "', but the default template is no gait of the gait file and has no name"));
  }
  return modeSequenceTemplateFromConfig(file.default_mode_sequence_template, "default_mode_sequence_template");
}

}  // namespace ocs2::humanoid
