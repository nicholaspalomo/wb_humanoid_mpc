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

#include <map>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/mode_schedule_config.nproto.h"
#include "humanoid_mpc_config/mode_sequence_template_config.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {

/**
 * The gait pattern `config` as a ModeSequenceTemplate. It has at least one mode, each a name of modeNumber2String()
 * (FLY, RF, LF, STANCE), and one more switching time than modes, finite and strictly increasing: the templates that
 * describe a gait. The name is not read.
 *
 * @param config The pattern.
 * @param path Where it is in its file, for the errors: "gaits[walk]", "default_mode_sequence_template".
 * @return The template; InvalidArgument naming `path` and what is wrong.
 */
absl::StatusOr<ModeSequenceTemplate> modeSequenceTemplateFromConfig(const mpc_config::ModeSequenceTemplateConfig& config,
                                                                    absl::string_view path);

/**
 * The mode schedule `config` as a ModeSchedule: at least one mode, each a name of modeNumber2String(), and one event time
 * fewer than modes, each finite (the sizes ModeSchedule's constructor asserts).
 *
 * @param config The schedule.
 * @param path Where it is in its file, for the errors: "initial_mode_schedule".
 * @return The schedule; InvalidArgument naming `path` and what is wrong.
 */
absl::StatusOr<ModeSchedule> modeScheduleFromConfig(const mpc_config::ModeScheduleConfig& config, absl::string_view path);

/**
 * Every gait of the gait file by name, as the procedural motion manager selects them. gait_list names each gait of
 * gaits exactly once: a name it lists twice, a name no gait has, a gait it does not list, a gait without a name and two
 * gaits of one name are refused, as is a gait that modeSequenceTemplateFromConfig() refuses.
 *
 * @return The gaits; InvalidArgument naming the gait and what is wrong.
 */
absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaitMapFromConfig(const mpc_config::GaitFile& file);

/**
 * The mode schedule the gait schedule starts from: the reference file's initial_mode_schedule (modeScheduleFromConfig()),
 * with at least one event time, which GaitSchedule::getModeSchedule() needs; InvalidArgument otherwise.
 */
absl::StatusOr<ModeSchedule> initialModeScheduleFromConfig(const mpc_config::ReferenceFile& file);

/**
 * The gait the gait schedule repeats until the motion manager inserts another: the reference file's
 * default_mode_sequence_template (modeSequenceTemplateFromConfig()), which has no name; InvalidArgument when it has one.
 */
absl::StatusOr<ModeSequenceTemplate> defaultModeSequenceTemplateFromConfig(const mpc_config::ReferenceFile& file);

}  // namespace ocs2::humanoid
