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

#include "absl/strings/string_view.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/rollout/RolloutSettings.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "tools/config_dump/ValueDump.h"

namespace ocs2::humanoid::config_dump {

// Every value of a settings struct the start-up of the stack builds, one line each, under the path `path`.

void dumpModelSettings(ValueDump& dump, absl::string_view path, const ModelSettings& settings);
void dumpSqpSettings(ValueDump& dump, absl::string_view path, const sqp::Settings& settings);
void dumpRolloutSettings(ValueDump& dump, absl::string_view path, const rollout::Settings& settings);
void dumpMpcSettings(ValueDump& dump, absl::string_view path, const mpc::Settings& settings);
void dumpSwingSettings(ValueDump& dump, absl::string_view path, const SwingTrajectoryPlanner::Config& settings);
void dumpModeSchedule(ValueDump& dump, absl::string_view path, const ModeSchedule& schedule);
void dumpGaitMap(ValueDump& dump, absl::string_view path, const std::map<std::string, ModeSequenceTemplate>& gaits);
void dumpTargetTrajectories(ValueDump& dump, absl::string_view path, const TargetTrajectories& targets);
void dumpJointPdGains(ValueDump& dump, absl::string_view path, const JointPdGains& gains);
void dumpRobotProcessSettings(ValueDump& dump, absl::string_view path, const RobotProcessSettings& settings);
void dumpVisualizationConfig(ValueDump& dump, absl::string_view path, const visualization::VisualizationConfig& config);
void dumpKeyboardCommandLimits(ValueDump& dump, absl::string_view path, const teleop::KeyboardCommandLimits& limits);

}  // namespace ocs2::humanoid::config_dump
