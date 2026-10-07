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

#include <string>

#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid::config_dump {

/** The configuration files of an MPC that its typed conversions read, parsed. The contact planner's is not one of them. */
struct ConversionInputs {
  mpc_config::TaskFile task;
  mpc_config::ReferenceFile reference;
  mpc_config::JointPdGainsFile pdGains;
  mpc_config::GaitFile gait;
};

/**
 * Converts `inputs` with every typed conversion of the configuration that the start-up of `mpc` and of its robot
 * process makes (the conversions of the packages' config/ directories), and dumps what each one makes,
 * in the format of ValueDump: the model settings from `urdfFile`, the solver, the formulation's lists, every weight
 * matrix and value on the MPC's state and input layout, the contacts, constraints, costs and their barriers, the swing
 * trajectories, the locomotion heuristics, the reference, the gaits, the PD gains over fixed defaults, and the robot
 * process's, the controller side's, the visualization's and the keyboard teleoperation's settings. A conversion that
 * refuses its input is dumped as its error, and what depends on it is left out.
 *
 * Builds no Pinocchio model but the URDF's ModelSettings and no CppAD library, so it is cheap enough to run once per
 * value of a file (//tools/config_dump:every_field_is_consumed_test). It keeps no state of its own.
 */
std::string dumpConversions(const ConversionInputs& inputs, const std::string& urdfFile, StateInputLayout::Mpc mpc);

}  // namespace ocs2::humanoid::config_dump
