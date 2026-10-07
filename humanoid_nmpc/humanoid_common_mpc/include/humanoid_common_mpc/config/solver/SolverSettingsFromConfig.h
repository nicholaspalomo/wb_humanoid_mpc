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
#include <vector>

#include "absl/status/statusor.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/rollout/RolloutSettings.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_mpc_config/mpc_settings_config.nproto.h"
#include "humanoid_mpc_config/rollout_settings_config.nproto.h"
#include "humanoid_mpc_config/sqp_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/**
 * The settings of the solver an MPC interface sets up from its task file: the SQP solver (multiple_shooting), the
 * rollouts (rollout), the MPC loop (mpc) and the interface's verbosity (interface.verbose).
 */
struct SolverSettings {
  sqp::Settings sqpSettings;
  rollout::Settings rolloutSettings;
  mpc::Settings mpcSettings;
  // Whether the MPC interface logs its settings as it loads them.
  bool verbose = false;
};

/** Every name multiple_shooting.integrator_type accepts: the SQP solver's sensitivity integrators, by OCS2's names. */
std::vector<std::string> sensitivityIntegratorNames();

/** Every name rollout.integrator_type accepts: the rollouts' integrators, by OCS2's names. */
std::vector<std::string> rolloutIntegratorNames();

/** Every name rollout.root_finder_type accepts: the state-triggered rollout's root-finding algorithms. */
std::vector<std::string> rootFinderNames();

/**
 * The SQP settings of a task file's multiple_shooting block. A field the file leaves out has the default of
 * sqp::Settings, which the schema repeats, so an empty block gives sqp::Settings{}; the HPIPM settings, which no file
 * sets, keep theirs.
 *
 * @param config The block.
 * @return InvalidArgument naming the field (multiple_shooting.<field>) when integrator_type names no sensitivity
 *         integrator (the message lists them), or when sqp_iteration, log_size or n_threads is negative.
 */
absl::StatusOr<sqp::Settings> toSqpSettings(const mpc_config::SqpSettingsConfig& config);

/**
 * The rollout settings of a task file's rollout block, with the defaults of rollout::Settings for what it leaves out.
 *
 * @param config The block.
 * @return InvalidArgument naming the field (rollout.<field>) when integrator_type names no integrator (the message
 *         lists them), when root_finder_type names no root finder (the message lists them), or when
 *         max_num_steps_per_second is negative.
 */
absl::StatusOr<rollout::Settings> toRolloutSettings(const mpc_config::RolloutSettingsConfig& config);

/** The MPC settings of a task file's mpc block, with the defaults of mpc::Settings for what it leaves out. */
mpc::Settings toMpcSettings(const mpc_config::MpcSettingsConfig& config);

/**
 * The solver settings of a task file: toSqpSettings(), toRolloutSettings() and toMpcSettings() of its blocks, and its
 * interface.verbose.
 *
 * @param task The task file.
 * @return The first error of the conversions.
 */
absl::StatusOr<SolverSettings> solverSettingsFromConfig(const mpc_config::TaskFile& task);

}  // namespace ocs2::humanoid
