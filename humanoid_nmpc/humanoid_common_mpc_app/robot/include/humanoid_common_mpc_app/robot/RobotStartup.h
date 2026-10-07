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

#include <functional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc_app/robot/RobotAppOptions.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/RobotStack.h"

/**
 * The start of a robot binary around its formulation's set-up (humanoid_nmpc/humanoid_common_mpc_app/robot/README.md,
 * "The configuration store"): the configuration directory, the set-up with its one retry on the bundled files, the
 * start, the boot confirmation and the run. For the main thread.
 */
namespace ocs2::humanoid {

/** How long the realtime loop runs before a start counts as booted (RobotConfigDirectory::confirmBoot()). */
// LINT.IfChange(boot_confirmation_time)
inline constexpr absl::Duration kBootConfirmationTime = absl::Seconds(10);
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:boot_confirmation_time, //tools/deploy/README.md:boot_confirmation_time)
// clang-format on

/** A formulation's set-up of the files of `directory` (RobotConfigDirectory::files()), up to but not including the start. */
using RobotSetUp = std::function<absl::StatusOr<RobotStack>(const RobotConfigDirectory& directory)>;

/** A set-up and start of the files of `directory`: the started stack, or why it did not start (destroyed). */
using RobotBringUp = std::function<absl::StatusOr<RobotStack>(const RobotConfigDirectory& directory)>;

/** The configuration directory of the binary's options: --config_store_dir and --config_seed over the bundled files. */
absl::StatusOr<RobotConfigDirectory> openRobotConfigDirectory(const RobotAppOptions& options);

/** model_settings.robot_name of the bundled task file: the robot every stored and saved task file must be. */
absl::StatusOr<std::string> bundledRobotName(const RobotConfigDirectory& directory);

/**
 * Brings the robot up on `directory` with `bringUp`, rejecting the stored copies only for a start they broke: a start
 * that fails on stored copies is tried on the bundle in place (RobotConfigDirectory::bundleInPlace(); the trial stack is
 * destroyed again). When the bundle starts, the stored copies are rejected (RobotConfigDirectory::rejectStoredCopies())
 * and the robot is brought up on the store once more; when it does not, the stored copies are kept, the boot marker is
 * withdrawn, and the first failure is returned. Logs each step.
 *
 * @return The started stack; the first start's error when the bundle fails too; the error of the start after the
 *         rejection, or of the rejection.
 */
absl::StatusOr<RobotStack> bringUpRobot(RobotConfigDirectory& directory, const RobotBringUp& bringUp);

/**
 * Runs a robot binary: opens the configuration directory, sets up (`setUpRobot`) and starts the bus and the process on
 * it with the fallback of bringUpRobot(), and waits until `shutdownRequested` or a fault
 * (RobotProcess::runUntilShutdown()). The boot is confirmed once the loop has run for kBootConfirmationTime, or when the
 * run ends without a fault, so that a start that dies earlier falls back to the bundled files at the next one.
 *
 * @return The start's error (bringUpRobot()), or the run's (the realtime loop's fault).
 */
absl::Status runRobot(const RobotAppOptions& options, const RobotSetUp& setUpRobot, const std::function<bool()>& shutdownRequested);

}  // namespace ocs2::humanoid
