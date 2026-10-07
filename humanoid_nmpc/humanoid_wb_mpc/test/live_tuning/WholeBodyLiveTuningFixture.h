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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

/*
 * What the live-tuning tests of the whole-body MPC share: the G1 whole-body files, the variant of its task file whose
 * problem carries the terms the shipped one leaves out, the MPCs built from them, and the states they are evaluated at.
 * Every MPC of the test program tapes its CppAD libraries into one working directory made for the run, so that each
 * library is generated once and every later MPC and term loads it by its model name.
 */
namespace ocs2::humanoid::live_tuning_test {

/** The G1 whole-body files, as absolute paths into the runfiles. */
struct WholeBodyFiles {
  std::string taskFile;
  std::string urdfFile;
  std::string referenceFile;
  std::string gaitFile;
};

/** Returns the G1 whole-body files; a test failure and empty paths when they are not in the runfiles. */
const WholeBodyFiles& g1WholeBodyFiles();

/** Returns the shipped G1 whole-body task file; a test failure and an empty file when it does not load. */
mpc_config::TaskFile shippedTaskFile();

/**
 * Returns the shipped task file with the terms its problem lacks: joint_torque_cost listed (with a weight for every joint),
 * zero_velocity and normal_velocity soft instead of hard, and contact_wrench_cone instead of friction_force_cone, so that
 * joint_torque_weights, contacts.contact_wrench_cone_soft_constraint and the soft weights of model_settings.foot_constraint
 * are carried by a running problem; and a swing pitch, which the swing foot's cost then holds the foot to.
 */
mpc_config::TaskFile variantTaskFile();

/**
 * Returns the whole-body MPC of `task` with the G1 URDF and reference file, built in the run's working directory; a test
 * failure and nullptr when it is refused.
 */
std::unique_ptr<WBMpcInterface> createWholeBodyMpc(const mpc_config::TaskFile& task);

/** Returns the MPC of shippedTaskFile(), built once for the test program and never destroyed; nullptr when refused. */
WBMpcInterface* absl_nullable shippedWholeBodyMpc();

/** Returns the MPC of variantTaskFile(), built once for the test program and never destroyed; nullptr when refused. */
WBMpcInterface* absl_nullable variantWholeBodyMpc();

// [s] The swing of scheduleOneSwing().
inline constexpr scalar_t kSwingStart = 0.0;
inline constexpr scalar_t kSwingEnd = 0.6;

/**
 * Replaces the mode schedule of `interface`'s reference manager with one in which the contact `swingFoot` swings over
 * [kSwingStart, kSwingEnd] between two stance phases, and runs the reference manager over it, so that the swing
 * trajectories are planned. The brackets far outside the window keep the gait schedule from tiling its template over
 * it.
 */
void scheduleOneSwing(WBMpcInterface& interface, size_t swingFoot);

/** A time, a state and an input to evaluate a term at. */
struct EvaluationPoint {
  scalar_t time = 0.0;
  vector_t state;
  vector_t input;
};

/**
 * Returns `count` points around the initial state of `interface`, at times inside the swing of scheduleOneSwing() and
 * in stance on either side of it, from a generator of a fixed seed: the base and joints moved by a few centimeters and
 * radians, velocities and inputs drawn from a unit range scaled to the contact wrenches.
 */
std::vector<EvaluationPoint> evaluationPoints(const WBMpcInterface& interface, size_t count);

}  // namespace ocs2::humanoid::live_tuning_test
