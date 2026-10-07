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

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_msgs/mode_schedule.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/performance_index.pb.h"
#include "humanoid_mpc_msgs/system_observation.pb.h"
#include "humanoid_mpc_msgs/target_trajectories.pb.h"

/**
 * Lossless conversions between the OCS2 types the MPC link carries and their humanoid_mpc_msgs messages.
 *
 * Every double travels as a double, and every mode and post-event index as a uint64, so that a round trip reproduces
 * the OCS2 object exactly; a policy rebuilt from a message evaluates to the same inputs as the policy it was made from.
 *
 * The toProto() functions overwrite the fields they own and reuse the capacity the message already has: the repeated
 * fields are resized rather than cleared, and the nested Vector messages are kept. A caller that keeps one message
 * object and encodes into it again and again therefore allocates only while the message grows, which is the first call
 * and any call with longer trajectories or larger vectors than before.
 *
 * The fromProto() functions validate the whole message before they write anything, so the output is unchanged when they
 * return an error: every double is finite, the times are sorted, the array lengths agree, and the controller data match
 * the dimensions of the nodes. An InvalidArgument error names the offending field with its path in the message, e.g.
 * "MpcPolicy.state_trajectory[3].data[2]". Whether the dimensions are those of the robot model is a separate question
 * the conversions cannot answer; checkDimensions() answers it for the caller that knows the model.
 *
 * None of these functions is meant for a realtime thread: the decoding side allocates the OCS2 arrays it fills.
 */
namespace ocs2::humanoid::ipc {

/** The dimensions of the model a message must match, for checkDimensions(). */
struct ModelDimensions {
  size_t stateDim = 0;
  size_t inputDim = 0;
  /**
   * The number of modes of the model: every mode a message carries (an observation's, each mode of a policy's mode
   * schedule) must be below it. The humanoids' modes are the contact flags of the two feet, 0 to 3
   * (humanoid_common_mpc/gait/MotionPhaseDefinition.h), so 4; another mode would reach modeNumber2StanceLeg(), which
   * has no contact flags for it.
   */
  size_t numModes = 0;
};

// ---------------------------------------------------------------------------------------------------------------------
// SystemObservation
// ---------------------------------------------------------------------------------------------------------------------

/** Writes `observation` into `message`. Allocates only while the message's state or input grows. */
void toProto(const SystemObservation& observation, humanoid_mpc_msgs::SystemObservation* absl_nonnull message);

/**
 * Reads `message` into `observation`. Requires a finite time, state and input. Decoding into the same observation
 * again allocates nothing while the dimensions stay the same, as Eigen keeps a vector's storage when its size does not
 * change.
 */
absl::Status fromProto(const humanoid_mpc_msgs::SystemObservation& message, SystemObservation* absl_nonnull observation);

/**
 * Checks that `message` has a state of dimensions.stateDim and an input of dimensions.inputDim entries, and a mode
 * below dimensions.numModes.
 */
absl::Status checkDimensions(const humanoid_mpc_msgs::SystemObservation& message, const ModelDimensions& dimensions);

// ---------------------------------------------------------------------------------------------------------------------
// ModeSchedule
// ---------------------------------------------------------------------------------------------------------------------

/** Writes `modeSchedule` into `message`. Allocates only while the message's arrays grow. */
void toProto(const ModeSchedule& modeSchedule, humanoid_mpc_msgs::ModeSchedule* absl_nonnull message);

/**
 * Reads `message` into `modeSchedule`. Requires at least one mode, one mode more than there are event times, and
 * finite event times that do not decrease (OCS2 looks modes up by binary search).
 */
absl::Status fromProto(const humanoid_mpc_msgs::ModeSchedule& message, ModeSchedule* absl_nonnull modeSchedule);

// ---------------------------------------------------------------------------------------------------------------------
// TargetTrajectories
// ---------------------------------------------------------------------------------------------------------------------

/** Writes `targetTrajectories` into `message`, reusing its Vector messages. Allocates only while the message grows. */
void toProto(const TargetTrajectories& targetTrajectories, humanoid_mpc_msgs::TargetTrajectories* absl_nonnull message);

/**
 * Reads `message` into `targetTrajectories`. Requires finite values, times that do not decrease, one state per time,
 * and either one input per time or none (OCS2 allows target trajectories without inputs). An empty message is the empty
 * TargetTrajectories. Decoding into the same object again allocates nothing while the sizes stay the same.
 */
absl::Status fromProto(const humanoid_mpc_msgs::TargetTrajectories& message, TargetTrajectories* absl_nonnull targetTrajectories);

// ---------------------------------------------------------------------------------------------------------------------
// PerformanceIndex
// ---------------------------------------------------------------------------------------------------------------------

/** Writes `performanceIndex` into `message`. Never allocates. */
void toProto(const PerformanceIndex& performanceIndex, humanoid_mpc_msgs::PerformanceIndex* absl_nonnull message);

/**
 * Reads `message` into `performanceIndex`. Accepts any value, NaN and infinity included: the performance index is a
 * diagnostic that no controller evaluates, and a NaN merit is what a diverged solve reports, which the receiver should
 * see rather than lose.
 */
absl::Status fromProto(const humanoid_mpc_msgs::PerformanceIndex& message, PerformanceIndex* absl_nonnull performanceIndex);

// ---------------------------------------------------------------------------------------------------------------------
// The policy: CommandData + PrimalSolution + PerformanceIndex <-> MpcPolicy
// ---------------------------------------------------------------------------------------------------------------------

/**
 * Writes one MPC solution into `message`: the command data (init_observation, target_trajectories), the primal
 * solution (time, state and input trajectories, post_event_indices, mode_schedule, controller_type, controller_data) and
 * the performance index. The fields the MPC node owns (resets_served, full_resets_served, solver_status, annotations)
 * are left as they are.
 *
 * The controller travels as one Vector per time node, laid out as OCS2's flattenSingle() writes it
 * (FeedforwardController: uff; LinearController: [uff_0, K_0,:, uff_1, K_1,:, ...]). A controller whose time stamps are
 * the primal solution's time trajectory, which is how OCS2's multiple-shooting solvers build it, travels as its own node
 * values. That keeps the two nodes of an event, which share one time, apart; ControllerBase::flatten() would sample the
 * pre-event node for both and change the policy after every event. A controller on other time stamps is sampled at the
 * time trajectory with ControllerBase::flatten(), as the ROS interface did; that path allocates scratch arrays on every
 * call, the other allocates only while the message grows.
 *
 * Fails, leaving `message` unchanged, if the primal solution has no controller, a controller other than a
 * FeedforwardController or a LinearController, an empty controller next to a non-empty time trajectory, or a controller
 * whose arrays disagree in length.
 */
absl::Status policyToProto(const CommandData& commandData,
                           const PrimalSolution& primalSolution,
                           const PerformanceIndex& performanceIndex,
                           humanoid_mpc_msgs::MpcPolicy* absl_nonnull message);

/**
 * Reads the command data, the primal solution and the performance index of `message`, and rebuilds the controller:
 * a FeedforwardController or a LinearController on the time trajectory, equal node for node to the one that was sent.
 *
 * Besides the checks of the fromProto() functions above, requires a non-empty time trajectory (a policy without nodes
 * cannot be evaluated), one state, one input and one controller entry per time, finite values throughout, strictly
 * increasing post-event indices in [1, number of nodes], a known controller type, and controller data of the size the
 * node's state and input dimensions imply. Allocates the arrays of the primal solution and a new controller.
 */
absl::Status policyFromProto(const humanoid_mpc_msgs::MpcPolicy& message,
                             CommandData* absl_nonnull commandData,
                             PrimalSolution* absl_nonnull primalSolution,
                             PerformanceIndex* absl_nonnull performanceIndex);

/**
 * Checks that every state of `message` (init_observation, target_trajectories and state_trajectory) has
 * dimensions.stateDim entries, every input dimensions.inputDim, and that the mode of init_observation and every mode of
 * mode_schedule is below dimensions.numModes. Together with policyFromProto(), which ties the controller data to the
 * node dimensions, that makes the policy one the model can evaluate. Target trajectories without inputs pass.
 */
absl::Status checkDimensions(const humanoid_mpc_msgs::MpcPolicy& message, const ModelDimensions& dimensions);

}  // namespace ocs2::humanoid::ipc
