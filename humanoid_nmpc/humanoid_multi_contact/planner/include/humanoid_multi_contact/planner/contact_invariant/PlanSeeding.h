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

#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/planner/contact_invariant/PlanTrajectory.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The configuration of the multi-contact centroidal MPC planner A solves on:
 * the scenario's mpc_task_file and mpc_reference_file, found in the binary's
 * runfiles (robot::resolveResourcePath()), with planner A's own base
 * position weights in place of the tracking MPC's. FailedPrecondition when
 * the scenario names none; the loaders' errors.
 */
absl::StatusOr<CentroidalMpcConfig> plannerMpcConfigOf(
    const LoadedScenario& loaded);

/** The knot times of a motion of `duration` [s] at `dt`: 0, dt, ..., the
 * last one `duration` (the step shortened or stretched by at most half). */
std::vector<double> knotGrid(double duration, double dt);

/**
 * The MPC state (of `model`'s layout) of `configuration`, with zero momentum:
 * its joints by name, a joint the configuration leaves out at 0.
 */
vector_t mpcStateOf(const MpcRobotModelBase<scalar_t>& model,
                    const RobotConfiguration& configuration);

/**
 * `trajectory` at `times`: the states interpolated on `manifold` (nullptr:
 * linearly), the inputs and the stances of the last node at or before each
 * time (the solver's zero-order hold).
 */
MpcTrajectory resampledTrajectory(const MpcTrajectory& trajectory,
                                  const std::vector<double>& times,
                                  const StateManifold* absl_nullable manifold);

/**
 * `plan` resampled on the MPC's layout at `times` (within the plan's): the
 * base position, joints and momenta interpolated linearly and the base
 * orientation spherically between the knots around each time, the knot
 * before the time's contact mode, wrenches and joint velocities.
 * InvalidArgument for a plan without knots or of the wrong sizes.
 */
absl::StatusOr<MpcTrajectory> mpcTrajectoryOfPlan(
    const config::MultiContactPlan& plan,
    const MpcRobotModelBase<scalar_t>& model,
    const std::vector<std::string>& contactNames,
    const ContactSet& contactSet,
    double mass,
    const std::vector<double>& times);

/**
 * `plan` retimed to `duration` [s]: its runs of knots of one stance keep
 * their order and their knots' share of the run, each run lasting its old
 * duration times one factor, but at least `minimumRun` [s] (design section
 * 5.2: planner B's sequence as planner A's initialization). With a positive
 * `changeQuantum` [s] the stance changes are then rounded to its nearest
 * multiples (thresholdActivations()' quantum; a quantum shorter than the
 * minimum run keeps every run). The momenta and velocities are scaled with
 * the time. InvalidArgument for a plan without knots or a duration too short
 * for its runs.
 */
absl::StatusOr<config::MultiContactPlan> retimedPlan(
    const config::MultiContactPlan& plan,
    double duration,
    double minimumRun,
    double changeQuantum);

/**
 * OK when the last knot of `plan` is the goal stance of `loaded`, each goal
 * contact on its primitive (its last phase's), with the base in the goal
 * region; NotFound saying which is not, with the numbers.
 */
absl::Status checkGoalReached(const config::MultiContactPlan& plan,
                              const LoadedScenario& loaded);

}  // namespace ocs2::humanoid::multi_contact
