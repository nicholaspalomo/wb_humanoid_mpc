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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_multi_contact/planner/contact_invariant/PlanSeeding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_oc/oc_data/StateTrajectoryInterpolation.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

using PlanKnot = config::MultiContactPlan::Knot;

// [s] Times closer than this are one.
constexpr double kTimeTolerance = 1.0e-9;
// The bisection of the retiming's factor.
constexpr int kRetimingBisections = 100;
// Planner A's own base position weights: the task file's are the tracking
// MPC's, tuned to follow a plan, and the plans must not move with them.
constexpr double kBaseHorizontalWeight = 25.0;
constexpr double kBaseVerticalWeight = 15.0;

/** The joint positions `values` of `names`, in the MPC's joint order. */
vector_t mpcJointsOf(const MpcRobotModelBase<scalar_t>& model,
                     const std::vector<std::string>& names,
                     const Eigen::VectorXd& values) {
  absl::flat_hash_map<std::string, double> byName;
  for (size_t j = 0; j < names.size(); ++j) byName[names[j]] = values(j);
  const std::vector<std::string>& mpcNames =
      model.modelSettings.mpcModelJointNames;
  vector_t joints = vector_t::Zero(static_cast<Eigen::Index>(mpcNames.size()));
  for (size_t j = 0; j < mpcNames.size(); ++j) {
    if (const absl::flat_hash_map<std::string, double>::const_iterator found =
            byName.find(mpcNames[j]);
        found != byName.end()) {
      joints(j) = found->second;
    }
  }
  return joints;
}

/** [s] The retimed duration of runs of `oldDurations` at `factor`, each at
 * least `minimumRun`. */
double retimedDuration(const std::vector<double>& oldDurations,
                       double minimumRun,
                       double factor) {
  double sum = 0.0;
  for (const double old : oldDurations) {
    sum += std::max(minimumRun, factor * old);
  }
  return sum;
}

/** The index of the last knot of `plan` at or before `time` (clamped). */
size_t knotBefore(const config::MultiContactPlan& plan, double time) {
  size_t j = 0;
  while (j + 1 < plan.knots.size() && plan.knots[j + 1].time <= time) ++j;
  return j;
}

}  // namespace

absl::StatusOr<CentroidalMpcConfig> plannerMpcConfigOf(
    const LoadedScenario& loaded) {
  const config::Scenario& scenario = loaded.scenario;
  if (scenario.mpc_task_file.empty() || scenario.mpc_reference_file.empty()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "scenario '", scenario.name,
        "': the contact-invariant planner needs an mpc_task_file and an "
        "mpc_reference_file"));
  }
  ASSIGN_OR_RETURN(const std::string task,
                   robot::resolveResourcePath(scenario.mpc_task_file));
  ASSIGN_OR_RETURN(const std::string reference,
                   robot::resolveResourcePath(scenario.mpc_reference_file));
  ASSIGN_OR_RETURN(CentroidalMpcConfig config,
                   loadCentroidalMpcConfig(task, reference));
  config.task.state_weights.pos_base_rt_world_ewrt_world =
      mpc_config::Xyz{.x = kBaseHorizontalWeight,
                      .y = kBaseHorizontalWeight,
                      .z = kBaseVerticalWeight};
  return config;
}

std::vector<double> knotGrid(double duration, double dt) {
  const int intervals =
      std::max(1, static_cast<int>(std::lround(duration / dt)));
  std::vector<double> times;
  for (int k = 0; k <= intervals; ++k) {
    times.push_back(duration * k / intervals);
  }
  return times;
}

MpcTrajectory resampledTrajectory(const MpcTrajectory& trajectory,
                                  const std::vector<double>& times,
                                  const StateManifold* absl_nullable manifold) {
  MpcTrajectory resampled;
  resampled.times = times;
  for (const double time : times) {
    resampled.states.push_back(interpolateStateTrajectory(
        manifold, time, trajectory.times, trajectory.states));
    size_t node = 0;
    while (node + 1 < trajectory.times.size() &&
           trajectory.times[node + 1] <= time + kTimeTolerance) {
      ++node;
    }
    resampled.inputs.push_back(trajectory.inputs[node]);
    resampled.masks.push_back(trajectory.masks[node]);
  }
  return resampled;
}

vector_t mpcStateOf(const MpcRobotModelBase<scalar_t>& model,
                    const RobotConfiguration& configuration) {
  vector_t state =
      vector_t::Zero(static_cast<Eigen::Index>(model.getStateDim()));
  model.setPosBaseRtWorldEwrtWorld(state, configuration.basePosition);
  model.setQuatBaseRtWorldXyzw(state, configuration.baseOrientation.coeffs());
  model.setJointAngles(state, mpcJointsOf(model, configuration.jointNames,
                                          configuration.jointPositions));
  return state;
}

absl::StatusOr<MpcTrajectory> mpcTrajectoryOfPlan(
    const config::MultiContactPlan& plan,
    const MpcRobotModelBase<scalar_t>& model,
    const std::vector<std::string>& contactNames,
    const ContactSet& contactSet,
    double mass,
    const std::vector<double>& times) {
  if (plan.knots.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("plan '", plan.scenario, "' has no knot"));
  }
  const std::vector<int> candidates =
      candidateIndicesOf(contactNames, contactSet);
  MpcTrajectory trajectory;
  trajectory.times = times;
  for (const double time : times) {
    const size_t j = knotBefore(plan, time);
    const size_t next = std::min(j + 1, plan.knots.size() - 1);
    const PlanKnot& a = plan.knots[j];
    const PlanKnot& b = plan.knots[next];
    if (a.base_position.size() != kPositionDim ||
        a.base_orientation_xyzw.size() != kQuaternionCoefficients ||
        a.joint_positions.size() !=
            static_cast<Eigen::Index>(plan.joint_names.size())) {
      return absl::InvalidArgumentError(
          absl::StrCat("plan '", plan.scenario, "': knot ", j,
                       " has vectors of the wrong "
                       "size"));
    }
    const double span = b.time - a.time;
    const double alpha =
        span > 0.0 ? std::clamp((time - a.time) / span, 0.0, 1.0) : 0.0;
    RobotConfiguration configuration;
    configuration.basePosition =
        (1.0 - alpha) * Eigen::Vector3d(a.base_position) +
        alpha * Eigen::Vector3d(b.base_position);
    const Eigen::Quaterniond qa(Eigen::Vector4d(a.base_orientation_xyzw));
    const Eigen::Quaterniond qb(Eigen::Vector4d(b.base_orientation_xyzw));
    configuration.baseOrientation = qa.slerp(alpha, qb).normalized();
    configuration.jointNames = plan.joint_names;
    configuration.jointPositions =
        (1.0 - alpha) * a.joint_positions + alpha * b.joint_positions;
    vector_t state = mpcStateOf(model, configuration);
    if (a.linear_momentum.size() == kPositionDim &&
        b.linear_momentum.size() == kPositionDim) {
      state.head<kPositionDim>() =
          ((1.0 - alpha) * a.linear_momentum + alpha * b.linear_momentum) /
          mass;
    }
    if (a.angular_momentum.size() == kPositionDim &&
        b.angular_momentum.size() == kPositionDim) {
      state.segment<kPositionDim>(kPositionDim) =
          ((1.0 - alpha) * a.angular_momentum + alpha * b.angular_momentum) /
          mass;
    }
    vector_t input =
        vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
    for (const config::MultiContactPlan::ContactWrench& wrench :
         a.contact_wrenches) {
      for (size_t i = 0; i < contactNames.size(); ++i) {
        if (contactNames[i] != wrench.contact) continue;
        if (wrench.force.size() == kForceDim) {
          model.setForceContactEwrtParam(input, wrench.force, i);
        }
        if (wrench.torque.size() == kTorqueDim &&
            model.getContactInputDim(i) == static_cast<size_t>(kWrenchDim)) {
          vector6_t full;
          full << model.getForceContactEwrtParam(input, i),
              Eigen::Vector3d(wrench.torque);
          model.setWrenchContactEwrtParam(input, full, i);
        }
      }
    }
    if (a.joint_velocities.size() == a.joint_positions.size()) {
      model.setJointVelocities(
          state, input,
          mpcJointsOf(model, plan.joint_names, a.joint_velocities));
    }
    trajectory.states.push_back(std::move(state));
    trajectory.inputs.push_back(std::move(input));
    trajectory.masks.push_back(alpha >= 1.0 ? b.active_contacts
                                            : a.active_contacts);
  }
  // A mask of a candidate the MPC lacks would be refused by the conversion.
  for (const ContactMask mask : trajectory.masks) {
    for (int c = 0; c < contactSet.size(); ++c) {
      if (isContactActive(mask, c) &&
          std::find(candidates.begin(), candidates.end(), c) ==
              candidates.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "plan '", plan.scenario, "': '", contactSet.candidate(c).name,
            "' is active but not a contact of the MPC"));
      }
    }
  }
  return trajectory;
}

absl::StatusOr<config::MultiContactPlan> retimedPlan(
    const config::MultiContactPlan& plan,
    double duration,
    double minimumRun,
    double changeQuantum) {
  if (plan.knots.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("plan '", plan.scenario, "' has no knot"));
  }
  // The runs of equal stances: knots [begins[r], begins[r + 1]).
  std::vector<size_t> begins = {0};
  for (size_t k = 1; k < plan.knots.size(); ++k) {
    if (plan.knots[k].active_contacts != plan.knots[k - 1].active_contacts) {
      begins.push_back(k);
    }
  }
  const size_t runs = begins.size();
  std::vector<double> oldDurations(runs);
  for (size_t r = 0; r < runs; ++r) {
    const size_t end = r + 1 < runs ? begins[r + 1] : plan.knots.size() - 1;
    oldDurations[r] = plan.knots[end].time - plan.knots[begins[r]].time;
  }
  if (minimumRun * static_cast<double>(runs) > duration) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "plan '%s': %d stances of at least %.2f s do not fit in %.2f s",
        plan.scenario, static_cast<int>(runs), minimumRun, duration));
  }
  double low = 0.0;
  double high =
      duration / std::max(kTimeTolerance,
                          plan.knots.back().time - plan.knots.front().time) +
      duration;
  for (int iteration = 0; iteration < kRetimingBisections; ++iteration) {
    const double middle = 0.5 * (low + high);
    if (retimedDuration(oldDurations, minimumRun, middle) < duration) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const double factor = 0.5 * (low + high);
  // The runs' start times, the changes rounded to the quantum.
  std::vector<double> starts(runs + 1, 0.0);
  for (size_t r = 0; r < runs; ++r) {
    starts[r + 1] = starts[r] + std::max(minimumRun, factor * oldDurations[r]);
  }
  if (changeQuantum > 0.0) {
    // n / (1 / q) rather than n q: for a quantum whose inverse is a whole
    // number the change is then the correctly rounded n / 10 (for 0.1 s),
    // the very double knotGrid() puts a knot at, not 0.30000000000000004.
    const double perSecond = 1.0 / changeQuantum;
    for (size_t r = 1; r < runs; ++r) {
      starts[r] = std::round(starts[r] * perSecond) / perSecond;
    }
  }
  config::MultiContactPlan retimed = plan;
  for (size_t r = 0; r < runs; ++r) {
    const double old = oldDurations[r];
    const double fresh = starts[r + 1] - starts[r];
    const double scale = old > 0.0 ? fresh / old : 1.0;
    const size_t end = r + 1 < runs ? begins[r + 1] : plan.knots.size();
    for (size_t k = begins[r]; k < end; ++k) {
      PlanKnot& knot = retimed.knots[k];
      knot.time =
          starts[r] + (plan.knots[k].time - plan.knots[begins[r]].time) * scale;
      knot.linear_momentum = plan.knots[k].linear_momentum / scale;
      knot.angular_momentum = plan.knots[k].angular_momentum / scale;
      knot.joint_velocities = plan.knots[k].joint_velocities / scale;
    }
  }
  retimed.knots.back().time = std::max(retimed.knots.back().time, duration);
  for (config::MultiContactPlan::ContactPhase& phase : retimed.phases) {
    const size_t first = knotBefore(plan, phase.start_time);
    const size_t last = knotBefore(plan, phase.end_time);
    phase.start_time = retimed.knots[first].time;
    phase.end_time = retimed.knots[last].time;
  }
  retimed.report.reset();
  return retimed;
}

absl::Status checkGoalReached(const config::MultiContactPlan& plan,
                              const LoadedScenario& loaded) {
  if (plan.knots.empty()) {
    return absl::NotFoundError("the plan has no knot");
  }
  const config::Scenario::Goal& goal = loaded.scenario.goal;
  std::vector<std::string> goalNames;
  for (const config::Scenario::GoalContact& contact : goal.contacts) {
    goalNames.push_back(contact.contact);
  }
  ASSIGN_OR_RETURN(const ContactMask goalMask,
                   loaded.contactSet.maskOf(goalNames));
  const PlanKnot& last = plan.knots.back();
  if (last.active_contacts != goalMask) {
    return absl::NotFoundError(absl::StrFormat(
        "the plan ends on the stance %#x, not the goal stance %#x",
        last.active_contacts, goalMask));
  }
  for (const config::Scenario::GoalContact& contact : goal.contacts) {
    const config::MultiContactPlan::ContactPhase* absl_nullable phase = nullptr;
    for (const config::MultiContactPlan::ContactPhase& candidate :
         plan.phases) {
      if (candidate.contact == contact.contact) phase = &candidate;
    }
    if (phase == nullptr || phase->primitive != contact.primitive) {
      return absl::NotFoundError(
          absl::StrCat("the plan ends with '", contact.contact, "' on '",
                       phase == nullptr ? "nothing" : phase->primitive,
                       "', not on '", contact.primitive, "'"));
    }
  }
  if (goal.base_region.has_value()) {
    const Eigen::Vector3d offset = (Eigen::Vector3d(last.base_position) -
                                    Eigen::Vector3d(goal.base_region->center))
                                       .cwiseAbs();
    if ((offset.array() >
         Eigen::Vector3d(goal.base_region->half_extents).array())
            .any()) {
      return absl::NotFoundError(absl::StrFormat(
          "the plan ends with the base at (%.3f, %.3f, %.3f), outside the "
          "goal region",
          Eigen::Vector3d(last.base_position).x(),
          Eigen::Vector3d(last.base_position).y(),
          Eigen::Vector3d(last.base_position).z()));
    }
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::multi_contact
