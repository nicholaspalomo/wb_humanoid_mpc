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

#include "humanoid_multi_contact/planner/contact_invariant/PlanTrajectory.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/terrain/PatchFrame.h"

namespace ocs2::humanoid::multi_contact {
namespace {

using PlanKnot = config::MultiContactPlan::Knot;
using PlanPhase = config::MultiContactPlan::ContactPhase;

// [m] How far inside a pad, against its z axis, the patch it rests on is
// looked up.
constexpr double kPatchProbeDepth = 0.02;
// The least alignment of a pad's normal with the face it is assigned to,
// cos(45 degrees), below which the face its normal faces is taken.
constexpr double kMinimumFaceAlignment = 0.7071;

/** The configuration of the MPC state `state` by joint name. */
RobotConfiguration configurationOf(const MpcRobotModelBase<scalar_t>& model,
                                   const vector_t& state) {
  RobotConfiguration configuration;
  configuration.posBaseRtWorldEwrtWorld =
      model.getPosBaseRtWorldEwrtWorld(state);
  configuration.quatBaseRtWorld.coeffs() = model.getQuatBaseRtWorldXyzw(state);
  configuration.quatBaseRtWorld.normalize();
  configuration.jointNames = model.modelSettings.mpcModelJointNames;
  configuration.jointPositions = model.getJointAngles(state);
  return configuration;
}

/** The knot of node `k` of `trajectory`, without its phases. */
PlanKnot knotOf(const MpcTrajectory& trajectory,
                const PlanConversion& conversion,
                const std::vector<int>& candidates,
                size_t k) {
  const MpcRobotModelBase<scalar_t>& model = *conversion.model;
  const vector_t& state = trajectory.states[k];
  const vector_t& input = trajectory.inputs[k];
  PlanKnot knot;
  knot.time = trajectory.times[k];
  knot.pos_base_rt_world_ewrt_world = model.getPosBaseRtWorldEwrtWorld(state);
  Eigen::Vector4d quatBaseRtWorldXyzw = model.getQuatBaseRtWorldXyzw(state);
  quatBaseRtWorldXyzw.normalize();
  knot.quat_base_rt_world_xyzw = quatBaseRtWorldXyzw;
  knot.joint_positions = model.getJointAngles(state);
  knot.joint_velocities = model.getJointVelocities(state, input);
  knot.linmom_com_rt_world_ewrt_world =
      conversion.mass * state.head<kPositionDim>().eval();
  knot.angmom_com_rt_world_ewrt_world =
      conversion.mass * state.segment<kPositionDim>(kPositionDim).eval();
  knot.active_contacts = trajectory.masks[k];
  const ContactSet& contactSet = conversion.scenario->contactSet;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i] < 0 ||
        !isContactActive(trajectory.masks[k], candidates[i])) {
      continue;
    }
    config::MultiContactPlan::ContactWrench wrenchMsg;
    wrenchMsg.contact = contactSet.candidate(candidates[i]).name;
    // The world-frame wrench, whatever frame the input parameterizes it in.
    // NOLINTBEGIN(frame-naming): the components of a ContactWrench, whose frames its field names
    wrenchMsg.force = model.getForceContactEwrtWorld(state, input, i);
    if (contactSet.candidate(candidates[i]).kind == ContactKind::kSurface) {
      wrenchMsg.torque = model.getTorqueContactEwrtWorld(state, input, i);
    }
    // NOLINTEND(frame-naming)
    knot.wrench_contacts_ewrt_world.push_back(std::move(wrenchMsg));
  }
  return knot;
}

/** The phase of candidate `candidate` over knots [begin, end): on the patch
 * under its mean pad position, found from a point kPatchProbeDepth inside the
 * pad (against its mean z axis), so that a sole near a riser's foot is
 * assigned the tread it stands on rather than the riser. */
PlanPhase phaseOf(
    const std::vector<Eigen::Vector3d>& posPadRtWorldEwrtWorldTraj,
    const std::vector<Eigen::Vector3d>& zAxisPadEwrtWorldTraj,
    const MpcTrajectory& trajectory,
    const TerrainModel& terrain,
    const std::string& name,
    size_t begin,
    size_t end) {
  Eigen::Vector3d meanPosPadRtWorldEwrtWorld = Eigen::Vector3d::Zero();
  Eigen::Vector3d meanZAxisPadEwrtWorld = Eigen::Vector3d::Zero();
  for (size_t k = begin; k < end; ++k) {
    meanPosPadRtWorldEwrtWorld += posPadRtWorldEwrtWorldTraj[k];
    meanZAxisPadEwrtWorld += zAxisPadEwrtWorldTraj[k];
  }
  meanPosPadRtWorldEwrtWorld /= static_cast<double>(end - begin);
  if (meanZAxisPadEwrtWorld.norm() > 0.0) meanZAxisPadEwrtWorld.normalize();
  PatchFrame patch = terrain.patchFrameAt(
      meanPosPadRtWorldEwrtWorld - kPatchProbeDepth * meanZAxisPadEwrtWorld);
  // A pad at a box's edge: the face its normal faces, of the box it is on.
  if (patch.face.has_value() &&
      patch.getNormalSurfaceEwrtWorld().dot(meanZAxisPadEwrtWorld) <
          kMinimumFaceAlignment) {
    double best = patch.getNormalSurfaceEwrtWorld().dot(meanZAxisPadEwrtWorld);
    for (int face = 0; face < kBoxFaceCount; ++face) {
      const absl::StatusOr<PatchFrame> frame = terrain.boxFaceFrame(
          patch.primitiveIndex, static_cast<BoxFace>(face));
      if (frame.ok() && frame->getNormalSurfaceEwrtWorld().dot(
                            meanZAxisPadEwrtWorld) > best) {
        best = frame->getNormalSurfaceEwrtWorld().dot(meanZAxisPadEwrtWorld);
        patch = *frame;
      }
    }
  }
  const Eigen::Vector3d normalSurfaceEwrtWorld =
      patch.getNormalSurfaceEwrtWorld();
  const Eigen::Vector3d desPosPadRtWorldEwrtWorld =
      meanPosPadRtWorldEwrtWorld -
      normalSurfaceEwrtWorld.dot(meanPosPadRtWorldEwrtWorld -
                                 patch.posSurfaceRtWorldEwrtWorld) *
          normalSurfaceEwrtWorld;
  PlanPhase phase;
  phase.contact = name;
  // The phase lasts until the knot that breaks the contact: the MPC's mode
  // keeps the contact until then, and the patch of a time between the last
  // active knot and that one must be this phase's, not the next one's.
  phase.start_time = trajectory.times[begin];
  phase.end_time = trajectory.times[std::min(end, trajectory.times.size() - 1)];
  phase.primitive = terrain.primitive(patch.primitiveIndex).name;
  phase.face = patch.face.has_value() ? static_cast<int>(*patch.face) : 0;
  Eigen::VectorXd desPosePadRtWorld(kPoseTangentDim);
  desPosePadRtWorld << desPosPadRtWorldEwrtWorld, normalSurfaceEwrtWorld;
  phase.des_pose_pad_rt_world = desPosePadRtWorld;
  return phase;
}

}  // namespace

std::vector<int> candidateIndicesOf(
    const std::vector<std::string>& contactNames,
    const ContactSet& contactSet) {
  std::vector<int> indices;
  for (const std::string& name : contactNames) {
    const absl::StatusOr<int> index = contactSet.indexOf(name);
    indices.push_back(index.ok() ? *index : -1);
  }
  return indices;
}

absl::StatusOr<config::MultiContactPlan> planOfTrajectory(
    const MpcTrajectory& trajectory, const PlanConversion& conversion) {
  const size_t n = trajectory.times.size();
  if (n == 0 || trajectory.states.size() != n ||
      trajectory.inputs.size() != n || trajectory.masks.size() != n) {
    return absl::InvalidArgumentError(absl::StrCat(
        "plan of a trajectory: ", n, " times, ", trajectory.states.size(),
        " states, ", trajectory.inputs.size(), " inputs and ",
        trajectory.masks.size(), " masks"));
  }
  const LoadedScenario& scenario = *conversion.scenario;
  const ContactSet& contactSet = scenario.contactSet;
  const std::vector<int> candidates =
      candidateIndicesOf(conversion.contactNames, contactSet);
  ContactMask mpcMask = 0;
  for (const int candidate : candidates) {
    if (candidate >= 0) mpcMask |= contactBit(candidate);
  }
  config::MultiContactPlan plan;
  plan.scenario = scenario.scenario.name;
  plan.planner = conversion.planner;
  plan.joint_names = conversion.model->modelSettings.mpcModelJointNames;
  stampPlanHashes(scenario.contactSetConfig, scenario.scenario.terrain_scene,
                  plan);
  // The pads of every candidate at every knot.
  std::vector<std::vector<Eigen::Vector3d>> posPadsRtWorldEwrtWorldTraj(
      contactSet.size(), std::vector<Eigen::Vector3d>(n));
  std::vector<std::vector<Eigen::Vector3d>> zAxisPadsEwrtWorldTraj(
      contactSet.size(), std::vector<Eigen::Vector3d>(n));
  for (size_t k = 0; k < n; ++k) {
    if ((trajectory.masks[k] & ~mpcMask) != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("plan of a trajectory: knot ", k,
                       " has an active contact the MPC does not have"));
    }
    plan.knots.push_back(knotOf(trajectory, conversion, candidates, k));
    RETURN_IF_ERROR(conversion.stanceModel->setConfiguration(
        configurationOf(*conversion.model, trajectory.states[k])));
    for (int c = 0; c < contactSet.size(); ++c) {
      posPadsRtWorldEwrtWorldTraj[c][k] =
          conversion.stanceModel->getPosPadRtWorldEwrtWorld(c);
      zAxisPadsEwrtWorldTraj[c][k] =
          conversion.stanceModel->getRotmatPadRtWorld(c).col(kZAxisIndex);
    }
  }
  // One phase per run of active knots, in time order.
  for (int c = 0; c < contactSet.size(); ++c) {
    size_t k = 0;
    while (k < n) {
      if (!isContactActive(trajectory.masks[k], c)) {
        ++k;
        continue;
      }
      size_t end = k;
      while (end < n && isContactActive(trajectory.masks[end], c)) ++end;
      plan.phases.push_back(phaseOf(
          posPadsRtWorldEwrtWorldTraj[c], zAxisPadsEwrtWorldTraj[c], trajectory,
          scenario.terrain, contactSet.candidate(c).name, k, end));
      k = end;
    }
  }
  return plan;
}

}  // namespace ocs2::humanoid::multi_contact
