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

#include "humanoid_centroidal_mpc/multi_contact/PlannedMotion.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_centroidal_mpc/multi_contact/PlanContactOrder.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"

namespace ocs2::humanoid {
namespace {

/** How far from 1 the norm of a knot's quaternion may be. */
constexpr scalar_t kUnitQuaternionTolerance = 1.0e-6;
/** How far from 1 the norm of a phase's patch normal may be. */
constexpr scalar_t kUnitNormalTolerance = 1.0e-3;
/** [rad] The largest absolute position of a joint the MPC fixes that the plan
 * may give it: the MPC model holds it at 0. */
constexpr scalar_t kFixedJointTolerance = 1.0e-9;
/** [s] How far outside a phase's interval a knot it covers may lie. */
constexpr scalar_t kPhaseCoverageTolerance = 1.0e-6;
/** The entries of a target pad pose: the center, then the normal. */
constexpr size_t kTargetPadPoseDim = 2 * kPositionDim;
/** Below this |n . x| the world x axis is the patch's x direction, otherwise
 * the world y axis is. */
constexpr scalar_t kWorldXAxisAlignment = 0.9;
/** How far the last knot's state of a periodic plan may be from its first. */
constexpr scalar_t kPeriodicSeamTolerance = 1.0e-9;
/** The fewest knots of a periodic plan: its first, and its first again. */
constexpr size_t kMinPeriodicKnots = 2;

/** floor(numerator / denominator) for a positive denominator. */
int64_t floorDivide(int64_t numerator, int64_t denominator) {
  const int64_t quotient = numerator / denominator;
  return numerator % denominator < 0 ? quotient - 1 : quotient;
}

/** InvalidArgument of the knot `index` saying `what`. */
absl::Status knotError(size_t index, absl::string_view what) {
  return absl::InvalidArgumentError(
      absl::StrCat("[PlannedMotion] knot ", index, ": ", what));
}

/** Whether a knot's vector may be left out, which makes it zero. */
enum class Presence { kRequired, kOptional };

/** `values` as a 3-vector; InvalidArgument of knot `index` naming `field`
 * unless it has exactly three finite entries, or none when it is optional. */
absl::StatusOr<vector3_t> knotVector(const vector_t& values,
                                     size_t index,
                                     absl::string_view field,
                                     Presence presence) {
  if (values.size() == 0 && presence == Presence::kOptional) {
    return vector3_t(vector3_t::Zero());
  }
  if (values.size() != kPositionDim) {
    return knotError(index, absl::StrCat(field, " has ", values.size(),
                                         " entries, not ", kPositionDim, "."));
  }
  vector3_t vector = values.head<kPositionDim>();
  if (!vector.allFinite()) {
    return knotError(index, absl::StrCat(field, " is not finite."));
  }
  return vector;
}

/** Where each plan joint goes: the MPC joint's index, or none for a joint the
 * MPC fixes. */
struct PlanJointColumns {
  std::vector<std::optional<size_t>> mpcJointIndex;
};

/** The columns of the plan's joints in the MPC's joints; InvalidArgument for a
 * joint the URDF does not have or one named twice. */
absl::StatusOr<PlanJointColumns> planJointColumns(
    const std::vector<std::string>& planJoints,
    const ModelSettings& modelSettings) {
  absl::flat_hash_map<std::string, size_t> mpcJointByName;
  for (size_t i = 0; i < modelSettings.mpcModelJointNames.size(); ++i) {
    mpcJointByName.try_emplace(modelSettings.mpcModelJointNames[i], i);
  }
  const absl::flat_hash_set<std::string> urdfJoints(
      modelSettings.fullJointNames.begin(), modelSettings.fullJointNames.end());
  absl::flat_hash_set<std::string> seen;
  PlanJointColumns columns;
  for (const std::string& joint : planJoints) {
    if (!seen.insert(joint).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[PlannedMotion] the plan names the joint '", joint, "' twice."));
    }
    if (!urdfJoints.contains(joint)) {
      return absl::InvalidArgumentError(
          absl::StrCat("[PlannedMotion] the plan's joint '", joint,
                       "' is not a joint of the robot's URDF."));
    }
    const absl::flat_hash_map<std::string, size_t>::const_iterator mpc =
        mpcJointByName.find(joint);
    columns.mpcJointIndex.push_back(mpc == mpcJointByName.end()
                                        ? std::nullopt
                                        : std::optional<size_t>(mpc->second));
  }
  return columns;
}

/** The MPC state of knot `index` of `plan`. */
absl::StatusOr<vector_t> knotState(
    const multi_contact::config::MultiContactPlan& plan,
    size_t index,
    const PlanJointColumns& columns,
    const MpcRobotModelBase<scalar_t>& model,
    scalar_t robotMass) {
  const multi_contact::config::MultiContactPlan::Knot& knot = plan.knots[index];
  vector_t state = model.getNeutralState();
  ASSIGN_OR_RETURN(
      const vector3_t posBaseRtWorldEwrtWorld,
      knotVector(knot.pos_base_rt_world_ewrt_world, index,
                 "pos_base_rt_world_ewrt_world", Presence::kRequired));
  model.setPosBaseRtWorldEwrtWorld(state, posBaseRtWorldEwrtWorld);
  if (knot.quat_base_rt_world_xyzw.size() != kQuaternionCoefficients) {
    return knotError(index, "quat_base_rt_world_xyzw is not 4 coefficients.");
  }
  const vector4_t quatBaseRtWorldXyzw =
      knot.quat_base_rt_world_xyzw.head<kQuaternionCoefficients>();
  if (!quatBaseRtWorldXyzw.allFinite() ||
      std::abs(quatBaseRtWorldXyzw.norm() - 1.0) > kUnitQuaternionTolerance) {
    return knotError(index,
                     "quat_base_rt_world_xyzw is not a unit quaternion.");
  }
  model.setQuatBaseRtWorldXyzw(state, quatBaseRtWorldXyzw);

  if (static_cast<size_t>(knot.joint_positions.size()) !=
      plan.joint_names.size()) {
    return knotError(
        index, absl::StrCat("joint_positions has ", knot.joint_positions.size(),
                            " entries for the plan's ", plan.joint_names.size(),
                            " joint names."));
  }
  vector_t joints = vector_t::Zero(
      static_cast<Eigen::Index>(model.modelSettings.mpc_joint_dim));
  for (size_t j = 0; j < plan.joint_names.size(); ++j) {
    const double jointPosition =
        knot.joint_positions(static_cast<Eigen::Index>(j));
    if (!std::isfinite(jointPosition)) {
      return knotError(
          index, absl::StrCat("the position of joint '", plan.joint_names[j],
                              "' is not finite."));
    }
    const std::optional<size_t> column = columns.mpcJointIndex[j];
    if (column.has_value()) {
      joints(static_cast<Eigen::Index>(*column)) = jointPosition;
    } else if (std::abs(jointPosition) > kFixedJointTolerance) {
      return knotError(
          index, absl::StrCat("the MPC fixes the joint '", plan.joint_names[j],
                              "' at 0, but the plan puts it at ", jointPosition,
                              " rad: unlock it in model_settings."));
    }
  }
  model.setJointAngles(state, joints);

  ASSIGN_OR_RETURN(
      const vector3_t linmomComRtWorldEwrtWorld,
      knotVector(knot.linmom_com_rt_world_ewrt_world, index,
                 "linmom_com_rt_world_ewrt_world", Presence::kOptional));
  ASSIGN_OR_RETURN(
      const vector3_t angmomComRtWorldEwrtWorld,
      knotVector(knot.angmom_com_rt_world_ewrt_world, index,
                 "angmom_com_rt_world_ewrt_world", Presence::kOptional));
  // The normalized centroidal momentum h = [m c_dot; L] / m at the head of the
  // state.
  state.head<centroidal_model::kLinearMomentumDim>() =
      linmomComRtWorldEwrtWorld / robotMass;
  state.segment<centroidal_model::kAngularMomentumDim>(
      centroidal_model::kAngularMomentumStartIndex) =
      angmomComRtWorldEwrtWorld / robotMass;
  return state;
}

/** The phase of `phase`, or InvalidArgument naming its index. */
absl::StatusOr<PlannedContactPhase> plannedPhase(
    const multi_contact::config::MultiContactPlan::ContactPhase& phase,
    size_t index) {
  if (!std::isfinite(phase.start_time) || !std::isfinite(phase.end_time) ||
      phase.end_time < phase.start_time) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[PlannedMotion] phase ", index, " of '", phase.contact, "' ends at ",
        phase.end_time, " s, before its start at ", phase.start_time, " s."));
  }
  if (static_cast<size_t>(phase.des_pose_pad_rt_world.size()) !=
      kTargetPadPoseDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[PlannedMotion] phase ", index, " of '", phase.contact,
        "' has a des_pose_pad_rt_world of ", phase.des_pose_pad_rt_world.size(),
        " values, not ", kTargetPadPoseDim, " (its center, then its normal)."));
  }
  const vector3_t desPosPadRtWorldEwrtWorld =
      phase.des_pose_pad_rt_world.head<kPositionDim>();
  const vector3_t desNormalSurfaceEwrtWorld =
      phase.des_pose_pad_rt_world.tail<kPositionDim>();
  if (!desPosPadRtWorldEwrtWorld.allFinite() ||
      !desNormalSurfaceEwrtWorld.allFinite() ||
      std::abs(desNormalSurfaceEwrtWorld.norm() - 1.0) > kUnitNormalTolerance) {
    return absl::InvalidArgumentError(
        absl::StrCat("[PlannedMotion] phase ", index, " of '", phase.contact,
                     "' has a target pad normal that is not a unit vector."));
  }
  const vector3_t unitDesNormalSurfaceEwrtWorld =
      desNormalSurfaceEwrtWorld.normalized();
  return PlannedContactPhase{
      .startTime = phase.start_time,
      .endTime = phase.end_time,
      .desPosContactRtWorldEwrtWorld = desPosPadRtWorldEwrtWorld,
      .desNormalSurfaceEwrtWorld = unitDesNormalSurfaceEwrtWorld,
      .patch = PlannedMotion::patchFrameOf(desPosPadRtWorldEwrtWorld,
                                           unitDesNormalSurfaceEwrtWorld)};
}

}  // namespace

ContactPatchFrame PlannedMotion::patchFrameOf(
    const vector3_t& posSurfaceRtWorldEwrtWorld,
    const vector3_t& normalSurfaceEwrtWorld) {
  // The world axis the patch's x axis is made from.
  const vector3_t seedDirEwrtWorld =
      std::abs(normalSurfaceEwrtWorld.x()) < kWorldXAxisAlignment
          ? vector3_t(vector3_t::UnitX())
          : vector3_t(vector3_t::UnitY());
  // p_R_w: the rows are the patch axes in world coordinates (the columns of
  // rotmatSurfaceRtWorld).
  matrix3_t rotmatWorldRtSurface;
  rotmatWorldRtSurface.row(0) =
      (seedDirEwrtWorld -
       seedDirEwrtWorld.dot(normalSurfaceEwrtWorld) * normalSurfaceEwrtWorld)
          .normalized()
          .transpose();
  rotmatWorldRtSurface.row(1) =
      normalSurfaceEwrtWorld
          .cross(vector3_t(rotmatWorldRtSurface.row(0).transpose()))
          .transpose();
  rotmatWorldRtSurface.row(kZAxisIndex) = normalSurfaceEwrtWorld.transpose();
  return ContactPatchFrame{
      .rotmatWorldRtSurface = rotmatWorldRtSurface,
      .posSurfaceRtWorldEwrtSurfaceZ =
          normalSurfaceEwrtWorld.dot(posSurfaceRtWorldEwrtWorld)};
}

namespace {

/** The wrenches of knot `index` of a plan in the MPC's input layout, from
 * `zeroInput`. The plan's wrenches are in world axes, so they go through the
 * state-aware world setters at the knot's `state`: under basis-vector inputs
 * the input is in the contact frame, which the state places. InvalidArgument
 * for a wrench of an unknown contact or of the wrong size. */
absl::StatusOr<vector_t> plannedInputOfKnot(
    const multi_contact::config::MultiContactPlan::Knot& knot,
    size_t index,
    const PlanContactOrder& order,
    const MpcRobotModelBase<scalar_t>& model,
    const vector_t& state,
    const vector_t& zeroInput) {
  vector_t input = zeroInput;
  for (const multi_contact::config::MultiContactPlan::ContactWrench& wrenchMsg :
       knot.wrench_contacts_ewrt_world) {
    const std::optional<size_t> planIndex =
        order.planIndexOfName(wrenchMsg.contact);
    const std::optional<size_t> mpcIndex =
        planIndex.has_value() ? order.mpcIndexOf(*planIndex) : std::nullopt;
    if (!mpcIndex.has_value()) {
      return knotError(index, absl::StrCat("its wrench of '", wrenchMsg.contact,
                                           "' is of no contact of the MPC."));
    }
    // NOLINTBEGIN(frame-naming): the components of a ContactWrench, whose frames its field names
    if (wrenchMsg.force.size() != kForceDim ||
        (wrenchMsg.torque.size() != 0 &&
         wrenchMsg.torque.size() != kTorqueDim)) {
      return knotError(index, absl::StrCat("its wrench of '", wrenchMsg.contact,
                                           "' has vectors of the wrong size."));
    }
    // A 6-DoF contact carries the plan's moment; under basis-vector inputs its
    // input block is wider than a wrench, so the contact is asked, not the
    // input width.
    if (wrenchMsg.torque.size() == kTorqueDim &&
        !model.modelSettings.isPointContact(*mpcIndex)) {
      vector6_t wrenchContactEwrtWorld;
      wrenchContactEwrtWorld << vector3_t(wrenchMsg.force),
          vector3_t(wrenchMsg.torque);
      model.setWrenchContactEwrtWorld(state, input, wrenchContactEwrtWorld,
                                      *mpcIndex);
    } else {
      model.setForceContactEwrtWorld(state, input, vector3_t(wrenchMsg.force),
                                     *mpcIndex);
    }
    // NOLINTEND(frame-naming)
  }
  return input;
}

}  // namespace

absl::StatusOr<PlannedMotion> PlannedMotion::Create(
    const multi_contact::config::MultiContactPlan& plan,
    const PlanContactOrder& order,
    const MpcRobotModelBase<scalar_t>& model,
    scalar_t robotMass,
    PlanRepetition repetition) {
  if (plan.knots.empty()) {
    return absl::InvalidArgumentError("[PlannedMotion] the plan has no knots.");
  }
  if (!(robotMass > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[PlannedMotion] the robot mass is ", robotMass, " kg."));
  }
  ASSIGN_OR_RETURN(const PlanJointColumns columns,
                   planJointColumns(plan.joint_names, model.modelSettings));
  PlannedMotion motion(order);
  const vector_t zeroInput =
      vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
  std::vector<size_t> knotModes;
  for (size_t i = 0; i < plan.knots.size(); ++i) {
    const multi_contact::config::MultiContactPlan::Knot& knot = plan.knots[i];
    if (!std::isfinite(knot.time) ||
        (i > 0 && !(knot.time > plan.knots[i - 1].time))) {
      return knotError(i, "its time does not increase.");
    }
    ASSIGN_OR_RETURN(vector_t state,
                     knotState(plan, i, columns, model, robotMass));
    ASSIGN_OR_RETURN(
        vector_t plannedInput,
        plannedInputOfKnot(knot, i, order, model, state, zeroInput));
    motion.targets_.timeTrajectory.push_back(knot.time);
    motion.targets_.stateTrajectory.push_back(std::move(state));
    motion.targets_.inputTrajectory.push_back(zeroInput);
    motion.plannedInputs_.push_back(std::move(plannedInput));
    absl::StatusOr<size_t> mode = order.modeOf(knot.active_contacts);
    if (!mode.ok()) return knotError(i, mode.status().message());
    knotModes.push_back(*mode);
  }
  motion.modeSequence_.push_back(knotModes.front());
  for (size_t i = 1; i < knotModes.size(); ++i) {
    if (knotModes[i] == knotModes[i - 1]) continue;
    motion.eventTimes_.push_back(plan.knots[i].time);
    motion.modeSequence_.push_back(knotModes[i]);
  }
  if (repetition == PlanRepetition::kPeriodic) {
    const TargetTrajectories& targets = motion.targets_;
    if (targets.size() < kMinPeriodicKnots) {
      return absl::InvalidArgumentError(
          "[PlannedMotion] a periodic plan needs two knots at least.");
    }
    if (motion.modeSequence_.front() != motion.modeSequence_.back() ||
        (targets.stateTrajectory.front() - targets.stateTrajectory.back())
                .lpNorm<Eigen::Infinity>() > kPeriodicSeamTolerance) {
      return absl::InvalidArgumentError(
          "[PlannedMotion] a periodic plan must end in the state and the "
          "stance it starts in (forwardAndReverseCycle()).");
    }
    motion.period_ = motion.endTime() - motion.startTime();
  }

  // The phases, by MPC contact.
  motion.phases_.assign(order.numMpcContacts(), {});
  for (size_t p = 0; p < plan.phases.size(); ++p) {
    const multi_contact::config::MultiContactPlan::ContactPhase& phase =
        plan.phases[p];
    const std::optional<size_t> planIndex =
        order.planIndexOfName(phase.contact);
    if (!planIndex.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[PlannedMotion] phase ", p, " names the contact '", phase.contact,
          "', which is not a candidate of the plan's contact set."));
    }
    const std::optional<size_t> mpcIndex = order.mpcIndexOf(*planIndex);
    // A candidate the MPC does not have is never active (modeOf() refuses it).
    if (!mpcIndex.has_value()) continue;
    ASSIGN_OR_RETURN(PlannedContactPhase planned, plannedPhase(phase, p));
    motion.phases_[*mpcIndex].push_back(std::move(planned));
  }
  for (std::vector<PlannedContactPhase>& phases : motion.phases_) {
    std::sort(phases.begin(), phases.end(),
              [](const PlannedContactPhase& a, const PlannedContactPhase& b) {
                return a.startTime < b.startTime;
              });
  }
  // Every active contact of every knot rests on a phase of it.
  const size_t numMpc = order.numMpcContacts();
  for (size_t i = 0; i < knotModes.size(); ++i) {
    for (size_t contact = 0; contact < numMpc; ++contact) {
      if (!contactIsActiveInMode(knotModes[i], contact, numMpc)) continue;
      if (!motion.phaseAt(contact, plan.knots[i].time).has_value()) {
        return knotError(
            i, absl::StrCat("the contact '", order.mpcContactName(contact),
                            "' is active, but no phase of it covers the "
                            "knot's time ",
                            plan.knots[i].time, " s."));
      }
    }
  }
  motion.buildSwingWindows();
  return motion;
}

void PlannedMotion::buildSwingWindows() {
  // The swings, from the stances: an interval out of contact that ends at a
  // change into contact.
  const size_t numMpc = order_.numMpcContacts();
  swingWindows_.assign(numMpc, {});
  for (size_t contact = 0; contact < numMpc; ++contact) {
    std::optional<scalar_t> liftOff;
    bool fromStart = false;
    if (!contactIsActiveInMode(modeSequence_.front(), contact, numMpc)) {
      liftOff = startTime();
      fromStart = true;
      // Periodic: in the air across the start since the period's last
      // lift-off, a period before.
      for (size_t k = 0; period_.has_value() && k < eventTimes_.size(); ++k) {
        if (contactIsActiveInMode(modeSequence_[k], contact, numMpc) &&
            !contactIsActiveInMode(modeSequence_[k + 1], contact, numMpc)) {
          liftOff = eventTimes_[k] - *period_;
          fromStart = false;
        }
      }
    }
    for (size_t k = 0; k < eventTimes_.size(); ++k) {
      const bool before =
          contactIsActiveInMode(modeSequence_[k], contact, numMpc);
      const bool after =
          contactIsActiveInMode(modeSequence_[k + 1], contact, numMpc);
      if (before && !after) {
        liftOff = eventTimes_[k];
        fromStart = false;
      } else if (!before && after && liftOff.has_value()) {
        swingWindows_[contact].push_back(
            PlannedSwingWindow{.liftOffTime = *liftOff,
                               .touchDownTime = eventTimes_[k],
                               .fromStart = fromStart});
        liftOff.reset();
      }
    }
  }
}

int64_t PlannedMotion::cycleIndex(scalar_t planTime) const {
  if (!period_.has_value()) return 0;
  return static_cast<int64_t>(std::floor((planTime - startTime()) / *period_));
}

scalar_t PlannedMotion::cycleTime(scalar_t planTime) const {
  if (!period_.has_value()) return planTime;
  // Rounding may put the time a hair outside the period.
  return std::clamp(
      planTime - static_cast<scalar_t>(cycleIndex(planTime)) * *period_,
      startTime(), endTime());
}

PlannedContactPhase PlannedMotion::shifted(PlannedContactPhase phase,
                                           int64_t cycles) const {
  if (!period_.has_value() || cycles == 0) return phase;
  const scalar_t shift = static_cast<scalar_t>(cycles) * *period_;
  phase.startTime += shift;
  phase.endTime += shift;
  return phase;
}

TargetTrajectories PlannedMotion::targetsOver(scalar_t from,
                                              scalar_t to) const {
  const scalar_array_t& times = targets_.timeTrajectory;
  TargetTrajectories targets;
  if (!period_.has_value()) {
    const scalar_array_t::const_iterator first =
        std::upper_bound(times.begin(), times.end(), from);
    const scalar_array_t::const_iterator last =
        std::lower_bound(times.begin(), times.end(), to);
    const size_t begin = first == times.begin()
                             ? 0
                             : static_cast<size_t>(first - times.begin()) - 1;
    const size_t end =
        std::min(static_cast<size_t>(last - times.begin()) + 1, times.size());
    for (size_t k = begin; k < end; ++k) {
      targets.timeTrajectory.push_back(times[k]);
      targets.stateTrajectory.push_back(targets_.stateTrajectory[k]);
      targets.inputTrajectory.push_back(targets_.inputTrajectory[k]);
    }
    return targets;
  }
  // The knots of all periods in one sequence: knot j is knot j mod n of
  // period j / n, n the knots of a period (the last knot is the next
  // period's first).
  const int64_t perCycle = static_cast<int64_t>(times.size()) - 1;
  const scalar_array_t::const_iterator cycleEnd = times.begin() + perCycle;
  const int64_t atOrBefore = std::max<int64_t>(
      std::upper_bound(times.begin(), cycleEnd, cycleTime(from)) -
          times.begin() - 1,
      0);
  const int64_t atOrAfter =
      std::lower_bound(times.begin(), times.end(), cycleTime(to)) -
      times.begin();
  const int64_t first = cycleIndex(from) * perCycle + atOrBefore;
  const int64_t last = cycleIndex(to) * perCycle + atOrAfter;
  for (int64_t j = first; j <= last; ++j) {
    const int64_t cycle = floorDivide(j, perCycle);
    const size_t k = static_cast<size_t>(j - cycle * perCycle);
    targets.timeTrajectory.push_back(times[k] +
                                     static_cast<scalar_t>(cycle) * *period_);
    targets.stateTrajectory.push_back(targets_.stateTrajectory[k]);
    targets.inputTrajectory.push_back(targets_.inputTrajectory[k]);
  }
  return targets;
}

size_t PlannedMotion::modeAt(scalar_t planTime) const {
  const std::vector<scalar_t>::const_iterator next = std::upper_bound(
      eventTimes_.begin(), eventTimes_.end(), cycleTime(planTime));
  return modeSequence_[static_cast<size_t>(next - eventTimes_.begin())];
}

size_t PlannedMotion::modeBeforeEvent(size_t event) const {
  if (!period_.has_value()) return modeSequence_[event];
  if (eventTimes_.empty()) return modeSequence_.front();
  return modeSequence_[event % eventTimes_.size()];
}

ModeSchedule PlannedMotion::modeScheduleOver(scalar_t from, scalar_t to) const {
  if (!period_.has_value()) return ModeSchedule(eventTimes_, modeSequence_);
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence = {modeSequence_.front()};
  for (int64_t cycle = cycleIndex(from); cycle <= cycleIndex(to); ++cycle) {
    const scalar_t shift = static_cast<scalar_t>(cycle) * *period_;
    for (size_t k = 0; k < eventTimes_.size(); ++k) {
      eventTimes.push_back(eventTimes_[k] + shift);
      modeSequence.push_back(modeSequence_[k + 1]);
    }
  }
  return ModeSchedule(std::move(eventTimes), std::move(modeSequence));
}

std::vector<PlannedSwingWindow> PlannedMotion::swingWindowsOver(
    size_t mpcIndex, scalar_t from, scalar_t to) const {
  if (!period_.has_value()) return swingWindows_[mpcIndex];
  std::vector<PlannedSwingWindow> windows;
  // A swing of a period touches down in it and may lift off in the one
  // before.
  for (int64_t cycle = cycleIndex(from) - 1; cycle <= cycleIndex(to) + 1;
       ++cycle) {
    const scalar_t shift = static_cast<scalar_t>(cycle) * *period_;
    for (const PlannedSwingWindow& window : swingWindows_[mpcIndex]) {
      const PlannedSwingWindow moved{
          .liftOffTime = window.liftOffTime + shift,
          .touchDownTime = window.touchDownTime + shift,
          .fromStart = false};
      if (moved.touchDownTime >= from && moved.liftOffTime <= to) {
        windows.push_back(moved);
      }
    }
  }
  return windows;
}

std::optional<PlannedContactPhase> PlannedMotion::cyclePhaseAt(
    size_t mpcIndex, scalar_t cycleTime) const {
  if (mpcIndex >= phases_.size()) return std::nullopt;
  for (const PlannedContactPhase& phase : phases_[mpcIndex]) {
    if (cycleTime >= phase.startTime - kPhaseCoverageTolerance &&
        cycleTime <= phase.endTime + kPhaseCoverageTolerance) {
      return phase;
    }
  }
  return std::nullopt;
}

std::optional<PlannedContactPhase> PlannedMotion::phaseAt(
    size_t mpcIndex, scalar_t planTime) const {
  const std::optional<PlannedContactPhase> phase =
      cyclePhaseAt(mpcIndex, cycleTime(planTime));
  if (!phase.has_value()) return std::nullopt;
  return shifted(*phase, cycleIndex(planTime));
}

std::optional<PlannedContactPhase> PlannedMotion::nextPhase(
    size_t mpcIndex, scalar_t planTime) const {
  if (mpcIndex >= phases_.size()) return std::nullopt;
  const scalar_t time = cycleTime(planTime);
  const int64_t cycle = cycleIndex(planTime);
  for (const PlannedContactPhase& phase : phases_[mpcIndex]) {
    if (phase.startTime > time) return shifted(phase, cycle);
  }
  // Periodic: the first of the next period.
  if (!period_.has_value() || phases_[mpcIndex].empty()) return std::nullopt;
  return shifted(phases_[mpcIndex].front(), cycle + 1);
}

std::optional<PlannedContactPhase> PlannedMotion::previousPhase(
    size_t mpcIndex, scalar_t planTime) const {
  if (mpcIndex >= phases_.size()) return std::nullopt;
  const scalar_t time = cycleTime(planTime);
  const int64_t cycle = cycleIndex(planTime);
  std::optional<PlannedContactPhase> previous;
  for (const PlannedContactPhase& phase : phases_[mpcIndex]) {
    if (phase.endTime <= time) previous = shifted(phase, cycle);
  }
  // Periodic: the last of the period before.
  if (previous.has_value() || !period_.has_value() ||
      phases_[mpcIndex].empty()) {
    return previous;
  }
  return shifted(phases_[mpcIndex].back(), cycle - 1);
}

const vector_t& PlannedMotion::plannedInputAt(scalar_t planTime) const {
  const std::vector<scalar_t>& times = targets_.timeTrajectory;
  const size_t after = static_cast<size_t>(
      std::upper_bound(times.begin(), times.end(), cycleTime(planTime)) -
      times.begin());
  return plannedInputs_[after == 0 ? 0 : after - 1];
}

}  // namespace ocs2::humanoid
