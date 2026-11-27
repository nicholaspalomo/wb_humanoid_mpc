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

#include "support/StairStepPlan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/spatial.hpp"
#include "pinocchio/spatial/explog.hpp"  // IWYU pragma: keep
#include "pinocchio/spatial/fwd.hpp"

#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact_swing/ContactSwingTrajectory.h"
#include "legged_multi_contact/contact_set/ContactSet.h"
#include "legged_multi_contact/plan/PlanFile.h"
#include "legged_multi_contact/plan/multi_contact_plan.nproto.h"
#include "legged_multi_contact/scenarios/Scenario.h"
#include "legged_multi_contact/scenarios/scenario.nproto.h"

namespace ocs2::legged {
namespace {

constexpr char kLeftFoot[] = "foot_l_contact";
constexpr char kRightFoot[] = "foot_r_contact";
constexpr char kStep[] = "step_1";
constexpr char kGround[] = "ground";
/** The legs' joints, hip to ankle. */
constexpr std::array<absl::string_view, 6> kLeftLeg = {
    "l_leg_hpz", "l_leg_hpx", "l_leg_hpy",
    "l_leg_kny", "l_leg_aky", "l_leg_akx"};
constexpr std::array<absl::string_view, 6> kRightLeg = {
    "r_leg_hpz", "r_leg_hpx", "r_leg_hpy",
    "r_leg_kny", "r_leg_aky", "r_leg_akx"};
/** The box face a foot rests on: +z (BoxFace order of multi_contact_plan). */
constexpr int kTopFace = 4;
/** The contact set's bits of the two feet. */
constexpr multi_contact::ContactMask kLeftFootBit = 1;
constexpr multi_contact::ContactMask kRightFootBit = 2;

constexpr int kIkIterations = 200;
constexpr scalar_t kIkTolerance = 1.0e-10;
constexpr scalar_t kIkDamping = 1.0e-9;
constexpr int kComIterations = 50;
constexpr scalar_t kMaxNormErrPosComRtWorldEwrtWorldXy = 1.0e-7;  // [m]
// [s] How close to a lift-off or touch-down a time counts as at it.
constexpr scalar_t kPhaseTimeTolerance = 1.0e-7;
/** [m] The base height of the crouch the steps are taken in. */
constexpr scalar_t kCrouchPosBaseRtWorldEwrtWorldZ = 0.84;
/** [m] Where the center of mass is put over a stance foot's pad: forward
 * over the right foot on the floor, back over the left foot on the step. */
constexpr scalar_t kDesPosComRtRfootEwrtWorldX = 0.01;
constexpr scalar_t kDesPosLfootRtComEwrtWorldX = 0.06;
constexpr scalar_t kDesPosLfootRtComEwrtWorldY = 0.02;
/** [m] Where the left foot lands on the step, ahead of where it stands. */
constexpr scalar_t kErrPosFootRtWorldEwrtWorldX = 0.292;
/** [m] A height above every step, where the distance to a step's top face is
 * the vertical one. */
constexpr scalar_t kProbePosQueryRtWorldEwrtWorldZ = 10.0;

/** The model's leg of `joints` and its foot frame. */
struct Leg {
  pinocchio::FrameIndex foot = 0;
  std::vector<Eigen::Index> velocityIndices;
};

absl::StatusOr<Leg> legOf(const pinocchio::Model& model,
                          absl::string_view foot,
                          const std::array<absl::string_view, 6>& joints) {
  if (!model.existFrame(std::string(foot))) {
    return absl::InvalidArgumentError(
        absl::StrCat("[makeStairStepPlan] the model has no frame ", foot));
  }
  Leg leg;
  leg.foot = model.getFrameId(std::string(foot));
  for (const absl::string_view joint : joints) {
    if (!model.existJointName(std::string(joint))) {
      return absl::InvalidArgumentError(
          absl::StrCat("[makeStairStepPlan] the model has no joint ", joint));
    }
    leg.velocityIndices.push_back(static_cast<Eigen::Index>(
        model.joints[model.getJointId(std::string(joint))].idx_v()));
  }
  return leg;
}

/** Moves the joints of `leg` so that its foot frame is at
 * `desPosFootRtWorldEwrtWorld` with a level sole (world orientation), the base
 * held. */
absl::Status solveLeg(const pinocchio::Model& model,
                      pinocchio::Data& data,
                      const Leg& leg,
                      const vector3_t& desPosFootRtWorldEwrtWorld,
                      vector_t& q) {
  const pinocchio::SE3 desPoseFootRtWorld(matrix3_t::Identity(),
                                          desPosFootRtWorldEwrtWorld);
  const Eigen::Index numColumns =
      static_cast<Eigen::Index>(leg.velocityIndices.size());
  for (int iteration = 0; iteration < kIkIterations; ++iteration) {
    pinocchio::forwardKinematics(model, data, q);
    const pinocchio::SE3& poseFootRtWorld =
        pinocchio::updateFramePlacement(model, data, leg.foot);
    const vector6_t error =
        pinocchio::log6(poseFootRtWorld.actInv(desPoseFootRtWorld)).toVector();
    if (error.norm() < kIkTolerance) return absl::OkStatus();
    pinocchio::computeJointJacobians(model, data, q);
    matrix_t jacobian = matrix_t::Zero(kTwistDim, model.nv);
    pinocchio::getFrameJacobian(model, data, leg.foot, pinocchio::LOCAL,
                                jacobian);
    matrix_t legJacobian(kTwistDim, numColumns);
    for (Eigen::Index c = 0; c < numColumns; ++c) {
      legJacobian.col(c) = jacobian.col(leg.velocityIndices[c]);
    }
    const vector_t step =
        (legJacobian.transpose() * legJacobian +
         kIkDamping * matrix_t::Identity(numColumns, numColumns))
            .ldlt()
            .solve(legJacobian.transpose() * error);
    vector_t generalizedVelocity = vector_t::Zero(model.nv);
    for (Eigen::Index c = 0; c < numColumns; ++c) {
      generalizedVelocity(leg.velocityIndices[c]) = step(c);
    }
    q = pinocchio::integrate(model, q, generalizedVelocity);
  }
  return absl::InternalError(
      "[makeStairStepPlan] the inverse kinematics of a leg did not converge");
}

/** The legs, their targets and the model the configurations are solved on. */
struct
    Solver {  // NOLINT(cppcoreguidelines-pro-type-member-init): absl_nonnull pointers have no null default; built by designated initializers
  const pinocchio::Model* absl_nonnull model;
  pinocchio::Data* absl_nonnull data;
  Leg left;
  Leg right;
};

/** Solves both legs of `q` for the feet at `desPosLfootRtWorldEwrtWorld` and
 * `desPosRfootRtWorldEwrtWorld`. */
absl::Status solveFeet(const Solver& solver,
                       const vector3_t& desPosLfootRtWorldEwrtWorld,
                       const vector3_t& desPosRfootRtWorldEwrtWorld,
                       vector_t& q) {
  RETURN_IF_ERROR(solveLeg(*solver.model, *solver.data, solver.left,
                           desPosLfootRtWorldEwrtWorld, q));
  return solveLeg(*solver.model, *solver.data, solver.right,
                  desPosRfootRtWorldEwrtWorld, q);
}

/** Moves the base of `q` horizontally, the feet held at
 * `desPosLfootRtWorldEwrtWorld` and `desPosRfootRtWorldEwrtWorld`, until the
 * center of mass is over `desPosComRtWorldEwrtWorldXy`. */
absl::Status placeCenterOfMass(const Solver& solver,
                               const vector3_t& desPosLfootRtWorldEwrtWorld,
                               const vector3_t& desPosRfootRtWorldEwrtWorld,
                               const vector2_t& desPosComRtWorldEwrtWorldXy,
                               vector_t& q) {
  for (int iteration = 0; iteration < kComIterations; ++iteration) {
    RETURN_IF_ERROR(solveFeet(solver, desPosLfootRtWorldEwrtWorld,
                              desPosRfootRtWorldEwrtWorld, q));
    const vector3_t posComRtWorldEwrtWorld =
        pinocchio::centerOfMass(*solver.model, *solver.data, q);
    // The desired minus the current CoM position.
    const vector2_t errPosComRtWorldEwrtWorldXy =
        desPosComRtWorldEwrtWorldXy -
        posComRtWorldEwrtWorld.head<kHorizontalPositionDim>();
    if (errPosComRtWorldEwrtWorldXy.norm() <
        kMaxNormErrPosComRtWorldEwrtWorldXy) {
      return absl::OkStatus();
    }
    q.head<kHorizontalPositionDim>() += errPosComRtWorldEwrtWorldXy;
  }
  return absl::InternalError(
      "[makeStairStepPlan] the center of mass could not be placed");
}

/** s = 10 sigma^3 - 15 sigma^4 + 6 sigma^5 of sigma in [0, 1]: at rest and
 * without acceleration at both ends, so that the base's acceleration is
 * continuous across the phases. */
scalar_t smootherstep(scalar_t sigma) {
  const scalar_t x = std::clamp(sigma, 0.0, 1.0);
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

/** One phase of the base: from `initPosBaseRtWorldEwrtWorld` to
 * `finalPosBaseRtWorldEwrtWorld` over [start, end]. */
struct BaseSegment {
  scalar_t start = 0.0;
  scalar_t end = 0.0;
  vector3_t initPosBaseRtWorldEwrtWorld = vector3_t::Zero();
  vector3_t finalPosBaseRtWorldEwrtWorld = vector3_t::Zero();
};

/** The contact phase of `contact` on `primitive` over [start, end] at `pad`.
 */
multi_contact::config::MultiContactPlan::ContactPhase phaseOf(
    absl::string_view contact,
    absl::string_view primitive,
    scalar_t start,
    scalar_t end,
    const vector3_t& desPosPadRtWorldEwrtWorld) {
  multi_contact::config::MultiContactPlan::ContactPhase phase;
  phase.contact = std::string(contact);
  phase.primitive = std::string(primitive);
  phase.face = kTopFace;
  phase.start_time = start;
  phase.end_time = end;
  phase.des_pose_pad_rt_world.resize(2 * kPositionDim);
  phase.des_pose_pad_rt_world << desPosPadRtWorldEwrtWorld, vector3_t::UnitZ();
  return phase;
}

/** The vertical force `weight` [N] on `contact`. */
multi_contact::config::MultiContactPlan::ContactWrench weightOn(
    absl::string_view contact, scalar_t weight) {
  multi_contact::config::MultiContactPlan::ContactWrench wrenchMsg;
  wrenchMsg.contact = std::string(contact);
  // NOLINTBEGIN(frame-naming): the components of a ContactWrench, whose frames its field names
  wrenchMsg.force = vector3_t(0.0, 0.0, weight);
  wrenchMsg.torque = vector3_t::Zero();
  // NOLINTEND(frame-naming)
  return wrenchMsg;
}

}  // namespace

absl::StatusOr<multi_contact::config::MultiContactPlan> makeStairStepPlan(
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& model,
    const multi_contact::LoadedScenario& scenario) {
  PinocchioInterface pinocchio = pinocchioInterface;
  const pinocchio::Model& pinocchioModel = pinocchio.getModel();
  pinocchio::Data& data = pinocchio.getData();
  ASSIGN_OR_RETURN(Leg left, legOf(pinocchioModel, kLeftFoot, kLeftLeg));
  ASSIGN_OR_RETURN(Leg right, legOf(pinocchioModel, kRightFoot, kRightLeg));
  const Solver solver{.model = &pinocchioModel,
                      .data = &data,
                      .left = std::move(left),
                      .right = std::move(right)};
  ASSIGN_OR_RETURN(const int stepIndex, scenario.terrain.primitiveIndex(kStep));

  // The initial configuration of the scenario.
  const multi_contact::config::Scenario::Pose& initialPoseMsg =
      scenario.scenario.initial_posture;
  const std::vector<std::string>& joints =
      model.modelSettings.mpcModelJointNames;
  vector_t q0 = vector_t::Zero(pinocchioModel.nq);
  q0.head<kPositionDim>() =
      initialPoseMsg.pos_base_rt_world_ewrt_world.head<kPositionDim>();
  q0.segment<kQuaternionCoefficients>(kPositionDim) =
      initialPoseMsg.quat_base_rt_world_xyzw.head<kQuaternionCoefficients>();
  for (const multi_contact::config::Scenario::JointPosition& joint :
       initialPoseMsg.joint_positions) {
    const std::vector<std::string>::const_iterator found =
        std::find(joints.begin(), joints.end(), joint.joint);
    if (found == joints.end()) continue;
    q0(kPositionDim + kQuaternionCoefficients +
       static_cast<Eigen::Index>(found - joints.begin())) =
        joint.joint_position;
  }
  pinocchio::forwardKinematics(pinocchioModel, data, q0);
  pinocchio::updateFramePlacements(pinocchioModel, data);
  const vector3_t initPosLfootRtWorldEwrtWorld =
      data.oMf[solver.left.foot].translation();
  const vector3_t initPosRfootRtWorldEwrtWorld =
      data.oMf[solver.right.foot].translation();
  // On the step's top face, a step length ahead: the height of a point far
  // above it less its distance to the step.
  const scalar_t posSurfaceRtWorldEwrtWorldZ =
      kProbePosQueryRtWorldEwrtWorldZ -
      scenario.terrain
          .primitiveQuery<scalar_t>(
              stepIndex, vector3_t(initPosLfootRtWorldEwrtWorld.x() +
                                       kErrPosFootRtWorldEwrtWorldX,
                                   0.0, kProbePosQueryRtWorldEwrtWorldZ))
          .distance;
  const vector3_t desPosLfootRtWorldEwrtWorld(
      initPosLfootRtWorldEwrtWorld.x() + kErrPosFootRtWorldEwrtWorldX,
      initPosLfootRtWorldEwrtWorld.y(), posSurfaceRtWorldEwrtWorldZ);
  const vector3_t desPosRfootRtWorldEwrtWorld(
      initPosRfootRtWorldEwrtWorld.x() + kErrPosFootRtWorldEwrtWorldX,
      initPosRfootRtWorldEwrtWorld.y(), posSurfaceRtWorldEwrtWorldZ);

  // The waypoints of the base: crouched over the right foot, the left foot on
  // the step, over the left foot, both feet on the step, standing.
  vector_t q1 = q0;
  q1(kZAxisIndex) = kCrouchPosBaseRtWorldEwrtWorldZ;
  const vector2_t crouchDesPosComRtWorldEwrtWorldXy(
      initPosRfootRtWorldEwrtWorld.x() + kDesPosComRtRfootEwrtWorldX,
      initPosRfootRtWorldEwrtWorld.y());
  RETURN_IF_ERROR(placeCenterOfMass(solver, initPosLfootRtWorldEwrtWorld,
                                    initPosRfootRtWorldEwrtWorld,
                                    crouchDesPosComRtWorldEwrtWorldXy, q1));
  vector_t q2 = q1;
  RETURN_IF_ERROR(placeCenterOfMass(solver, desPosLfootRtWorldEwrtWorld,
                                    initPosRfootRtWorldEwrtWorld,
                                    crouchDesPosComRtWorldEwrtWorldXy, q2));
  vector_t q3 = q2;
  const vector2_t stepDesPosComRtWorldEwrtWorldXy(
      desPosLfootRtWorldEwrtWorld.x() - kDesPosLfootRtComEwrtWorldX,
      desPosLfootRtWorldEwrtWorld.y() - kDesPosLfootRtComEwrtWorldY);
  RETURN_IF_ERROR(placeCenterOfMass(solver, desPosLfootRtWorldEwrtWorld,
                                    initPosRfootRtWorldEwrtWorld,
                                    stepDesPosComRtWorldEwrtWorldXy, q3));
  vector_t q4 = q3;
  RETURN_IF_ERROR(placeCenterOfMass(solver, desPosLfootRtWorldEwrtWorld,
                                    desPosRfootRtWorldEwrtWorld,
                                    stepDesPosComRtWorldEwrtWorldXy, q4));
  vector_t q5 = q0;
  q5.head<kPositionDim>() +=
      desPosLfootRtWorldEwrtWorld - initPosLfootRtWorldEwrtWorld;
  RETURN_IF_ERROR(solveFeet(solver, desPosLfootRtWorldEwrtWorld,
                            desPosRfootRtWorldEwrtWorld, q5));
  const std::vector<BaseSegment> segments = {
      {.start = 0.0,
       .end = StairStepTimes::kLeftLiftOff,
       .initPosBaseRtWorldEwrtWorld = q0.head<kPositionDim>(),
       .finalPosBaseRtWorldEwrtWorld = q1.head<kPositionDim>()},
      {.start = StairStepTimes::kLeftLiftOff,
       .end = StairStepTimes::kLeftTouchDown,
       .initPosBaseRtWorldEwrtWorld = q1.head<kPositionDim>(),
       .finalPosBaseRtWorldEwrtWorld = q2.head<kPositionDim>()},
      {.start = StairStepTimes::kLeftTouchDown,
       .end = StairStepTimes::kRightLiftOff,
       .initPosBaseRtWorldEwrtWorld = q2.head<kPositionDim>(),
       .finalPosBaseRtWorldEwrtWorld = q3.head<kPositionDim>()},
      {.start = StairStepTimes::kRightLiftOff,
       .end = StairStepTimes::kRightTouchDown,
       .initPosBaseRtWorldEwrtWorld = q3.head<kPositionDim>(),
       .finalPosBaseRtWorldEwrtWorld = q4.head<kPositionDim>()},
      {.start = StairStepTimes::kRightTouchDown,
       .end = StairStepTimes::kEnd,
       .initPosBaseRtWorldEwrtWorld = q4.head<kPositionDim>(),
       .finalPosBaseRtWorldEwrtWorld = q5.head<kPositionDim>()},
  };

  // The swings, on the generalized swing planner's arcs.
  const ContactSwingSettings swingSettings;
  const multi_contact::TerrainModel& terrain = scenario.terrain;
  ASSIGN_OR_RETURN(
      const ContactSwingTrajectory leftSwing,
      ContactSwingTrajectory::Create(
          StairStepTimes::kLeftLiftOff, StairStepTimes::kLeftTouchDown,
          ContactSwingEndpoint{
              .posContactRtWorldEwrtWorld = initPosLfootRtWorldEwrtWorld,
              .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
          ContactSwingEndpoint{
              .posContactRtWorldEwrtWorld = desPosLfootRtWorldEwrtWorld,
              .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
          swingSettings, [&terrain](const vector3_t& posQueryRtWorldEwrtWorld) {
            return terrain.signedDistance<scalar_t>(posQueryRtWorldEwrtWorld);
          }));
  ASSIGN_OR_RETURN(
      const ContactSwingTrajectory rightSwing,
      ContactSwingTrajectory::Create(
          StairStepTimes::kRightLiftOff, StairStepTimes::kRightTouchDown,
          ContactSwingEndpoint{
              .posContactRtWorldEwrtWorld = initPosRfootRtWorldEwrtWorld,
              .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
          ContactSwingEndpoint{
              .posContactRtWorldEwrtWorld = desPosRfootRtWorldEwrtWorld,
              .normalSurfaceEwrtWorld = vector3_t::UnitZ()},
          swingSettings, [&terrain](const vector3_t& posQueryRtWorldEwrtWorld) {
            return terrain.signedDistance<scalar_t>(posQueryRtWorldEwrtWorld);
          }));

  // The knots: the base on its segment, the feet on their pads or arcs.
  std::vector<scalar_t> times;
  std::vector<vector_t> configurations;
  std::vector<multi_contact::ContactMask> masks;
  vector_t q = q0;
  const int numKnots =
      static_cast<int>(
          std::lround(StairStepTimes::kEnd / StairStepTimes::kKnotSpacing)) +
      1;
  for (int k = 0; k < numKnots; ++k) {
    const scalar_t t = static_cast<scalar_t>(k) * StairStepTimes::kKnotSpacing;
    for (const BaseSegment& segment : segments) {
      if (t < segment.start || t > segment.end) continue;
      const scalar_t blend =
          smootherstep((t - segment.start) / (segment.end - segment.start));
      q.head<kPositionDim>() = segment.initPosBaseRtWorldEwrtWorld +
                               blend * (segment.finalPosBaseRtWorldEwrtWorld -
                                        segment.initPosBaseRtWorldEwrtWorld);
    }
    const vector3_t posLfootRtWorldEwrtWorld =
        t < StairStepTimes::kLeftLiftOff
            ? initPosLfootRtWorldEwrtWorld
            : leftSwing.reference(t).posContactRtWorldEwrtWorld;
    const vector3_t posRfootRtWorldEwrtWorld =
        t < StairStepTimes::kRightLiftOff
            ? initPosRfootRtWorldEwrtWorld
            : rightSwing.reference(t).posContactRtWorldEwrtWorld;
    RETURN_IF_ERROR(solveFeet(solver, posLfootRtWorldEwrtWorld,
                              posRfootRtWorldEwrtWorld, q));
    multi_contact::ContactMask mask = 0;
    if (t < StairStepTimes::kLeftLiftOff - kPhaseTimeTolerance ||
        t >= StairStepTimes::kLeftTouchDown - kPhaseTimeTolerance) {
      mask |= kLeftFootBit;
    }
    if (t < StairStepTimes::kRightLiftOff - kPhaseTimeTolerance ||
        t >= StairStepTimes::kRightTouchDown - kPhaseTimeTolerance) {
      mask |= kRightFootBit;
    }
    times.push_back(t);
    configurations.push_back(q);
    masks.push_back(mask);
  }

  // The plan, with the momenta of the configurations' central differences.
  multi_contact::config::MultiContactPlan plan;
  plan.scenario = scenario.scenario.name;
  plan.planner = "hand_written";
  plan.joint_names = joints;
  const scalar_t weight =
      pinocchio::computeTotalMass(pinocchioModel) * kGravityMagnitude;
  for (size_t k = 0; k < configurations.size(); ++k) {
    const size_t previous = k == 0 ? k : k - 1;
    const size_t next = std::min(k + 1, configurations.size() - 1);
    const vector_t generalizedVelocity =
        pinocchio::difference(pinocchioModel, configurations[previous],
                              configurations[next]) /
        (times[next] - times[previous]);
    pinocchio::computeCentroidalMomentum(
        pinocchioModel, data, configurations[k], generalizedVelocity);
    multi_contact::config::MultiContactPlan::Knot knot;
    knot.time = times[k];
    knot.pos_base_rt_world_ewrt_world = configurations[k].head<kPositionDim>();
    knot.quat_base_rt_world_xyzw =
        configurations[k].segment<kQuaternionCoefficients>(kPositionDim);
    knot.joint_positions =
        configurations[k].tail(static_cast<Eigen::Index>(joints.size()));
    knot.linmom_com_rt_world_ewrt_world = data.hg.linear();
    knot.angmom_com_rt_world_ewrt_world = data.hg.angular();
    knot.active_contacts = masks[k];
    // The weight shared by the feet down, for the playback.
    const bool leftDown = (masks[k] & kLeftFootBit) != 0;
    const bool rightDown = (masks[k] & kRightFootBit) != 0;
    const scalar_t share =
        weight / static_cast<scalar_t>(static_cast<int>(leftDown) +
                                       static_cast<int>(rightDown));
    if (leftDown) {
      knot.wrench_contacts_ewrt_world.push_back(weightOn(kLeftFoot, share));
    }
    if (rightDown) {
      knot.wrench_contacts_ewrt_world.push_back(weightOn(kRightFoot, share));
    }
    plan.knots.push_back(std::move(knot));
  }
  plan.phases = {
      phaseOf(kLeftFoot, kGround, /*start=*/0.0, StairStepTimes::kLeftLiftOff,
              initPosLfootRtWorldEwrtWorld),
      phaseOf(kLeftFoot, kStep, StairStepTimes::kLeftTouchDown,
              StairStepTimes::kEnd, desPosLfootRtWorldEwrtWorld),
      phaseOf(kRightFoot, kGround, /*start=*/0.0, StairStepTimes::kRightLiftOff,
              initPosRfootRtWorldEwrtWorld),
      phaseOf(kRightFoot, kStep, StairStepTimes::kRightTouchDown,
              StairStepTimes::kEnd, desPosRfootRtWorldEwrtWorld),
  };
  multi_contact::stampPlanHashes(scenario.contactSetConfig,
                                 scenario.scenario.terrain_scene, plan);
  return plan;
}

}  // namespace ocs2::legged
