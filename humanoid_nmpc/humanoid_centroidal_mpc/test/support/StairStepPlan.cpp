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
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/PhysicalConstants.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/spatial/explog.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_swing/ContactSwingTrajectory.h"
#include "humanoid_multi_contact/plan/PlanFile.h"

namespace ocs2::humanoid {
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
constexpr scalar_t kComTolerance = 1.0e-7;  // [m]
/** [m] The base height of the crouch the steps are taken in. */
constexpr scalar_t kCrouchHeight = 0.84;
/** [m] Where the center of mass is put over a stance foot's pad: forward
 * over the right foot on the floor, back over the left foot on the step. */
constexpr scalar_t kComAheadOfRightPad = 0.01;
constexpr scalar_t kComBehindLeftPad = 0.06;
constexpr scalar_t kComInsideLeftPad = 0.02;
/** [m] Where the left foot lands on the step, ahead of where it stands. */
constexpr scalar_t kStepLength = 0.292;
/** [m] A height above every step, where the distance to a step's top face is
 * the vertical one. */
constexpr scalar_t kProbeHeight = 10.0;

/** The model's leg of `joints` and its foot frame. */
struct Leg {
  pinocchio::FrameIndex foot = 0;
  std::vector<Eigen::Index> velocityColumns;
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
    leg.velocityColumns.push_back(static_cast<Eigen::Index>(
        model.joints[model.getJointId(std::string(joint))].idx_v()));
  }
  return leg;
}

/** Moves the joints of `leg` so that its foot frame is at `target` with a
 * level sole (world orientation), the base held. */
absl::Status solveLeg(const pinocchio::Model& model,
                      pinocchio::Data& data,
                      const Leg& leg,
                      const vector3_t& target,
                      vector_t& q) {
  const pinocchio::SE3 goal(matrix3_t::Identity(), target);
  const Eigen::Index numColumns =
      static_cast<Eigen::Index>(leg.velocityColumns.size());
  for (int iteration = 0; iteration < kIkIterations; ++iteration) {
    pinocchio::forwardKinematics(model, data, q);
    const pinocchio::SE3& placement =
        pinocchio::updateFramePlacement(model, data, leg.foot);
    const vector6_t error = pinocchio::log6(placement.actInv(goal)).toVector();
    if (error.norm() < kIkTolerance) return absl::OkStatus();
    pinocchio::computeJointJacobians(model, data, q);
    matrix_t jacobian = matrix_t::Zero(kTwistDim, model.nv);
    pinocchio::getFrameJacobian(model, data, leg.foot, pinocchio::LOCAL,
                                jacobian);
    matrix_t legJacobian(kTwistDim, numColumns);
    for (Eigen::Index c = 0; c < numColumns; ++c) {
      legJacobian.col(c) = jacobian.col(leg.velocityColumns[c]);
    }
    const vector_t step =
        (legJacobian.transpose() * legJacobian +
         kIkDamping * matrix_t::Identity(numColumns, numColumns))
            .ldlt()
            .solve(legJacobian.transpose() * error);
    vector_t velocity = vector_t::Zero(model.nv);
    for (Eigen::Index c = 0; c < numColumns; ++c) {
      velocity(leg.velocityColumns[c]) = step(c);
    }
    q = pinocchio::integrate(model, q, velocity);
  }
  return absl::InternalError(
      "[makeStairStepPlan] the inverse kinematics of a leg did not converge");
}

/** The legs, their targets and the model the configurations are solved on. */
struct Solver {
  const pinocchio::Model* absl_nonnull model;
  pinocchio::Data* absl_nonnull data;
  Leg left;
  Leg right;
};

/** Solves both legs of `q` for the feet at `left` and `right`. */
absl::Status solveFeet(const Solver& solver,
                       const vector3_t& left,
                       const vector3_t& right,
                       vector_t& q) {
  RETURN_IF_ERROR(solveLeg(*solver.model, *solver.data, solver.left, left, q));
  return solveLeg(*solver.model, *solver.data, solver.right, right, q);
}

/** Moves the base of `q` horizontally, the feet held at `left` and `right`,
 * until the center of mass is over `com`. */
absl::Status placeCenterOfMass(const Solver& solver,
                               const vector3_t& left,
                               const vector3_t& right,
                               const vector2_t& com,
                               vector_t& q) {
  for (int iteration = 0; iteration < kComIterations; ++iteration) {
    RETURN_IF_ERROR(solveFeet(solver, left, right, q));
    const vector3_t current =
        pinocchio::centerOfMass(*solver.model, *solver.data, q);
    const vector2_t error = com - current.head<kHorizontalPositionDim>();
    if (error.norm() < kComTolerance) return absl::OkStatus();
    q.head<kHorizontalPositionDim>() += error;
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

/** One phase of the base: from `from` to `to` over [start, end]. */
struct BaseSegment {
  scalar_t start = 0.0;
  scalar_t end = 0.0;
  vector3_t from = vector3_t::Zero();
  vector3_t to = vector3_t::Zero();
};

/** The contact phase of `contact` on `primitive` over [start, end] at `pad`.
 */
multi_contact::config::MultiContactPlan::ContactPhase phaseOf(
    absl::string_view contact,
    absl::string_view primitive,
    scalar_t start,
    scalar_t end,
    const vector3_t& pad) {
  multi_contact::config::MultiContactPlan::ContactPhase phase;
  phase.contact = std::string(contact);
  phase.primitive = std::string(primitive);
  phase.face = kTopFace;
  phase.start_time = start;
  phase.end_time = end;
  phase.target_pad_pose.resize(2 * kPositionDim);
  phase.target_pad_pose << pad, vector3_t::UnitZ();
  return phase;
}

/** The vertical force `weight` [N] on `contact`. */
multi_contact::config::MultiContactPlan::ContactWrench weightOn(
    absl::string_view contact, scalar_t weight) {
  multi_contact::config::MultiContactPlan::ContactWrench wrench;
  wrench.contact = std::string(contact);
  wrench.force = vector3_t(0.0, 0.0, weight);
  wrench.torque = vector3_t::Zero();
  return wrench;
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
  const multi_contact::config::Scenario::Pose& pose =
      scenario.scenario.initial_pose;
  const std::vector<std::string>& joints =
      model.modelSettings.mpcModelJointNames;
  vector_t q0 = vector_t::Zero(pinocchioModel.nq);
  q0.head<kPositionDim>() = pose.base_position.head<kPositionDim>();
  q0.segment<kQuaternionCoefficients>(kPositionDim) =
      pose.base_orientation_xyzw.head<kQuaternionCoefficients>();
  for (const multi_contact::config::Scenario::JointPosition& joint :
       pose.joint_positions) {
    const std::vector<std::string>::const_iterator found =
        std::find(joints.begin(), joints.end(), joint.joint);
    if (found == joints.end()) continue;
    q0(kPositionDim + kQuaternionCoefficients +
       static_cast<Eigen::Index>(found - joints.begin())) = joint.position;
  }
  pinocchio::forwardKinematics(pinocchioModel, data, q0);
  pinocchio::updateFramePlacements(pinocchioModel, data);
  const vector3_t leftPad = data.oMf[solver.left.foot].translation();
  const vector3_t rightPad = data.oMf[solver.right.foot].translation();
  // On the step's top face, a step length ahead: the height of a point far
  // above it less its distance to the step.
  const scalar_t stepTop =
      kProbeHeight - scenario.terrain
                         .primitiveQuery<scalar_t>(
                             stepIndex, vector3_t(leftPad.x() + kStepLength,
                                                  0.0, kProbeHeight))
                         .distance;
  const vector3_t leftTarget(leftPad.x() + kStepLength, leftPad.y(), stepTop);
  const vector3_t rightTarget(rightPad.x() + kStepLength, rightPad.y(),
                              stepTop);

  // The waypoints of the base: crouched over the right foot, the left foot on
  // the step, over the left foot, both feet on the step, standing.
  vector_t q1 = q0;
  q1(kZAxisIndex) = kCrouchHeight;
  const vector2_t overRight(rightPad.x() + kComAheadOfRightPad, rightPad.y());
  RETURN_IF_ERROR(placeCenterOfMass(solver, leftPad, rightPad, overRight, q1));
  vector_t q2 = q1;
  RETURN_IF_ERROR(
      placeCenterOfMass(solver, leftTarget, rightPad, overRight, q2));
  vector_t q3 = q2;
  const vector2_t overLeft(leftTarget.x() - kComBehindLeftPad,
                           leftTarget.y() - kComInsideLeftPad);
  RETURN_IF_ERROR(
      placeCenterOfMass(solver, leftTarget, rightPad, overLeft, q3));
  vector_t q4 = q3;
  RETURN_IF_ERROR(
      placeCenterOfMass(solver, leftTarget, rightTarget, overLeft, q4));
  vector_t q5 = q0;
  q5.head<kPositionDim>() += leftTarget - leftPad;
  RETURN_IF_ERROR(solveFeet(solver, leftTarget, rightTarget, q5));
  const std::vector<BaseSegment> segments = {
      {.start = 0.0,
       .end = StairStepTimes::kLeftLiftOff,
       .from = q0.head<kPositionDim>(),
       .to = q1.head<kPositionDim>()},
      {.start = StairStepTimes::kLeftLiftOff,
       .end = StairStepTimes::kLeftTouchDown,
       .from = q1.head<kPositionDim>(),
       .to = q2.head<kPositionDim>()},
      {.start = StairStepTimes::kLeftTouchDown,
       .end = StairStepTimes::kRightLiftOff,
       .from = q2.head<kPositionDim>(),
       .to = q3.head<kPositionDim>()},
      {.start = StairStepTimes::kRightLiftOff,
       .end = StairStepTimes::kRightTouchDown,
       .from = q3.head<kPositionDim>(),
       .to = q4.head<kPositionDim>()},
      {.start = StairStepTimes::kRightTouchDown,
       .end = StairStepTimes::kEnd,
       .from = q4.head<kPositionDim>(),
       .to = q5.head<kPositionDim>()},
  };

  // The swings, on the generalized swing planner's arcs.
  const ContactSwingSettings swingSettings;
  const multi_contact::TerrainModel& terrain = scenario.terrain;
  ASSIGN_OR_RETURN(
      const ContactSwingTrajectory leftSwing,
      ContactSwingTrajectory::Create(
          StairStepTimes::kLeftLiftOff, StairStepTimes::kLeftTouchDown,
          ContactSwingEndpoint{.position = leftPad,
                               .normal = vector3_t::UnitZ()},
          ContactSwingEndpoint{.position = leftTarget,
                               .normal = vector3_t::UnitZ()},
          swingSettings, [&terrain](const vector3_t& point) {
            return terrain.signedDistance<scalar_t>(point);
          }));
  ASSIGN_OR_RETURN(
      const ContactSwingTrajectory rightSwing,
      ContactSwingTrajectory::Create(
          StairStepTimes::kRightLiftOff, StairStepTimes::kRightTouchDown,
          ContactSwingEndpoint{.position = rightPad,
                               .normal = vector3_t::UnitZ()},
          ContactSwingEndpoint{.position = rightTarget,
                               .normal = vector3_t::UnitZ()},
          swingSettings, [&terrain](const vector3_t& point) {
            return terrain.signedDistance<scalar_t>(point);
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
      q.head<kPositionDim>() =
          segment.from + blend * (segment.to - segment.from);
    }
    const vector3_t leftFoot = t < StairStepTimes::kLeftLiftOff
                                   ? leftPad
                                   : leftSwing.reference(t).position;
    const vector3_t rightFoot = t < StairStepTimes::kRightLiftOff
                                    ? rightPad
                                    : rightSwing.reference(t).position;
    RETURN_IF_ERROR(solveFeet(solver, leftFoot, rightFoot, q));
    multi_contact::ContactMask mask = 0;
    if (t < StairStepTimes::kLeftLiftOff - kComTolerance ||
        t >= StairStepTimes::kLeftTouchDown - kComTolerance) {
      mask |= kLeftFootBit;
    }
    if (t < StairStepTimes::kRightLiftOff - kComTolerance ||
        t >= StairStepTimes::kRightTouchDown - kComTolerance) {
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
      pinocchio::computeTotalMass(pinocchioModel) * kGravitationalAcceleration;
  for (size_t k = 0; k < configurations.size(); ++k) {
    const size_t previous = k == 0 ? k : k - 1;
    const size_t next = std::min(k + 1, configurations.size() - 1);
    const vector_t velocity =
        pinocchio::difference(pinocchioModel, configurations[previous],
                              configurations[next]) /
        (times[next] - times[previous]);
    pinocchio::computeCentroidalMomentum(pinocchioModel, data,
                                         configurations[k], velocity);
    multi_contact::config::MultiContactPlan::Knot knot;
    knot.time = times[k];
    knot.base_position = configurations[k].head<kPositionDim>();
    knot.base_orientation_xyzw =
        configurations[k].segment<kQuaternionCoefficients>(kPositionDim);
    knot.joint_positions =
        configurations[k].tail(static_cast<Eigen::Index>(joints.size()));
    knot.linear_momentum = data.hg.linear();
    knot.angular_momentum = data.hg.angular();
    knot.active_contacts = masks[k];
    // The weight shared by the feet down, for the playback.
    const bool leftDown = (masks[k] & kLeftFootBit) != 0;
    const bool rightDown = (masks[k] & kRightFootBit) != 0;
    const scalar_t share =
        weight / static_cast<scalar_t>(static_cast<int>(leftDown) +
                                       static_cast<int>(rightDown));
    if (leftDown) knot.contact_wrenches.push_back(weightOn(kLeftFoot, share));
    if (rightDown) knot.contact_wrenches.push_back(weightOn(kRightFoot, share));
    plan.knots.push_back(std::move(knot));
  }
  plan.phases = {
      phaseOf(kLeftFoot, kGround, /*start=*/0.0, StairStepTimes::kLeftLiftOff,
              leftPad),
      phaseOf(kLeftFoot, kStep, StairStepTimes::kLeftTouchDown,
              StairStepTimes::kEnd, leftTarget),
      phaseOf(kRightFoot, kGround, /*start=*/0.0, StairStepTimes::kRightLiftOff,
              rightPad),
      phaseOf(kRightFoot, kStep, StairStepTimes::kRightTouchDown,
              StairStepTimes::kEnd, rightTarget),
  };
  multi_contact::stampPlanHashes(scenario.contactSetConfig,
                                 scenario.scenario.terrain_scene, plan);
  return plan;
}

}  // namespace ocs2::humanoid
