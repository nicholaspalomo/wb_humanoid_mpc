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

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/spatial/se3.hpp"

#include "humanoid_multi_contact/contact_set/ContactSet.h"

namespace ocs2::humanoid::multi_contact {

/** A whole-body configuration by name: the floating base's pose and the joint
 * positions; a joint left out is at 0. Passive data. */
struct RobotConfiguration {
  Eigen::Vector3d basePosition = Eigen::Vector3d::Zero();
  Eigen::Quaterniond baseOrientation = Eigen::Quaterniond::Identity();
  std::vector<std::string> jointNames;
  // One per name.
  Eigen::VectorXd jointPositions;
};

/** One actuated joint of the model: a joint of one degree of freedom. */
struct ActuatedJoint {
  std::string name;
  // Its index in the generalized velocity and force vectors.
  int velocityIndex = 0;
  // Its index in the generalized position vector.
  int positionIndex = 0;
  // [N m] The URDF's effort limit; a joint without one (0) is not checked.
  double effortLimit = 0.0;
  // [rad] The URDF's position limits, lowerLimit <= upperLimit.
  double lowerLimit = 0.0;
  double upperLimit = 0.0;
};

/** The centroidal momentum about the center of mass, in world-aligned axes.
 * Passive data. */
struct CentroidalMomentum {
  // [kg m/s] m c_dot.
  Eigen::Vector3d linear = Eigen::Vector3d::Zero();
  // [kg m^2/s] The angular momentum about the center of mass.
  Eigen::Vector3d angular = Eigen::Vector3d::Zero();
};

/**
 * The robot's whole-body kinematics and statics for the stability certifier
 * (design sections 2 and 6): a Pinocchio model of the URDF with a free-flyer
 * root ("root_joint"), the parent joint of each contact candidate, and at one
 * configuration the pads' poses, the center of mass, the gravity torques g(q)
 * and the Jacobians of points on the pads' links.
 *
 * Not thread-safe: setConfiguration() and setGeneralizedPosition() write the
 * Pinocchio data that the const accessors read.
 */
class RobotStanceModel {
 public:
  /**
   * Builds the model of the URDF at `urdfPath` for the candidates of
   * `contacts` (copied). InvalidArgument when Pinocchio cannot read the URDF;
   * NotFound naming the candidate when its parent joint is not a joint of the
   * URDF (or "root_joint").
   */
  static absl::StatusOr<std::unique_ptr<RobotStanceModel>> Create(
      absl::string_view urdfPath, const ContactSet& contacts);

  RobotStanceModel(const RobotStanceModel&) = delete;
  RobotStanceModel& operator=(const RobotStanceModel&) = delete;
  ~RobotStanceModel() = default;

  /**
   * Computes the kinematics, the center of mass and its Jacobian, g(q) and the
   * joint Jacobians at `configuration`. InvalidArgument for a name that is not
   * a joint of one degree of freedom, a name given twice, a size mismatch, a
   * quaternion that is not unit (within 1e-6) or a value that is not finite;
   * the previous configuration then stays.
   */
  absl::Status setConfiguration(const RobotConfiguration& configuration);

  /**
   * The configuration as Pinocchio's generalized position (model().nq values:
   * the base position, its quaternion x, y, z, w, then the joints in model
   * order); the errors of setConfiguration() but for the base quaternion's
   * norm, which is normalized. Does not change the model's configuration.
   */
  absl::StatusOr<Eigen::VectorXd> generalizedPositionOf(
      const RobotConfiguration& configuration) const;

  /** The configuration of the generalized position `q` by name, with every
   * actuated joint named, in model order; requires q.size() == model().nq. */
  RobotConfiguration configurationOf(const Eigen::VectorXd& q) const;

  /**
   * Computes what setConfiguration() computes at the generalized position `q`.
   * InvalidArgument for a vector of the wrong size, a value that is not finite
   * or a base quaternion that is not unit (within 1e-6); the previous
   * configuration then stays.
   */
  absl::Status setGeneralizedPosition(const Eigen::VectorXd& q);

  /**
   * The centroidal momentum at the generalized position `q` and velocity `v`
   * (model().nv values, the base's twist in its own frame as Pinocchio's
   * free-flyer takes it). Sets the model's configuration to `q`, as
   * setGeneralizedPosition(); its errors, and InvalidArgument for a velocity of
   * the wrong size or not finite.
   */
  absl::StatusOr<CentroidalMomentum> centroidalMomentum(
      const Eigen::VectorXd& q, const Eigen::VectorXd& v);

  const ContactSet& contactSet() const { return contacts_; }
  const pinocchio::Model& model() const { return model_; }
  /** [kg] */
  double mass() const { return mass_; }
  /** The actuated joints, in the model's order. */
  const std::vector<ActuatedJoint>& actuatedJoints() const {
    return actuatedJoints_;
  }
  /** Widens every actuated joint's position limits (actuatedJoints()) by
   * `relaxation` [rad] on both sides: JointLimits::kRelaxedPosition.
   * InvalidArgument for a relaxation that is negative or not finite. */
  absl::Status relaxPositionLimits(double relaxation);
  /** Whether `name` is a joint of the model ("root_joint" included). */
  bool hasJoint(absl::string_view name) const;
  /** The index of the joint `name` ("root_joint" included); NotFound names
   * it. */
  absl::StatusOr<pinocchio::JointIndex> jointIndex(
      absl::string_view name) const;
  /** The parent joint of candidate `index`; requires 0 <= index <
   * contactSet().size(). */
  pinocchio::JointIndex padJoint(int index) const { return padJoints_[index]; }

  // At the configuration of the last successful setConfiguration():

  /** [m] The center of mass in the world frame. */
  const Eigen::Vector3d& centerOfMass() const { return centerOfMass_; }
  /** [m] The center of candidate `index`'s pad in the world frame. */
  Eigen::Vector3d padPosition(int index) const;
  /** world_R_pad of candidate `index`: z = -padNormal (ContactCandidate). */
  Eigen::Matrix3d padRotation(int index) const;
  /** [N m] The generalized gravity forces g(q), of the size of the velocity. */
  const Eigen::VectorXd& generalizedGravity() const { return data_.g; }
  /**
   * The Jacobian (3 x nv) of the velocity of `point` (world frame) moving with
   * candidate `index`'s parent link, whose transpose maps a force at the point
   * to generalized forces.
   */
  Eigen::Matrix<double, kPositionDim, Eigen::Dynamic> pointJacobian(
      int index, const Eigen::Vector3d& point) const;

  /** The generalized position of the current configuration. */
  const Eigen::VectorXd& generalizedPosition() const { return q_; }
  /** The placement of joint `joint`'s frame in the world; requires a joint of
   * the model. */
  const pinocchio::SE3& jointPlacement(pinocchio::JointIndex joint) const {
    return data_.oMi[joint];
  }
  /** The Jacobian (3 x nv) of the velocity of `point` (world frame) moving
   * with joint `joint`'s link. */
  Eigen::Matrix<double, kPositionDim, Eigen::Dynamic> jointPointJacobian(
      pinocchio::JointIndex joint, const Eigen::Vector3d& point) const;
  /** The Jacobian (3 x nv) of the angular velocity of joint `joint`'s link, in
   * world axes. */
  Eigen::Matrix<double, kRotationTangentDim, Eigen::Dynamic>
  jointAngularJacobian(pinocchio::JointIndex joint) const;
  /** The Jacobian (3 x nv) of the center of mass. */
  const Eigen::Matrix<double, kPositionDim, Eigen::Dynamic>&
  centerOfMassJacobian() const {
    return centerOfMassJacobian_;
  }

 private:
  RobotStanceModel(pinocchio::Model model,
                   ContactSet contacts,
                   std::vector<pinocchio::JointIndex> padJoints);

  pinocchio::Model model_;
  pinocchio::Data data_;
  ContactSet contacts_;
  // The parent joint of each candidate.
  std::vector<pinocchio::JointIndex> padJoints_;
  std::vector<ActuatedJoint> actuatedJoints_;
  double mass_ = 0.0;
  Eigen::Vector3d centerOfMass_ = Eigen::Vector3d::Zero();
  // The generalized position of the last successful configuration.
  Eigen::VectorXd q_;
  Eigen::Matrix<double, kPositionDim, Eigen::Dynamic> centerOfMassJacobian_;
};

}  // namespace ocs2::humanoid::multi_contact
