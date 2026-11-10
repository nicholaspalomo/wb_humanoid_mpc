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
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid::estimation {

/**
 * The name of the MPC model's free-flyer joint: the root joint Pinocchio's URDF
 * parser adds (pinocchio::urdf::buildModel() with a root joint, as
 * getPinocchioInterfaceFromUrdfFile() builds the MPC's model), from which a
 * contact on the base hangs (model_settings.contact_parent_joint_names). In the
 * fixed-base model of ContactKinematics it is the URDF's root link.
 */
inline constexpr char kMpcRootJointName[] = "root_joint";

/**
 * A contact point of the robot: a URDF link, which is a BODY frame of the
 * fixed-base Pinocchio model, and the point in that link's frame.
 */
struct ContactPointSpec {
  /** The link's name in the URDF. */
  std::string linkName;
  /** [m] The contact point in the link's frame. */
  vector3_t posContactRtLinkEwrtLink = vector3_t::Zero();
};

/**
 * Returns the child link of each joint of `jointNames` in the URDF `urdfFile`,
 * in their order: the link whose frame the joint places. kMpcRootJointName maps
 * to the URDF's root link. A URDF child link's frame is its joint's frame, so a
 * point given relative to a joint's frame (an MPC contact's offset from its
 * parent joint) is the same point relative to the child link. A joint of type
 * fixed in the URDF has a child link too. Parses the URDF once.
 *
 * @return InvalidArgument for a URDF that does not load, or naming a joint the
 * URDF does not have.
 */
absl::StatusOr<std::vector<std::string>> childLinkNamesOfJoints(
    absl::string_view urdfFile, absl::Span<const std::string> jointNames);

/**
 * The leg kinematics of a floating-base robot, relative to its base: where each
 * contact point is in the base frame and how fast it moves relative to the
 * base, from the joint positions and velocities alone.
 *
 * The URDF is built as a FIXED-BASE Pinocchio model
 * (pinocchio::urdf::buildModel() without a root joint), so the base link is the
 * model's universe and every result is in the base frame: p_rel = p_contact -
 * p_base expressed in the base axes, and v_rel = J(q) q_dot, the velocity of
 * the contact point relative to the base, in the base axes. The world velocity
 * of a contact point follows as v_contact = v_base + R (omega x p_rel + v_rel),
 * with R the base's orientation and omega its angular velocity in the base
 * frame.
 *
 * The joint vectors given to update() are in the robot's joint order (the joint
 * names given to Create(), by robot joint index); a model joint the robot does
 * not name stays at its neutral position with zero velocity. Revolute and
 * prismatic joints (one coordinate) and continuous joints (Pinocchio's
 * unbounded revolute, q = (cos, sin)) are supported.
 *
 * update() allocates nothing; the class is not thread-safe.
 */
class ContactKinematics {
 public:
  /**
   * The kinematics of `contacts` on the URDF `urdfFile`, with the joint vectors
   * in the order of `jointNames`. InvalidArgument for a URDF that does not
   * parse, a contact whose link is not in it (the message lists its links), a
   * joint of an unsupported type, or no contact.
   */
  static absl::StatusOr<std::unique_ptr<ContactKinematics>> Create(
      absl::string_view urdfFile,
      absl::Span<const ContactPointSpec> contacts,
      absl::Span<const std::string> jointNames);

  ContactKinematics(const ContactKinematics&) = delete;
  ContactKinematics& operator=(const ContactKinematics&) = delete;
  ContactKinematics(ContactKinematics&&) = delete;
  ContactKinematics& operator=(ContactKinematics&&) = delete;
  ~ContactKinematics();

  size_t numContacts() const { return contactLinkNames_.size(); }
  /** The link of contact `contact` (ContactPointSpec::linkName). */
  const std::string& contactLinkName(size_t contact) const {
    return contactLinkNames_[contact];
  }
  /** The number of joints update() takes: the size of the joint names given to
   * Create(). */
  size_t numJoints() const { return numRobotJoints_; }
  /** [m/s^2] The gravity of the model, in the world (base-aligned at rest)
   * axes: (0, 0, -9.81). */
  const vector3_t& gravityEwrtWorld() const { return gravityEwrtWorld_; }

  /**
   * Computes every contact point's position and velocity relative to the base.
   * `jointPositions` and `jointVelocities` have numJoints() entries in the
   * robot's joint order. Allocates nothing.
   */
  void update(const vector_t& jointPositions, const vector_t& jointVelocities);

  /** [m] Contact `contact`'s position relative to the base, in the base frame,
   * as of the last update(). */
  const vector3_t& posContactRtBaseEwrtBase(size_t contact) const {
    return posContactsRtBaseEwrtBase_[contact];
  }
  /** [m/s] Contact `contact`'s velocity relative to the base (J q_dot), in the
   * base frame, as of the last update(). */
  const vector3_t& linvelContactRtBaseEwrtBase(size_t contact) const {
    return linvelContactsRtBaseEwrtBase_[contact];
  }

 private:
  /** The Pinocchio model and data and how the robot's joints map onto them
   * (defined in ContactKinematics.cpp). */
  struct Model;

  ContactKinematics(std::unique_ptr<Model> model,
                    std::vector<std::string> contactLinkNames,
                    size_t numRobotJoints);

  /** Never null. */
  std::unique_ptr<Model> model_;
  std::vector<std::string> contactLinkNames_;
  size_t numRobotJoints_ = 0;
  vector3_t gravityEwrtWorld_ = vector3_t::Zero();
  std::vector<vector3_t> posContactsRtBaseEwrtBase_;
  std::vector<vector3_t> linvelContactsRtBaseEwrtBase_;
};

}  // namespace ocs2::humanoid::estimation
