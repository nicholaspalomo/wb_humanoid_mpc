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

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The rigid-body dynamics the MRT joint controllers evaluate on the realtime
 * thread, without a heap allocation: the gravity torques of the passive modes,
 * the floating-base inverse dynamics of a policy (base free) and of the weight
 * compensation (base held), the centroidal momentum of the observation and the
 * orientations of the sole frames.
 *
 * Every function takes and returns the MPC model's coordinates - the
 * configuration q = [p_W, unit quaternion (x, y, z, w), q_j] and the velocity v
 * = [dp_W, w_B, dq_j] of its Composite(Translation, Spherical) root - but runs
 * Pinocchio on a twin of the model whose root joint is a FreeFlyer: the same
 * joint indices, bodies, inertias and frames (frame indices included), the same
 * q, and the velocity [R' dp_W, w_B, dq_j]. Pinocchio's passes over the
 * composite joint build its dynamic-size motion subspace into a heap temporary
 * in every call (crba(), rnea(), nonLinearEffects(), computeJointJacobians(),
 * computeCentroidalMap()), the FreeFlyer's subspace is the fixed-size identity;
 * the joint torques, the momentum and the frame placements are the robot's and
 * do not depend on how the base velocity is coordinatized
 * (testRealtimeRigidBodyDynamics holds them to the MPC model's own passes and
 * RNEA). Each result lives in a workspace sized by Create() and stays valid
 * until the next call of the same function.
 *
 * Not thread-safe: one per control thread. Copying is disabled; it is owned
 * through the std::unique_ptr of Create().
 */
class RealtimeRigidBodyDynamics {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * The twin of `mpcModel`, whose root (joint 1) must be the MPC models'
   * Composite(Translation, Spherical) with nothing between its two parts, and
   * the sole frames `contactFrames` of the contacts, in the order of the
   * contacts.
   *
   * @return InvalidArgument for another root, NotFound for a contact frame
   * `mpcModel` does not have.
   */
  static absl::StatusOr<std::unique_ptr<RealtimeRigidBodyDynamics>> Create(
      const PinocchioInterface::Model& mpcModel,
      const feet_array_t<pinocchio::FrameIndex>& contactFrames);

  RealtimeRigidBodyDynamics(const RealtimeRigidBodyDynamics&) = delete;
  RealtimeRigidBodyDynamics& operator=(const RealtimeRigidBodyDynamics&) =
      delete;
  RealtimeRigidBodyDynamics(RealtimeRigidBodyDynamics&&) = delete;
  RealtimeRigidBodyDynamics& operator=(RealtimeRigidBodyDynamics&&) = delete;
  ~RealtimeRigidBodyDynamics() = default;

  /** The joint rows of the generalized gravity g(q) (nv - 6): the torques that
   * hold the joints with the base held. */
  const vector_t& gravityTorques(const Eigen::Ref<const vector_t>& q);

  /**
   * The joint torques that produce the joint accelerations `qddJoints` (nv - 6)
   * at (q, v) while the feet push on the ground with the world-frame wrenches
   * `footWrenches` (force, then moment about the origin of the sole frame), the
   * base being free:
   *   a_b = M_bb^-1 ((J^T W)_b - nle_b - M_bj qdd_j),    tau_j = M_jb a_b +
   * M_jj qdd_j + nle_j - (J^T W)_j, the joint rows of M qdd + nle = S^T tau +
   * J^T W (computeJointTorques() of DynamicsHelperFunctions.h, which builds the
   * same on the MPC model with vectors of its own). The base acceleration it
   * solved is getBaseAcceleration().
   */
  const vector_t& freeBaseJointTorques(
      const Eigen::Ref<const vector_t>& q,
      const Eigen::Ref<const vector_t>& v,
      const Eigen::Ref<const vector_t>& qddJoints,
      const feet_array_t<vector6_t>& footWrenches);

  /**
   * As freeBaseJointTorques(), the base held instead (by the gantry, or as if
   * it were): its acceleration [ddp_W, dw_B] is zero, tau_j = M_jj qdd_j +
   * nle_j - (J^T W)_j (computeBaseHeldJointTorques() of
   * DynamicsHelperFunctions.h).
   */
  const vector_t& heldBaseJointTorques(
      const Eigen::Ref<const vector_t>& q,
      const Eigen::Ref<const vector_t>& v,
      const Eigen::Ref<const vector_t>& qddJoints,
      const feet_array_t<vector6_t>& footWrenches);

  /** The base acceleration [ddp_W, dw_B] of the MPC model's coordinates that
   * the last freeBaseJointTorques() solved. */
  const vector6_t& getBaseAcceleration() const { return baseAcceleration_; }

  /**
   * The centroidal momentum h = A(q) v at (q, v): the linear momentum, then the
   * angular momentum about the center of mass, both in world axes (Pinocchio's
   * ccrba(); OCS2's computeCentroidalMap() times v on the MPC model).
   */
  const vector6_t& centroidalMomentum(const Eigen::Ref<const vector_t>& q,
                                      const Eigen::Ref<const vector_t>& v);

  /** The orientations (sole frame to world) of the contacts' sole frames at q,
   * in the order of the contacts. */
  const feet_array_t<matrix3_t>& contactFrameRotations(
      const Eigen::Ref<const vector_t>& q);

  /** The twin: the MPC model with a FreeFlyer root. */
  const PinocchioInterface::Model& getModel() const { return model_; }

 private:
  RealtimeRigidBodyDynamics(
      const PinocchioInterface::Model& twin,
      const feet_array_t<pinocchio::FrameIndex>& contactFrames);

  /** q into configuration_ and the twin's velocity [R' dp_W, w_B, dq_j] of (q,
   * v) into velocity_. */
  void setState(const Eigen::Ref<const vector_t>& q,
                const Eigen::Ref<const vector_t>& v);
  /** J^T W of the sole frames' LOCAL_WORLD_ALIGNED Jacobians at configuration_
   * into generalizedContactForce_. */
  void computeGeneralizedContactForce(
      const feet_array_t<vector6_t>& footWrenches);
  /** The rotation (base to world) of configuration_'s base quaternion. */
  matrix3_t baseRotation() const;

  PinocchioInterface::Model model_;
  PinocchioInterface::Data data_;
  feet_array_t<pinocchio::FrameIndex> contactFrames_;
  vector_t configuration_;  ///< nq
  vector_t velocity_;       ///< nv, the twin's
  vector_t acceleration_;   ///< nv, the twin's
  matrix_t soleJacobian_;   ///< 6 x nv, the twin's LOCAL_WORLD_ALIGNED Jacobian
                            ///< of one sole frame
  vector_t generalizedContactForce_;  ///< nv, J^T W; its joint rows are those
                                      ///< of the MPC model
  vector_t jointTorques_;             ///< nv - 6
  vector_t gravityTorques_;           ///< nv - 6
  vector6_t baseAcceleration_ = vector6_t::Zero();
  vector6_t centroidalMomentum_ = vector6_t::Zero();
  feet_array_t<matrix3_t> contactFrameRotations_;
};

}  // namespace ocs2::humanoid
