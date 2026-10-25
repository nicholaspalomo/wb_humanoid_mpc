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

#include "Eigen/Core"
#include "Eigen/Dense"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * Structure holding parameters for a single SIREN layer in C++.
 */
struct SirenLayerWeights {
  matrix_t weight;  // (out_dim x in_dim)
  vector_t bias;    // (out_dim)
};

/**
 * The body tangent Jacobian of the ACoM attitude xi_a = xi_hat (x)
 * Exp(Delta_theta(q_j)) (AngularCenterOfMass:: acomAttitude()): when the base
 * orientation moves to xi_hat (x) Exp(dphi) and the joints to q_j + dq_j, the
 * attitude moves to xi_a (x) Exp(dphi_a) with, to first order,
 *
 *   dphi_a = baseOrientation * dphi + jointAngles * dq_j.
 *
 * Neither block depends on the base orientation, only on the joints.
 */
struct AcomAttitudeTangentJacobian {
  /** R(Delta_theta)^T, 3 x 3: the adjoint that carries a body perturbation of
   * the base into the frame of the ACoM. */
  matrix3_t baseOrientation;
  /** J_r(Delta_theta) J_Delta_theta(q_j), 3 x n_j: the right Jacobian of SO(3)
   * times the network's joint Jacobian. */
  matrix_t jointAngles;
};

/**
 * Angular Center of Mass (ACoM) evaluator: the whole-body orientation of Chen,
 * Nelson, Griffin, Posa and Pratt (IROS 2023) as a unit quaternion of the base
 * orientation and a learned, joint-induced rotation (design section 2.8.3 of
 * humanoid_nmpc/docs/quaternion_base_orientation/README.md):
 *
 *   xi_a(xi, q_j) = xi_hat (x) Exp(Delta_theta(q_j)).
 *
 * Delta_theta(q_j) is the rotation vector, in the base frame, that the SIREN
 * network of humanoid_learning/acom emits: its components are about the base's
 * x, y and z axes, in that order. The network was fitted at the identity base
 * (dataset_generator.py), which is what makes it a function of the joints alone
 * and lets the base rotation compose on its left. A rotation of the whole robot
 * rotates the ACoM one for one: xi_a(q (x) xi, q_j) = q (x) xi_a(xi, q_j).
 *
 * Build one with Create(), which checks the network against the MPC model's
 * joints. The evaluation methods are const, keep no state between calls and are
 * safe to call concurrently; loadWeights() and setJointNames() are not.
 */
class AngularCenterOfMass {
 public:
  /** The frequency omega_0 of the sinusoidal layers that the training code
   * (humanoid_learning/acom) defaults to. */
  static constexpr double kDefaultOmega0 = 30.0;

  AngularCenterOfMass(size_t inputDim,
                      size_t numLayers,
                      double omega0 = kDefaultOmega0);

  // A value type: copying it copies the network.
  AngularCenterOfMass(const AngularCenterOfMass&) = default;
  AngularCenterOfMass& operator=(const AngularCenterOfMass&) = default;
  AngularCenterOfMass(AngularCenterOfMass&&) = default;
  AngularCenterOfMass& operator=(AngularCenterOfMass&&) = default;
  ~AngularCenterOfMass() = default;

  /**
   * The values of model_settings.robot_name that have a network compiled into
   * this binary, sorted. This is the registry Create() resolves names against,
   * and the list its unknown-robot error prints.
   */
  static std::vector<std::string> registeredRobotNames();

  /**
   * Creates the evaluator for `robotName` (model_settings.robot_name) and
   * checks it against the model it will be fed.
   *
   * The network is a function of a VECTOR of joint angles, so it is only
   * meaningful when `modelJointNames` - the MPC model's joints, in the order
   * the MPC hands them over (ModelSettings::mpcModelJointNames, or equivalently
   * the reduced Pinocchio model's joints after the floating base) - is exactly
   * the joint list recorded in the weight header at training time. A
   * permutation, or a different set of fixed joints of the same size, keeps
   * every dimension check green while the network silently evaluates a
   * different robot, so the names are compared one by one here rather than only
   * counted.
   *
   * @return NotFound if no network is registered for `robotName`, listing the
   * registered names; FailedPrecondition if the joint list differs from the one
   * the network was trained on, naming the robot and the first joint that
   * differs.
   */
  static absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> Create(
      absl::string_view robotName,
      const std::vector<std::string>& modelJointNames);

  /**
   * Validates and loads layer weights and biases. Returns InvalidArgument, and
   * leaves the evaluator unchanged, if the layer count, a layer's shape or the
   * 3-wide readout does not match the architecture this evaluator was built
   * for.
   */
  absl::Status loadWeights(const std::vector<SirenLayerWeights>& layers);

  /**
   * Records the joint names, in input order, that the network was trained on.
   * Returns InvalidArgument unless there is exactly one name per network input.
   */
  absl::Status setJointNames(std::vector<std::string> jointNames);

  /** The joint names recorded by setJointNames(), in input order; empty if none
   * were recorded. */
  const std::vector<std::string>& getJointNames() const { return jointNames_; }

  /**
   * Computes the joint-induced orientation offset Delta_theta(q_j) in R^3: the
   * rotation vector, in the base frame, of the ACoM relative to the base,
   * components about the base's x, y and z axes (the network's native order).
   */
  vector3_t computeJointOrientationOffset(const vector_t& qJoints) const;

  /**
   * Computes the exact analytical Jacobian d(Delta_theta)/d(q_j) in R^(3 x
   * n_j), rows as computeJointOrientationOffset().
   */
  matrix_t computeJointOffsetJacobian(const vector_t& qJoints) const;

  /**
   * Computes the ACoM attitude xi_a = xi_hat (x) Exp(Delta_theta(q_j)), the
   * quaternion coefficients (x, y, z, w) of the rotation from the ACoM frame to
   * the world.
   *
   * @param xi       The base orientation, quaternion coefficients (x, y, z, w),
   * read through safelyNormalizedQuaternion() (orientation/BaseOrientation.h):
   * any nonzero scale gives the same attitude.
   * @param qJoints  The network's joints, getInputDim() of them, in the order
   * of getJointNames().
   * @return A unit quaternion for every |xi| above kQuaternionNormGuard, in the
   * hemisphere of xi_hat: -xi gives -xi_a.
   */
  vector4_t acomAttitude(const vector4_t& xi, const vector_t& qJoints) const;

  /**
   * Computes the body tangent Jacobian of acomAttitude()
   * (AcomAttitudeTangentJacobian): R(Delta_theta)^T for the base orientation
   * and J_r(Delta_theta) J_Delta_theta(q_j) for the joints. `qJoints` as for
   * acomAttitude(). One forward and one chain-rule pass of the network, as
   * computeJointOffsetJacobian().
   */
  AcomAttitudeTangentJacobian acomAttitudeTangentJacobian(
      const vector_t& qJoints) const;

  size_t getInputDim() const { return inputDim_; }

 private:
  /** Delta_theta(q_j) and d(Delta_theta)/d(q_j) of one forward and chain-rule
   * pass. */
  struct JointOffsetAndJacobian {
    vector3_t offset;
    matrix_t jacobian;
  };

  /**
   * Computes Delta_theta(q_j) and its Jacobian in one pass: the offset is
   * computeJointOrientationOffset()'s, bit for bit (the same operations in the
   * same order), the Jacobian computeJointOffsetJacobian()'s.
   */
  JointOffsetAndJacobian computeJointOffsetAndJacobian(
      const vector_t& qJoints) const;

  // The two checks below guard the evaluation hot path against PROGRAMMING
  // errors - a joint vector of the wrong size, or an evaluator used before its
  // weights were loaded - which Create() has already ruled out for every
  // configuration it accepts. They are ABSL_CHECKs, so that the evaluation
  // methods keep returning plain values: an invariant that does not hold ends
  // the process with the message instead of evaluating out of bounds.

  /** Checks that qJoints has exactly inputDim_ entries. */
  void checkJointVectorSize(const vector_t& qJoints) const;

  /** Checks that weights have been loaded. */
  void checkWeightsLoaded() const;

  size_t inputDim_;
  size_t numLayers_;
  double omega0_;
  std::vector<SirenLayerWeights> layers_;
  std::vector<std::string> jointNames_;
};

}  // namespace ocs2::humanoid
