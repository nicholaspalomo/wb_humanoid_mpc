/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

namespace ocs2 {

/**
 * The end-effector quantities the whole-body MPC's foot constraints read, as
 * functions of the state and the input, all in the world frame: the position
 * and its first two time derivatives (the swing foot's normal-motion
 * constraint, EndEffectorDynamicsLinearAccConstraint), and the pose with
 * respect to a plane - the position and the orientation error to the plane -
 * with its first two time derivatives (the stance foot's constraint,
 * EndEffectorDynamicsAccelerationsConstraint). Each function returns one value
 * per end effector of getIds(), in that order, and each linear approximation
 * holds the value it linearizes. An implementation need not be safe to call
 * concurrently: the terms clone it (clone()), and the solver clones the terms
 * per worker thread.
 */
template <typename SCALAR_T>
class EndEffectorDynamics {
 public:
  // A position, a velocity, an acceleration or an orientation error, and the
  // pose [p; e] with respect to a plane or one of its time derivatives.
  using vector3_t = Eigen::Matrix<SCALAR_T, kPositionDim, 1>;
  using vector6_t = Eigen::Matrix<SCALAR_T, kPoseTangentDim, 1>;
  using vector_t = Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>;

  EndEffectorDynamics() = default;
  virtual ~EndEffectorDynamics() = default;
  /** Returns an owning copy. */
  virtual EndEffectorDynamics* absl_nonnull clone() const = 0;
  EndEffectorDynamics& operator=(const EndEffectorDynamics&) = delete;
  EndEffectorDynamics(EndEffectorDynamics&&) = delete;
  EndEffectorDynamics& operator=(EndEffectorDynamics&&) = delete;

  /** Returns the names of the end-effector frames, in the order of every
   * function's values. */
  virtual const std::vector<std::string>& getIds() const = 0;

  /** Returns the position of every end effector. */
  virtual std::vector<vector3_t> getPosition(const vector_t& state) const = 0;

  /** Returns the linear velocity of every end effector: the time derivative of
   * getPosition(). */
  virtual std::vector<vector3_t> getVelocity(const vector_t& state,
                                             const vector_t& input) const = 0;

  /** Returns the linear (classical) acceleration of every end effector: the
   * time derivative of getVelocity(). */
  virtual std::vector<vector3_t> getLinearAcceleration(
      const vector_t& state, const vector_t& input) const = 0;

  /**
   * Returns the orientation error of every end effector with respect to the
   * plane of the unit normal (world frame) of the same index in `planeNormals`:
   * rotationMatrixDistanceToPlane() of
   * ocs2_robotic_tools/common/RotationTransforms.h.
   */
  virtual std::vector<vector3_t> getOrientationErrorWrtPlane(
      const vector_t& state,
      const std::vector<vector3_t>& planeNormals) const = 0;

  /**
   * Returns, per end effector, the rate of its pose with respect to the plane
   * of the unit normal (world frame) of the same index in `planeNormals`:
   * [dp/dt; de/dt + 1/2 n (v.w)], p the position (getPosition()) and e the
   * orientation error with respect to the plane
   * (getOrientationErrorWrtPlane()), v the end-effector normal and w the
   * world-frame angular velocity. dp/dt is getVelocity(). e has no component
   * along n, whose row carries half the rotation rate about the end-effector
   * normal instead; where the normal points away from the plane the orientation
   * rows are w / 2 (orientationErrorWrtPlaneRate() of
   * end_effector/OrientationErrorWrtPlaneRates.h has the formulas).
   *
   * @param [in] state: state vector
   * @param [in] input: input vector
   * @param [in] planeNormals: one unit plane normal per end effector, in the
   * world frame
   * @return array of 6D rates
   */
  virtual std::vector<vector6_t> getPoseWrtPlaneRate(
      const vector_t& state,
      const vector_t& input,
      const std::vector<vector3_t>& planeNormals) const = 0;

  /**
   * Returns, per end effector, the time derivative of getPoseWrtPlaneRate()
   * along the motion: [d2p/dt2; d2e/dt2 + 1/2 n (v.alpha)], alpha = dw/dt;
   * d2p/dt2 is getLinearAcceleration(). Where the normal points away from the
   * plane the orientation rows are alpha / 2
   * (orientationErrorWrtPlaneAcceleration()).
   *
   * @param [in] state: state vector
   * @param [in] input: input vector
   * @param [in] planeNormals: one unit plane normal per end effector, in the
   * world frame
   * @return array of 6D accelerations
   */
  virtual std::vector<vector6_t> getPoseWrtPlaneAcceleration(
      const vector_t& state,
      const vector_t& input,
      const std::vector<vector3_t>& planeNormals) const = 0;

  /** Returns the linear approximations of getPosition() in the state. */
  virtual std::vector<VectorFunctionLinearApproximation>
  getPositionLinearApproximation(const vector_t& state) const = 0;

  /** Returns the linear approximations of getVelocity(). */
  virtual std::vector<VectorFunctionLinearApproximation>
  getVelocityLinearApproximation(const vector_t& state,
                                 const vector_t& input) const = 0;

  /** Returns the linear approximations of getLinearAcceleration(). */
  virtual std::vector<VectorFunctionLinearApproximation>
  getLinearAccelerationLinearApproximation(const vector_t& state,
                                           const vector_t& input) const = 0;

  /** Returns the linear approximations of getOrientationErrorWrtPlane() in the
   * state. */
  virtual std::vector<VectorFunctionLinearApproximation>
  getOrientationErrorWrtPlaneLinearApproximation(
      const vector_t& state,
      const std::vector<vector3_t>& planeNormals) const = 0;

  /** Returns the linear approximations of getPoseWrtPlaneRate(), whose values
   * they hold. */
  virtual std::vector<VectorFunctionLinearApproximation>
  getPoseWrtPlaneRateLinearApproximation(
      const vector_t& state,
      const vector_t& input,
      const std::vector<vector3_t>& planeNormals) const = 0;

  /** Returns the linear approximations of getPoseWrtPlaneAcceleration(), whose
   * values they hold. */
  virtual std::vector<VectorFunctionLinearApproximation>
  getPoseWrtPlaneAccelerationLinearApproximation(
      const vector_t& state,
      const vector_t& input,
      const std::vector<vector3_t>& planeNormals) const = 0;

 protected:
  // For the clone() of a derived class.
  EndEffectorDynamics(const EndEffectorDynamics&) = default;
};

}  // namespace ocs2
