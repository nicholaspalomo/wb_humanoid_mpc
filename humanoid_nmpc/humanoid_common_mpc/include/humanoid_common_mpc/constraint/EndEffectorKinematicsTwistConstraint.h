/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
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

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * Defines a linear constraint on an end-effector position (xee) and linear velocity (vee).
 * g(xee, vee) = Ax * xee + Av * vee + b
 * - For defining constraint of type g(xee), set Av to matrix_t(0, 0)
 * - For defining constraint of type g(vee), set Ax to matrix_t(0, 0)
 *
 * When the orientation columns of Av are active, the angular velocity is mapped through the kinematic Jacobian of the
 * orientation error so that the constraint expresses a proper PD law:
 *   Kp * orientationError + Kd * d(orientationError)/dt = 0
 * rather than mixing quaternion distance with raw angular velocity.
 *
 * The orientation error with respect to a plane only measures the tilt of the end-effector normal, so it has two
 * degrees of freedom and its rate alone leaves the rotation about the contact normal unconstrained. The mapping
 * therefore also carries the rotation rate about the end-effector normal in the plane-normal direction, which for a
 * stance foot is what stops it from pivoting on the spot. See getAngularVelocityToOrientationErrorRateMap().
 */
class EndEffectorKinematicsTwistConstraint final : public StateInputConstraint {
 public:
  /**
   * Coefficients of the linear constraints of the form:
   * g(xee, vee) = Ax * xee + Av * vee + b = 0
   */
  struct Config {
    vector_t b;
    matrix_t Ax;
    matrix_t Av;
  };

  /**
   * Constructor
   * @param [in] endEffectorKinematics: The kinematic interface to the target end-effector.
   * @param [in] numConstraints: The number of constraints {3, 6}. 3 = translation-only, 6 = full pose.
   * @param [in] config: The constraint coefficients, g(xee, vee) = Ax * xee + Av * vee + b
   */
  EndEffectorKinematicsTwistConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                                       size_t numConstraints,
                                       Config config = Config());

  ~EndEffectorKinematicsTwistConstraint() override = default;
  EndEffectorKinematicsTwistConstraint* clone() const override { return new EndEffectorKinematicsTwistConstraint(*this); }

  /** Sets a new constraint coefficients. */
  void configure(Config&& config);
  /** Sets a new constraint coefficients. */
  void configure(const Config& config) { this->configure(Config(config)); }
  /** Gets a reference of the config to allow to modify it. */
  Config& getConfig() { return config_; }

  /** Sets the number of constraint rows (3 = translation-only, 6 = full pose). */
  void setNumConstraints(size_t numConstraints) { numConstraints_ = numConstraints; }

  /** Gets the underlying end-effector kinematics interface. */
  EndEffectorKinematics<scalar_t>& getEndEffectorKinematics() { return *endEffectorKinematicsPtr_; }

  /**
   * Adds the rotation rate about the end-effector normal to the plane-normal row of the orientation block. Off by
   * default: it makes a 6D constraint pin all six degrees of freedom, which is correct but removes one input degree of
   * freedom per stance foot from a controller tuned without it. See getAngularVelocityToOrientationErrorRateMap().
   */
  void setConstrainYawRateAboutNormal(bool constrain) { constrainYawRateAboutNormal_ = constrain; }
  bool getConstrainYawRateAboutNormal() const { return constrainYawRateAboutNormal_; }

  /** Sets the ground contact plane normal (default: {0, 0, 1} for flat ground). */
  void setGroundPlaneNormal(const vector3_t& normal) { ground_plane_normal_ = normal.normalized(); }

  /** Gets the current ground contact plane normal. */
  const vector3_t& getGroundPlaneNormal() const { return ground_plane_normal_; }

  size_t getNumConstraints(scalar_t time) const override { return numConstraints_; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

 private:
  EndEffectorKinematicsTwistConstraint(const EndEffectorKinematicsTwistConstraint& rhs);

  /**
   * The 3x3 matrix M that maps the end-effector angular velocity to the rate of the orientation constraint residual.
   *
   * With v the end-effector normal (R * e_z) and n the plane normal, the orientation error of
   * rotationMatrixDistanceToPlane is e = -(v x n) / sqrt(2 (1 + v.n)), whose analytic derivative is
   *
   *   de/d(omega) = [ (v.n) I - v n^T ] / sqrt(2 (1 + v.n)) + (v x n)(v x n)^T / (2 (1 + v.n))^(3/2).
   *
   * That matrix has rank two: e is always perpendicular to n, and rotating about v does not change v, so a rotation
   * about the end-effector normal maps to zero. Used on its own it would leave a stance foot free to pivot about the
   * contact normal. When setConstrainYawRateAboutNormal() is set, the term 0.5 * n * v^T is added, which puts the
   * rotation rate about the end-effector normal into the plane-normal row with the same half-angle scaling the tilt
   * rows have, making the result full rank.
   */
  matrix3_t getAngularVelocityToOrientationErrorRateMap(const vector_t& state) const;

  vector3_t ground_plane_normal_;
  bool constrainYawRateAboutNormal_{false};
  std::unique_ptr<EndEffectorKinematics<scalar_t>> endEffectorKinematicsPtr_;
  size_t numConstraints_;
  Config config_;
};

}  // namespace ocs2::humanoid
