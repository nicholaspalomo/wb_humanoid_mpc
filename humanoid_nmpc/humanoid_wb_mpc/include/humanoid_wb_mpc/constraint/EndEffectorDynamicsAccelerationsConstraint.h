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

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/end_effector/EndEffectorDynamics.h"

namespace ocs2::humanoid {

/**
 * Defines a linear constraint on an end-effector position (xee) and linear velocity (vee).
 * g(xee, vee, aee) = Ax * xee + Av * vee + Aa *aee + b
 * - For defining constraint of type g(xee), set Av to matrix_t(0, 0)
 * - For defining constraint of type g(vee), set Ax to matrix_t(0, 0)
 */
class EndEffectorDynamicsAccelerationsConstraint final : public StateInputConstraint {
 public:
  /**
   * Coefficients of the linear constraints of the form:
   * g(xee, vee, aee) = Ax * xee + Av * vee + Aa *aee + b
   */
  struct Config {
    vector_t b;
    matrix_t Ax;
    matrix_t Av;
    matrix_t Aa;
  };

  /**
   * Makes the constraint on the end effector of `endEffectorDynamics`, which it clones.
   * @param [in] endEffectorDynamics: The dynamics interface to the target end-effector; it has exactly one end effector.
   * @param [in] numConstraints: The number of constraints {1, 2, 3, 4, 5, 6}
   * @param [in] config: The constraint coefficients, g(xee, vee, aee) = Ax * xee + Av * vee + Aa *aee + b
   * @return InvalidArgument naming the end effectors when `endEffectorDynamics` has other than one.
   */
  static absl::StatusOr<std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>> Create(
      const EndEffectorDynamics<scalar_t>& endEffectorDynamics, size_t numConstraints, Config config = Config());

  ~EndEffectorDynamicsAccelerationsConstraint() override = default;
  EndEffectorDynamicsAccelerationsConstraint& operator=(const EndEffectorDynamicsAccelerationsConstraint&) = delete;
  EndEffectorDynamicsAccelerationsConstraint(EndEffectorDynamicsAccelerationsConstraint&&) = delete;
  EndEffectorDynamicsAccelerationsConstraint& operator=(EndEffectorDynamicsAccelerationsConstraint&&) = delete;
  EndEffectorDynamicsAccelerationsConstraint* absl_nonnull clone() const override {
    return new EndEffectorDynamicsAccelerationsConstraint(*this);
  }

  /**
   * Sets new constraint coefficients, unchecked (the pre-computation sets them on every evaluation): `config.b` has
   * getNumConstraints() rows, and each of Ax, Av and Aa is empty or getNumConstraints() x 6, at least one of Ax and Av
   * not empty.
   */
  void configure(Config&& config);
  /** Sets a new constraint coefficients. */
  void configure(const Config& config) { this->configure(Config(config)); }

  /** The constraint coefficients in effect (Create(), configure()). */
  const Config& getConfig() const { return config_; }

  /** Gets the underlying end-effector kinematics interface. */
  EndEffectorDynamics<scalar_t>& getEndEffectorDynamics() { return *endEffectorDynamicsPtr_; }

  size_t getNumConstraints(scalar_t /*time*/) const override { return numConstraints_; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

 private:
  EndEffectorDynamicsAccelerationsConstraint(const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
                                             size_t numConstraints,
                                             Config config);
  EndEffectorDynamicsAccelerationsConstraint(const EndEffectorDynamicsAccelerationsConstraint& rhs);

  vector3_t ground_plane_normal_;
  std::unique_ptr<EndEffectorDynamics<scalar_t>> endEffectorDynamicsPtr_;
  const size_t numConstraints_;
  Config config_;
};

}  // namespace ocs2::humanoid
