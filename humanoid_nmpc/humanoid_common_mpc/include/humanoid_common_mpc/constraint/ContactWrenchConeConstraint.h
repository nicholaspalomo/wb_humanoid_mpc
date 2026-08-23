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

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Analytical linear StateInputConstraint enforcing the contact wrench cone on a contact point.
 *
 * Implements N + 7 linear inequality constraints:
 *  1. Friction cone approximation with numBasisVectors basis vectors:
 *     mu * (Fz_local + F_grip) - (cos(theta_k) * Fx_local + sin(theta_k) * Fy_local) >= 0, theta_k = 2*pi*k/N
 *  2. Normal force lower bound:
 *     Fz_local - minNormalForce >= 0
 *  3. Center of pressure (CoP) / Contact moments within rectangular footprint [x_min, x_max] x [y_min, y_max]:
 *     tau_x_local - y_min * Fz_local >= 0
 *     -tau_x_local + y_max * Fz_local >= 0
 *     -tau_y_local - x_min * Fz_local >= 0
 *     tau_y_local + x_max * Fz_local >= 0
 *  4. Torsional yaw friction moment about the contact patch center:
 *     mu_torsion * (Fz_local + F_grip) +/- (tau_z_local - (x_offset * Fy_local - y_offset * Fx_local)) >= 0
 */
class ContactWrenchConeConstraint final : public StateInputConstraint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Config {
    explicit Config(size_t numBasisVectorsParam = 4,
                    scalar_t frictionCoefficientParam = 0.7,
                    scalar_t torsionalFrictionCoefficientParam = 0.05,
                    scalar_t minNormalForceParam = 5.0,
                    scalar_t gripperForceParam = 0.0,
                    vector3_t patchOffsetParam = vector3_t::Zero())
        : numBasisVectors(numBasisVectorsParam),
          frictionCoefficient(frictionCoefficientParam),
          torsionalFrictionCoefficient(torsionalFrictionCoefficientParam),
          minNormalForce(minNormalForceParam),
          gripperForce(gripperForceParam),
          patchOffset(std::move(patchOffsetParam)) {
      assert(numBasisVectors >= 3);
      assert(frictionCoefficient > 0.0);
      assert(torsionalFrictionCoefficient >= 0.0);
      assert(minNormalForce >= 0.0);
      assert(gripperForce >= 0.0);
    }

    size_t numBasisVectors;
    scalar_t frictionCoefficient;
    scalar_t torsionalFrictionCoefficient;
    scalar_t minNormalForce;
    scalar_t gripperForce;
    vector3_t patchOffset;
  };

  ContactWrenchConeConstraint(const SwitchedModelReferenceManager& referenceManager,
                              const ContactRectangle& contactRectangle,
                              size_t contactPointIndex,
                              const PinocchioInterface& pinocchioInterface,
                              const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                              Config config = Config());

  ~ContactWrenchConeConstraint() override = default;
  ContactWrenchConeConstraint(const ContactWrenchConeConstraint& other);
  ContactWrenchConeConstraint* clone() const override { return new ContactWrenchConeConstraint(*this); }

  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t time) const override { return numConstraints_; }

  const Config& getConfig() const { return config_; }

  vector_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const PreComputation& preComp) const override;

  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

  VectorFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time,
                                                                 const vector_t& state,
                                                                 const vector_t& input,
                                                                 const PreComputation& preComp) const override;

 private:
  void initializeLocalConstraintMatrix();

  const SwitchedModelReferenceManager* referenceManagerPtr_;
  const PinocchioInterface* pinocchioInterfacePtr_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;
  const ContactRectangle contactRectangle_;
  const size_t contactPointIndex_;
  const Config config_;

  size_t numConstraints_;
  bool isActive_ = true;

  matrix_t A_f_local_;
  matrix_t A_tau_local_;
  vector_t b_local_;
};

}  // namespace ocs2::humanoid
