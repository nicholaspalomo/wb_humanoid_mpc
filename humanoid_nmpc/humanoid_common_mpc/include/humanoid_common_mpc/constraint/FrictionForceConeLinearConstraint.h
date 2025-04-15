/******************************************************************************
Copyright (c) 2025, Nicholas Palomo and Manuel Yves Galliker. All rights
reserved.

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

#include <absl/log/check.h>
#include <memory>
#include <ocs2_core/constraint/StateInputConstraint.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

class FrictionForceConeLinearConstraint final
    : public ocs2::StateInputConstraint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using PreComputationCallback =
      std::function<matrix3_t(const vector_t& state, const vector_t& input,
                              const PreComputation& preComp)>;

  struct Config {
    explicit Config(scalar_t frictionCoefficient = 0.5,
                    scalar_t minimumNormalForce = 1.0,
                    size_t numBasisVectors = 4)
        : frictionCoefficient(frictionCoefficient),
          minimumNormalForce(minimumNormalForce),
          numBasisVectors(numBasisVectors) {
      CHECK_GT(frictionCoefficient, 0.0);
      CHECK_GE(minimumNormalForce, 0.0);
      CHECK_GE(numBasisVectors, 4);
    }

    scalar_t frictionCoefficient;
    scalar_t minimumNormalForce;  // [N]
    size_t numBasisVectors;
  };

  FrictionForceConeLinearConstraint(
      const SwitchedModelReferenceManager& referenceManager, Config config,
      size_t contactPointIndex,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      PreComputationCallback callback);

  ~FrictionForceConeLinearConstraint() override = default;
  FrictionForceConeLinearConstraint* clone() const override {
    return new FrictionForceConeLinearConstraint(*this);
  }

  bool isActive(scalar_t time) const override;
  void setActive(bool active) override { isActive_ = active; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t time) const override;
  matrix_t getBasisVectors(scalar_t time) const;
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input,
                    const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(
      scalar_t time, const vector_t& state, const vector_t& input,
      const PreComputation& preComp) const override;

 private:
  const SwitchedModelReferenceManager* referenceManagerPtr_;
  const Config config_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;
  const size_t contactPointIndex_;
  PreComputationCallback preCompCallback_;
  bool isActive_ = true;
};

}  // namespace ocs2::humanoid
