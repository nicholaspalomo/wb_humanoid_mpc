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

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"

namespace ocs2::humanoid::test {

/**
 * A robot model that is nothing but the dimensions of a StateInputLayout - the state and input sizes of a centroidal or
 * whole-body MPC of a robot's ModelSettings - for tests of what is built from the configuration (the factory's
 * weights, the reference manager, the target calculator, the motion manager) without a robot model package and its
 * CppAD libraries. Every accessor returns zeros and every setter does nothing.
 */
template <typename SCALAR_T>
class LayoutRobotModel final : public MpcRobotModelBase<SCALAR_T> {
 public:
  LayoutRobotModel(const ModelSettings& modelSettings, StateInputLayout::Mpc mpc)
      : MpcRobotModelBase<SCALAR_T>(modelSettings,
                                    /*state_dim=*/static_cast<scalar_t>(stateDimension(stateInputLayout(modelSettings, mpc))),
                                    /*input_dim=*/static_cast<scalar_t>(inputDimension(stateInputLayout(modelSettings, mpc)))),
        mpc_(mpc) {}

  LayoutRobotModel* absl_nonnull clone() const override { return new LayoutRobotModel(this->modelSettings, mpc_); }

  size_t getBaseStartindex() const override { return mpc_ == StateInputLayout::Mpc::kCentroidal ? 6 : 0; }
  size_t getJointStartindex() const override { return mpc_ == StateInputLayout::Mpc::kCentroidal ? 12 : 6; }
  size_t getJointVelocitiesStartindex() const override { return 6 * this->modelSettings.contactNames.size(); }
  size_t getContactWrenchStartIndices(size_t contactIndex) const override { return 6 * contactIndex; }

  VECTOR_T<SCALAR_T> getGeneralizedCoordinates(const VECTOR_T<SCALAR_T>& /*state*/) const override {
    return VECTOR_T<SCALAR_T>::Zero(this->getGenCoordinatesDim());
  }
  VECTOR6_T<SCALAR_T> getBasePose(const VECTOR_T<SCALAR_T>& /*state*/) const override { return VECTOR6_T<SCALAR_T>::Zero(); }
  VECTOR3_T<SCALAR_T> getBasePosition(const VECTOR_T<SCALAR_T>& /*state*/) const override { return VECTOR3_T<SCALAR_T>::Zero(); }
  VECTOR3_T<SCALAR_T> getBaseOrientationEulerZYX(const VECTOR_T<SCALAR_T>& /*state*/) const override { return VECTOR3_T<SCALAR_T>::Zero(); }
  VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(const VECTOR_T<SCALAR_T>& /*state*/) const override { return VECTOR3_T<SCALAR_T>::Zero(); }
  VECTOR6_T<SCALAR_T> getBaseComVelocity(const VECTOR_T<SCALAR_T>& /*state*/) const override { return VECTOR6_T<SCALAR_T>::Zero(); }
  VECTOR_T<SCALAR_T> getJointAngles(const VECTOR_T<SCALAR_T>& /*state*/) const override {
    return VECTOR_T<SCALAR_T>::Zero(this->getJointDim());
  }
  VECTOR_T<SCALAR_T> getJointVelocities(const VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& /*input*/) const override {
    return VECTOR_T<SCALAR_T>::Zero(this->getJointDim());
  }
  VECTOR_T<SCALAR_T> getGeneralizedVelocities(const VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& /*input*/) override {
    return VECTOR_T<SCALAR_T>::Zero(this->getGenCoordinatesDim());
  }

  void setGeneralizedCoordinates(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& /*generalizedCorrdinates*/) const override {}
  void setBasePose(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR6_T<SCALAR_T>& /*basePose*/) const override {}
  void setBasePosition(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR3_T<SCALAR_T>& /*position*/) const override {}
  void setBaseOrientationEulerZYX(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR3_T<SCALAR_T>& /*eulerAnglesZYX*/) const override {}
  void setBaseComLinearVelocity(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR3_T<SCALAR_T>& /*velocity*/) const override {}
  void setJointAngles(VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& /*jointAngles*/) const override {}
  void setJointVelocities(VECTOR_T<SCALAR_T>& /*state*/,
                          VECTOR_T<SCALAR_T>& /*input*/,
                          const VECTOR_T<SCALAR_T>& /*jointVelocities*/) const override {}
  void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& /*state*/, scalar_t /*heightChange*/) const override {}

  VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& /*input*/, size_t /*contactIndex*/) const override {
    return VECTOR6_T<SCALAR_T>::Zero();
  }
  VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& /*input*/, size_t /*contactIndex*/) const override {
    return VECTOR3_T<SCALAR_T>::Zero();
  }
  VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& /*input*/, size_t /*contactIndex*/) const override {
    return VECTOR3_T<SCALAR_T>::Zero();
  }
  void setContactWrench(VECTOR_T<SCALAR_T>& /*input*/, const VECTOR6_T<SCALAR_T>& /*wrench*/, size_t /*contactIndex*/) const override {}
  void setContactForce(VECTOR_T<SCALAR_T>& /*input*/, const VECTOR3_T<SCALAR_T>& /*force*/, size_t /*contactIndex*/) const override {}
  void setContactMoment(VECTOR_T<SCALAR_T>& /*input*/, const VECTOR3_T<SCALAR_T>& /*moment*/, size_t /*contactIndex*/) const override {}

 private:
  StateInputLayout::Mpc mpc_;
};

}  // namespace ocs2::humanoid::test
