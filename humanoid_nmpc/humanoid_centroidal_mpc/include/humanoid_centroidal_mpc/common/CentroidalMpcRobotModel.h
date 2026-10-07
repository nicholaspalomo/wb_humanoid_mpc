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

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_pinocchio_interface/PinocchioStateInputMapping.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/* State Vector definition

  Define the state vector x = [h, q_b, q_j]^T

    Centroidal momentum h = [vcom_x, vcom_y, vcom_z, L_x / mass, L_y / mass, L_z / mass]^T
    Base pose q_b = [p_base_x, p_base_y, p_base_z, theta_base_z, theta_base_y, theta_base_x]
    Joint angles q_j

*/
/******************************************************************************************************/

/******************************************************************************************************/
/* Input Vector definition

  Define the input vector u = [W_l, W_r, q_dot_j]^T

    Left contact Wrench W_l in inertial frame
    Right contact Wrench W_r in inertial frame
    Joint velocities q_dot_j

*/
/******************************************************************************************************/

/**
 * The state and input layout of the centroidal MPC defined above, for the scalar type of the solver or of CppAD: the
 * normalized centroidal momentum, the base pose and the joint angles of the state; the contact wrenches and the joint
 * velocities of the input. getGeneralizedVelocities() updates the model's own copy of the Pinocchio interface, so a
 * model is not thread-safe; clone() one for each thread. Every accessor takes a state of getStateDim(), an input of
 * getInputDim() and generalized coordinates of 6 + mpc_joint_dim entries, unchecked: the solver and the control thread
 * call them on every evaluation.
 */
template <typename SCALAR_T>
class CentroidalMpcRobotModel : public MpcRobotModelBase<SCALAR_T> {
 public:
  CentroidalMpcRobotModel(const ModelSettings& modelSettings,
                          const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                          const CentroidalModelInfoTpl<SCALAR_T>& centroidalModelInfo)
      : MpcRobotModelBase<SCALAR_T>(modelSettings, 12 + modelSettings.mpc_joint_dim, 6 * kNumContacts + modelSettings.mpc_joint_dim),
        pinocchioInterface_(pinocchioInterface),
        centroidalModelInfo_(centroidalModelInfo),
        pinocchioMappingPtr_(std::make_unique<CentroidalModelPinocchioMappingTpl<SCALAR_T>>(centroidalModelInfo)) {
    pinocchioMappingPtr_->setPinocchioInterface(pinocchioInterface_);
  }

  ~CentroidalMpcRobotModel() override = default;
  CentroidalMpcRobotModel* absl_nonnull clone() const override { return new CentroidalMpcRobotModel(*this); }
  // Copied only by clone(), whose copy constructor is private; never assigned or moved.
  CentroidalMpcRobotModel& operator=(const CentroidalMpcRobotModel&) = delete;
  CentroidalMpcRobotModel(CentroidalMpcRobotModel&&) = delete;
  CentroidalMpcRobotModel& operator=(CentroidalMpcRobotModel&&) = delete;

  /******************************************************************************************************/
  /*                                          Start indices                                             */
  /******************************************************************************************************/

  // LINT.IfChange(state_input_indices)
  size_t getBaseStartindex() const override { return 6; }
  size_t getJointStartindex() const override { return 12; }
  size_t getJointVelocitiesStartindex() const override { return 6 * kNumContacts; }

  // Assumes contact wrench [f_x, f_y, f_z, M_x, M_y, M_z]^T
  size_t getContactWrenchStartIndices(size_t contactIndex) const override { return 6 * contactIndex; }
  size_t getContactForceStartIndices(size_t contactIndex) const override { return getContactWrenchStartIndices(contactIndex); }
  size_t getContactMomentStartIndices(size_t contactIndex) const override { return getContactWrenchStartIndices(contactIndex) + 3; }
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/weights/StateInputWeightsFromConfig.cpp:state_input_offsets)

  /******************************************************************************************************/
  /*                                     Generalized coordinates                                        */
  /******************************************************************************************************/

  VECTOR_T<SCALAR_T> getGeneralizedCoordinates(const VECTOR_T<SCALAR_T>& state) const override {
    return state.tail(6 + this->modelSettings.mpc_joint_dim);
  }

  VECTOR6_T<SCALAR_T> getBasePose(const VECTOR_T<SCALAR_T>& state) const override { return state.segment(6, 6); }

  VECTOR3_T<SCALAR_T> getBasePosition(const VECTOR_T<SCALAR_T>& state) const override { return state.segment(6, 3); }

  VECTOR3_T<SCALAR_T> getBaseOrientationEulerZYX(const VECTOR_T<SCALAR_T>& state) const override { return state.segment(6 + 3, 3); }

  // Return Com linear velocity
  VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(const VECTOR_T<SCALAR_T>& state) const override { return state.head(3); }

  VECTOR6_T<SCALAR_T> getBaseComVelocity(const VECTOR_T<SCALAR_T>& state) const override { return state.head(6); }

  void setBaseComLinearVelocity(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& velocity) const override { state.head(3) = velocity; }

  VECTOR_T<SCALAR_T> getJointAngles(const VECTOR_T<SCALAR_T>& state) const override {
    return state.tail(this->modelSettings.mpc_joint_dim);
  }

  VECTOR_T<SCALAR_T> getJointVelocities(const VECTOR_T<SCALAR_T>& /*state*/, const VECTOR_T<SCALAR_T>& input) const override {
    return input.tail(this->modelSettings.mpc_joint_dim);
  }

  VECTOR_T<SCALAR_T> getGeneralizedVelocities(const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& input) override {
    updateCentroidalDynamics<SCALAR_T>(pinocchioInterface_, centroidalModelInfo_, pinocchioMappingPtr_->getPinocchioJointPosition(state));
    return pinocchioMappingPtr_->getPinocchioJointVelocity(state, input);
  }

  void setGeneralizedCoordinates(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& generalizedCorrdinates) const override {
    state.tail(6 + this->modelSettings.mpc_joint_dim) = generalizedCorrdinates;
  }

  void setBasePose(VECTOR_T<SCALAR_T>& state, const VECTOR6_T<SCALAR_T>& basePose) const override { state.segment(6, 6) = basePose; }

  void setBasePosition(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& position) const override { state.segment(6, 3) = position; }

  void setBaseOrientationEulerZYX(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& eulerAnglesZYX) const override {
    state.segment(6 + 3, 3) = eulerAnglesZYX;
  }

  void setJointAngles(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& jointAngles) const override {
    state.tail(this->modelSettings.mpc_joint_dim) = jointAngles;
  }

  void setJointVelocities(VECTOR_T<SCALAR_T>& /*state*/,
                          VECTOR_T<SCALAR_T>& input,
                          const VECTOR_T<SCALAR_T>& jointVelocities) const override {
    input.tail(this->modelSettings.mpc_joint_dim) = jointVelocities;
  }

  void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& state, scalar_t heightChange) const override { state[6 + 2] += heightChange; }

  /******************************************************************************************************/
  /*                                          Contacts                                                  */
  /******************************************************************************************************/

  VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment(getContactWrenchStartIndices(contactIndex), kContactWrenchDim);
  }

  VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment(getContactWrenchStartIndices(contactIndex), 3);
  }

  VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const override {
    return input.segment((getContactWrenchStartIndices(contactIndex) + 3), 3);
  }

  void setContactWrench(VECTOR_T<SCALAR_T>& input, const VECTOR6_T<SCALAR_T>& wrench, size_t contactIndex) const override {
    input.segment(getContactWrenchStartIndices(contactIndex), 6) = wrench;
  }

  void setContactForce(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& force, size_t contactIndex) const override {
    input.segment(getContactWrenchStartIndices(contactIndex), 3) = force;
  }

  void setContactMoment(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& moment, size_t contactIndex) const override {
    input.segment(getContactWrenchStartIndices(contactIndex) + 3, 3) = moment;
  }

  /******************************************************************************************************/
  /*                                    Custom Centroidal Methods                                       */
  /******************************************************************************************************/

  VECTOR_T<SCALAR_T> getCentroidalMomentum(const VECTOR_T<SCALAR_T>& state) const { return state.head(6); }

  const CentroidalModelInfoTpl<SCALAR_T>& getCentroidalModelInfo() const { return centroidalModelInfo_; }

 private:
  CentroidalMpcRobotModel(const CentroidalMpcRobotModel& rhs)
      : MpcRobotModelBase<SCALAR_T>(rhs),
        pinocchioInterface_(rhs.pinocchioInterface_),
        centroidalModelInfo_(rhs.centroidalModelInfo_),
        pinocchioMappingPtr_(rhs.pinocchioMappingPtr_->clone()) {
    pinocchioMappingPtr_->setPinocchioInterface(pinocchioInterface_);
  }

  PinocchioInterfaceTpl<SCALAR_T> pinocchioInterface_;
  const CentroidalModelInfoTpl<SCALAR_T> centroidalModelInfo_;
  const std::unique_ptr<CentroidalModelPinocchioMappingTpl<SCALAR_T>> pinocchioMappingPtr_;
};

}  // namespace ocs2::humanoid
