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

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

namespace ocs2::humanoid {

template <typename SCALAR_T>
class MockMpcRobotModel : public MpcRobotModelBase<SCALAR_T> {
 public:
  /**
   * Constructor for the mock robot model
   * @param modelSettings Model settings object
   * @param stateDim Dimension of the state vector
   * @param inputDim Dimension of the input vector
   */
  MockMpcRobotModel(const ModelSettings& modelSettings, size_t stateDim = 12,
                    size_t inputDim = 12)
      : MpcRobotModelBase<SCALAR_T>(modelSettings, stateDim, inputDim),
        baseStartIndex_(0),
        jointStartIndex_(6),
        jointVelocitiesStartIndex_(6 + modelSettings.mpc_joint_dim) {}

  /**
   * Virtual destructor
   */
  ~MockMpcRobotModel() override = default;

  /**
   * Clone method for the model
   * @return A pointer to a new instance of this model
   */
  MockMpcRobotModel* clone() const override {
    return new MockMpcRobotModel(*this);
  }

  /**
   * Get base start index in state vector
   * @return Base start index
   */
  size_t getBaseStartindex() const override { return baseStartIndex_; }

  /**
   * Get joint angles start index in state vector
   * @return Joint angles start index
   */
  size_t getJointStartindex() const override { return jointStartIndex_; }

  /**
   * Get joint velocities start index in state vector
   * @return Joint velocities start index
   */
  size_t getJointVelocitiesStartindex() const override {
    return jointVelocitiesStartIndex_;
  }

  /**
   * Get the start index for the contact wrench in the input vector
   * @param contactIndex The index of the contact point
   * @return The start index in the input vector
   */
  size_t getContactWrenchStartIndices(size_t contactIndex) const override {
    return contactIndex * 6;
  }

  /**
   * Get generalized coordinates from state
   * @param state The state vector
   * @return Generalized coordinates vector
   */
  VECTOR_T<SCALAR_T> getGeneralizedCoordinates(
      const VECTOR_T<SCALAR_T>& state) const override {
    return state.template head<12>();
  }

  /**
   * Get base pose from state
   * @param state The state vector
   * @return Base pose as a 6D vector
   */
  VECTOR6_T<SCALAR_T> getBasePose(
      const VECTOR_T<SCALAR_T>& state) const override {
    return state.template head<6>();
  }

  /**
   * Get base position from state
   * @param state The state vector
   * @return Base position as a 3D vector
   */
  VECTOR3_T<SCALAR_T> getBasePosition(
      const VECTOR_T<SCALAR_T>& state) const override {
    return state.template head<3>();
  }

  /**
   * Get base orientation as Euler angles (ZYX)
   * @param state The state vector
   * @return Base orientation as a 3D vector of Euler angles
   */
  VECTOR3_T<SCALAR_T> getBaseOrientationEulerZYX(
      const VECTOR_T<SCALAR_T>& state) const override {
    return state.template segment<3>(3);
  }

  /**
   * Get base CoM linear velocity
   * @param state The state vector
   * @return Base CoM linear velocity as a 3D vector
   */
  VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(
      const VECTOR_T<SCALAR_T>& state) const override {
    return VECTOR3_T<SCALAR_T>::Zero();
  }

  /**
   * Get base CoM velocity (linear and angular)
   * @param state The state vector
   * @return Base CoM velocity as a 6D vector
   */
  VECTOR6_T<SCALAR_T> getBaseComVelocity(
      const VECTOR_T<SCALAR_T>& state) const override {
    return VECTOR6_T<SCALAR_T>::Zero();
  }

  /**
   * Get joint angles from state
   * @param state The state vector
   * @return Joint angles vector
   */
  VECTOR_T<SCALAR_T> getJointAngles(
      const VECTOR_T<SCALAR_T>& state) const override {
    return state.template segment(jointStartIndex_,
                                  this->modelSettings.mpc_joint_dim);
  }

  /**
   * Get joint velocities from state and input
   * @param state The state vector
   * @param input The input vector
   * @return Joint velocities vector
   */
  VECTOR_T<SCALAR_T> getJointVelocities(
      const VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& input) const override {
    return VECTOR_T<SCALAR_T>::Zero(this->modelSettings.mpc_joint_dim);
  }

  /**
   * Get generalized velocities from state and input
   * @param state The state vector
   * @param input The input vector
   * @return Generalized velocities vector
   */
  VECTOR_T<SCALAR_T> getGeneralizedVelocities(
      const VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& input) override {
    return VECTOR_T<SCALAR_T>::Zero(this->gen_coordinates_dim);
  }

  /**
   * Set generalized coordinates in state
   * @param state The state vector to modify
   * @param generalizedCoordinates The generalized coordinates to set
   */
  void setGeneralizedCoordinates(
      VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& generalizedCoordinates) const override {
    state.template head(this->gen_coordinates_dim) = generalizedCoordinates;
  }

  /**
   * Set base pose in state
   * @param state The state vector to modify
   * @param basePose The base pose to set
   */
  void setBasePose(VECTOR_T<SCALAR_T>& state,
                   const VECTOR6_T<SCALAR_T>& basePose) const override {
    state.template head<6>() = basePose;
  }

  /**
   * Set base position in state
   * @param state The state vector to modify
   * @param position The base position to set
   */
  void setBasePosition(VECTOR_T<SCALAR_T>& state,
                       const VECTOR3_T<SCALAR_T>& position) const override {
    state.template head<3>() = position;
  }

  /**
   * Set base orientation as Euler angles (ZYX) in state
   * @param state The state vector to modify
   * @param eulerAnglesZYX The base orientation to set
   */
  void setBaseOrientationEulerZYX(
      VECTOR_T<SCALAR_T>& state,
      const VECTOR3_T<SCALAR_T>& eulerAnglesZYX) const override {
    state.template segment<3>(3) = eulerAnglesZYX;
  }

  /**
   * Set joint angles in state
   * @param state The state vector to modify
   * @param jointAngles The joint angles to set
   */
  void setJointAngles(VECTOR_T<SCALAR_T>& state,
                      const VECTOR_T<SCALAR_T>& jointAngles) const override {
    state.template segment(jointStartIndex_,
                           this->modelSettings.mpc_joint_dim) = jointAngles;
  }

  /**
   * Set joint velocities in state and input
   * @param state The state vector to modify
   * @param input The input vector to modify
   * @param jointVelocities The joint velocities to set
   */
  void setJointVelocities(
      VECTOR_T<SCALAR_T>& state, VECTOR_T<SCALAR_T>& input,
      const VECTOR_T<SCALAR_T>& jointVelocities) const override {
    // Mock implementation - in real systems this might set velocities in state
    // or modify inputs
  }

  /**
   * Adapt the base pose height
   * @param state The state vector to modify
   * @param heightChange The height change to apply
   */
  void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& state,
                           scalar_t heightChange) const override {
    state[2] += heightChange;
  }

  /**
   * Get contact wrench from input
   * @param input The input vector
   * @param contactIndex The index of the contact point
   * @return Contact wrench as a 6D vector
   */
  VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& input,
                                       size_t contactIndex) const override {
    return input.template segment<6>(
        getContactWrenchStartIndices(contactIndex));
  }

  /**
   * Get contact force from input
   * @param input The input vector
   * @param contactIndex The index of the contact point
   * @return Contact force as a 3D vector
   */
  VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& input,
                                      size_t contactIndex) const override {
    return input.template segment<3>(
        getContactWrenchStartIndices(contactIndex));
  }

  /**
   * Get contact moment from input
   * @param input The input vector
   * @param contactIndex The index of the contact point
   * @return Contact moment as a 3D vector
   */
  VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& input,
                                       size_t contactIndex) const override {
    return input.template segment<3>(
        getContactWrenchStartIndices(contactIndex) + 3);
  }

  /**
   * Set contact wrench in input
   * @param input The input vector to modify
   * @param wrench The contact wrench to set
   * @param contactIndex The index of the contact point
   */
  void setContactWrench(VECTOR_T<SCALAR_T>& input,
                        const VECTOR6_T<SCALAR_T>& wrench,
                        size_t contactIndex) const override {
    input.template segment<6>(getContactWrenchStartIndices(contactIndex)) =
        wrench;
  }

  /**
   * Set contact force in input
   * @param input The input vector to modify
   * @param force The contact force to set
   * @param contactIndex The index of the contact point
   */
  void setContactForce(VECTOR_T<SCALAR_T>& input,
                       const VECTOR3_T<SCALAR_T>& force,
                       size_t contactIndex) const override {
    input.template segment<3>(getContactWrenchStartIndices(contactIndex)) =
        force;
  }

  /**
   * Set contact moment in input
   * @param input The input vector to modify
   * @param moment The contact moment to set
   * @param contactIndex The index of the contact point
   */
  void setContactMoment(VECTOR_T<SCALAR_T>& input,
                        const VECTOR3_T<SCALAR_T>& moment,
                        size_t contactIndex) const override {
    input.template segment<3>(getContactWrenchStartIndices(contactIndex) + 3) =
        moment;
  }

  // Public setters to modify mock behavior in tests
  void setBaseStartIndex(size_t index) { baseStartIndex_ = index; }
  void setJointStartIndex(size_t index) { jointStartIndex_ = index; }
  void setJointVelocitiesStartIndex(size_t index) {
    jointVelocitiesStartIndex_ = index;
  }

 private:
  size_t baseStartIndex_;
  size_t jointStartIndex_;
  size_t jointVelocitiesStartIndex_;
};

}  // namespace ocs2::humanoid
