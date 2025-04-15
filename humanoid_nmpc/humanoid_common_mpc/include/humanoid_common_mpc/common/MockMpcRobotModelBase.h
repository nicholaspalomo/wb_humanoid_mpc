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

#include <filesystem>

#include <gmock/gmock.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

namespace ocs2::humanoid {

class MockMpcRobotModel : public MpcRobotModelBase<scalar_t> {
 public:
  MockMpcRobotModel(const std::filesystem::path& taskFilePath,
                    const std::filesystem::path& urdfFilePath, size_t stateDim,
                    size_t inputDim, std::string_view robotName,
                    bool verbose = false)
      : MpcRobotModelBase<scalar_t>(
            ModelSettings(std::string(taskFilePath), std::string(urdfFilePath),
                          std::string(robotName), verbose),
            stateDim, inputDim) {}

  MOCK_METHOD(size_t, getContactForceStartIndices, (size_t), (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getContactForce,
              (const VECTOR_T<scalar_t>&, size_t), (const, override));

  // Other virtual methods need stubs
  MOCK_METHOD(MpcRobotModelBase<scalar_t>*, clone, (), (const, override));
  MOCK_METHOD(size_t, getBaseStartindex, (), (const, override));
  MOCK_METHOD(size_t, getJointStartindex, (), (const, override));
  MOCK_METHOD(size_t, getJointVelocitiesStartindex, (), (const, override));
  MOCK_METHOD(size_t, getContactWrenchStartIndices, (size_t),
              (const, override));
  MOCK_METHOD(VECTOR_T<scalar_t>, getGeneralizedCoordinates,
              (const VECTOR_T<scalar_t>&), (const, override));
  MOCK_METHOD(VECTOR6_T<scalar_t>, getBasePose, (const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getBasePosition, (const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getBaseOrientationEulerZYX,
              (const VECTOR_T<scalar_t>&), (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getBaseComLinearVelocity,
              (const VECTOR_T<scalar_t>&), (const, override));
  MOCK_METHOD(VECTOR6_T<scalar_t>, getBaseComVelocity,
              (const VECTOR_T<scalar_t>&), (const, override));
  MOCK_METHOD(VECTOR_T<scalar_t>, getJointAngles, (const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(VECTOR_T<scalar_t>, getJointVelocities,
              (const VECTOR_T<scalar_t>&, const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(VECTOR_T<scalar_t>, getGeneralizedVelocities,
              (const VECTOR_T<scalar_t>&, const VECTOR_T<scalar_t>&),
              (override));
  MOCK_METHOD(void, setGeneralizedCoordinates,
              (VECTOR_T<scalar_t>&, const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setBasePose,
              (VECTOR_T<scalar_t>&, const VECTOR6_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setBasePosition,
              (VECTOR_T<scalar_t>&, const VECTOR3_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setBaseOrientationEulerZYX,
              (VECTOR_T<scalar_t>&, const VECTOR3_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setJointAngles,
              (VECTOR_T<scalar_t>&, const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, setJointVelocities,
              (VECTOR_T<scalar_t>&, VECTOR_T<scalar_t>&,
               const VECTOR_T<scalar_t>&),
              (const, override));
  MOCK_METHOD(void, adaptBasePoseHeight, (VECTOR_T<scalar_t>&, scalar_t),
              (const, override));
  MOCK_METHOD(VECTOR6_T<scalar_t>, getContactWrench,
              (const VECTOR_T<scalar_t>&, size_t), (const, override));
  MOCK_METHOD(VECTOR3_T<scalar_t>, getContactMoment,
              (const VECTOR_T<scalar_t>&, size_t), (const, override));
  MOCK_METHOD(void, setContactWrench,
              (VECTOR_T<scalar_t>&, const VECTOR6_T<scalar_t>&, size_t),
              (const, override));
  MOCK_METHOD(void, setContactForce,
              (VECTOR_T<scalar_t>&, const VECTOR3_T<scalar_t>&, size_t),
              (const, override));
  MOCK_METHOD(void, setContactMoment,
              (VECTOR_T<scalar_t>&, const VECTOR3_T<scalar_t>&, size_t),
              (const, override));
};

}  // namespace ocs2::humanoid
