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

#include <gmock/gmock.h>
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlannerBase.h"

namespace ocs2::humanoid {

/**
 * @brief GMock implementation of the SwingTrajectoryPlannerBase class
 *
 * This mock version allows setting expectations on method calls
 * and verifying interactions in unit tests.
 */
class MockSwingTrajectoryPlanner : public SwingTrajectoryPlannerBase {
 public:
  MockSwingTrajectoryPlanner() = default;
  ~MockSwingTrajectoryPlanner() override = default;

  // Mock methods with MOCK_METHOD macro
  MOCK_METHOD(void, update, (const ModeSchedule&, scalar_t), (override));

  MOCK_METHOD(void, update,
              (const ModeSchedule&, const feet_array_t<scalar_array_t>&,
               const feet_array_t<scalar_array_t>&),
              (override));

  MOCK_METHOD(scalar_t, getZaccelerationConstraint, (size_t, scalar_t),
              (const, override));

  MOCK_METHOD(scalar_t, getZvelocityConstraint, (size_t, scalar_t),
              (const, override));

  MOCK_METHOD(scalar_t, getZpositionConstraint, (size_t, scalar_t),
              (const, override));

  MOCK_METHOD(scalar_t, getImpactProximityFactor, (size_t, scalar_t),
              (const, override));

  // Helper method to set up default values for all constraint methods
  void setDefaultConstraintValues(scalar_t zAcceleration = 0.0,
                                  scalar_t zVelocity = 0.0,
                                  scalar_t zPosition = 0.0,
                                  scalar_t impactFactor = 0.0) {
    ON_CALL(*this, getZaccelerationConstraint(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(zAcceleration));

    ON_CALL(*this, getZvelocityConstraint(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(zVelocity));

    ON_CALL(*this, getZpositionConstraint(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(zPosition));

    ON_CALL(*this, getImpactProximityFactor(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(impactFactor));
  }
};

}  // namespace ocs2::humanoid
