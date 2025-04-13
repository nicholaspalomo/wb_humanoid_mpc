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

#include <ocs2_core/reference/ModeSchedule.h>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

class SwingTrajectoryPlannerBase {
 public:
  struct Config {
    scalar_t liftOffVelocity = 0.0;
    scalar_t touchDownVelocity = 0.0;
    scalar_t swingHeight = 0.1;
    scalar_t swingTimeScale = 0.15;  // swing phases shorter than this time will
                                     // be scaled down in height and velocity
    scalar_t touchDownHeightOffset = 0.0;

    scalar_t impactProximityFactorLiftOffVelocity =
        0;  // should lesser or equal 0
    scalar_t impactProximityFactorTouchDownVelocity =
        0;  // should be greater or equal to 0
    scalar_t impactProximityFactorMidPointValue =
        0.1;  // should be between 0 and 1
  };

  virtual ~SwingTrajectoryPlannerBase() = default;

  /**
   * Update the swing trajectory planner with a mode schedule and terrain height
   *
   * @param modeSchedule Current mode schedule for contact planning
   * @param terrainHeight Height of the terrain
   */
  virtual void update(const ModeSchedule& modeSchedule,
                      scalar_t terrainHeight) = 0;

  /**
   * Update the swing trajectory planner with a mode schedule and custom height
   * sequences
   *
   * @param modeSchedule Current mode schedule for contact planning
   * @param liftOffHeightSequence Sequence of lift-off heights for each foot
   * @param touchDownHeightSequence Sequence of touch-down heights for each foot
   */
  virtual void update(
      const ModeSchedule& modeSchedule,
      const feet_array_t<scalar_array_t>& liftOffHeightSequence,
      const feet_array_t<scalar_array_t>& touchDownHeightSequence) = 0;

  /**
   * Get the z-acceleration constraint for a leg at a specific time
   *
   * @param leg Index of the leg
   * @param time Current time
   * @return z-acceleration constraint value
   */
  virtual scalar_t getZaccelerationConstraint(size_t leg,
                                              scalar_t time) const = 0;

  /**
   * Get the z-velocity constraint for a leg at a specific time
   *
   * @param leg Index of the leg
   * @param time Current time
   * @return z-velocity constraint value
   */
  virtual scalar_t getZvelocityConstraint(size_t leg, scalar_t time) const = 0;

  /**
   * Get the z-position constraint for a leg at a specific time
   *
   * @param leg Index of the leg
   * @param time Current time
   * @return z-position constraint value
   */
  virtual scalar_t getZpositionConstraint(size_t leg, scalar_t time) const = 0;

  /**
   * Get the impact proximity factor for a leg at a specific time
   *
   * @param leg Index of the leg
   * @param time Current time
   * @return Impact proximity factor (between 0 and 1)
   */
  virtual scalar_t getImpactProximityFactor(size_t leg,
                                            scalar_t time) const = 0;
};

/**
 * Load swing trajectory settings from a configuration file
 *
 * @param fileName Path to the configuration file
 * @param fieldName Name of the field containing the configuration
 * @param verbose Whether to print verbose output
 * @return Configured SwingTrajectoryPlannerBase::Config object
 */
SwingTrajectoryPlannerBase::Config loadSwingTrajectorySettings(
    const std::string& fileName,
    const std::string& fieldName = "swing_trajectory_config",
    bool verbose = true);

}  // namespace ocs2::humanoid
