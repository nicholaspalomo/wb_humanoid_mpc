/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"

namespace ocs2::humanoid {

/**
 * Where the controller wants the contact frame of one foot to be on the ground, in the world frame: the landing pose of
 * the foot's swing (the one in flight, or else the next one in the executed schedule) or, without an upcoming swing, the
 * foot's current placement. The MuJoCo viewer draws the foot's contact patch at this pose, so that the planned position
 * and yaw of every step can be seen against the robot.
 */
struct TargetContactPose {
  enum class Kind {
    STANCE,           // no upcoming swing with a touch-down in the schedule: the foot's current placement
    SWING_IN_FLIGHT,  // landing pose of the swing in flight
    NEXT_SWING        // landing pose of the foot's next swing (the foot is still in contact)
  };
  bool valid = false;
  Kind kind = Kind::STANCE;
  vector2_t position = vector2_t::Zero();  // [m] contact frame origin, xy
  scalar_t height = 0.0;                   // [m] contact surface height
  scalar_t yaw = 0.0;                      // [rad] contact frame yaw about the world z axis
  bool yawPlanned = false;                 // the yaw comes from the heading model; otherwise it is the measured foot yaw
  scalar_t touchDownTime = 0.0;            // [s] when the foot is expected in contact there (the query time for STANCE)
};

/** Measured foot state and corrections that enter the target poses besides the plan and the schedule. */
struct TargetContactPoseInputs {
  scalar_t time = 0.0;
  feet_array_t<vector3_t> footPositions = makeFeetArray(vector3_t(vector3_t::Zero()));      // measured contact frame origins
  feet_array_t<scalar_t> footYaws = makeFeetArray(0.0);                                     // measured contact frame yaws
  feet_array_t<vector2_t> dcmStepAdjustment = makeFeetArray(vector2_t(vector2_t::Zero()));  // landing offset (zero when off)
  scalar_t terrainHeight = 0.0;                                                             // [m] ground height the swing targets land on
};

/**
 * Target contact pose of every foot at `inputs.time` from the active `plan` and the executed `schedule`. A foot in the
 * air takes the plan's foothold at its touch-down plus the DCM step adjustment; a foot in contact takes the landing pose
 * of its next swing, if the schedule has one with a touch-down, and its measured placement otherwise. The yaw is the
 * plan's landing yaw with the heading model and the measured foot yaw without it. Every pose is invalid when the plan is
 * not valid.
 */
feet_array_t<TargetContactPose> computeTargetContactPoses(const ContactPlan& plan,
                                                          const ModeSchedule& schedule,
                                                          const TargetContactPoseInputs& inputs);

}  // namespace ocs2::humanoid
