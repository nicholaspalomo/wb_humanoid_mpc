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
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"

namespace ocs2::humanoid {

// Pure abstract base class for gait scheduler that allows for mocking
class GaitScheduleBase {
 public:
  virtual ~GaitScheduleBase() = default;

  /**
   * @param [in] lowerBoundTime: The smallest time for which the ModeSchedule
   * should be defined.
   * @param [in] upperBoundTime: The greatest time for which the ModeSchedule
   * should be defined.
   */
  virtual ModeSchedule getModeSchedule(scalar_t lowerBoundTime,
                                       scalar_t upperBoundTime) = 0;

  /**
   * Get the current mode schedule without extending it.
   */
  virtual ModeSchedule getCurrentModeSchedule() const = 0;

  /**
   * Used to insert a new user defined logic in the given time period.
   *
   * @param [in] modeSequenceTemplate: The new mode sequence template to insert.
   * @param [in] startTime: The initial time from which the new mode sequence
   * template should start.
   * @param [in] finalTime: The final time until when the new mode sequence
   * needs to be defined.
   */
  virtual void insertModeSequenceTemplate(
      const ModeSequenceTemplate& modeSequenceTemplate, scalar_t startTime,
      scalar_t finalTime) = 0;

  /**
   * Update the current mode schedule.
   *
   * @param [in] modeSchedule: The new mode schedule.
   */
  virtual void updateModeSchedule(const ModeSchedule& modeSchedule) = 0;
};

}  // namespace ocs2::humanoid
