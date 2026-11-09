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

#include "absl/types/span.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_state_estimator/ContactEstimationInputs.h"

namespace ocs2::humanoid {

/**
 * Where the robot process takes the surface height of each contact from for
 * the state estimator's height rows (estimation::ContactEstimationInputs):
 * the flat ground while walking (FlatGroundSurfaceHeights), the patches of the
 * multi-contact plan the MPC tracks (PlannedSurfaceHeights). Built at set-up
 * (contactSurfaceHeightSource()) and owned by RobotProcess::Config.
 *
 * REALTIME. fill() runs on the realtime thread every cycle, before
 * StateEstimator::estimate(): it allocates nothing, takes no lock and does no
 * I/O. Immutable after construction, so fill() is safe to call concurrently.
 */
class ContactSurfaceHeightSource {
 public:
  ContactSurfaceHeightSource() = default;
  ContactSurfaceHeightSource(const ContactSurfaceHeightSource&) = delete;
  ContactSurfaceHeightSource& operator=(const ContactSurfaceHeightSource&) =
      delete;
  virtual ~ContactSurfaceHeightSource() = default;

  /**
   * Writes the surface height of every contact of `inputs`, or clears it where
   * no surface is known, from the state estimator's estimated contact
   * positions `estPosContactsRtWorldEwrtWorld` (StateEstimator::
   * estPosContactsRtWorldEwrtWorld(): the previous cycle's estimate, in contact
   * order; empty for an estimator that keeps none). Allocates nothing.
   */
  virtual void fill(absl::Span<const vector3_t> estPosContactsRtWorldEwrtWorld,
                    estimation::ContactEstimationInputs& inputs) const = 0;
};

}  // namespace ocs2::humanoid
