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

#include <array>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * Gate of the planned contact wrenches in the inverse dynamics of the MRT joint controllers, driven by the measured
 * contact state (robot_model/ContactEstimator.h).
 *
 * A contact point that is not measured in contact cannot transmit a wrench, so its planned wrench is dropped whatever
 * the executed plan expects there. With the default configuration the planned wrench of a foot is applied in full from
 * the first control cycle its contact is measured (an instantaneous gate). Two optional shaping terms soften the load
 * onset at touch-down, where a heel or toe strike trips the contact detection while the sole is still landing:
 *  - debounceTime: measured contact must persist this long before the foot's planned wrench is applied at all;
 *  - rampTime: the wrench then ramps linearly from zero to the planned value over this time.
 * Lift-off is never shaped: a foot that leaves the ground loses its wrench at once. Both times default to zero, which
 * reproduces the instantaneous gate exactly.
 */
class ContactWrenchGate {
 public:
  struct Config {
    scalar_t debounceTime{0.0};  // [s] >= 0
    scalar_t rampTime{0.0};      // [s] >= 0
  };

  ContactWrenchGate();
  explicit ContactWrenchGate(const Config& config);

  void setConfig(const Config& config);
  const Config& getConfig() const { return config_; }

  /**
   * Advances the gate to `time` with the measured contact state of this control cycle and returns the per-foot scale in
   * [0, 1] of the planned wrench. Call once per control cycle; time must not decrease between calls (a decrease restarts
   * the onset of every foot in contact).
   */
  const feet_array_t<scalar_t>& update(scalar_t time, const contact_flag_t& measuredContactFlags);

  /** The scales of the last update (all ones before the first). */
  const feet_array_t<scalar_t>& getScales() const { return scales_; }

  /** The planned wrenches scaled by the last update. */
  std::array<vector6_t, 2> apply(std::array<vector6_t, 2> plannedWrenches) const;

  /** Forgets every contact onset; the next update starts the debounce and ramp of every foot in contact anew. */
  void reset();

 private:
  Config config_;
  feet_array_t<scalar_t> onsetTime_{};  // time the measured contact of each foot began; NaN while not in contact
  feet_array_t<scalar_t> scales_{};
};

}  // namespace ocs2::humanoid
