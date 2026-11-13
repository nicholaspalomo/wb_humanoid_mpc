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

#include <cstddef>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/ContactMask.h"
#include "legged_common_mpc/common/Types.h"

namespace ocs2::legged {

/**
 * Gate of the planned contact wrenches in the inverse dynamics of the MRT joint
 * controllers, driven by the measured contact state
 * (robot_model/ContactEstimator.h).
 *
 * A contact point that is not measured in contact cannot transmit a wrench, so
 * its planned wrench is dropped whatever the executed plan expects there. With
 * the default configuration the planned wrench of a contact is applied in full
 * from the first control cycle it is measured in contact (an instantaneous
 * gate). The gate has one entry per contact of the contact set, in contact
 * order, fixed at construction.
 * Two optional shaping terms soften the load onset at touch-down, where a heel
 * or toe strike trips the contact detection while the sole is still landing:
 *  - debounceTime: measured contact must persist this long before the
 * contact's planned wrench is applied at all;
 *  - rampTime: the wrench then ramps linearly from zero to the planned value
 * over this time. Lift-off is never shaped: a contact that leaves the ground
 * loses its wrench at once. Both times default to zero, which reproduces the
 * instantaneous gate exactly.
 *
 * Copyable value type. Not thread-safe: update() and setConfig() mutate it, so
 * one thread (the realtime thread of the joint controller) owns it; its const
 * methods are safe to call concurrently with each other.
 */
class ContactWrenchGate {
 public:
  struct Config {
    scalar_t debounceTime = 0.0;  // [s] >= 0
    scalar_t rampTime = 0.0;      // [s] >= 0
  };

  /** The instantaneous gate (the default Config) of `numContacts` contacts.
   * Requires numContacts <= kMaxNumContacts (checked); ModelSettings::Create()
   * refuses a larger contact set. */
  explicit ContactWrenchGate(size_t numContacts)
      : onsetTime_(numContacts, /*value=*/std::nullopt),
        scales_(numContacts, /*value=*/1.0) {}
  /** The gate of `config` for `numContacts` contacts, or InvalidArgument when
   * validateConfig(config) refuses the configuration or there are more than
   * kMaxNumContacts contacts. */
  static absl::StatusOr<ContactWrenchGate> Create(const Config& config,
                                                  size_t numContacts);

  ContactWrenchGate(const ContactWrenchGate&) = default;
  ContactWrenchGate& operator=(const ContactWrenchGate&) = default;
  ContactWrenchGate(ContactWrenchGate&&) = default;
  ContactWrenchGate& operator=(ContactWrenchGate&&) = default;
  ~ContactWrenchGate() = default;

  /**
   * InvalidArgument unless both times are non-negative (NaN is refused).
   * Validate a configuration with it before setConfig(): a configuration read
   * from a file or sent by the operator is input, and setConfig() takes only a
   * valid one.
   */
  static absl::Status validateConfig(const Config& config);
  /** validateConfig(config).ok(), without building a Status: allocation-free.
   */
  static bool isValidConfig(const Config& config);

  /**
   * Replaces the configuration when validateConfig(config) is OK and returns
   * true; the onsets of the contacts measured in contact are kept. An invalid
   * configuration keeps the one in use and returns false: the realtime thread
   * calls this, where nothing may end the process, and the callers validate
   * what they read, so it does not happen. Allocation-free.
   */
  bool setConfig(const Config& config);
  const Config& getConfig() const { return config_; }

  /**
   * Advances the gate to `time` with the measured contact state of this control
   * cycle and returns the per-contact scale in [0, 1] of the planned wrench.
   * Call once per control cycle; time must not decrease between calls (a
   * decrease restarts the onset of every contact measured in contact).
   * Contact i reads entry i of `measuredContactFlags`; a contact past its size
   * is not in contact. Allocation-free.
   */
  const ContactArray<scalar_t>& update(scalar_t time,
                                       ContactMask measuredContactFlags);

  /** The scales of the last update (all ones before the first), one per
   * contact. */
  const ContactArray<scalar_t>& getScales() const { return scales_; }

  /** Scales each of the planned wrenches `plannedWrenchContactsEwrtWorld` (one
   * per contact, in place: the preallocated buffer of the realtime thread) by
   * the last update. An entry past numContacts() is left as it is.
   * Allocation-free. */
  void apply(ContactArray<vector6_t>& plannedWrenchContactsEwrtWorld) const;

  /** The number of contacts of the gate, fixed at construction. */
  size_t numContacts() const { return scales_.size(); }

  /** Forgets every contact onset; the next update starts the debounce and ramp
   * of every contact measured in contact anew. */
  void reset();

 private:
  ContactWrenchGate(const Config& config, size_t numContacts)
      : config_(config),
        onsetTime_(numContacts, /*value=*/std::nullopt),
        scales_(numContacts, /*value=*/1.0) {}

  Config config_;
  // [s] When the measured contact of each contact began; empty while not in
  // contact. One entry per contact.
  ContactArray<std::optional<scalar_t>> onsetTime_;
  // One per contact, all ones before the first update.
  ContactArray<scalar_t> scales_;
};

}  // namespace ocs2::legged
