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
#include <utility>
#include <vector>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_swing/ContactSwingTrajectory.h"

namespace ocs2::humanoid {

/**
 * The generalized swing planner of a multi-contact plan (design section 7.2,
 * humanoid_nmpc/docs/multi_contact_planning/README.md): the swings of every
 * contact of the model's contact set (ModelSettings::contactNames), each a
 * ContactSwingTrajectory from where the contact lifted off to its target
 * patch, and the reference of a contact at a time.
 *
 * A reference manager that plans swings owns one, rebuilds it before every
 * solve (setSwings()) and hands it out through
 * SwitchedModelReferenceManager::getContactSwingPlanner(); the
 * pre-computation and the swing terms read it during the solve. Not
 * thread-safe for writes: setSwings() and clear() run on the solver thread
 * between solves, and the const methods are then safe to call concurrently.
 */
class ContactSwingPlanner {
 public:
  /** A planner of `numContacts` contacts, none of them swinging. */
  explicit ContactSwingPlanner(size_t numContacts) : swings_(numContacts) {}

  ContactSwingPlanner(const ContactSwingPlanner&) = default;
  ContactSwingPlanner& operator=(const ContactSwingPlanner&) = default;
  ContactSwingPlanner(ContactSwingPlanner&&) noexcept = default;
  ContactSwingPlanner& operator=(ContactSwingPlanner&&) noexcept = default;
  ~ContactSwingPlanner() = default;

  size_t numContacts() const { return swings_.size(); }

  /** Replaces the swings of `contactIndex` (< numContacts()) by `swings`, in
   * time order and not overlapping. */
  void setSwings(size_t contactIndex,
                 std::vector<ContactSwingTrajectory> swings) {
    swings_[contactIndex] = std::move(swings);
  }

  /** Removes every swing. */
  void clear() {
    for (std::vector<ContactSwingTrajectory>& swings : swings_) swings.clear();
  }

  /** The swings of `contactIndex` (< numContacts()), in time order. */
  const std::vector<ContactSwingTrajectory>& swings(size_t contactIndex) const {
    return swings_[contactIndex];
  }

  /**
   * The reference of contact `contactIndex` at `time` [s]: that of its swing
   * whose window holds `time`, or std::nullopt when none does (the contact is
   * in contact, or swings without a planned touch-down) or the index is not a
   * contact of the planner. Linear in the contact's swings.
   */
  std::optional<ContactSwingReference> reference(size_t contactIndex,
                                                 scalar_t time) const;

 private:
  // One list per contact, each in time order.
  std::vector<std::vector<ContactSwingTrajectory>> swings_;
};

}  // namespace ocs2::humanoid
