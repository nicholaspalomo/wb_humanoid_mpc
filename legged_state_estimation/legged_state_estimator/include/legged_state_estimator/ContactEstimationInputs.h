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
#include <vector>

#include "ocs2_core/Types.h"

namespace ocs2::legged::estimation {

/**
 * What the caller of a state estimator knows about the contacts this cycle
 * beyond the backend's state: the world height of the surface each contact
 * rests on (the right-hand side of KalmanBaseEstimator's height row), and the
 * times since and until each contact's planned switch, which the estimator's
 * contact-trust ramp reads when it is on. A contact without a height has no
 * height row; its relative position and velocity rows still apply.
 *
 * The robot process owns one, sized to its contacts, and fills it every cycle
 * before StateEstimator::estimate() (ContactSurfaceHeightSource): the flat
 * ground's height while walking, the planned patch's under a multi-contact
 * plan; and the planned switch times from the MPC policy's mode schedule
 * (RobotController::plannedContactSwitchTimes()). A freshly constructed object
 * gives every contact the height 0.0, flat ground at z = 0, and no known
 * switch (+infinity both ways).
 *
 * REALTIME. The storage is sized by the constructor; no method allocates
 * afterwards. The per-contact accessors take `contact` < numContacts() as an
 * unchecked precondition. Not thread-safe.
 */
class ContactEstimationInputs {
 public:
  /** Sizes the inputs for `numContacts` contacts, each at height 0.0 with no
   * known switch. Allocates. */
  explicit ContactEstimationInputs(size_t numContacts);

  ContactEstimationInputs(const ContactEstimationInputs&) = default;
  ContactEstimationInputs& operator=(const ContactEstimationInputs&) = default;
  ContactEstimationInputs(ContactEstimationInputs&&) noexcept = default;
  ContactEstimationInputs& operator=(ContactEstimationInputs&&) noexcept =
      default;
  ~ContactEstimationInputs() = default;

  size_t numContacts() const { return nomPosContactsRtWorldEwrtWorldZ_.size(); }

  /** Gives contact `contact` the surface height `height` [m]. */
  void setNomPosContactRtWorldEwrtWorldZ(size_t contact, scalar_t height) {
    nomPosContactsRtWorldEwrtWorldZ_[contact] = height;
    hasNomPosContactsRtWorldEwrtWorldZ_[contact] = true;
  }
  /** Leaves contact `contact` without a surface height: no height row. */
  void clearNomPosContactRtWorldEwrtWorldZ(size_t contact) {
    hasNomPosContactsRtWorldEwrtWorldZ_[contact] = false;
  }
  /** Gives every contact the surface height `height` [m]. */
  void setAllNomPosContactsRtWorldEwrtWorldZ(scalar_t height);
  /** Returns whether contact `contact` has a surface height. */
  bool hasNomPosContactRtWorldEwrtWorldZ(size_t contact) const {
    return hasNomPosContactsRtWorldEwrtWorldZ_[contact];
  }
  /** Returns contact `contact`'s surface height [m]; precondition:
   * hasNomPosContactRtWorldEwrtWorldZ(contact). */
  scalar_t nomPosContactRtWorldEwrtWorldZ(size_t contact) const {
    return nomPosContactsRtWorldEwrtWorldZ_[contact];
  }

  /** Gives contact `contact` the time [s] since its planned contact state last
   * changed and the time [s] until it next changes, each >= 0 or +infinity
   * where the plan shows no such switch. */
  void setPlannedContactSwitchTimes(size_t contact,
                                    scalar_t timeSinceSwitch,
                                    scalar_t timeToSwitch) {
    plannedTimesSinceSwitch_[contact] = timeSinceSwitch;
    plannedTimesToSwitch_[contact] = timeToSwitch;
  }
  /** Forgets every contact's planned switch times, setting both to
   * +infinity. */
  void clearPlannedContactSwitchTimes();
  /** Returns the time [s] since contact `contact`'s last planned switch;
   * +infinity when unknown. */
  scalar_t plannedTimeSinceSwitch(size_t contact) const {
    return plannedTimesSinceSwitch_[contact];
  }
  /** Returns the time [s] until contact `contact`'s next planned switch;
   * +infinity when unknown. */
  scalar_t plannedTimeToSwitch(size_t contact) const {
    return plannedTimesToSwitch_[contact];
  }

 private:
  /** [m] One per contact; meaningful where the flag below is set. */
  std::vector<scalar_t> nomPosContactsRtWorldEwrtWorldZ_;
  std::vector<bool> hasNomPosContactsRtWorldEwrtWorldZ_;
  /** [s] One per contact; +infinity: no switch known. */
  std::vector<scalar_t> plannedTimesSinceSwitch_;
  std::vector<scalar_t> plannedTimesToSwitch_;
};

}  // namespace ocs2::legged::estimation
