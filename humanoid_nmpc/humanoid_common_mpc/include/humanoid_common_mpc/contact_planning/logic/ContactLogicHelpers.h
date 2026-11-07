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
#include <cstdint>

#include "absl/types/span.h"

#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"

namespace ocs2::humanoid {

/** Fixes a free binary; false when it is already fixed to the other value (a
 * contradiction). */
inline bool fixBinary(MiqpAssignment& a,
                      int index,
                      int8_t value,
                      bool& changed) {
  int8_t& entry = a[static_cast<size_t>(index)];
  if (entry == kMiqpFree) {
    entry = value;
    changed = true;
    return true;
  }
  return entry == value;
}

/**
 * Propagates, at one node, that a foot of `boundFeet` lifts only while at
 * least `minContactCount` of the node's feet stay in contact: the counting
 * core of the rules written over the number of feet (no_flight with a count of
 * one, minimum_double_support with its minimum support count).
 *
 * `nodeBinaries` holds the binary index of every foot at the node, in foot
 * order, and `boundFeet` has one entry per foot. A free binary counts as a
 * possible contact, so the pass is sound on a partial assignment: a bound foot
 * is fixed down only when the other feet cannot reach the count in any
 * completion, and a contradiction is reported only when a bound foot is
 * already lifted while they cannot. Fixes binaries of `a` and sets `changed`
 * through fixBinary(). Returns false on a contradiction.
 */
inline bool propagateMinimumContactCount(absl::Span<const int> nodeBinaries,
                                         ContactMask boundFeet,
                                         size_t minContactCount,
                                         MiqpAssignment& a,
                                         bool& changed) {
  // The feet that are or may still be in contact at the node. Fixing a free
  // binary to 1 below leaves it unchanged.
  size_t possibleContacts = 0;
  for (const int index : nodeBinaries) {
    if (a[static_cast<size_t>(index)] != 0) ++possibleContacts;
  }
  for (size_t foot = 0; foot < nodeBinaries.size(); ++foot) {
    if (!boundFeet[foot]) continue;
    const int index = nodeBinaries[foot];
    const int8_t value = a[static_cast<size_t>(index)];
    const size_t possibleOtherContacts =
        possibleContacts - (value != 0 ? 1 : 0);
    if (possibleOtherContacts >= minContactCount) continue;
    if (value == 0) return false;
    fixBinary(a, index, /*value=*/1, changed);
  }
  return true;
}

/** What the lift of an overdue foot at a node depends on, for one other foot.
 */
struct OtherFootAtNode {
  // Its binary at the node and its contact state just before it are fixed.
  bool decided = false;
  // Its binary at the node is 1 (read only when `decided`).
  bool inContact = false;
  // It supports at the node, is overdue as well, and the tie break lets it
  // lift first.
  bool liftsFirst = false;
};

/** Whether the double-support hold after the latest touch-down keeps an
 * overdue foot down at a node (ContactLogicState::heldAfterTouchDown()). */
enum class DoubleSupportHold { kNone, kHeld };

/**
 * Returns whether a foot that is overdue to lift at a node (its maximum
 * contact duration reached) must lift there, given what the other feet do at
 * the node: only once every other foot is decided, at least one of them
 * supports (otherwise the lift would be a flight phase), the double-support
 * hold after the latest touch-down does not keep the foot down (`hold`), and
 * no supporting other foot that is overdue too lifts first. For a biped this
 * is the one other foot. The counting core of phase_durations.
 */
inline bool overdueLiftIsImplied(absl::Span<const OtherFootAtNode> otherFeet,
                                 DoubleSupportHold hold) {
  bool anyOtherSupports = false;
  for (const OtherFootAtNode& other : otherFeet) {
    if (!other.decided) return false;
    if (!other.inContact) continue;
    anyOtherSupports = true;
    if (other.liftsFirst) return false;
  }
  return anyOtherSupports && hold == DoubleSupportHold::kNone;
}

}  // namespace ocs2::humanoid
