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

#include <bit>
#include <cstddef>
#include <cstdint>

#include "absl/log/absl_check.h"

#include "humanoid_common_mpc/common/ContactArray.h"

namespace ocs2::humanoid {

/**
 * The active contacts of a contact set of at most kMaxNumContacts contacts,
 * stored as their mode number: contact i is bit numContacts - 1 - i, so the
 * first contact is the most significant bit (contactIsActiveInMode() in
 * gait/MotionPhaseDefinition.h is the reference reading). For a biped's two
 * feet, left then right, FLY = 0, RF = 1, LF = 2 and STANCE = 3; in a larger
 * set whose last two contacts are the feet those numbers mean the same feet.
 *
 *   ContactMask mask = ContactMask::fromModeNumber(modeNumber, numContacts);
 *   for (size_t i = 0; i < mask.size(); ++i) {
 *     if (mask[i]) ...
 *   }
 *
 * A trivially copyable value of a few bytes that never allocates: pass and
 * return it by value. Not thread-safe for writes; its const methods are safe to
 * call concurrently.
 */
class ContactMask {
 public:
  /** The mask of a set of no contacts. */
  ContactMask() = default;

  /** The mask of a set of `numContacts` contacts, none of them active.
   * Requires numContacts <= kMaxNumContacts (checked). */
  explicit ContactMask(size_t numContacts) : numContacts_(numContacts) {
    ABSL_CHECK_LE(numContacts, kMaxNumContacts);
  }

  ContactMask(const ContactMask&) = default;
  ContactMask& operator=(const ContactMask&) = default;
  ContactMask(ContactMask&&) noexcept = default;
  ContactMask& operator=(ContactMask&&) noexcept = default;
  ~ContactMask() = default;

  /** The mask of a set of `numContacts` contacts, all of them active. Requires
   * numContacts <= kMaxNumContacts (checked). */
  static ContactMask allActive(size_t numContacts) {
    ContactMask mask(numContacts);
    mask.bits_ = fullBits(numContacts);
    return mask;
  }

  /**
   * The mask of `modeNumber` in a set of `numContacts` contacts. A number with
   * a bit at or past numContacts is no mode of the set, and gives a mask with
   * no active contact, as contactIsActiveInMode() reads it. Unchecked, for the
   * realtime path: requires numContacts <= kMaxNumContacts.
   */
  static ContactMask fromModeNumber(size_t modeNumber, size_t numContacts) {
    ContactMask mask;
    mask.numContacts_ = numContacts;
    if ((static_cast<uint64_t>(modeNumber) >> numContacts) == 0) {
      mask.bits_ = static_cast<uint32_t>(modeNumber);
    }
    return mask;
  }

  /** The mode number of the active contacts. */
  size_t modeNumber() const { return bits_; }

  /** The number of contacts of the set, active or not. */
  size_t size() const { return numContacts_; }

  /** Whether contact `contactIndex` is active. Unchecked: requires
   * contactIndex < size(). */
  bool operator[](size_t contactIndex) const {
    return ((bits_ >> bitOf(contactIndex)) & 1U) != 0;
  }

  /** Makes contact `contactIndex` active or inactive. Unchecked: requires
   * contactIndex < size(). */
  void set(size_t contactIndex, bool active) {
    const uint32_t bit = uint32_t{1} << bitOf(contactIndex);
    bits_ = active ? (bits_ | bit) : (bits_ & ~bit);
  }

  /** The number of active contacts. */
  size_t numActive() const { return static_cast<size_t>(std::popcount(bits_)); }

  /** Whether no contact is active (true for an empty set). */
  bool none() const { return bits_ == 0; }

  /** Whether every contact is active (true for an empty set). */
  bool all() const { return bits_ == fullBits(numContacts_); }

  /** Appends the flags in contact order, e.g. "{1,0,1,1}". */
  template <typename Sink>
  friend void AbslStringify(Sink& sink, ContactMask mask) {
    sink.Append("{");
    for (size_t i = 0; i < mask.size(); ++i) {
      if (i > 0) sink.Append(",");
      sink.Append(mask[i] ? "1" : "0");
    }
    sink.Append("}");
  }

 private:
  /** The bit of contact `contactIndex`. */
  size_t bitOf(size_t contactIndex) const {
    return numContacts_ - 1 - contactIndex;
  }

  /** The mode number in which all `numContacts` contacts are active. */
  static uint32_t fullBits(size_t numContacts) {
    return static_cast<uint32_t>((uint64_t{1} << numContacts) - 1);
  }

  // The mode number; no bit at or past numContacts_ is set.
  uint32_t bits_ = 0;
  // The number of contacts of the set, at most kMaxNumContacts.
  size_t numContacts_ = 0;
};

/** Whether `a` and `b` are masks of sets of the same size with the same active
 * contacts. */
inline bool operator==(ContactMask a, ContactMask b) {
  return a.size() == b.size() && a.modeNumber() == b.modeNumber();
}

inline bool operator!=(ContactMask a, ContactMask b) {
  return !(a == b);
}

}  // namespace ocs2::humanoid
