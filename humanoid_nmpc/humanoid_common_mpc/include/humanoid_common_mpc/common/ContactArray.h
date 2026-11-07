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
#include <cstddef>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"

namespace ocs2::humanoid {

/**
 * The most contacts a contact set (ModelSettings::contactNames) may have: one
 * bit of a uint32_t mode number each (ContactMask), as many as the MuJoCo
 * ground-truth contact mask covers. ModelSettings::Create() refuses a larger
 * set.
 */
// LINT.IfChange(max_num_contacts)
inline constexpr size_t kMaxNumContacts = 32;
// LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/MujocoUtils.h:contact_mask_capacity)

/**
 * One value per contact of a contact set, in contact order
 * (ModelSettings::contactNames: the 3-DoF contacts first, then the 6-DoF ones).
 *
 * The storage is a std::array of kMaxNumContacts entries, of which the first
 * size() are the contacts; the size is set at construction, or by assign(),
 * from the model's contact count. Indexing, iteration, fill() and copies never
 * allocate (an entry type that owns heap memory, such as std::string, still
 * allocates in its own copies), so the realtime path uses it in place of a
 * std::vector. Iteration and equality cover the first size() entries only.
 *
 *   ContactArray<vector6_t> wrenches(modelSettings.numContacts(),
 *                                    vector6_t::Zero());
 *   for (vector6_t& wrench : wrenches) wrench.setZero();
 *
 * Not thread-safe for writes; its const methods are safe to call concurrently.
 */
template <typename T>
class ContactArray {
 public:
  /** An array of no contacts. */
  ContactArray() = default;

  /** An array of `numContacts` value-initialized entries (T{}: zero for a
   * number, unset coefficients for an Eigen fixed-size type). Requires
   * numContacts <= kMaxNumContacts (checked). */
  explicit ContactArray(size_t numContacts) : size_(numContacts) {
    ABSL_CHECK_LE(numContacts, kMaxNumContacts);
  }

  /** An array of `numContacts` entries, each `value`. Requires numContacts <=
   * kMaxNumContacts (checked). */
  ContactArray(size_t numContacts, const T& value) {
    assign(numContacts, value);
  }

  ContactArray(const ContactArray&) = default;
  ContactArray& operator=(const ContactArray&) = default;
  ContactArray(ContactArray&&) noexcept = default;
  ContactArray& operator=(ContactArray&&) noexcept = default;
  ~ContactArray() = default;

  /** Resizes the array to `numContacts` entries and sets each to `value`.
   * Requires numContacts <= kMaxNumContacts (checked); meant for construction
   * and resets, not for every tick. */
  void assign(size_t numContacts, const T& value) {
    ABSL_CHECK_LE(numContacts, kMaxNumContacts);
    size_ = numContacts;
    fill(value);
  }

  /** Sets each of the size() entries to `value`. */
  void fill(const T& value) {
    for (size_t i = 0; i < size_; ++i) values_[i] = value;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  /** The entry of contact `contactIndex`. Unchecked: requires contactIndex <
   * size(). */
  T& operator[](size_t contactIndex) { return values_[contactIndex]; }
  const T& operator[](size_t contactIndex) const {
    return values_[contactIndex];
  }

  /** The first entry, never null, also for an empty array. */
  T* absl_nonnull data() { return values_.data(); }
  const T* absl_nonnull data() const { return values_.data(); }

  /** The size() entries, for range-for and the standard algorithms. */
  T* absl_nonnull begin() { return values_.data(); }
  T* absl_nonnull end() { return values_.data() + size_; }
  const T* absl_nonnull begin() const { return values_.data(); }
  const T* absl_nonnull end() const { return values_.data() + size_; }

 private:
  // Entries at and past size_ are storage, never read as contacts.
  std::array<T, kMaxNumContacts> values_{};
  // The number of contacts, at most kMaxNumContacts.
  size_t size_ = 0;
};

/** Whether `a` and `b` have the same size and equal entries. */
template <typename T>
bool operator==(const ContactArray<T>& a, const ContactArray<T>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!(a[i] == b[i])) return false;
  }
  return true;
}

template <typename T>
bool operator!=(const ContactArray<T>& a, const ContactArray<T>& b) {
  return !(a == b);
}

}  // namespace ocs2::humanoid
