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
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"

namespace ocs2::humanoid {

/**
 * The correspondence, by name, between the candidates of a multi-contact
 * plan's contact set (multi_contact::ContactSet, whose order numbers the bits
 * of a plan's ContactMask, bit i for candidate i) and the contacts of the MPC
 * (ModelSettings::contactNames: the 3-DoF contacts, then the 6-DoF ones, whose
 * order numbers the bits of a mode number, the first contact the most
 * significant bit; contactIsActiveInMode()). The two orders differ - the MPC
 * lists its point contacts first and its feet last - so a contact is never
 * mapped by its bit position: the map is built once, by name, at load.
 *
 * Either side may hold contacts the other has not: an MPC contact that is not
 * a candidate is never active in a plan's mode, and a candidate that is not an
 * MPC contact must never be active in a plan the MPC tracks (modeOf()).
 *
 * A value type, immutable after Create(): every method is const and safe to
 * call concurrently.
 */
class PlanContactOrder {
 public:
  /**
   * The map between `planContacts` and `mpcContactNames`. InvalidArgument
   * when an MPC name is listed twice, or there are too many MPC contacts for a
   * mode number.
   */
  static absl::StatusOr<PlanContactOrder> Create(
      const multi_contact::ContactSet& planContacts,
      const std::vector<std::string>& mpcContactNames);

  PlanContactOrder(const PlanContactOrder&) = default;
  PlanContactOrder& operator=(const PlanContactOrder&) = default;
  PlanContactOrder(PlanContactOrder&&) noexcept = default;
  PlanContactOrder& operator=(PlanContactOrder&&) noexcept = default;
  ~PlanContactOrder() = default;

  size_t numPlanContacts() const { return mpcIndexByPlanIndex_.size(); }
  size_t numMpcContacts() const { return planIndexByMpcIndex_.size(); }

  /** The MPC contact of candidate `planIndex`, or std::nullopt when the MPC
   * has no contact of its name or the index is not a candidate. */
  std::optional<size_t> mpcIndexOf(size_t planIndex) const;
  /** The candidate of MPC contact `mpcIndex`, or std::nullopt when the plan's
   * contact set has none of its name or the index is not an MPC contact. */
  std::optional<size_t> planIndexOf(size_t mpcIndex) const;
  /** The candidate named `name`, or std::nullopt when the plan's contact set
   * has none. */
  std::optional<size_t> planIndexOfName(absl::string_view name) const;
  /** The name of candidate `planIndex` (< numPlanContacts()). */
  const std::string& planContactName(size_t planIndex) const {
    return planContactNames_[planIndex];
  }
  /** The name of MPC contact `mpcIndex` (< numMpcContacts()). */
  const std::string& mpcContactName(size_t mpcIndex) const {
    return mpcContactNames_[mpcIndex];
  }

  /**
   * The MPC mode number of the plan's contact mode `mask`: for every active
   * candidate, the bit of its MPC contact. InvalidArgument naming the
   * candidate when one that is active has no MPC contact, or when `mask` sets
   * a bit beyond the candidates.
   */
  absl::StatusOr<size_t> modeOf(multi_contact::ContactMask mask) const;

  /**
   * The plan's contact mode of the MPC mode number `mode`. InvalidArgument
   * naming the contact when an active MPC contact is no candidate of the
   * plan's set, or when `mode` sets a bit beyond the MPC contacts.
   */
  absl::StatusOr<multi_contact::ContactMask> maskOf(size_t mode) const;

 private:
  PlanContactOrder() = default;

  std::vector<std::string> mpcContactNames_;
  std::vector<std::string> planContactNames_;
  // Indexed by candidate, then by MPC contact.
  std::vector<std::optional<size_t>> mpcIndexByPlanIndex_;
  std::vector<std::optional<size_t>> planIndexByMpcIndex_;
  absl::flat_hash_map<std::string, size_t> planIndexByName_;
};

}  // namespace ocs2::humanoid
