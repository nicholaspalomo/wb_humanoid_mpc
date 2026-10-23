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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"

namespace ocs2::humanoid {

/** The static field list of an applier class (HotFieldApplier: staticFields()). */
using HotFieldList = absl::Span<const absl::string_view> (*absl_nonnull)();

/**
 * One applier of a formulation's list: its class's static field list, and how to make it from the formulation's MPC
 * interface. A formulation keeps one constexpr array of these, from which its appliers are made, in its order
 * (makeHotFieldAppliers()), and its hot field names are read (hotFieldNames()), so that the two cannot disagree; the
 * updater's appliedFields() is the same union over the appliers it was given.
 */
template <typename Interface>
struct HotFieldApplierEntry {
  HotFieldList staticFields;
  // Makes the applier for the interface, which it may keep; the refusal of an applier that cannot be built for it.
  absl::StatusOr<std::unique_ptr<HotFieldApplier>> (*absl_nonnull make)(const Interface& interface);
};

/** make() of an applier that needs nothing of the interface. */
template <typename Applier, typename Interface>
absl::StatusOr<std::unique_ptr<HotFieldApplier>> makeHotFieldApplier(const Interface& /*interface*/) {
  return std::make_unique<Applier>();
}

/** Returns the fields of `fieldLists`, sorted and each once. Allocates; not for the realtime thread. */
std::vector<std::string> sortedFieldUnion(absl::Span<const absl::Span<const absl::string_view>> fieldLists);

/** Returns the static fields of the appliers of `entries`, sorted and each once: a formulation's hot field names. */
template <typename Interface>
std::vector<std::string> hotFieldNames(absl::Span<const HotFieldApplierEntry<Interface>> entries) {
  std::vector<absl::Span<const absl::string_view>> fieldLists;
  fieldLists.reserve(entries.size());
  for (const HotFieldApplierEntry<Interface>& entry : entries) fieldLists.push_back(entry.staticFields());
  return sortedFieldUnion(fieldLists);
}

/** Returns the appliers of `entries` made for `interface`, in their order, or the first refusal of a make(). */
template <typename Interface>
absl::StatusOr<std::vector<std::unique_ptr<HotFieldApplier>>> makeHotFieldAppliers(
    absl::Span<const HotFieldApplierEntry<Interface>> entries, const Interface& interface) {
  std::vector<std::unique_ptr<HotFieldApplier>> appliers;
  appliers.reserve(entries.size());
  for (const HotFieldApplierEntry<Interface>& entry : entries) {
    absl::StatusOr<std::unique_ptr<HotFieldApplier>> applier = entry.make(interface);
    if (!applier.ok()) return applier.status();
    appliers.push_back(*std::move(applier));
  }
  return appliers;
}

}  // namespace ocs2::humanoid
