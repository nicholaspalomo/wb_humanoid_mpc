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

#include "humanoid_multi_contact/planner/MultiContactPlannerRegistry.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/planner/ContactInvariantPlanner.h"
#include "humanoid_multi_contact/planner/StanceGraphPlanner.h"
#include "humanoid_multi_contact/planner/StanceGraphThenContactInvariantPlanner.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid::multi_contact {

MultiContactPlannerRegistry::MultiContactPlannerRegistry() {
  add(std::string(kStanceGraphPlannerName),
      "the stance-graph planner, design section 5",
      []() -> absl::StatusOr<std::unique_ptr<MultiContactPlanner>> {
        absl::StatusOr<std::unique_ptr<StanceGraphPlanner>> planner =
            StanceGraphPlanner::Create(StanceGraphPlannerOptions());
        if (!planner.ok()) return planner.status();
        return std::unique_ptr<MultiContactPlanner>(*std::move(planner));
      });
  add(std::string(kContactInvariantPlannerName),
      "contact-invariant optimization, design section 4",
      []() -> absl::StatusOr<std::unique_ptr<MultiContactPlanner>> {
        absl::StatusOr<std::unique_ptr<ContactInvariantPlanner>> planner =
            ContactInvariantPlanner::Create(ContactInvariantPlannerOptions());
        if (!planner.ok()) return planner.status();
        return std::unique_ptr<MultiContactPlanner>(*std::move(planner));
      });
  add(std::string(kStanceGraphThenContactInvariantPlannerName),
      "the stance graph initializing contact-invariant optimization, design "
      "section 5.2",
      []() -> absl::StatusOr<std::unique_ptr<MultiContactPlanner>> {
        absl::StatusOr<std::unique_ptr<StanceGraphThenContactInvariantPlanner>>
            planner = StanceGraphThenContactInvariantPlanner::Create(
                StanceGraphThenContactInvariantOptions());
        if (!planner.ok()) return planner.status();
        return std::unique_ptr<MultiContactPlanner>(*std::move(planner));
      });
}

void MultiContactPlannerRegistry::add(std::string name,
                                      std::string description,
                                      Factory factory) {
  for (Entry& entry : entries_) {
    if (entry.name == name) {
      entry.description = std::move(description);
      entry.factory = std::move(factory);
      return;
    }
  }
  entries_.push_back(Entry{.name = std::move(name),
                           .description = std::move(description),
                           .factory = std::move(factory)});
}

bool MultiContactPlannerRegistry::has(absl::string_view name) const {
  return find(name) != nullptr;
}

std::vector<std::string> MultiContactPlannerRegistry::names() const {
  std::vector<std::string> result;
  for (const Entry& entry : entries_) result.push_back(entry.name);
  return result;
}

std::string MultiContactPlannerRegistry::availableNames() const {
  std::vector<std::string> parts;
  for (const Entry& entry : entries_) {
    parts.push_back(absl::StrCat(entry.name, " (", entry.description, ")"));
  }
  return absl::StrJoin(parts, ", ");
}

absl::StatusOr<std::unique_ptr<MultiContactPlanner>>
MultiContactPlannerRegistry::create(absl::string_view name) const {
  if (const Entry* absl_nullable entry = find(name); entry != nullptr) {
    return entry->factory();
  }
  if (std::find(kPlannerNames.begin(), kPlannerNames.end(), name) !=
      kPlannerNames.end()) {
    return absl::UnimplementedError(
        absl::StrCat("the planner '", name,
                     "' of the design is not implemented yet; available: ",
                     availableNames()));
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "unknown planner '", name, "'; available: ", availableNames()));
}

const MultiContactPlannerRegistry::Entry* absl_nullable
MultiContactPlannerRegistry::find(absl::string_view name) const {
  for (const Entry& entry : entries_) {
    if (entry.name == name) return &entry;
  }
  return nullptr;
}

}  // namespace ocs2::humanoid::multi_contact
