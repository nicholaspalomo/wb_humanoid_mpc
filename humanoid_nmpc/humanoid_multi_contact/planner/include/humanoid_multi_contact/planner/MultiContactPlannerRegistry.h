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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/planner/MultiContactPlanner.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The multi-contact planners by name (design section 5.3): the only place a
 * planner is added. A scenario names its planner (Scenario.planner), and the
 * registry builds it. A name of the design (kPlannerNames) that no planner
 * implements yet is refused as Unimplemented, any other unknown name as
 * InvalidArgument; both list the available planners.
 *
 * Not thread-safe: built and used on one thread.
 */
class MultiContactPlannerRegistry {
 public:
  using Factory =
      std::function<absl::StatusOr<std::unique_ptr<MultiContactPlanner>>()>;

  /** A registry with the built-in planners: "stance_graph",
   * "contact_invariant" and "stance_graph_then_contact_invariant". */
  MultiContactPlannerRegistry();

  /** Adds the planner `name`, replacing one of that name. */
  void add(std::string name, std::string description, Factory factory);
  bool has(absl::string_view name) const;
  std::vector<std::string> names() const;
  /** "stance_graph (the stance-graph planner), ...". */
  std::string availableNames() const;

  /** The planner `name` names. */
  absl::StatusOr<std::unique_ptr<MultiContactPlanner>> create(
      absl::string_view name) const;

 private:
  struct Entry {
    std::string name;
    std::string description;
    Factory factory;
  };
  /** The entry `name` names; nullptr for an unknown one. */
  const Entry* absl_nullable find(absl::string_view name) const;

  std::vector<Entry> entries_;
};

}  // namespace ocs2::humanoid::multi_contact
