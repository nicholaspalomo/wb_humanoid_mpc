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

#include "robot_model/ContactEstimatorRegistry.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"

#include "robot_model/AlwaysInContactEstimator.h"
#include "robot_model/RobotStateContactEstimator.h"

namespace robot::model {

// LINT.IfChange(contact_estimator_names)
ContactEstimatorRegistry::ContactEstimatorRegistry() {
  add(kRobotState, "the contact flags the hardware or simulator interface writes into the RobotState",
      [] { return std::make_shared<RobotStateContactEstimator>(); });
  add(kAlwaysInContact, "every contact point touching: the executed schedule is taken as the measured contact state",
      [] { return std::make_shared<AlwaysInContactEstimator>(); });
}
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:contact_estimator, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto:contact_estimator, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto:contact_estimator, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto:contact_estimator, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:contact_estimator, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:contact_estimator)
// clang-format on

std::string ContactEstimatorRegistry::canonicalName(absl::string_view name) {
  return absl::AsciiStrToLower(absl::StripAsciiWhitespace(name));
}

void ContactEstimatorRegistry::add(absl::string_view name, absl::string_view description, Factory factory) {
  const std::string canonical = canonicalName(name);
  ABSL_CHECK(!canonical.empty()) << "ContactEstimatorRegistry: an estimator needs a name";
  ABSL_CHECK(factory != nullptr) << "ContactEstimatorRegistry: estimator '" << canonical << "' has no factory";
  ABSL_CHECK(!has(canonical)) << "ContactEstimatorRegistry: estimator '" << canonical << "' is already registered";
  estimators_.push_back({{.name = canonical, .description = std::string(description)}, std::move(factory)});
}

bool ContactEstimatorRegistry::has(absl::string_view name) const {
  const std::string canonical = canonicalName(name);
  return std::any_of(estimators_.begin(), estimators_.end(), [&canonical](const Registered& r) { return r.entry.name == canonical; });
}

std::shared_ptr<ContactEstimator> ContactEstimatorRegistry::create(absl::string_view name) const {
  const std::string canonical = canonicalName(name);
  const std::vector<Registered>::const_iterator found =
      std::find_if(estimators_.begin(), estimators_.end(), [&canonical](const Registered& r) { return r.entry.name == canonical; });
  ABSL_CHECK(found != estimators_.end()) << "ContactEstimatorRegistry: unknown contact estimator '" << name
                                         << "'; available: " << availableNames()
                                         << ". Check has() first: a name from a file is the caller's to validate.";
  return found->factory();
}

std::vector<ContactEstimatorRegistry::Entry> ContactEstimatorRegistry::available() const {
  std::vector<Entry> entries;
  entries.reserve(estimators_.size());
  for (const Registered& registered : estimators_) entries.push_back(registered.entry);
  return entries;
}

std::string ContactEstimatorRegistry::availableNames() const {
  return absl::StrJoin(estimators_, ", ",
                       [](std::string* absl_nonnull out, const Registered& registered) { out->append(registered.entry.name); });
}

}  // namespace robot::model
