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

#include <robot_model/ContactEstimatorRegistry.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <robot_model/AlwaysInContactEstimator.h>
#include <robot_model/RobotStateContactEstimator.h>

namespace robot::model {

// LINT.IfChange(contact_estimator_names)
ContactEstimatorRegistry::ContactEstimatorRegistry() {
  add(kRobotState, "the contact flags the hardware or simulator interface writes into the RobotState",
      [] { return std::make_shared<RobotStateContactEstimator>(); });
  add(kAlwaysInContact, "every contact point touching: the executed schedule is taken as the measured contact state",
      [] { return std::make_shared<AlwaysInContactEstimator>(); });
}
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:contact_estimator, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:contact_estimator, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml:contact_estimator, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml:contact_estimator, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:contact_estimator)
// clang-format on

std::string ContactEstimatorRegistry::canonicalName(const std::string& name) {
  const auto isBlank = [](unsigned char c) { return std::isspace(c) != 0; };
  auto begin = std::find_if_not(name.begin(), name.end(), isBlank);
  auto end = std::find_if_not(name.rbegin(), name.rend(), isBlank).base();
  std::string canonical = begin < end ? std::string(begin, end) : std::string();
  std::transform(canonical.begin(), canonical.end(), canonical.begin(), [](unsigned char c) { return std::tolower(c); });
  return canonical;
}

void ContactEstimatorRegistry::add(const std::string& name, const std::string& description, Factory factory) {
  const std::string canonical = canonicalName(name);
  if (canonical.empty()) throw std::invalid_argument("ContactEstimatorRegistry: an estimator needs a name");
  if (!factory) throw std::invalid_argument("ContactEstimatorRegistry: estimator '" + canonical + "' has no factory");
  if (has(canonical)) throw std::invalid_argument("ContactEstimatorRegistry: estimator '" + canonical + "' is already registered");
  estimators_.push_back({{canonical, description}, std::move(factory)});
}

bool ContactEstimatorRegistry::has(const std::string& name) const {
  const std::string canonical = canonicalName(name);
  return std::any_of(estimators_.begin(), estimators_.end(), [&](const Registered& r) { return r.entry.name == canonical; });
}

std::shared_ptr<ContactEstimator> ContactEstimatorRegistry::create(const std::string& name) const {
  const std::string canonical = canonicalName(name);
  for (const Registered& registered : estimators_) {
    if (registered.entry.name == canonical) return registered.factory();
  }
  throw std::invalid_argument("ContactEstimatorRegistry: unknown contact estimator '" + name + "'; available: " + availableNames());
}

std::vector<ContactEstimatorRegistry::Entry> ContactEstimatorRegistry::available() const {
  std::vector<Entry> entries;
  for (const Registered& registered : estimators_) entries.push_back(registered.entry);
  return entries;
}

std::string ContactEstimatorRegistry::availableNames() const {
  std::string names;
  for (const Registered& registered : estimators_) {
    if (!names.empty()) names += ", ";
    names += registered.entry.name;
  }
  return names;
}

}  // namespace robot::model
