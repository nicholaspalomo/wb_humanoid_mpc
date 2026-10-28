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

#include "humanoid_centroidal_mpc/multi_contact/PlanContactOrder.h"

#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

absl::StatusOr<PlanContactOrder> PlanContactOrder::Create(
    const multi_contact::ContactSet& planContacts,
    const std::vector<std::string>& mpcContactNames) {
  constexpr size_t kModeNumberBits = std::numeric_limits<size_t>::digits;
  if (mpcContactNames.size() >= kModeNumberBits) {
    return absl::InvalidArgumentError(
        absl::StrCat("[PlanContactOrder] the MPC has ", mpcContactNames.size(),
                     " contacts, but a mode number holds fewer than ",
                     kModeNumberBits, "."));
  }
  absl::flat_hash_map<std::string, size_t> mpcIndexByName;
  for (size_t i = 0; i < mpcContactNames.size(); ++i) {
    if (!mpcIndexByName.try_emplace(mpcContactNames[i], i).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("[PlanContactOrder] the MPC lists the contact '",
                       mpcContactNames[i], "' twice."));
    }
  }
  PlanContactOrder order;
  order.mpcContactNames_ = mpcContactNames;
  order.planIndexByMpcIndex_.assign(mpcContactNames.size(), std::nullopt);
  order.mpcIndexByPlanIndex_.assign(static_cast<size_t>(planContacts.size()),
                                    std::nullopt);
  for (int planIndex = 0; planIndex < planContacts.size(); ++planIndex) {
    const std::string& name = planContacts.candidate(planIndex).name;
    order.planContactNames_.push_back(name);
    order.planIndexByName_.try_emplace(name, static_cast<size_t>(planIndex));
    const absl::flat_hash_map<std::string, size_t>::const_iterator mpc =
        mpcIndexByName.find(name);
    if (mpc == mpcIndexByName.end()) continue;
    order.mpcIndexByPlanIndex_[static_cast<size_t>(planIndex)] = mpc->second;
    order.planIndexByMpcIndex_[mpc->second] = static_cast<size_t>(planIndex);
  }
  return order;
}

std::optional<size_t> PlanContactOrder::mpcIndexOf(size_t planIndex) const {
  if (planIndex >= mpcIndexByPlanIndex_.size()) return std::nullopt;
  return mpcIndexByPlanIndex_[planIndex];
}

std::optional<size_t> PlanContactOrder::planIndexOf(size_t mpcIndex) const {
  if (mpcIndex >= planIndexByMpcIndex_.size()) return std::nullopt;
  return planIndexByMpcIndex_[mpcIndex];
}

std::optional<size_t> PlanContactOrder::planIndexOfName(
    absl::string_view name) const {
  const absl::flat_hash_map<std::string, size_t>::const_iterator found =
      planIndexByName_.find(name);
  if (found == planIndexByName_.end()) return std::nullopt;
  return found->second;
}

absl::StatusOr<size_t> PlanContactOrder::modeOf(
    multi_contact::ContactMask mask) const {
  const size_t numPlan = numPlanContacts();
  if (numPlan < static_cast<size_t>(multi_contact::kMaxContactCandidates) &&
      (mask >> numPlan) != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[PlanContactOrder] the contact mode ", mask, " sets a bit beyond the ",
        numPlan, " candidates of the plan."));
  }
  size_t mode = 0;
  for (size_t planIndex = 0; planIndex < numPlan; ++planIndex) {
    if (!multi_contact::isContactActive(mask, static_cast<int>(planIndex))) {
      continue;
    }
    const std::optional<size_t> mpcIndex = mpcIndexByPlanIndex_[planIndex];
    if (!mpcIndex.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[PlanContactOrder] the plan's contact '",
          planContactNames_[planIndex],
          "' is active, but the MPC has no contact of that name; list it in "
          "the task file's model_settings (contacts_3dof or "
          "contact_names_6dof)."));
    }
    mode |= contactModeBit(*mpcIndex, numMpcContacts());
  }
  return mode;
}

absl::StatusOr<multi_contact::ContactMask> PlanContactOrder::maskOf(
    size_t mode) const {
  const size_t numMpc = numMpcContacts();
  if ((mode >> numMpc) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[PlanContactOrder] the mode ", mode,
                     " sets a bit beyond the ", numMpc, " MPC contacts."));
  }
  multi_contact::ContactMask mask = 0;
  for (size_t mpcIndex = 0; mpcIndex < numMpc; ++mpcIndex) {
    if (!contactIsActiveInMode(mode, mpcIndex, numMpc)) continue;
    const std::optional<size_t> planIndex = planIndexByMpcIndex_[mpcIndex];
    if (!planIndex.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[PlanContactOrder] the MPC contact '", mpcContactNames_[mpcIndex],
          "' is active, but the plan's contact set has no candidate of that "
          "name."));
    }
    mask |= multi_contact::contactBit(static_cast<int>(*planIndex));
  }
  return mask;
}

}  // namespace ocs2::humanoid
