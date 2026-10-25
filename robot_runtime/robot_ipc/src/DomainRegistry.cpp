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

#include "robot_runtime/robot_ipc/src/DomainRegistry.h"

#include <string>

#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace robot::ipc {

absl::Status DomainRegistry::acquire(int domainId, const std::string& config) {
  absl::MutexLock lock(mutex_);
  const absl::flat_hash_map<int, Domain>::iterator it = domains_.find(domainId);
  if (it != domains_.end()) {
    if (it->second.config != config) {
      return absl::FailedPreconditionError(absl::StrCat(
          "robot_ipc: DDS domain ", domainId,
          " is configured differently by another bus of this process (a bus of "
          "another node or network); use one network file and node per "
          "process"));
    }
    ++it->second.users;
    return absl::OkStatus();
  }
  const dds_entity_t domain =
      dds_create_domain(static_cast<dds_domainid_t>(domainId), config.c_str());
  if (domain < 0) {
    return absl::InternalError(
        absl::StrCat("robot_ipc: cannot create DDS domain ", domainId, ": ",
                     dds_strretcode(domain), " (configuration: ", config, ")"));
  }
  domains_.emplace(domainId,
                   Domain{.entity = domain, .config = config, .users = 1});
  return absl::OkStatus();
}

void DomainRegistry::release(int domainId) {
  absl::MutexLock lock(mutex_);
  const absl::flat_hash_map<int, Domain>::iterator it = domains_.find(domainId);
  if (it == domains_.end()) {
    return;
  }
  if (--it->second.users > 0) {
    return;
  }
  const dds_return_t deleted = dds_delete(it->second.entity);
  if (deleted < 0) {
    LOG(ERROR) << "robot_ipc: deleting DDS domain " << domainId
               << " failed: " << dds_strretcode(deleted);
  }
  domains_.erase(it);
}

DomainRegistry& domainRegistry() {
  static absl::NoDestructor<DomainRegistry> registry;
  return *registry;
}

}  // namespace robot::ipc
