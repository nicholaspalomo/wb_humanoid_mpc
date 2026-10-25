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

// The DDS domains of this process, which every Bus of a domain shares. Private
// to the library: not in include/.

#pragma once

#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "dds/dds.h"

namespace robot::ipc {

/**
 * The Cyclone DDS domains of this process, with the configuration each was
 * created with. A process configures a domain once (dds_create_domain() refuses
 * a second time), so the buses of one domain share it: the first bus creates
 * it, the last one to release it deletes it, and a later bus may then create it
 * again, with another configuration. Thread-safe.
 */
class DomainRegistry {
 public:
  DomainRegistry() = default;
  ~DomainRegistry() = default;
  DomainRegistry(const DomainRegistry&) = delete;
  DomainRegistry& operator=(const DomainRegistry&) = delete;
  DomainRegistry(DomainRegistry&&) = delete;
  DomainRegistry& operator=(DomainRegistry&&) = delete;

  /**
   * Creates domain `domainId` with `config` (Cyclone DDS XML), or counts one
   * more user of it when it exists with the same configuration. One that exists
   * with another configuration is FailedPrecondition: two buses of one process
   * would disagree about the network. A DDS failure (an invalid configuration,
   * an interface that does not exist) is Internal.
   */
  absl::Status acquire(int domainId, const std::string& config);

  /** Releases one use of `domainId` from acquire(); the last release deletes
   * the domain. */
  void release(int domainId);

 private:
  struct Domain {
    dds_entity_t entity = 0;
    std::string config;
    int users = 0;
  };

  absl::Mutex mutex_;
  absl::flat_hash_map<int, Domain> domains_ ABSL_GUARDED_BY(mutex_);
};

/** The registry of this process. */
DomainRegistry& domainRegistry();

}  // namespace robot::ipc
