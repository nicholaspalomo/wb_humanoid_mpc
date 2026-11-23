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

// The Zenoh sessions of this process, which every Bus of a domain shares.
// Private to the library: not in include/.

#pragma once

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "zenoh.h"  // NOLINT(build/include_subdir): zenoh-c installs its headers in no directory

#include "robot_ipc/NetworkConfig.h"

namespace robot::ipc {

/** A share of a session from SessionRegistry::acquire(). */
struct SessionLease {
  /** Valid until the matching SessionRegistry::release(). */
  const z_loaned_session_t* absl_nonnull session;
  /** The configuration the session was opened with (SessionConfig.h). */
  std::string config;
  /** The TCP port the session listens on. */
  int listenPort = 0;
};

/**
 * The Zenoh sessions of this process, one per domain, with the configuration
 * each was opened with. The buses of one domain share its session, which
 * listens on one TCP port: the first bus opens it, on the lowest free port of
 * the domain (tcpPortRange()), and the last one to release it closes it; a
 * later bus may then open it again, with another configuration. Thread-safe.
 */
class SessionRegistry {
 public:
  SessionRegistry() = default;
  ~SessionRegistry() = default;
  SessionRegistry(const SessionRegistry&) = delete;
  SessionRegistry& operator=(const SessionRegistry&) = delete;
  SessionRegistry(SessionRegistry&&) = delete;
  SessionRegistry& operator=(SessionRegistry&&) = delete;

  /**
   * Opens the session of `network`'s domain for a bus of `nodeName`, or counts
   * one more user of it when it is open with the same configuration. One open
   * with another configuration is FailedPrecondition: two buses of one process
   * would disagree about the network. The errors of zenohSessionConfig()
   * (Unavailable for a node of another machine) pass through; a domain whose
   * every port is taken is Unavailable, and an invalid configuration is
   * Internal.
   */
  absl::StatusOr<SessionLease> acquire(const NetworkConfig& network,
                                       absl::string_view nodeName);

  /** Releases one use of `domainId` from acquire(); the last release closes
   * the session. */
  void release(int domainId);

 private:
  struct Session {
    z_owned_session_t session;
    // The configuration with the listen port left out, which the buses that
    // share the session must agree on.
    std::string sharedConfig;
    SessionLease lease;
    int users = 0;
  };

  absl::Mutex mutex_;
  absl::flat_hash_map<int, std::unique_ptr<Session>> sessions_
      ABSL_GUARDED_BY(mutex_);
};

/** The registry of this process. */
SessionRegistry& sessionRegistry();

}  // namespace robot::ipc
