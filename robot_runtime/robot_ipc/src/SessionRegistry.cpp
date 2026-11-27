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

#include "robot_runtime/robot_ipc/src/SessionRegistry.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "zenoh.h"  // NOLINT(build/include_subdir): zenoh-c installs its headers in no directory

#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/SessionConfig.h"
#include "robot_runtime/robot_ipc/src/StatusMacros.h"

namespace robot::ipc {
namespace {

// Zenoh's own log: errors only, unless RUST_LOG says otherwise.
constexpr char kZenohLogFilter[] = "error";
// The listen port of the configuration that buses sharing a session compare.
constexpr int kSharedConfigPort = 0;

void initializeZenohLog() {
  static absl::once_flag once;
  absl::call_once(once, [] {
    if (zc_init_log_from_env_or(kZenohLogFilter) != Z_OK) {
      LOG(WARNING) << "robot_ipc: cannot set up Zenoh's log";
    }
  });
}

// Opens a session of `config`; false when Zenoh refuses it (a taken port).
bool openSession(absl::string_view config, z_owned_session_t& session) {
  // zc_config_from_str() needs a NUL-terminated text.
  const std::string configText(config);
  z_owned_config_t zenohConfig;
  if (zc_config_from_str(&zenohConfig, configText.c_str()) != Z_OK) {
    return false;
  }
  return z_open(&session, z_config_move(&zenohConfig),
                /*_options=*/nullptr) == Z_OK;
}

}  // namespace

absl::StatusOr<SessionLease> SessionRegistry::acquire(
    const NetworkConfig& network, absl::string_view nodeName) {
  absl::StatusOr<std::string> sharedConfig =
      zenohSessionConfig(network, nodeName, kSharedConfigPort);
  if (!sharedConfig.ok()) {
    return sharedConfig.status();
  }
  absl::MutexLock lock(mutex_);
  const absl::flat_hash_map<int, std::unique_ptr<Session>>::iterator it =
      sessions_.find(network.domainId);
  if (it != sessions_.end()) {
    if (it->second->sharedConfig != *sharedConfig) {
      return absl::FailedPreconditionError(absl::StrCat(
          "robot_ipc: domain ", network.domainId,
          " is configured differently by another bus of this process (a bus of "
          "another node or network); use one network file and node per "
          "process"));
    }
    ++it->second->users;
    return it->second->lease;
  }
  {
    z_owned_config_t probe;
    if (zc_config_from_str(&probe, sharedConfig->c_str()) != Z_OK) {
      return absl::InternalError(absl::StrCat(
          "robot_ipc: Zenoh refuses the configuration ", *sharedConfig));
    }
    z_config_drop(z_config_move(&probe));
  }
  initializeZenohLog();
  ROBOT_IPC_ASSIGN_OR_RETURN(const std::string listenAddress,
                             interfaceAddressFor(network, nodeName));
  const PortRange ports = tcpPortRange(network);
  for (int port = ports.first; port <= ports.last; ++port) {
    // A taken port is skipped before Zenoh tries it, which would log an error.
    if (!isTcpPortFree(listenAddress, port)) {
      continue;
    }
    absl::StatusOr<std::string> config =
        zenohSessionConfig(network, nodeName, port);
    if (!config.ok()) {
      return config.status();
    }
    std::unique_ptr<Session> entry = std::make_unique<Session>();
    if (!openSession(*config, entry->session)) {
      continue;
    }
    entry->sharedConfig = *std::move(sharedConfig);
    entry->lease = SessionLease{.session = z_session_loan(&entry->session),
                                .config = *std::move(config),
                                .listenPort = port};
    entry->users = 1;
    SessionLease lease = entry->lease;
    sessions_.emplace(network.domainId, std::move(entry));
    return lease;
  }
  return absl::UnavailableError(absl::StrCat(
      "robot_ipc: no Zenoh session of domain ", network.domainId,
      " could listen on any TCP port of ", ports.first, "-", ports.last,
      ": every port is taken on this machine (stop a process of the domain, or "
      "raise transport.max_processes_per_host), or the listen address is not "
      "this machine's"));
}

void SessionRegistry::release(int domainId) {
  absl::MutexLock lock(mutex_);
  const absl::flat_hash_map<int, std::unique_ptr<Session>>::iterator it =
      sessions_.find(domainId);
  if (it == sessions_.end()) {
    return;
  }
  if (--it->second->users > 0) {
    return;
  }
  if (z_close(z_session_loan_mut(&it->second->session),
              /*options=*/nullptr) != Z_OK) {
    LOG(ERROR) << "robot_ipc: closing the Zenoh session of domain " << domainId
               << " failed";
  }
  z_session_drop(z_session_move(&it->second->session));
  sessions_.erase(it);
}

SessionRegistry& sessionRegistry() {
  static absl::NoDestructor<SessionRegistry> registry;
  return *registry;
}

}  // namespace robot::ipc
