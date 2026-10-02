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

#include "humanoid_common_mpc_app/robot/RemoteMpcLinkAdapter.h"

#include <utility>

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<RemoteMpcLinkAdapter>> RemoteMpcLinkAdapter::Create(robot::ipc::Bus& bus,
                                                                                   ipc::RemoteMpcLink::Config config,
                                                                                   MpcResetSupervisor::Config supervisorConfig) {
  std::unique_ptr<RemoteMpcLinkAdapter> adapter(new RemoteMpcLinkAdapter(supervisorConfig));
  absl::StatusOr<std::unique_ptr<ipc::RemoteMpcLink>> link = ipc::RemoteMpcLink::Create(bus, adapter->resetSupervisor(), std::move(config));
  if (!link.ok()) {
    return link.status();
  }
  adapter->link_ = *std::move(link);
  return adapter;
}

RemoteMpcLinkAdapter::RemoteMpcLinkAdapter(MpcResetSupervisor::Config supervisorConfig) : MpcLink(supervisorConfig) {}

// The remote link goes first (members are destroyed before the base class): its destructor detaches the bus's
// callbacks, which reach the supervisor the base class owns.
RemoteMpcLinkAdapter::~RemoteMpcLinkAdapter() {
  link_.reset();
}

void RemoteMpcLinkAdapter::start(const SystemObservation& initialObservation) {
  link_->setCurrentObservation(initialObservation);
}

MpcLinkFactory handOverMpcLink(std::unique_ptr<MpcLink> link) {
  // std::function needs a copyable callable, and the link is handed over once: a shared slot holds it until then.
  std::shared_ptr<std::unique_ptr<MpcLink>> slot = std::make_shared<std::unique_ptr<MpcLink>>(std::move(link));
  return [slot](MpcLink::ResetTargetFunction /*resetTarget*/) -> std::unique_ptr<MpcLink> { return std::move(*slot); };
}

}  // namespace ocs2::humanoid
