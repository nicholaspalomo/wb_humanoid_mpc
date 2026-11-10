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

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "ocs2_mpc/SystemObservation.h"

#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid {
namespace {

/**
 * An MpcLinkFactory that hands over the link it owns on the first call and
 * returns nullptr after that. Not thread-safe; move-only.
 */
class LinkHandOver {
 public:
  explicit LinkHandOver(std::unique_ptr<MpcLink> link)
      : link_(std::move(link)) {}

  LinkHandOver(const LinkHandOver&) = delete;
  LinkHandOver& operator=(const LinkHandOver&) = delete;
  LinkHandOver(LinkHandOver&&) noexcept = default;
  LinkHandOver& operator=(LinkHandOver&&) noexcept = default;
  ~LinkHandOver() = default;

  // MpcLinkFactory calls it through a const reference, hence the mutable link.
  std::unique_ptr<MpcLink> operator()(
      MpcLink::ResetTargetFunction /*resetTarget*/) const {
    return std::move(link_);
  }

 private:
  /** The link until the first call; nullptr after it. */
  mutable std::unique_ptr<MpcLink> link_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<RemoteMpcLinkAdapter>>
RemoteMpcLinkAdapter::Create(robot::ipc::Bus* absl_nonnull bus,
                             ipc::RemoteMpcLink::Config config,
                             MpcResetSupervisor::Config supervisorConfig) {
  std::unique_ptr<RemoteMpcLinkAdapter> adapter =
      absl::WrapUnique(new RemoteMpcLinkAdapter(supervisorConfig));
  absl::StatusOr<std::unique_ptr<ipc::RemoteMpcLink>> link =
      ipc::RemoteMpcLink::Create(bus, &adapter->resetSupervisor(),
                                 std::move(config));
  if (!link.ok()) {
    return link.status();
  }
  adapter->link_ = *std::move(link);
  return adapter;
}

RemoteMpcLinkAdapter::RemoteMpcLinkAdapter(
    MpcResetSupervisor::Config supervisorConfig)
    : MpcLink(supervisorConfig) {}

// The remote link goes first (members are destroyed before the base class): its
// destructor detaches the bus's callbacks, which reach the supervisor the base
// class owns.
RemoteMpcLinkAdapter::~RemoteMpcLinkAdapter() {
  link_.reset();
}

void RemoteMpcLinkAdapter::start(const SystemObservation& initialObservation) {
  link_->setCurrentObservation(initialObservation);
}

MpcLinkFactory handOverMpcLink(std::unique_ptr<MpcLink> link) {
  return LinkHandOver(std::move(link));
}

}  // namespace ocs2::humanoid
