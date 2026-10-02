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

#include <memory>

#include "absl/status/statusor.h"

#include <ocs2_mpc/MRT_BASE.h>
#include <ocs2_mpc/SystemObservation.h>

#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid {

/**
 * The MpcLink of a controller whose MPC is the MPC node (MpcServer) on the bus: an ipc::RemoteMpcLink, which serves
 * this link's MpcResetSupervisor, as the controllers' MpcLink interface (humanoid_common_mpc/mrt/MpcLink.h). The
 * controller calls setCurrentObservation(), updatePolicy() and the predicates on it as on InProcessMpcLink; the
 * observation goes into a triple buffer that the bus's IO thread publishes on robot/mpc_observation, and policies come
 * in on mpc/policy. Its isHealthy() is the supervisor's, which the remote link feeds with the MPC node's health and the
 * policy timeout (link loss), so the controller holds JOINT_PD on link loss as for a failing solver.
 *
 * The link registers on a bus that is not running yet; the robot process starts the bus once everything is registered.
 * start() hands the first observation to the link (the MPC node serves its first reset as a full one by itself). stop()
 * has no thread to join - the IO thread is the bus's - and the callbacks end with the link, whose destructor detaches
 * them from the bus.
 */
class RemoteMpcLinkAdapter final : public MpcLink {
 public:
  /**
   * A link on `bus` (not running yet), which must outlive it. `config` is ipc::RemoteMpcLink's: the dimensions of the
   * controller's model (the effective input dimension under basis-vector contact inputs) and mpcLink.policyTimeout.
   */
  static absl::StatusOr<std::unique_ptr<RemoteMpcLinkAdapter>> Create(robot::ipc::Bus& bus,
                                                                      ipc::RemoteMpcLink::Config config,
                                                                      MpcResetSupervisor::Config supervisorConfig = {});

  ~RemoteMpcLinkAdapter() override;

  void start(const SystemObservation& initialObservation) override;
  void stop() override {}

  MRT_BASE& mrt() override { return *link_; }
  const MRT_BASE& mrt() const override { return *link_; }

  /** The remote link: its statistics (LoopTiming's stale policies and policy age) and the viewer annotations. */
  ipc::RemoteMpcLink& remote() { return *link_; }
  const ipc::RemoteMpcLink& remote() const { return *link_; }

 private:
  explicit RemoteMpcLinkAdapter(MpcResetSupervisor::Config supervisorConfig);

  std::unique_ptr<ipc::RemoteMpcLink> link_;
};

/**
 * An MpcLinkFactory that hands the controller `link`, made before the controller: the controller calls its factory once,
 * from its constructor. The reset target the controller passes is not used (a remote link's MPC node resets to its own).
 * A second call returns nullptr.
 */
MpcLinkFactory handOverMpcLink(std::unique_ptr<MpcLink> link);

}  // namespace ocs2::humanoid
