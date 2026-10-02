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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <robot_model/ContactEstimator.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "robot_ipc/Bus.h"

/** What the tests of the robot process share: the Atlas files, headless simulators, loopback buses and a controller. */
namespace ocs2::humanoid::robot_test {

inline constexpr const char* kAtlasScene = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
inline constexpr const char* kAtlasUrdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
inline constexpr const char* kAtlasTask = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml";
inline constexpr const char* kAtlasGains = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.yaml";

/** A headless simulator on the Atlas scene, held by the gantry (weld) or not. */
std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface> makeHeadlessAtlas(bool gantryLocked);

/** A bus publishing as `nodeName` on an ephemeral loopback port, alone in its network. */
std::unique_ptr<robot::ipc::Bus> createLoopbackBus(const std::string& nodeName);

/** Connects each bus's subscriber to the other's publisher. Before either starts. */
void connectBoth(robot::ipc::Bus& first, robot::ipc::Bus& second);

/** Polls `condition` every millisecond until it holds or `timeout` passes; whether it held. */
bool waitFor(const std::function<bool()>& condition, absl::Duration timeout = absl::Seconds(10));

/**
 * A RobotController that plans nothing: every active mode commands a joint PD to the posture it is handed (gravity is
 * carried by the gantry in the tests), ZERO_TORQUE no gain and no torque as the MRT controllers do, and it records what
 * the robot process hands it. The realtime-thread methods are called by the loop only; the getters below are for the
 * test thread.
 */
class ScriptedRobotController final : public RobotController {
 public:
  void prepareCycle(absl::string_view controlMode, const std::vector<scalar_t>& nominalJointPositions) override;
  void computeJointControlAction(const robot::model::RobotState& robotState, robot::model::RobotJointAction& jointAction) override;
  std::optional<contact_flag_t> plannedContactFlags() const override;
  const contact_flag_t& measuredContactFlags() const override { return measuredContactFlags_; }
  void requestMpcReset() override { resets_.fetch_add(1); }
  void requestMpcResetAndHold() override { resetsAndHolds_.fetch_add(1); }
  bool isMpcHealthy() const override { return healthy_.load(); }
  void setContactEstimator(std::shared_ptr<robot::model::ContactEstimator> contactEstimator) override;
  const ContactWrenchGate::Config& contactWrenchGateConfig() const override { return gate_; }
  void setContactWrenchGateConfig(const ContactWrenchGate::Config& config) override;
  absl::Status setPdGainsYaml(absl::string_view yamlText) override;
  void pollPdGainsFile() override { pdGainsPolls_.fetch_add(1); }
  void setEventSink(ControllerEventSink* eventSink) override { eventSink_.store(eventSink); }
  void startMpc(const robot::model::RobotState& /*initialState*/) override { started_.store(true); }
  bool policyReady() override { return true; }

  // ---- The test thread.
  void setHealthy(bool healthy) { healthy_.store(healthy); }
  /** The next computeJointControlAction() throws std::runtime_error(`message`). */
  void throwInNextCycle(const std::string& message);
  ControllerEventSink* eventSink() const { return eventSink_.load(); }
  std::string mode() const;
  std::string contactEstimatorName() const;
  ContactWrenchGate::Config gate() const;
  std::uint64_t cycles() const { return cycles_.load(); }
  std::uint64_t resets() const { return resets_.load(); }
  std::uint64_t resetsAndHolds() const { return resetsAndHolds_.load(); }
  std::uint64_t pdGainsDocuments() const { return pdGainsDocuments_.load(); }
  std::uint64_t pdGainsPolls() const { return pdGainsPolls_.load(); }
  bool started() const { return started_.load(); }
  /** The posture handed over with the newest cycle. */
  std::vector<scalar_t> nominal() const;

 private:
  mutable absl::Mutex mutex_;
  std::string mode_ ABSL_GUARDED_BY(mutex_);
  std::string contactEstimatorName_ ABSL_GUARDED_BY(mutex_);
  std::vector<scalar_t> nominal_ ABSL_GUARDED_BY(mutex_);
  ContactWrenchGate::Config gate_;
  std::shared_ptr<robot::model::ContactEstimator> contactEstimator_;
  contact_flag_t measuredContactFlags_{};
  std::vector<bool> estimatedFlags_;
  std::atomic<bool> healthy_{true};
  std::atomic<bool> started_{false};
  std::atomic<ControllerEventSink*> eventSink_{nullptr};
  std::string throwMessage_;  // written before throwInNextCycle_ is set
  std::atomic<bool> throwInNextCycle_{false};
  std::atomic<std::uint64_t> cycles_{0};
  std::atomic<std::uint64_t> resets_{0};
  std::atomic<std::uint64_t> resetsAndHolds_{0};
  std::atomic<std::uint64_t> pdGainsDocuments_{0};
  std::atomic<std::uint64_t> pdGainsPolls_{0};
};

}  // namespace ocs2::humanoid::robot_test
