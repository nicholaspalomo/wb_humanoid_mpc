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

#include "humanoid_common_mpc_app/visualization/IoThreadOutbox.h"

#include <cstddef>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/message.h"

#include "humanoid_mpc_msgs/telemetry_series.pb.h"
#include "humanoid_mpc_msgs/visualization_scene.pb.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid::visualization {
namespace {

/** [s] How often a recurring publish failure is logged at most. */
constexpr double kLogPeriodSeconds = 5.0;

}  // namespace

absl::StatusOr<std::shared_ptr<IoThreadOutbox>> IoThreadOutbox::Attach(robot::ipc::Bus& bus, absl::Duration period, size_t capacity) {
  if (capacity == 0) {
    return absl::InvalidArgumentError("the visualization's outbox must hold at least one message.");
  }
  std::shared_ptr<IoThreadOutbox> outbox = absl::WrapUnique(new IoThreadOutbox(&bus, capacity));
  // The callback keeps the outbox: it runs on the IO thread until the bus stops, whatever the poster does meanwhile.
  if (absl::Status added = bus.addPeriodicCallback(period, [outbox] { outbox->drain(); }); !added.ok()) {
    return added;
  }
  return outbox;
}

IoThreadOutbox::IoThreadOutbox(robot::ipc::Bus* absl_nonnull bus, size_t capacity) : bus_(bus), queue_(capacity) {}

absl::Status IoThreadOutbox::post(absl::string_view topic, const google::protobuf::Message& message) {
  const google::protobuf::Descriptor* absl_nonnull type = message.GetDescriptor();
  const bool isTelemetry = type == humanoid_mpc_msgs::TelemetrySeries::descriptor();
  if (!isTelemetry && type != humanoid_mpc_msgs::VisualizationScene::descriptor()) {
    return absl::InvalidArgumentError(
        absl::StrCat("the visualization's outbox carries TelemetrySeries and VisualizationScene, not ", type->full_name()));
  }
  const bool queued = queue_.tryPushInPlace([&](Slot& slot) {
    slot.topic.assign(topic.data(), topic.size());
    if (isTelemetry) {
      slot.kind = Slot::Kind::kTelemetry;
      slot.telemetry = static_cast<const humanoid_mpc_msgs::TelemetrySeries&>(message);
    } else {
      slot.kind = Slot::Kind::kScene;
      slot.scene = static_cast<const humanoid_mpc_msgs::VisualizationScene&>(message);
    }
  });
  if (!queued) {
    return absl::ResourceExhaustedError(absl::StrCat("the visualization's outbox to the bus's IO thread is full (", queue_.capacity(),
                                                     " messages); dropped one on '", topic, "'"));
  }
  return absl::OkStatus();
}

void IoThreadOutbox::drain() {
  // At most a queue's worth per call, so that a poster that keeps up cannot keep the IO thread here.
  for (size_t count = 0; count < queue_.capacity(); ++count) {
    const bool popped = queue_.tryPopInPlace([this](const Slot& slot) { publish(slot); });
    if (!popped) {
      return;
    }
  }
}

void IoThreadOutbox::publish(const Slot& slot) {
  const google::protobuf::Message& message =
      slot.kind == Slot::Kind::kTelemetry ? static_cast<const google::protobuf::Message&>(slot.telemetry) : slot.scene;
  if (const absl::Status published = bus_->publishFromIoThread(slot.topic, message); !published.ok()) {
    publishFailures_.fetch_add(1, std::memory_order_relaxed);
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds) << "[IoThreadOutbox] " << slot.topic << " was not published: " << published;
  }
}

}  // namespace ocs2::humanoid::visualization
