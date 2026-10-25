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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/message.h"

#include "humanoid_mpc_msgs/telemetry_series.pb.h"
#include "humanoid_mpc_msgs/visualization_scene.pb.h"
#include "robot_ipc/Bus.h"
#include "robot_realtime/SpscQueue.h"

namespace ocs2::humanoid::visualization {

/**
 * Hands the visualization thread's messages (viz/telemetry, viz/scene) to the
 * bus's IO thread, which publishes them with Bus::publishFromIoThread(). The
 * visualization thread runs under SCHED_IDLE; Bus::publish() would take the
 * bus's queue and counter mutexes, which the solver thread's own Bus::publish()
 * takes too, and a SCHED_IDLE thread preempted while it holds one holds up the
 * solver until a core is idle. Through the outbox it takes no lock: a lock-free
 * queue of preallocated messages, drained by a periodic callback of the bus.
 *
 * Usage, before bus.start():
 *
 *   ASSIGN_OR_RETURN(std::shared_ptr<IoThreadOutbox> outbox,
 * IoThreadOutbox::Attach(bus, period, capacity)); outbox->post(topic, message);
 * // the visualization thread
 *
 * THREADS. post() on one thread at a time (the visualization thread); drain()
 * on the bus's IO thread only (Attach() registers it). publishFailures() on any
 * thread. Shared: the bus's callback and the poster each hold it, so it lives
 * as long as the longer of the two (the reason for std::shared_ptr:
 * Bus::addPeriodicCallback() keeps a copyable std::function).
 */
class IoThreadOutbox {
 public:
  /** How many messages may wait for the IO thread: a few polls' worth of the
   * visualization's messages. */
  static constexpr size_t kDefaultCapacity = 32;

  /**
   * An outbox of `capacity` messages that `bus` drains every `period` on its IO
   * thread. `bus` must not be running, and must outlive every post() that is to
   * be published (the callback runs only while the bus does).
   *
   * @return the errors of Bus::addPeriodicCallback() (a running bus, a
   * non-positive period); InvalidArgument for a zero capacity.
   */
  static absl::StatusOr<std::shared_ptr<IoThreadOutbox>> Attach(
      robot::ipc::Bus& bus, absl::Duration period, size_t capacity);

  IoThreadOutbox(const IoThreadOutbox&) = delete;
  IoThreadOutbox& operator=(const IoThreadOutbox&) = delete;
  ~IoThreadOutbox() = default;

  /**
   * Copies `message` into the queue for `topic`; the copy reuses its slot's
   * storage, so it allocates only when a message outgrows what the slot held.
   * ResourceExhausted when the queue is full (the message is dropped);
   * InvalidArgument for a message that is neither a TelemetrySeries nor a
   * VisualizationScene.
   */
  absl::Status post(absl::string_view topic,
                    const google::protobuf::Message& message);

  /**
   * Publishes the queued messages on the bus, oldest first, at most a queue's
   * worth; IO thread only. A message the bus refuses is counted and logged.
   */
  void drain();

  /** Messages the bus refused in drain(). */
  uint64_t publishFailures() const {
    return publishFailures_.load(std::memory_order_relaxed);
  }

 private:
  /** One queued message: `topic` and the message of `kind`. */
  struct Slot {
    enum class Kind { kTelemetry, kScene };
    std::string topic;
    Kind kind = Kind::kTelemetry;
    humanoid_mpc_msgs::TelemetrySeries telemetry;
    humanoid_mpc_msgs::VisualizationScene scene;
  };

  IoThreadOutbox(robot::ipc::Bus* absl_nonnull bus, size_t capacity);

  /** Publishes one queued message from the IO thread. */
  void publish(const Slot& slot);

  // The bus whose IO thread calls drain(); it outlives every drain(), which
  // only it calls.
  robot::ipc::Bus* absl_nonnull bus_;
  robot::realtime::SpscQueue<Slot> queue_;
  std::atomic<uint64_t> publishFailures_{0};
};

}  // namespace ocs2::humanoid::visualization
