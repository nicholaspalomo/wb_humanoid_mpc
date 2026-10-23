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


#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"

#include <ocs2_core/control/ControllerBase.h>

#include "ocs2_mpc/MRT_BASE.h"

/*
 * The policy buffer of MRT_BASE as the control thread sees it: updatePolicy() never waits for the buffer and never logs
 * (a failed try-lock is counted, for the controllers to report off the realtime thread), and the solver side never
 * holds the buffer's lock while it destroys a policy, which frees its trajectories and controller.
 */

namespace ocs2 {
namespace {

/** A controller that calls `onDestroy` from its destructor: to see whether a policy is destroyed under the lock. */
class DestructionProbeController final : public ControllerBase {
 public:
  explicit DestructionProbeController(std::function<void()> onDestroy) : onDestroy_(std::move(onDestroy)) {}
  ~DestructionProbeController() override {
    if (onDestroy_) onDestroy_();
  }
  DestructionProbeController(const DestructionProbeController&) = delete;
  DestructionProbeController& operator=(const DestructionProbeController&) = delete;

  vector_t computeInput(scalar_t /*t*/, const vector_t& /*x*/) override { return vector_t(); }
  int size() const override { return 0; }
  ControllerType getType() const override { return ControllerType::FEEDFORWARD; }
  void clear() override {}
  bool empty() const override { return true; }
  DestructionProbeController* absl_nonnull clone() const override { return new DestructionProbeController(onDestroy_); }

 private:
  std::function<void()> onDestroy_;
};

/** An MRT the test fills directly, and whose buffer lock it can hold. */
class ProbeMrt final : public MRT_BASE {
 public:
  void resetMpcNode(const TargetTrajectories& /*initTargetTrajectories*/) override {}
  void setCurrentObservation(const SystemObservation& /*observation*/) override {}

  /** Moves a policy whose controller calls `onDestroy` when it is destroyed to the buffer. */
  void handOver(std::function<void()> onDestroy) {
    std::unique_ptr<PrimalSolution> policy = std::make_unique<PrimalSolution>();
    policy->controllerPtr_ = std::make_unique<DestructionProbeController>(std::move(onDestroy));
    moveToBuffer(std::make_unique<CommandData>(), std::move(policy), std::make_unique<PerformanceIndex>());
  }

  std::unique_lock<std::mutex> holdBuffer() const { return lockPolicyBuffer(); }
};

/** Whether updatePolicy(), called on another thread (the control thread's role), takes the buffer's lock now. */
bool controlThreadCanTakeTheBuffer(ProbeMrt& mrt) {
  const size_t before = mrt.getFailedPolicyUpdateLocks();
  std::thread controlThread([&mrt] { mrt.updatePolicy(); });
  controlThread.join();
  return mrt.getFailedPolicyUpdateLocks() == before;
}

TEST(MrtPolicyBuffer, AFailedTryLockIsCountedAndNotLogged) {
  ProbeMrt mrt;
  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();
  {
    const std::unique_lock<std::mutex> held = mrt.holdBuffer();
    constexpr size_t kFailedCycles = 2 * MRT_BASE::kPolicyUpdateLockContentionThreshold;
    std::thread controlThread([&mrt] {
      for (size_t cycle = 0; cycle < kFailedCycles; ++cycle) EXPECT_FALSE(mrt.updatePolicy());
    });
    controlThread.join();
    EXPECT_EQ(mrt.getConsecutiveFailedPolicyUpdateLocks(), kFailedCycles);
    EXPECT_EQ(mrt.getFailedPolicyUpdateLocks(), kFailedCycles);
  }
  // The next successful try-lock ends the run; the lifetime count stays.
  std::thread controlThread([&mrt] { mrt.updatePolicy(); });
  controlThread.join();
  EXPECT_EQ(mrt.getConsecutiveFailedPolicyUpdateLocks(), 0u);
  EXPECT_EQ(mrt.getFailedPolicyUpdateLocks(), 2 * MRT_BASE::kPolicyUpdateLockContentionThreshold);
}

TEST(MrtPolicyBuffer, DiscardingTheBufferedPolicyDestroysItOutsideTheLock) {
  ProbeMrt mrt;
  std::atomic<int> destroyed{0};
  std::atomic<bool> lockFreeWhileDestroyed{false};
  mrt.handOver([&mrt, &destroyed, &lockFreeWhileDestroyed] {
    lockFreeWhileDestroyed.store(controlThreadCanTakeTheBuffer(mrt));
    destroyed.fetch_add(1);
  });
  EXPECT_TRUE(mrt.discardBufferedPolicy());
  EXPECT_EQ(destroyed.load(), 1);
  EXPECT_TRUE(lockFreeWhileDestroyed.load()) << "the policy was destroyed while the buffer was locked";
}

TEST(MrtPolicyBuffer, AResetDestroysThePoliciesOutsideTheLock) {
  ProbeMrt mrt;
  std::atomic<int> destroyed{0};
  std::atomic<int> destroyedUnderLock{0};
  const std::function<void()> probe = [&mrt, &destroyed, &destroyedUnderLock] {
    if (!controlThreadCanTakeTheBuffer(mrt)) destroyedUnderLock.fetch_add(1);
    destroyed.fetch_add(1);
  };
  // One policy in use, one in the buffer.
  mrt.handOver(probe);
  ASSERT_TRUE(mrt.updatePolicy());
  mrt.handOver(probe);
  mrt.reset();
  EXPECT_EQ(destroyed.load(), 2);
  EXPECT_EQ(destroyedUnderLock.load(), 0);
  EXPECT_FALSE(mrt.initialPolicyReceived());
}

}  // namespace
}  // namespace ocs2
