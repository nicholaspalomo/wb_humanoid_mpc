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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/mrt/ControllerEvent.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * The measured contact state an MRT joint controller takes from its contact estimator every control cycle
 * (robot_model/ContactEstimator.h), shared by CentroidalMpcMrtJointController and WBMpcMrtJointController.
 *
 * An estimate is taken when it carries one flag per contact point of the controller (kNumContacts). Any other estimate
 * is a configuration error of the estimator: it is refused, the measured flags stay those of the cycle before, and the
 * refusal is counted. The first refusal after an estimate was taken, or after the estimator was replaced, posts
 * ControllerEventCode::kContactEstimateRefused to the controller's event sink, so that a misconfigured estimator is
 * reported once rather than on every cycle; the robot process's communication thread logs it as a warning. The estimators
 * the robot process and the closed-loop driver install are checked before that (checkContactEstimator()), so a refusal
 * means an estimator whose answer changed while it ran.
 *
 * Not thread-safe: the control thread calls resetEstimator() and take(). numRefused() may be read from any thread.
 */
class ContactEstimateIntake {
 public:
  /** `controller` names the controller in the refusal report: a string literal, which must outlive the intake. */
  explicit ContactEstimateIntake(const char* absl_nonnull controller);

  ContactEstimateIntake(const ContactEstimateIntake&) = delete;
  ContactEstimateIntake& operator=(const ContactEstimateIntake&) = delete;
  ~ContactEstimateIntake() = default;

  /**
   * A new estimator, called `estimatorName` in the report of its first refused estimate (cut to the event's text). Copies
   * the name into the intake's own buffer and allocates nothing.
   */
  void resetEstimator(absl::string_view estimatorName);

  /**
   * Takes `estimated` into `measured` when it holds kNumContacts flags and returns true. Otherwise leaves `measured` as it
   * is, counts the refusal, posts kContactEstimateRefused to `sink` when it is the first refusal since an estimate was
   * taken or since resetEstimator(), and returns false. Allocates nothing.
   */
  bool take(const std::vector<bool>& estimated, contact_flag_t& measured, ControllerEventSink& sink);

  /** The estimates refused so far. Any thread. */
  uint64_t numRefused() const { return numRefused_.load(); }

 private:
  /** The report of a refusal, prepared by resetEstimator(): take() fills in the number of flags and posts it. */
  ControllerEvent refusal_;
  /** The current run of refused estimates has been reported. */
  bool refusalReported_ = false;
  std::atomic<uint64_t> numRefused_{0};
};

/**
 * OK when `estimator` writes one flag per contact point of the MRT joint controllers (kNumContacts) for `robotState`;
 * InvalidArgument naming `estimatorName` and both counts otherwise. For the thread that installs an estimator, before
 * the control thread uses it: it asks the estimator once, which allocates.
 */
absl::Status checkContactEstimator(robot::model::ContactEstimator& estimator,
                                   const robot::model::RobotState& robotState,
                                   absl::string_view estimatorName);

}  // namespace ocs2::humanoid
