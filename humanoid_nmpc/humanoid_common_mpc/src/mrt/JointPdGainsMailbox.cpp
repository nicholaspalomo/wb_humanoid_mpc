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

#include "humanoid_common_mpc/mrt/JointPdGainsMailbox.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

namespace ocs2::humanoid {

JointPdGainsMailbox::JointPdGainsMailbox(const JointPdGains& initial)
    : numMpcJoints_(static_cast<size_t>(initial.mpcJointKp.size())),
      numOtherJoints_(static_cast<size_t>(initial.otherJointKp.size())),
      buffer_(initial) {}

absl::Status JointPdGainsMailbox::post(const JointPdGains& gains, uint64_t ticket) {
  if (!gains.hasDimensions(numMpcJoints_, numOtherJoints_)) {
    return absl::InvalidArgumentError(absl::StrCat("[JointPdGainsMailbox] the gains are not sized for ", numMpcJoints_, " MPC joints and ",
                                                   numOtherJoints_, " other joints."));
  }
  absl::MutexLock lock(&producerMutex_);
  if (ticket <= lastPostedTicket_) {
    // Handed in before gains that are already posted: those superseded these.
    return absl::OkStatus();
  }
  lastPostedTicket_ = ticket;
  buffer_.writeSlot() = gains;
  buffer_.publishWrite();
  return absl::OkStatus();
}

bool JointPdGainsMailbox::receive(JointPdGains& active) {
  if (!buffer_.acquireRead()) {
    return false;
  }
  active = buffer_.readSlot();
  return true;
}

}  // namespace ocs2::humanoid
