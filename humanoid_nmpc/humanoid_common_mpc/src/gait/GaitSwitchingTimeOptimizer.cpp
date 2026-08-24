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

#include "humanoid_common_mpc/gait/GaitSwitchingTimeOptimizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <ocs2_core/misc/Lookup.h>
#include <ocs2_core/misc/Numerics.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace ocs2::humanoid {

namespace {

// Numerical threshold to prevent small precision flutter during gradient steps.
static constexpr scalar_t kWeakEpsilon = 1e-4;

// Safety clamp on the switching time sensitivity \partial J / \partial t_i to
// prevent erratic jumps.
static constexpr scalar_t kMaxSensitivityClamp = 50.0;

// Multiplicative factor 1/2 for quadratic input energy 1/2 ||u||^2.
static constexpr scalar_t kHalf = 0.5;

// Weight on CoM vertical velocity for switching time sensitivity.
// If CoM velocity is downward (v_z < 0), advancing touchdown helps decelerate
// falling early.
static constexpr scalar_t kComVelocityWeight = 10.0;

// Minimum number of points in discrete time trajectory to allow interior node
// lookup.
static constexpr size_t kMinTrajectorySizeForInteriorNodes = 3;

// Motion phase definitions matching MotionPhaseDefinition.h:
// Mode 3: Double Support (both feet in stance: stanceLegs = {true, true})
// Mode 1: Left Swing / Right Stance (stanceLegs = {false, true})
// Mode 2: Right Swing / Left Stance (stanceLegs = {true, false})
static constexpr size_t kDoubleSupportMode = 3;
static constexpr size_t kLeftSwingMode = 1;
static constexpr size_t kRightSwingMode = 2;

}  // namespace

GaitSwitchingTimeOptimizer::GaitSwitchingTimeOptimizer(
    GaitOptimizationSettings settings)
    : settings_(std::move(settings)) {}

absl::StatusOr<size_t> GaitSwitchingTimeOptimizer::findInteriorTimeIndex(
    const scalar_array_t& timeTrajectory, scalar_t eventTime) {
  // Horizon must contain enough discrete intervals to extract interior points.
  if (timeTrajectory.size() < kMinTrajectorySizeForInteriorNodes) {
    return absl::InvalidArgumentError(
        "Time trajectory must contain at least 3 points to define interior "
        "nodes.");
  }

  // Find the closest index in time array such that timeTrajectory[timeIdx] <=
  // eventTime.
  const size_t timeIdx =
      lookup::findIndexInTimeArray(timeTrajectory, eventTime);

  // Boundary event times (t <= t_0 or t >= t_f) cannot have interior
  // sensitivity computed.
  if (timeIdx == 0 || timeIdx >= timeTrajectory.size() - 1) {
    return absl::OutOfRangeError(
        "Event time is outside the interior time trajectory bounds.");
  }
  return timeIdx;
}

absl::StatusOr<scalar_t>
GaitSwitchingTimeOptimizer::computeSwitchingTimeSensitivity(
    scalar_t eventTime,
    size_t modeBefore,
    size_t modeAfter,
    const PrimalSolution& primalSolution) const {
  // Validate lookup within the MPC primal solution horizon.
  const absl::StatusOr<size_t> timeIdxStatus =
      findInteriorTimeIndex(primalSolution.timeTrajectory_, eventTime);
  if (!timeIdxStatus.ok()) {
    return timeIdxStatus.status();
  }

  const size_t timeIdx = timeIdxStatus.value();
  const vector_t& state = primalSolution.stateTrajectory_[timeIdx];
  const vector_t& inputBefore =
      primalSolution.inputTrajectory_[timeIdx - 1];  // u(t_i^-)
  const vector_t& inputAfter =
      primalSolution.inputTrajectory_[timeIdx];  // u(t_i^+)

  // --------------------------------------------------------------------------
  // Switching Time Sensitivity via Pontryagin Maximum Principle (PMP)
  //
  // For hybrid dynamical systems with mode sequence {m_{i-1}, m_i} separated
  // by switching time t_i, the gradient of the total cost functional J is given
  // by the Hamiltonian jump across the transition:
  //
  //   \frac{\partial J}{\partial t_i} = \mathcal{H}_{m_{i-1}}(x(t_i^-),
  //   u(t_i^-), \lambda(t_i^-))
  //                                    - \mathcal{H}_{m_i}(x(t_i^+), u(t_i^+),
  //                                    \lambda(t_i^+))
  //
  // where the Hamiltonian is:
  //   \mathcal{H}_m(x, u, \lambda) = L_m(x, u) + \lambda^T f_m(x, u)
  //
  // Along the optimal trajectory with continuous state derivatives, this
  // reduces to the difference in instantaneous running cost (control effort &
  // task tracking):
  //   \Delta \mathcal{H} \approx \frac{1}{2} \| u(t_i^-) \|_R^2 - \frac{1}{2}
  //   \| u(t_i^+) \|_R^2
  // --------------------------------------------------------------------------
  scalar_t sensitivity = 0.0;

  // 1. Control effort discontinuity:
  // If u(t_i^-) has higher norm than u(t_i^+), \Delta H > 0, indicating that
  // advancing the event time (reducing duration of phase i-1) will reduce the
  // total cost.
  const scalar_t inputNormDiff =
      inputBefore.squaredNorm() - inputAfter.squaredNorm();
  sensitivity += kHalf * inputNormDiff;

  // 2. Center of Mass (CoM) vertical momentum stability term:
  // State index 2 corresponds to CoM vertical velocity v_z (or momentum p_z).
  // If v_z < 0 (robot falling), a negative sensitivity encourages earlier
  // switching into double support to brake downward momentum.
  if (state.size() >= 3) {
    const scalar_t comVelocityZ = state[2];
    sensitivity += comVelocityZ * kComVelocityWeight;
  }

  return sensitivity;
}

bool GaitSwitchingTimeOptimizer::optimizeEventTimes(
    const PrimalSolution& primalSolution, ModeSchedule& modeSchedule) const {
  if (!settings_.enabled || !settings_.enableTrajectorySensitivity) {
    return false;
  }

  if (primalSolution.timeTrajectory_.empty() ||
      modeSchedule.eventTimes.empty()) {
    return false;
  }

  const scalar_t horizonStart = primalSolution.timeTrajectory_.front();
  const scalar_t horizonEnd = primalSolution.timeTrajectory_.back();

  bool modified = false;

  // Perform iterative projected gradient descent over switching times:
  //   t_i^{(k+1)} = \Pi_{[T_{\min}, T_{\max}]} \left( t_i^{(k)} - \alpha
  //   \frac{\partial J}{\partial t_i} \right)
  for (size_t iter = 0; iter < settings_.maxIterations; ++iter) {
    for (size_t i = 0; i < modeSchedule.eventTimes.size(); ++i) {
      const scalar_t eventTime = modeSchedule.eventTimes[i];

      // Skip event times that lie outside the active solver horizon.
      if (eventTime <= horizonStart + kWeakEpsilon ||
          eventTime >= horizonEnd - kWeakEpsilon) {
        continue;
      }

      const size_t modeBefore = modeSchedule.modeSequence[i];
      const size_t modeAfter = modeSchedule.modeSequence[i + 1];

      // Compute analytical Hamiltonian jump sensitivity \partial J / \partial
      // t_i.
      const absl::StatusOr<scalar_t> sensitivityStatus =
          computeSwitchingTimeSensitivity(eventTime, modeBefore, modeAfter,
                                          primalSolution);
      if (!sensitivityStatus.ok()) {
        continue;
      }

      const scalar_t sensitivity = sensitivityStatus.value();

      // Clamp sensitivity to guard against unconditioned spikes.
      const scalar_t clampedSensitivity =
          std::clamp(sensitivity, -kMaxSensitivityClamp, kMaxSensitivityClamp);

      // Apply gradient descent step: \Delta t_i = -\alpha \cdot (\partial J /
      // \partial t_i)
      if (std::abs(clampedSensitivity) > kWeakEpsilon) {
        modeSchedule.eventTimes[i] -= settings_.stepSize * clampedSensitivity;
        modified = true;
      }
    }

    // Project all modified switching times onto valid duration bounds [T_min,
    // T_max].
    enforceDurationBounds(modeSchedule);
  }

  return modified;
}

bool GaitSwitchingTimeOptimizer::adaptFromContactFeedback(
    const contact_flag_t& measuredContactFlags,
    scalar_t currentTime,
    ModeSchedule& modeSchedule) const {
  if (!settings_.enabled || !settings_.enableContactFeedback) {
    return false;
  }

  if (modeSchedule.eventTimes.empty()) {
    return false;
  }

  // Find the active mode index corresponding to currentTime.
  const size_t currentModeIdx =
      lookup::findIndexInTimeArray(modeSchedule.eventTimes, currentTime);
  if (currentModeIdx >= modeSchedule.modeSequence.size()) {
    return false;
  }

  const size_t currentMode = modeSchedule.modeSequence[currentModeIdx];
  const contact_flag_t plannedContact = modeNumber2StanceLeg(currentMode);

  bool modified = false;

  // --------------------------------------------------------------------------
  // Early Touchdown Reactive Adaptation:
  //
  // If the planned reference is in single support (swing phase for one foot),
  // but the hardware contact sensor detects touchdown ahead of schedule within
  // the early touchdown time window [t_i - \Delta t_{\text{window}}, t_i]:
  //
  //   Advance the next event time to the current timestamp:
  //     t_i \leftarrow \max(t_{\text{current}}, \, t_{i-1} +
  //     T_{\min,\text{single}})
  //
  // This immediately commands double support in the MPC reference manager to
  // accept ground reaction forces and prevent trajectory tracking windup.
  // --------------------------------------------------------------------------
  if (currentModeIdx < modeSchedule.eventTimes.size()) {
    const scalar_t nextEventTime = modeSchedule.eventTimes[currentModeIdx];
    const scalar_t timeToNextEvent = nextEventTime - currentTime;

    // Check if within the early touchdown detection window:
    if (timeToNextEvent > 0.0 &&
        timeToNextEvent <= settings_.earlyTouchDownTimeWindow) {
      const bool leftEarlyTouchdown =
          (!plannedContact[0] && measuredContactFlags[0]);
      const bool rightEarlyTouchdown =
          (!plannedContact[1] && measuredContactFlags[1]);

      if (leftEarlyTouchdown || rightEarlyTouchdown) {
        // Trigger early transition by advancing next event time to current
        // time.
        modeSchedule.eventTimes[currentModeIdx] = currentTime;
        modified = true;
      }
    }
  }

  // Enforce single/double support duration constraints after feedback
  // modification.
  if (modified) {
    enforceDurationBounds(modeSchedule);
  }

  return modified;
}

void GaitSwitchingTimeOptimizer::enforceDurationBounds(
    ModeSchedule& modeSchedule) const {
  if (modeSchedule.eventTimes.empty()) {
    return;
  }

  // --------------------------------------------------------------------------
  // Duration Bounds Constraint Projection:
  //
  // For each phase i in the mode schedule with duration \Delta t_i = t_i -
  // t_{i-1}:
  //
  //   \Delta t_i \leftarrow \operatorname{clip}\left( \Delta t_i, \,
  //   T_{\min}(m_i), \, T_{\max}(m_i) \right) t_i \leftarrow t_{i-1} + \Delta
  //   t_i
  //
  // where:
  //   - Single support modes (left swing = 1, right swing = 2):
  //       T_{\text{swing}} \in [0.25, 0.50] s
  //   - Double support mode (mode 3):
  //       T_{\text{double}} \in [0.05, 0.20] s
  // --------------------------------------------------------------------------
  for (size_t i = 0; i < modeSchedule.eventTimes.size(); ++i) {
    const scalar_t prevTime = (i == 0) ? 0.0 : modeSchedule.eventTimes[i - 1];
    const size_t currentMode = modeSchedule.modeSequence[i];

    scalar_t minDuration = settings_.minDoubleSupportDuration;
    scalar_t maxDuration = settings_.maxDoubleSupportDuration;

    if (currentMode == kLeftSwingMode || currentMode == kRightSwingMode) {
      minDuration = settings_.minSingleSupportDuration;
      maxDuration = settings_.maxSingleSupportDuration;
    }

    const scalar_t currentDuration = modeSchedule.eventTimes[i] - prevTime;
    const scalar_t clampedDuration =
        std::clamp(currentDuration, minDuration, maxDuration);

    modeSchedule.eventTimes[i] = prevTime + clampedDuration;
  }
}

}  // namespace ocs2::humanoid
