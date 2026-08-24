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

#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_core/reference/ModeSchedule.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/gait/GaitOptimizationSettings.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

/**
 * Optimizes gait switching times (eventTimes in ModeSchedule) based on:
 *  1. Trajectory sensitivities (Hamiltonian jump condition across mode switches):
 *       \frac{\partial J}{\partial t_i} = \mathcal{H}_{m_{i-1}}(t_i^-) - \mathcal{H}_{m_i}(t_i^+)
 *  2. Real-time contact sensor feedback (early touchdown / late touchdown adaptation):
 *       t_i \leftarrow \max(t_{\text{current}}, \, t_{i-1} + T_{\min})
 *  3. Configurable phase duration bounds:
 *       T_{\min}(m_i) \le t_i - t_{i-1} \le T_{\max}(m_i)
 */
class GaitSwitchingTimeOptimizer {
 public:
  explicit GaitSwitchingTimeOptimizer(GaitOptimizationSettings settings = GaitOptimizationSettings());

  ~GaitSwitchingTimeOptimizer() = default;

  const GaitOptimizationSettings& getSettings() const { return settings_; }
  void setSettings(GaitOptimizationSettings settings) { settings_ = std::move(settings); }

  /**
   * Optimizes the event times in the provided ModeSchedule via projected gradient descent:
   *   t_i^{(k+1)} = \Pi_{[T_{\min}, T_{\max}]} \left( t_i^{(k)} - \alpha \frac{\partial J}{\partial t_i} \right)
   *
   * @param [in] primalSolution: The most recent MPC primal solution containing time, state, and input trajectories.
   * @param [in,out] modeSchedule: The ModeSchedule whose eventTimes will be optimized.
   * @return true if event times were modified, false otherwise.
   */
  bool optimizeEventTimes(const PrimalSolution& primalSolution, ModeSchedule& modeSchedule) const;

  /**
   * Adapts event times based on instantaneous measured contact sensor flags (early touchdown detection):
   *   t_i \leftarrow \max(t_{\text{current}}, \, t_{i-1} + T_{\min,\text{single}})
   *
   * @param [in] measuredContactFlags: Contact state per leg (true = contact detected).
   * @param [in] currentTime: Current simulation / robot time.
   * @param [in,out] modeSchedule: The ModeSchedule to adapt.
   * @return true if modeSchedule was adapted, false otherwise.
   */
  bool adaptFromContactFeedback(const contact_flag_t& measuredContactFlags, scalar_t currentTime, ModeSchedule& modeSchedule) const;

  /**
   * Enforces single support and double support duration bounds on a ModeSchedule:
   *   \Delta t_i \leftarrow \operatorname{clip}(\Delta t_i, \, T_{\min}(m_i), \, T_{\max}(m_i))
   *
   * @param [in,out] modeSchedule: The ModeSchedule to enforce bounds on.
   */
  void enforceDurationBounds(ModeSchedule& modeSchedule) const;

  /**
   * Computes the switching time sensitivity across a mode transition at eventTime:
   *   \frac{\partial J}{\partial t_i} \approx \frac{1}{2} \| u(t_i^-) \|^2 - \frac{1}{2} \| u(t_i^+) \|^2 + w_z \cdot v_z(t_i)
   *
   * @param [in] eventTime: Switching time timestamp.
   * @param [in] modeBefore: Mode index active on [t_{i-1}, t_i).
   * @param [in] modeAfter: Mode index active on [t_i, t_{i+1}).
   * @param [in] primalSolution: MPC trajectory data.
   * @return absl::StatusOr containing the scalar sensitivity, or an error status if eventTime is out of bounds.
   */
  absl::StatusOr<scalar_t> computeSwitchingTimeSensitivity(scalar_t eventTime,
                                                           size_t modeBefore,
                                                           size_t modeAfter,
                                                           const PrimalSolution& primalSolution) const;

  /**
   * Finds the index of an eventTime in the given time trajectory within interior bounds [1, N-2].
   * Returns absl::OutOfRangeError if outside interior bounds, or absl::InvalidArgumentError if size < 3.
   *
   * @param [in] timeTrajectory: Monotonically increasing time vector.
   * @param [in] eventTime: Target timestamp to look up.
   * @return absl::StatusOr containing the interior index.
   */
  static absl::StatusOr<size_t> findInteriorTimeIndex(const scalar_array_t& timeTrajectory, scalar_t eventTime);

 private:
  GaitOptimizationSettings settings_;
};

}  // namespace ocs2::humanoid
