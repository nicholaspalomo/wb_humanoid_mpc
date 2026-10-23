/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "ocs2_core/misc/Lookup.h"
#include "ocs2_core/misc/Numerics.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {
namespace {

/** The coefficient of the quadratic term of the smoothstep 3 u^2 - 2 u^3 (swingPitchProfile()). */
constexpr scalar_t kSmoothStepQuadraticCoefficient = 3.0;

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::SwingTrajectoryPlanner(Config config, size_t numFeet) : config_(config), numFeet_(numFeet) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getZaccelerationConstraint(size_t leg, scalar_t time) const {
  const int index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  if (isSearching(swingWindows_[leg][index], time)) return 0.0;
  return feetHeightTrajectories_[leg][index].acceleration(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getZvelocityConstraint(size_t leg, scalar_t time) const {
  const int index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  if (isSearching(window, time)) return -window.descentVelocity;
  return feetHeightTrajectories_[leg][index].velocity(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t SwingTrajectoryPlanner::getZpositionConstraint(size_t leg, scalar_t time) const {
  const int index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  if (isSearching(window, time)) return window.descentStartHeight - window.descentVelocity * (time - window.finalTime);
  return feetHeightTrajectories_[leg][index].position(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getImpactProximityFactor(size_t leg, scalar_t time) const {
  const int index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  // A foot searching for the ground holds the factor of its planned touch-down.
  if (isSearching(window, time)) return impactProximityTrajectories_[leg][index].position(window.finalTime);
  return impactProximityTrajectories_[leg][index].position(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::swingPitchProfile(scalar_t tau) const {
  // Smoothstep up, hold, smoothstep down. The two ramps are clamped so that they never overlap even if the configured
  // fractions sum to more than one, in which case the profile simply never reaches its peak.
  const scalar_t rise = std::clamp(config_.swingPitchRiseFraction, 0.0, 1.0);
  const scalar_t fall = std::clamp(config_.swingPitchFallFraction, 0.0, 1.0 - rise);
  const std::function<scalar_t(scalar_t)> smoothStep = [](scalar_t u) { return u * u * (kSmoothStepQuadraticCoefficient - 2.0 * u); };

  if (tau <= 0.0 || tau >= 1.0) return 0.0;
  if (rise > 0.0 && tau < rise) return smoothStep(tau / rise);
  if (fall > 0.0 && tau > 1.0 - fall) return smoothStep((1.0 - tau) / fall);
  return 1.0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getSwingPitchAngle(size_t leg, scalar_t time) const {
  if (numerics::almost_eq(config_.swingPitchAngle, /*y=*/0.0) || swingWindows_[leg].empty()) return 0.0;

  const int index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  if (!window.isSwing) return 0.0;

  const scalar_t duration = window.finalTime - window.startTime;
  if (duration <= 0.0) return 0.0;

  const scalar_t tau = std::clamp((time - window.startTime) / duration, 0.0, 1.0);
  // Short swings get proportionally less pitch, for the same reason they get less height.
  return window.scaling * config_.swingPitchAngle * swingPitchProfile(tau);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SwingTrajectoryPlanner::update(const ModeSchedule& modeSchedule, scalar_t terrainHeight) {
  const scalar_array_t terrainHeightSequence(modeSchedule.modeSequence.size(), terrainHeight);
  const scalar_array_t touchDownTerrainHeightSequence(modeSchedule.modeSequence.size(), terrainHeight + config_.touchDownHeightOffset);
  feet_array_t<scalar_array_t> liftOffHeightSequence;
  liftOffHeightSequence.fill(terrainHeightSequence);
  feet_array_t<scalar_array_t> touchDownHeightSequence;
  touchDownHeightSequence.fill(touchDownTerrainHeightSequence);
  update(modeSchedule, liftOffHeightSequence, touchDownHeightSequence);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SwingTrajectoryPlanner::update(const ModeSchedule& modeSchedule,
                                    const feet_array_t<scalar_array_t>& liftOffHeightSequence,
                                    const feet_array_t<scalar_array_t>& touchDownHeightSequence) {
  update(modeSchedule, liftOffHeightSequence, touchDownHeightSequence, makeFeetArray(std::optional<GroundSearch>{}));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SwingTrajectoryPlanner::update(const ModeSchedule& modeSchedule,
                                    const feet_array_t<scalar_array_t>& liftOffHeightSequence,
                                    const feet_array_t<scalar_array_t>& touchDownHeightSequence,
                                    const feet_array_t<std::optional<GroundSearch>>& groundSearches) {
  constexpr scalar_t kSwingTimeTolerance = 1.0e-6;  // [s] event times that identify the same swing
  const std::vector<size_t>& modeSequence = modeSchedule.modeSequence;
  const std::vector<scalar_t>& eventTimes = modeSchedule.eventTimes;

  const feet_array_t<std::vector<bool>> eesContactFlagStocks = extractContactFlags(modeSequence);

  feet_array_t<std::vector<int>> startTimesIndices;
  feet_array_t<std::vector<int>> finalTimesIndices;
  for (size_t leg = 0; leg < numFeet_; ++leg) {
    FootSwingEventIndices swings = updateFootSchedule(eesContactFlagStocks[leg]);
    startTimesIndices[leg] = std::move(swings.startTimeIndices);
    finalTimesIndices[leg] = std::move(swings.finalTimeIndices);
  }

  for (size_t j = 0; j < numFeet_; ++j) {
    feetHeightTrajectories_[j].clear();
    feetHeightTrajectories_[j].reserve(modeSequence.size());
    impactProximityTrajectories_[j].clear();
    impactProximityTrajectories_[j].reserve(modeSequence.size());
    swingWindows_[j].clear();
    swingWindows_[j].reserve(modeSequence.size());
    for (int p = 0; p < static_cast<int>(modeSequence.size()); ++p) {
      const int swingStartIndex = startTimesIndices[j][p];
      const int swingFinalIndex = finalTimesIndices[j][p];
      checkThatIndicesAreValid(j, p, swingStartIndex, swingFinalIndex, modeSequence);

      const scalar_t swingStartTime = eventTimes[swingStartIndex];
      scalar_t swingFinalTime = eventTimes[swingFinalIndex];

      // A swing extended past its planned touch-down (GroundSearch): the spline and the pitch profile are fitted to the
      // planned swing, and the height reference continues from its end as a straight descent.
      bool searching = false;
      const std::optional<GroundSearch>& search = groundSearches[j];
      if (!eesContactFlagStocks[j][p] && search.has_value() && std::abs(search->liftOffTime - swingStartTime) <= kSwingTimeTolerance &&
          search->plannedTouchDownTime > swingStartTime + kSwingTimeTolerance &&
          search->plannedTouchDownTime < swingFinalTime - kSwingTimeTolerance) {
        searching = true;
        swingFinalTime = search->plannedTouchDownTime;
      }

      if (!eesContactFlagStocks[j][p]) {
        // Every phase of a swing, which spans several phases in a gait with flight phases (run: the swing of one foot is
        // FLY | other foot in stance | FLY), gets the same splines over the whole swing, so the references are
        // continuous at the inner events and have one apex and one short-swing scaling per swing. The heights are read
        // at the swing's first and last phase so that every phase of the swing reads the same ones.
        const scalar_t scaling = swingTrajectoryScaling(swingStartTime, swingFinalTime, config_.swingTimeScale);
        const scalar_t liftOffHeight = liftOffHeightSequence[j][swingStartIndex + 1];
        const scalar_t touchDownHeight = touchDownHeightSequence[j][swingFinalIndex];
        SwingWindow window;
        window.isSwing = true;
        window.startTime = swingStartTime;
        window.finalTime = swingFinalTime;
        window.scaling = scaling;
        window.searching = searching;
        window.descentStartHeight = touchDownHeight;
        window.descentVelocity = searching ? std::max(0.0, search->descentVelocity) : 0.0;
        swingWindows_[j].push_back(window);
        const SwingSplines splines = makeSwingSplines(window, liftOffHeight, touchDownHeight);
        feetHeightTrajectories_[j].push_back(splines.height);
        impactProximityTrajectories_[j].push_back(splines.impactProximity);
      } else {  // for a stance leg
        swingWindows_[j].push_back(SwingWindow{});
        // Note: setting the time here arbitrarily to 0.0 -> 1.0 makes the assert in CubicSpline fail
        const CubicSpline::Node liftOff{.time = 0.0, .position = liftOffHeightSequence[j][p], .velocity = 0.0};
        const CubicSpline::Node touchDown{.time = 1.0, .position = liftOffHeightSequence[j][p], .velocity = 0.0};
        feetHeightTrajectories_[j].emplace_back(liftOff, liftOffHeightSequence[j][p], touchDown);

        // If the foot is in contact the impact proximity factor is always 1.
        const CubicSpline::Node impactProximityTouchDown{.time = 0.0, .position = 1.0, .velocity = 0};
        const CubicSpline::Node impactProximityLiftOff{.time = 1.0, .position = 1.0, .velocity = 0};

        impactProximityTrajectories_[j].emplace_back(impactProximityTouchDown, 1.0, impactProximityLiftOff);
      }
    }
    feetHeightTrajectoriesEvents_[j] = eventTimes;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::SwingSplines SwingTrajectoryPlanner::makeSwingSplines(const SwingWindow& window,
                                                                              scalar_t liftOffHeight,
                                                                              scalar_t touchDownHeight) const {
  const scalar_t scaling = window.scaling;
  // The apex is swingHeight above the higher end, so that a step up taller than swingHeight still reaches the
  // touch-down height from above rather than from below.
  const scalar_t apexHeight = std::max(liftOffHeight, touchDownHeight) + scaling * config_.swingHeight;
  const CubicSpline::Node heightLiftOff{.time = window.startTime, .position = liftOffHeight, .velocity = scaling * config_.liftOffVelocity};
  const CubicSpline::Node heightTouchDown{
      .time = window.finalTime, .position = touchDownHeight, .velocity = scaling * config_.touchDownVelocity};

  const CubicSpline::Node impactProximityLiftOff{
      .time = window.startTime, .position = 1.0, .velocity = scaling * config_.impactProximityFactorLiftOffVelocity};
  const CubicSpline::Node impactProximityTouchDown{
      .time = window.finalTime, .position = 1.0, .velocity = scaling * config_.impactProximityFactorTouchDownVelocity};
  return SwingSplines{
      .height = SplineCpg(heightLiftOff, apexHeight, heightTouchDown),
      .impactProximity = SplineCpg(impactProximityLiftOff, config_.impactProximityFactorMidPointValue, impactProximityTouchDown),
  };
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::FootSwingEventIndices SwingTrajectoryPlanner::updateFootSchedule(const std::vector<bool>& contactFlagStock) {
  const size_t numPhases = contactFlagStock.size();

  std::vector<int> startTimeIndexStock(numPhases, 0);
  std::vector<int> finalTimeIndexStock(numPhases, 0);

  // find the startTime and finalTime indices for swing feet
  for (size_t i = 0; i < numPhases; ++i) {
    if (!contactFlagStock[i]) {
      const SwingEventIndices swing = findIndex(i, contactFlagStock);
      startTimeIndexStock[i] = swing.startTimeIndex;
      finalTimeIndexStock[i] = swing.finalTimeIndex;
    }
  }
  return FootSwingEventIndices{.startTimeIndices = std::move(startTimeIndexStock), .finalTimeIndices = std::move(finalTimeIndexStock)};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

feet_array_t<std::vector<bool>> SwingTrajectoryPlanner::extractContactFlags(const std::vector<size_t>& phaseIDsStock) const {
  const size_t numPhases = phaseIDsStock.size();

  feet_array_t<std::vector<bool>> contactFlagStock;
  std::fill(contactFlagStock.begin(), contactFlagStock.end(), std::vector<bool>(numPhases));

  for (size_t i = 0; i < numPhases; ++i) {
    const contact_flag_t contactFlag = modeNumber2StanceLeg(phaseIDsStock[i]);
    for (size_t j = 0; j < numFeet_; ++j) {
      contactFlagStock[j][i] = contactFlag[j];
    }
  }
  return contactFlagStock;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::SwingEventIndices SwingTrajectoryPlanner::findIndex(size_t index, const std::vector<bool>& contactFlagStock) {
  const size_t numPhases = contactFlagStock.size();

  // skip if it is a stance leg
  if (contactFlagStock[index]) {
    return SwingEventIndices{.startTimeIndex = 0, .finalTimeIndex = 0};
  }

  // find the starting time
  int startTimesIndex = -1;
  for (int ip = index - 1; ip >= 0; --ip) {
    if (contactFlagStock[ip]) {
      startTimesIndex = ip;
      break;
    }
  }

  // find the final time
  int finalTimesIndex = numPhases - 1;
  for (size_t ip = index + 1; ip < numPhases; ++ip) {
    if (contactFlagStock[ip]) {
      finalTimesIndex = ip - 1;
      break;
    }
  }

  return SwingEventIndices{.startTimeIndex = startTimesIndex, .finalTimeIndex = finalTimesIndex};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SwingTrajectoryPlanner::checkThatIndicesAreValid(
    int leg, int index, int startIndex, int finalIndex, const std::vector<size_t>& phaseIDsStock) {
  const size_t numSubsystems = phaseIDsStock.size();
  if (startIndex < 0) {
    LOG(INFO) << "Subsystem: " << index << " out of " << numSubsystems - 1;
    for (size_t i = 0; i < numSubsystems; ++i) {
      LOG(INFO) << "[" << i << "]: " << phaseIDsStock[i] << ",  ";
    }
    LOG(INFO);

    // NOLINTNEXTLINE(exceptions): update() runs in the solve, inside OCS2 overrides with no status channel; it fails the solve.
    throw std::runtime_error("The time of take-off for the first swing of the EE with ID " + std::to_string(leg) + " is not defined.");
  }
  if (static_cast<size_t>(finalIndex) >= numSubsystems - 1) {
    LOG(INFO) << "Subsystem: " << index << " out of " << numSubsystems - 1;
    for (size_t i = 0; i < numSubsystems; ++i) {
      LOG(INFO) << "[" << i << "]: " << phaseIDsStock[i] << ",  ";
    }
    LOG(INFO);

    // NOLINTNEXTLINE(exceptions): update() runs in the solve, inside OCS2 overrides with no status channel; it fails the solve.
    throw std::runtime_error("The time of touch-down for the last swing of the EE with ID " + std::to_string(leg) + " is not defined.");
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::swingTrajectoryScaling(scalar_t startTime, scalar_t finalTime, scalar_t swingTimeScale) {
  return std::min(1.0, (finalTime - startTime) / swingTimeScale);
}

}  // namespace ocs2::humanoid
