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

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

#include <ocs2_core/misc/Lookup.h>
#include <ocs2_core/misc/Numerics.h>

#include <algorithm>

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::SwingTrajectoryPlanner(Config config, size_t numFeet) : config_(std::move(config)), numFeet_(numFeet) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getZaccelerationConstraint(size_t leg, scalar_t time) const {
  const auto index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  if (isSearching(swingWindows_[leg][index], time)) return 0.0;
  return feetHeightTrajectories_[leg][index].acceleration(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getZvelocityConstraint(size_t leg, scalar_t time) const {
  const auto index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  if (isSearching(window, time)) return -window.descentVelocity;
  return feetHeightTrajectories_[leg][index].velocity(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t SwingTrajectoryPlanner::getZpositionConstraint(size_t leg, scalar_t time) const {
  const auto index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  if (isSearching(window, time)) return window.descentStartHeight - window.descentVelocity * (time - window.finalTime);
  return feetHeightTrajectories_[leg][index].position(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getImpactProximityFactor(size_t leg, scalar_t time) const {
  const auto index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
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
  const auto smoothStep = [](scalar_t u) { return u * u * (3.0 - 2.0 * u); };

  if (tau <= 0.0 || tau >= 1.0) return 0.0;
  if (rise > 0.0 && tau < rise) return smoothStep(tau / rise);
  if (fall > 0.0 && tau > 1.0 - fall) return smoothStep((1.0 - tau) / fall);
  return 1.0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getSwingPitchAngle(size_t leg, scalar_t time) const {
  if (numerics::almost_eq(config_.swingPitchAngle, 0.0) || swingWindows_[leg].empty()) return 0.0;

  const auto index = lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
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
  constexpr scalar_t kSwingTimeTolerance = 1e-6;  // [s] event times that identify the same swing
  const auto& modeSequence = modeSchedule.modeSequence;
  const auto& eventTimes = modeSchedule.eventTimes;

  const auto eesContactFlagStocks = extractContactFlags(modeSequence);

  feet_array_t<std::vector<int>> startTimesIndices;
  feet_array_t<std::vector<int>> finalTimesIndices;
  for (size_t leg = 0; leg < numFeet_; leg++) {
    std::tie(startTimesIndices[leg], finalTimesIndices[leg]) = updateFootSchedule(eesContactFlagStocks[leg]);
  }

  for (size_t j = 0; j < numFeet_; j++) {
    feetHeightTrajectories_[j].clear();
    feetHeightTrajectories_[j].reserve(modeSequence.size());
    impactProximityTrajectories_[j].clear();
    impactProximityTrajectories_[j].reserve(modeSequence.size());
    swingWindows_[j].clear();
    swingWindows_[j].reserve(modeSequence.size());
    for (int p = 0; p < modeSequence.size(); ++p) {
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
        const scalar_t scaling = swingTrajectoryScaling(swingStartTime, swingFinalTime, config_.swingTimeScale);  // for leg in the air
        // The pitch reference is evaluated on the whole swing, which may span several phases of the mode sequence.
        SwingWindow window;
        window.isSwing = true;
        window.startTime = swingStartTime;
        window.finalTime = swingFinalTime;
        window.scaling = scaling;
        window.searching = searching;
        window.descentStartHeight = touchDownHeightSequence[j][p];
        window.descentVelocity = searching ? std::max(0.0, search->descentVelocity) : 0.0;
        swingWindows_[j].push_back(window);
        if (eesContactFlagStocks[j][p - 1] && eesContactFlagStocks[j][p + 1]) {  // For a swing leg, only in the air for current mode

          const CubicSpline::Node liftOffHeight{swingStartTime, liftOffHeightSequence[j][p], scaling * config_.liftOffVelocity};
          const CubicSpline::Node touchDownHeight{swingFinalTime, touchDownHeightSequence[j][p], scaling * config_.touchDownVelocity};
          const scalar_t midHeight = std::min(liftOffHeightSequence[j][p], touchDownHeightSequence[j][p]) + scaling * config_.swingHeight;
          feetHeightTrajectories_[j].emplace_back(liftOffHeight, midHeight, touchDownHeight);

          const CubicSpline::Node impactProximityLiftOff{swingStartTime, 1.0, scaling * config_.impactProximityFactorLiftOffVelocity};
          const CubicSpline::Node impactProximityTouchDown{swingFinalTime, 1.0, scaling * config_.impactProximityFactorTouchDownVelocity};

          impactProximityTrajectories_[j].emplace_back(impactProximityLiftOff, config_.impactProximityFactorMidPointValue,
                                                       impactProximityTouchDown);
        } else if (eesContactFlagStocks[j][p - 1]) {  // For foot just leaving the ground and staying in the air
          const scalar_t midHeight = liftOffHeightSequence[j][p] + config_.swingHeight;
          const CubicSpline::Node liftOffHeight{swingStartTime, liftOffHeightSequence[j][p], config_.liftOffVelocity};
          const CubicSpline::Node touchDownHeight{swingFinalTime, midHeight, 0.0};
          feetHeightTrajectories_[j].emplace_back(liftOffHeight, midHeight, touchDownHeight);

          const CubicSpline::Node impactProximityLiftOff{swingStartTime, 1.0, config_.impactProximityFactorLiftOffVelocity};
          const CubicSpline::Node impactProximityTouchDown{swingFinalTime, config_.impactProximityFactorMidPointValue, 0.0};

          impactProximityTrajectories_[j].emplace_back(impactProximityLiftOff, config_.impactProximityFactorMidPointValue,
                                                       impactProximityTouchDown);
        } else if (eesContactFlagStocks[j][p + 1]) {  // For foot that was in the air and is impacting in the next mode
          const scalar_t midHeight = touchDownHeightSequence[j][p] + config_.swingHeight;
          const CubicSpline::Node liftOffHeight{swingStartTime, midHeight, 0.0};
          const CubicSpline::Node touchDownHeight{swingFinalTime, touchDownHeightSequence[j][p], config_.touchDownVelocity};
          feetHeightTrajectories_[j].emplace_back(liftOffHeight, midHeight, touchDownHeight);

          const CubicSpline::Node impactProximityLiftOff{swingStartTime, config_.impactProximityFactorMidPointValue, 0.0};
          const CubicSpline::Node impactProximityTouchDown{swingFinalTime, 1.0, config_.impactProximityFactorTouchDownVelocity};

          impactProximityTrajectories_[j].emplace_back(impactProximityLiftOff, config_.impactProximityFactorMidPointValue,
                                                       impactProximityTouchDown);
        } else {  // For foot in the air for last, current and next mode
          const scalar_t midHeight = touchDownHeightSequence[j][p] + config_.swingHeight;
          const CubicSpline::Node liftOffHeight{swingStartTime, midHeight, 0.0};
          const CubicSpline::Node touchDownHeight{swingFinalTime, midHeight, 0.0};
          feetHeightTrajectories_[j].emplace_back(liftOffHeight, midHeight, touchDownHeight);

          const CubicSpline::Node impactProximityLiftOff{swingStartTime, config_.impactProximityFactorMidPointValue, 0.0};
          const CubicSpline::Node impactProximityTouchDown{swingFinalTime, config_.impactProximityFactorMidPointValue, 0.0};

          impactProximityTrajectories_[j].emplace_back(impactProximityLiftOff, config_.impactProximityFactorMidPointValue,
                                                       impactProximityTouchDown);
        }
      } else {  // for a stance leg
        swingWindows_[j].push_back(SwingWindow{});
        // Note: setting the time here arbitrarily to 0.0 -> 1.0 makes the assert in CubicSpline fail
        const CubicSpline::Node liftOff{0.0, liftOffHeightSequence[j][p], 0.0};
        const CubicSpline::Node touchDown{1.0, liftOffHeightSequence[j][p], 0.0};
        feetHeightTrajectories_[j].emplace_back(liftOff, liftOffHeightSequence[j][p], touchDown);

        // If the foot is in contact the impact proximity factor is always 1.
        const CubicSpline::Node impactProximityTouchDown{0.0, 1.0, 0};
        const CubicSpline::Node impactProximityLiftOff{1.0, 1.0, 0};

        impactProximityTrajectories_[j].emplace_back(impactProximityTouchDown, 1.0, impactProximityLiftOff);
      }
    }
    feetHeightTrajectoriesEvents_[j] = eventTimes;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::pair<std::vector<int>, std::vector<int>> SwingTrajectoryPlanner::updateFootSchedule(const std::vector<bool>& contactFlagStock) {
  const size_t numPhases = contactFlagStock.size();

  std::vector<int> startTimeIndexStock(numPhases, 0);
  std::vector<int> finalTimeIndexStock(numPhases, 0);

  // find the startTime and finalTime indices for swing feet
  for (size_t i = 0; i < numPhases; i++) {
    if (!contactFlagStock[i]) {
      std::tie(startTimeIndexStock[i], finalTimeIndexStock[i]) = findIndex(i, contactFlagStock);
    }
  }
  return {startTimeIndexStock, finalTimeIndexStock};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

feet_array_t<std::vector<bool>> SwingTrajectoryPlanner::extractContactFlags(const std::vector<size_t>& phaseIDsStock) const {
  const size_t numPhases = phaseIDsStock.size();

  feet_array_t<std::vector<bool>> contactFlagStock;
  std::fill(contactFlagStock.begin(), contactFlagStock.end(), std::vector<bool>(numPhases));

  for (size_t i = 0; i < numPhases; i++) {
    const auto contactFlag = modeNumber2StanceLeg(phaseIDsStock[i]);
    for (size_t j = 0; j < numFeet_; j++) {
      contactFlagStock[j][i] = contactFlag[j];
    }
  }
  return contactFlagStock;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::pair<int, int> SwingTrajectoryPlanner::findIndex(size_t index, const std::vector<bool>& contactFlagStock) {
  const size_t numPhases = contactFlagStock.size();

  // skip if it is a stance leg
  if (contactFlagStock[index]) {
    return {0, 0};
  }

  // find the starting time
  int startTimesIndex = -1;
  for (int ip = index - 1; ip >= 0; ip--) {
    if (contactFlagStock[ip]) {
      startTimesIndex = ip;
      break;
    }
  }

  // find the final time
  int finalTimesIndex = numPhases - 1;
  for (size_t ip = index + 1; ip < numPhases; ip++) {
    if (contactFlagStock[ip]) {
      finalTimesIndex = ip - 1;
      break;
    }
  }

  return {startTimesIndex, finalTimesIndex};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SwingTrajectoryPlanner::checkThatIndicesAreValid(
    int leg, int index, int startIndex, int finalIndex, const std::vector<size_t>& phaseIDsStock) {
  const size_t numSubsystems = phaseIDsStock.size();
  if (startIndex < 0) {
    std::cerr << "Subsystem: " << index << " out of " << numSubsystems - 1 << std::endl;
    for (size_t i = 0; i < numSubsystems; i++) {
      std::cerr << "[" << i << "]: " << phaseIDsStock[i] << ",  ";
    }
    std::cerr << std::endl;

    throw std::runtime_error("The time of take-off for the first swing of the EE with ID " + std::to_string(leg) + " is not defined.");
  }
  if (finalIndex >= numSubsystems - 1) {
    std::cerr << "Subsystem: " << index << " out of " << numSubsystems - 1 << std::endl;
    for (size_t i = 0; i < numSubsystems; i++) {
      std::cerr << "[" << i << "]: " << phaseIDsStock[i] << ",  ";
    }
    std::cerr << std::endl;

    throw std::runtime_error("The time of touch-down for the last swing of the EE with ID " + std::to_string(leg) + " is not defined.");
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::swingTrajectoryScaling(scalar_t startTime, scalar_t finalTime, scalar_t swingTimeScale) {
  return std::min(1.0, (finalTime - startTime) / swingTimeScale);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::Config loadSwingTrajectorySettings(const std::string& fileName, const std::string& fieldName, bool verbose) {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(fileName, pt);

  if (verbose) {
    std::cerr << "\n #### Swing Trajectory Config:";
    std::cerr << "\n #### =============================================================================\n";
  }

  SwingTrajectoryPlanner::Config config;
  const std::string prefix = fieldName + ".";

  loadData::loadPtreeValue(pt, config.liftOffVelocity, prefix + "liftOffVelocity", verbose);
  loadData::loadPtreeValue(pt, config.touchDownVelocity, prefix + "touchDownVelocity", verbose);
  loadData::loadPtreeValue(pt, config.swingHeight, prefix + "swingHeight", verbose);
  loadData::loadPtreeValue(pt, config.swingTimeScale, prefix + "swingTimeScale", verbose);
  loadData::loadPtreeValue(pt, config.touchDownHeightOffset, prefix + "touchDownHeightOffset", verbose);

  loadData::loadPtreeValue(pt, config.impactProximityFactorLiftOffVelocity, prefix + "impactProximityFactorLiftOffVelocity", verbose);
  loadData::loadPtreeValue(pt, config.impactProximityFactorTouchDownVelocity, prefix + "impactProximityFactorTouchDownVelocity", verbose);
  loadData::loadPtreeValue(pt, config.impactProximityFactorMidPointValue, prefix + "impactProximityFactorMidPointValue", verbose);

  loadData::loadPtreeValue(pt, config.swingPitchAngle, prefix + "swingPitchAngle", verbose);
  loadData::loadPtreeValue(pt, config.swingPitchRiseFraction, prefix + "swingPitchRiseFraction", verbose);
  loadData::loadPtreeValue(pt, config.swingPitchFallFraction, prefix + "swingPitchFallFraction", verbose);

  if (verbose) {
    std::cerr << " #### =============================================================================" << std::endl;
  }

  return config;
}

}  // namespace ocs2::humanoid
