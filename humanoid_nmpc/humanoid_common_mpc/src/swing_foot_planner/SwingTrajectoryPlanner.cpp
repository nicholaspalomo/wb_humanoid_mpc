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
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "ocs2_core/Types.h"
#include "ocs2_core/misc/Lookup.h"
#include "ocs2_core/misc/Numerics.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/swing_foot_planner/CubicSpline.h"

namespace ocs2::humanoid {
namespace {

/** The coefficient of the quadratic term of the smoothstep 3 u^2 - 2 u^3
 * (swingPitchProfile()). */
constexpr scalar_t kSmoothStepQuadraticCoefficient = 3.0;

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::SwingTrajectoryPlanner(Config config, size_t numFeet)
    : config_(config),
      numFeet_(numFeet),
      swingWindows_(numFeet),
      impactProximityTrajectories_(numFeet),
      desPosFeetRtWorldEwrtWorldZTraj_(numFeet),
      feetHeightTrajectoriesEvents_(numFeet) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getDesLinaccFootRtWorldEwrtWorldZ(
    size_t leg, scalar_t time) const {
  const int index =
      lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  if (isSearching(swingWindows_[leg][index], time)) return 0.0;
  return desPosFeetRtWorldEwrtWorldZTraj_[leg][index].secondDerivative(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getDesLinvelFootRtWorldEwrtWorldZ(
    size_t leg, scalar_t time) const {
  const int index =
      lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  if (isSearching(window, time)) return -window.descentSpeedFootRtWorld;
  return desPosFeetRtWorldEwrtWorldZTraj_[leg][index].derivative(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t SwingTrajectoryPlanner::getDesPosFootRtWorldEwrtWorldZ(
    size_t leg, scalar_t time) const {
  const int index =
      lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  if (isSearching(window, time)) {
    return window.descentStartPosFootRtWorldEwrtWorldZ -
           window.descentSpeedFootRtWorld * (time - window.finalTime);
  }
  return desPosFeetRtWorldEwrtWorldZTraj_[leg][index].value(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getImpactProximityFactor(size_t leg,
                                                          scalar_t time) const {
  const int index =
      lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  // A foot searching for the ground holds the factor of its planned touch-down.
  if (isSearching(window, time)) {
    return impactProximityTrajectories_[leg][index].value(window.finalTime);
  }
  return impactProximityTrajectories_[leg][index].value(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::swingPitchProfile(scalar_t tau) const {
  // Smoothstep up, hold, smoothstep down. The two ramps are clamped so that
  // they never overlap even if the configured fractions sum to more than one,
  // in which case the profile simply never reaches its peak.
  const scalar_t rise = std::clamp(config_.swingPitchRiseFraction, 0.0, 1.0);
  const scalar_t fall =
      std::clamp(config_.swingPitchFallFraction, 0.0, 1.0 - rise);
  const std::function<scalar_t(scalar_t)> smoothStep = [](scalar_t u) {
    return u * u * (kSmoothStepQuadraticCoefficient - 2.0 * u);
  };

  if (tau <= 0.0 || tau >= 1.0) return 0.0;
  if (rise > 0.0 && tau < rise) return smoothStep(tau / rise);
  if (fall > 0.0 && tau > 1.0 - fall) return smoothStep((1.0 - tau) / fall);
  return 1.0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::getSwingRefTiltangleFootRtSurface(
    size_t leg, scalar_t time) const {
  if (numerics::almost_eq(config_.swingMaxTiltangleFootRtSurface, /*y=*/0.0) ||
      swingWindows_[leg].empty()) {
    return 0.0;
  }

  const int index =
      lookup::findIndexInTimeArray(feetHeightTrajectoriesEvents_[leg], time);
  const SwingWindow& window = swingWindows_[leg][index];
  if (!window.isSwing) return 0.0;

  const scalar_t duration = window.finalTime - window.startTime;
  if (duration <= 0.0) return 0.0;

  const scalar_t tau =
      std::clamp((time - window.startTime) / duration, 0.0, 1.0);
  // Short swings get proportionally less pitch, for the same reason they get
  // less height.
  return window.scaling * config_.swingMaxTiltangleFootRtSurface *
         swingPitchProfile(tau);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SwingTrajectoryPlanner::update(const ModeSchedule& modeSchedule,
                                    scalar_t posSurfaceRtWorldEwrtWorldZ) {
  const scalar_array_t posSurfaceRtWorldEwrtWorldZTraj(
      modeSchedule.modeSequence.size(), posSurfaceRtWorldEwrtWorldZ);
  const scalar_array_t touchdownPosFootRtWorldEwrtWorldZTraj(
      modeSchedule.modeSequence.size(),
      posSurfaceRtWorldEwrtWorldZ +
          config_.touchdownPosFootRtSurfaceEwrtWorldZ);
  const ContactArray<scalar_array_t> liftoffPosFeetRtWorldEwrtWorldZTraj(
      numFeet_, posSurfaceRtWorldEwrtWorldZTraj);
  const ContactArray<scalar_array_t> touchdownPosFeetRtWorldEwrtWorldZTraj(
      numFeet_, touchdownPosFootRtWorldEwrtWorldZTraj);
  update(modeSchedule, liftoffPosFeetRtWorldEwrtWorldZTraj,
         touchdownPosFeetRtWorldEwrtWorldZTraj);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SwingTrajectoryPlanner::update(
    const ModeSchedule& modeSchedule,
    const ContactArray<scalar_array_t>& liftoffPosFeetRtWorldEwrtWorldZTraj,
    const ContactArray<scalar_array_t>& touchdownPosFeetRtWorldEwrtWorldZTraj) {
  update(modeSchedule, liftoffPosFeetRtWorldEwrtWorldZTraj,
         touchdownPosFeetRtWorldEwrtWorldZTraj,
         ContactArray<std::optional<GroundSearch>>(
             numFeet_, /*value=*/std::optional<GroundSearch>{}));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void SwingTrajectoryPlanner::update(
    const ModeSchedule& modeSchedule,
    const ContactArray<scalar_array_t>& liftoffPosFeetRtWorldEwrtWorldZTraj,
    const ContactArray<scalar_array_t>& touchdownPosFeetRtWorldEwrtWorldZTraj,
    const ContactArray<std::optional<GroundSearch>>& groundSearches) {
  constexpr scalar_t kSwingTimeTolerance =
      1.0e-6;  // [s] event times that identify the same swing
  const std::vector<size_t>& modeSequence = modeSchedule.modeSequence;
  const std::vector<scalar_t>& eventTimes = modeSchedule.eventTimes;

  const ContactArray<std::vector<bool>> eesContactFlagStocks =
      extractContactFlags(modeSequence);

  ContactArray<std::vector<int>> startTimesIndices(numFeet_);
  ContactArray<std::vector<int>> finalTimesIndices(numFeet_);
  for (size_t leg = 0; leg < numFeet_; ++leg) {
    FootSwingEventIndices swings =
        updateFootSchedule(eesContactFlagStocks[leg]);
    startTimesIndices[leg] = std::move(swings.startTimeIndices);
    finalTimesIndices[leg] = std::move(swings.finalTimeIndices);
  }

  for (size_t j = 0; j < numFeet_; ++j) {
    desPosFeetRtWorldEwrtWorldZTraj_[j].clear();
    desPosFeetRtWorldEwrtWorldZTraj_[j].reserve(modeSequence.size());
    impactProximityTrajectories_[j].clear();
    impactProximityTrajectories_[j].reserve(modeSequence.size());
    swingWindows_[j].clear();
    swingWindows_[j].reserve(modeSequence.size());
    for (int p = 0; p < static_cast<int>(modeSequence.size()); ++p) {
      const int swingStartIndex = startTimesIndices[j][p];
      const int swingFinalIndex = finalTimesIndices[j][p];
      // A phase in contact, and a swing whose lift-off lies before the schedule
      // or whose touch-down lies after it - a contact out of contact at either
      // end, such as a hand that is placed for a while and lifted again, or one
      // the schedule never puts in contact: no swing is planned, and the
      // references are those of a contact in stance.
      if (eesContactFlagStocks[j][p] ||
          !isPlannableSwing(swingStartIndex, swingFinalIndex,
                            modeSequence.size())) {
        appendStancePhase(j, liftoffPosFeetRtWorldEwrtWorldZTraj[j][p]);
        continue;
      }

      const scalar_t swingStartTime = eventTimes[swingStartIndex];
      scalar_t swingFinalTime = eventTimes[swingFinalIndex];

      // A swing extended past its planned touch-down (GroundSearch): the spline
      // and the pitch profile are fitted to the planned swing, and the height
      // reference continues from its end as a straight descent.
      bool searching = false;
      const std::optional<GroundSearch>& search = groundSearches[j];
      if (search.has_value() &&
          std::abs(search->liftOffTime - swingStartTime) <=
              kSwingTimeTolerance &&
          search->plannedTouchDownTime > swingStartTime + kSwingTimeTolerance &&
          search->plannedTouchDownTime < swingFinalTime - kSwingTimeTolerance) {
        searching = true;
        swingFinalTime = search->plannedTouchDownTime;
      }

      // Every phase of a swing, which spans several phases in a gait with
      // flight phases (run: the swing of one foot is FLY | other foot in
      // stance | FLY), gets the same splines over the whole swing, so the
      // references are continuous at the inner events and have one apex and
      // one short-swing scaling per swing. The heights are read at the
      // swing's first and last phase so that every phase of the swing reads
      // the same ones.
      const scalar_t scaling = swingTrajectoryScaling(
          swingStartTime, swingFinalTime, config_.swingTimeScale);
      const scalar_t liftoffPosFootRtWorldEwrtWorldZ =
          liftoffPosFeetRtWorldEwrtWorldZTraj[j][swingStartIndex + 1];
      const scalar_t touchdownPosFootRtWorldEwrtWorldZ =
          touchdownPosFeetRtWorldEwrtWorldZTraj[j][swingFinalIndex];
      SwingWindow window;
      window.isSwing = true;
      window.startTime = swingStartTime;
      window.finalTime = swingFinalTime;
      window.scaling = scaling;
      window.searching = searching;
      window.descentStartPosFootRtWorldEwrtWorldZ =
          touchdownPosFootRtWorldEwrtWorldZ;
      window.descentSpeedFootRtWorld =
          searching ? std::max(0.0, search->descentSpeedFootRtWorld) : 0.0;
      swingWindows_[j].push_back(window);
      const SwingSplines splines =
          makeSwingSplines(window, liftoffPosFootRtWorldEwrtWorldZ,
                           touchdownPosFootRtWorldEwrtWorldZ);
      desPosFeetRtWorldEwrtWorldZTraj_[j].push_back(
          splines.desPosFootRtWorldEwrtWorldZ);
      impactProximityTrajectories_[j].push_back(splines.impactProximity);
    }
    feetHeightTrajectoriesEvents_[j] = eventTimes;
  }
}

void SwingTrajectoryPlanner::appendStancePhase(
    size_t contact, scalar_t posFootRtWorldEwrtWorldZ) {
  swingWindows_[contact].push_back(SwingWindow{});
  // Note: setting the time here arbitrarily to 0.0 -> 1.0 makes the
  // assert in CubicSpline fail
  const CubicSpline::Node liftOff{
      .time = 0.0, .value = posFootRtWorldEwrtWorldZ, .derivative = 0.0};
  const CubicSpline::Node touchDown{
      .time = 1.0, .value = posFootRtWorldEwrtWorldZ, .derivative = 0.0};
  desPosFeetRtWorldEwrtWorldZTraj_[contact].emplace_back(
      liftOff, posFootRtWorldEwrtWorldZ, touchDown);

  // If the foot is in contact the impact proximity factor is always 1.
  const CubicSpline::Node impactProximityTouchDown{
      .time = 0.0, .value = 1.0, .derivative = 0};
  const CubicSpline::Node impactProximityLiftOff{
      .time = 1.0, .value = 1.0, .derivative = 0};
  impactProximityTrajectories_[contact].emplace_back(
      impactProximityTouchDown, 1.0, impactProximityLiftOff);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::SwingSplines SwingTrajectoryPlanner::makeSwingSplines(
    const SwingWindow& window,
    scalar_t liftoffPosFootRtWorldEwrtWorldZ,
    scalar_t touchdownPosFootRtWorldEwrtWorldZ) const {
  const scalar_t scaling = window.scaling;
  // The apex is apexPosFootRtSwingendEwrtWorldZ above the higher end, so that a
  // step up taller than apexPosFootRtSwingendEwrtWorldZ still reaches the
  // touch-down height from above rather than from below.
  const scalar_t apexPosFootRtWorldEwrtWorldZ =
      std::max(liftoffPosFootRtWorldEwrtWorldZ,
               touchdownPosFootRtWorldEwrtWorldZ) +
      scaling * config_.apexPosFootRtSwingendEwrtWorldZ;
  const CubicSpline::Node liftoffNode{
      .time = window.startTime,
      .value = liftoffPosFootRtWorldEwrtWorldZ,
      .derivative = scaling * config_.liftoffLinvelFootRtWorldEwrtWorldZ};
  const CubicSpline::Node touchdownNode{
      .time = window.finalTime,
      .value = touchdownPosFootRtWorldEwrtWorldZ,
      .derivative = scaling * config_.touchdownLinvelFootRtWorldEwrtWorldZ};

  const CubicSpline::Node impactProximityLiftOff{
      .time = window.startTime,
      .value = 1.0,
      .derivative = scaling * config_.impactProximityFactorLiftoffSlope};
  const CubicSpline::Node impactProximityTouchDown{
      .time = window.finalTime,
      .value = 1.0,
      .derivative = scaling * config_.impactProximityFactorTouchdownSlope};
  return SwingSplines{
      .desPosFootRtWorldEwrtWorldZ =
          SplineCpg(liftoffNode, apexPosFootRtWorldEwrtWorldZ, touchdownNode),
      .impactProximity = SplineCpg(impactProximityLiftOff,
                                   config_.impactProximityFactorMidPointValue,
                                   impactProximityTouchDown),
  };
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::FootSwingEventIndices
SwingTrajectoryPlanner::updateFootSchedule(
    const std::vector<bool>& contactFlagStock) {
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
  return FootSwingEventIndices{
      .startTimeIndices = std::move(startTimeIndexStock),
      .finalTimeIndices = std::move(finalTimeIndexStock)};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ContactArray<std::vector<bool>> SwingTrajectoryPlanner::extractContactFlags(
    const std::vector<size_t>& phaseIDsStock) const {
  const size_t numPhases = phaseIDsStock.size();

  ContactArray<std::vector<bool>> contactFlagStock(
      numFeet_, std::vector<bool>(numPhases));

  for (size_t i = 0; i < numPhases; ++i) {
    const ContactMask contactFlag =
        ContactMask::fromModeNumber(phaseIDsStock[i], numFeet_);
    for (size_t j = 0; j < numFeet_; ++j) {
      contactFlagStock[j][i] = contactFlag[j];
    }
  }
  return contactFlagStock;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SwingTrajectoryPlanner::SwingEventIndices SwingTrajectoryPlanner::findIndex(
    size_t index, const std::vector<bool>& contactFlagStock) {
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

  return SwingEventIndices{.startTimeIndex = startTimesIndex,
                           .finalTimeIndex = finalTimesIndex};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool SwingTrajectoryPlanner::isPlannableSwing(int startIndex,
                                              int finalIndex,
                                              size_t numPhases) {
  return startIndex >= 0 && static_cast<size_t>(finalIndex) + 1 < numPhases;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwingTrajectoryPlanner::swingTrajectoryScaling(
    scalar_t startTime, scalar_t finalTime, scalar_t swingTimeScale) {
  return std::min(1.0, (finalTime - startTime) / swingTimeScale);
}

}  // namespace ocs2::humanoid
