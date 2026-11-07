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

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/swing_foot_planner/SplineCpg.h"

namespace ocs2::humanoid {

/**
 * Plans the swing-foot height, pitch and impact-proximity references of every
 * contact of the set over the mode schedule: update() rebuilds them from a
 * schedule, and the getters evaluate them at a time. Contact i is bit
 * numFeet - 1 - i of a mode number (ContactMask), so the planner covers any
 * contact count up to kMaxNumContacts; a planner of two feet reads a biped's
 * FLY, RF, LF and STANCE modes, and sees every other number as both feet in
 * swing. A swing that begins before the schedule or ends after it (a contact
 * the schedule never puts in contact, such as a hand of a set whose gait walks
 * on the feet alone, or one placed for a while and lifted again) has no planned
 * swing: its references are those of a contact in stance.
 *
 * Not thread-safe: SwitchedModelReferenceManager updates it on the solver
 * thread before each solve, and the costs read it during the solve.
 */
class SwingTrajectoryPlanner {
 public:
  struct Config {
    /** The defaults of the fields below, by field. */
    static constexpr scalar_t kNomApexPosFootRtSwingendEwrtWorldZ = 0.1;  // [m]
    static constexpr scalar_t kDefaultSwingTimeScale = 0.15;              // [s]
    static constexpr scalar_t kDefaultImpactProximityFactorMidPointValue = 0.1;
    static constexpr scalar_t kDefaultSwingPitchRiseFraction = 0.25;
    static constexpr scalar_t kDefaultSwingPitchFallFraction = 0.15;

    scalar_t liftoffLinvelFootRtWorldEwrtWorldZ = 0.0;
    scalar_t touchdownLinvelFootRtWorldEwrtWorldZ = 0.0;
    scalar_t apexPosFootRtSwingendEwrtWorldZ =
        kNomApexPosFootRtSwingendEwrtWorldZ;  // [m] apex above the higher of
                                              // the lift-off and touch-down
                                              // heights
    scalar_t swingTimeScale =
        kDefaultSwingTimeScale;  // swing phases shorter than this time will be
                                 // scaled down in height and velocity
    scalar_t touchdownPosFootRtSurfaceEwrtWorldZ = 0.0;

    scalar_t impactProximityFactorLiftoffSlope = 0;  // should lesser or equal 0
    scalar_t impactProximityFactorTouchdownSlope =
        0;  // should be greater or equal to 0
    scalar_t impactProximityFactorMidPointValue =
        kDefaultImpactProximityFactorMidPointValue;  // should be between 0 and
                                                     // 1

    // Toe-up pitch of the swing foot. The foot height reference is tracked at
    // the center of the sole, so the toe, half a foot length ahead of it,
    // reaches the ground before the reference does whenever the foot droops
    // toe-down. Pitching the reference toe-up through the swing lifts the toe
    // by (half foot length) * sin(angle) at no cost in sole clearance. The
    // pitch ramps up over the first swingPitchRiseFraction of the swing, is
    // held, and is back to flat at touch-down so that the contact is made with
    // the whole sole, as the contact model assumes.
    // [rad] The peak toe-up tilt from the ground, a pitch about the foot's
    // lateral axis; 0 (the default) reproduces the flat-foot reference.
    scalar_t swingMaxTiltangleFootRtSurface = 0.0;
    // The fraction of the swing spent ramping from flat up to
    // swingMaxTiltangleFootRtSurface.
    scalar_t swingPitchRiseFraction = kDefaultSwingPitchRiseFraction;
    // The fraction of the swing spent ramping back down to flat before
    // touch-down.
    scalar_t swingPitchFallFraction = kDefaultSwingPitchFallFraction;
  };

  /**
   * A swing extended past its planned touch-down while the foot searches for
   * the ground (late touch-down). The height reference of that swing is the
   * planned swing's spline up to the planned touch-down, continued as a
   * straight descent at `descentSpeedFootRtWorld` from the planned touch-down
   * height until the extended touch-down of the schedule; the pitch and the
   * impact proximity hold their touch-down values. Re-fitting the spline over
   * the extended swing instead moved its apex, raised the reference at the
   * current time by several millimeters at every extension step and then drove
   * the foot down at the spline's slope (about 0.35 m/s) rather than at the
   * search velocity.
   */
  struct GroundSearch {
    scalar_t liftOffTime = 0.0;  // identifies the swing in the schedule
    scalar_t plannedTouchDownTime =
        0.0;  // where the planned swing, and with it the spline, ends
    scalar_t descentSpeedFootRtWorld =
        0.0;  // [m/s] downward rate of the height reference past the planned
              // touch-down (>= 0)
  };

  /** A planner of `numFeet` contacts (ModelSettings::contactNames.size()),
   * at most kMaxNumContacts. */
  SwingTrajectoryPlanner(Config config, size_t numFeet);

  // Copyable, not assignable: the number of feet is fixed at construction.
  SwingTrajectoryPlanner(const SwingTrajectoryPlanner&) = default;
  SwingTrajectoryPlanner& operator=(const SwingTrajectoryPlanner&) = delete;
  SwingTrajectoryPlanner(SwingTrajectoryPlanner&&) = default;
  SwingTrajectoryPlanner& operator=(SwingTrajectoryPlanner&&) = delete;
  ~SwingTrajectoryPlanner() = default;

  void update(const ModeSchedule& modeSchedule,
              scalar_t posSurfaceRtWorldEwrtWorldZ);

  /** Rebuilds the references from `modeSchedule` with per-phase lift-off and
   * touch-down heights per contact; each array holds at least the planner's
   * number of feet, each entry one height per phase of the schedule. */
  void update(
      const ModeSchedule& modeSchedule,
      const ContactArray<scalar_array_t>& liftoffPosFeetRtWorldEwrtWorldZTraj,
      const ContactArray<scalar_array_t>&
          touchdownPosFeetRtWorldEwrtWorldZTraj);

  /** As above, with the swings that are searching for the ground (empty
   * entries: none). */
  void update(
      const ModeSchedule& modeSchedule,
      const ContactArray<scalar_array_t>& liftoffPosFeetRtWorldEwrtWorldZTraj,
      const ContactArray<scalar_array_t>& touchdownPosFeetRtWorldEwrtWorldZTraj,
      const ContactArray<std::optional<GroundSearch>>& groundSearches);

  /** The number of contacts the planner plans, fixed at construction. */
  size_t numFeet() const { return numFeet_; }

  scalar_t getDesLinaccFootRtWorldEwrtWorldZ(size_t leg, scalar_t time) const;

  scalar_t getDesLinvelFootRtWorldEwrtWorldZ(size_t leg, scalar_t time) const;

  scalar_t getDesPosFootRtWorldEwrtWorldZ(size_t leg, scalar_t time) const;

  scalar_t getImpactProximityFactor(size_t leg, scalar_t time) const;

  /**
   * Toe-up pitch of the swing foot reference at the given time, in radians,
   * positive toe-up (a rotation of -angle about the foot's lateral axis). Zero
   * for a foot in contact, and zero at both ends of a swing.
   */
  scalar_t getSwingRefTiltangleFootRtSurface(size_t leg, scalar_t time) const;

  /** Update swing trajectory config at runtime (e.g. from online parameter
   * tuning). */
  void setConfig(const Config& config) { config_ = config; }
  const Config& getConfig() const { return config_; }

 private:
  /**
   * Extracts for each leg the contact sequence over the motion phase sequence.
   * @param phaseIDsStock
   * @return contactFlagStock
   */
  ContactArray<std::vector<bool>> extractContactFlags(
      const std::vector<size_t>& phaseIDsStock) const;

  /**
   * The event-time indices that bound one swing phase:
   * eventTimes[startTimeIndex] is its take-off and eventTimes[finalTimeIndex]
   * its touch-down. Both are 0 for a stance phase; startTimeIndex is -1 for a
   * swing that began before the schedule and finalTimeIndex the last phase for
   * one that ends after it (isPlannableSwing()).
   */
  struct SwingEventIndices {
    int startTimeIndex = 0;
    int finalTimeIndex = 0;
  };

  /** The SwingEventIndices of every phase of one foot, by phase. */
  struct FootSwingEventIndices {
    std::vector<int> startTimeIndices;
    std::vector<int> finalTimeIndices;
  };

  /**
   * Finds the take-off and touch-down time indices of phase `index` of a foot
   * whose contact status per phase is `contactFlagStock`.
   */
  static SwingEventIndices findIndex(size_t index,
                                     const std::vector<bool>& contactFlagStock);

  /**
   * based on the input phaseIDsStock finds the start subsystem and final
   * subsystem of the swing phases of the a foot in each subsystem.
   *
   * startTimeIndexStock: eventTimes[startTimesIndex] will be the take-off time
   * for the requested leg. finalTimeIndexStock: eventTimes[finalTimesIndex]
   * will be the touch-down time for the requested leg.
   *
   * @param [in] footIndex: Foot index
   * @param [in] phaseIDsStock: The sequence of the motion phase IDs.
   * @param [in] contactFlagStock: The sequence of the contact status for the
   * requested leg.
   * @return the take-off and touch-down time indices of every phase.
   */
  static FootSwingEventIndices updateFootSchedule(
      const std::vector<bool>& contactFlagStock);

  /**
   * Whether the swing phase whose lift-off and touch-down are the event-time
   * indices `startIndex` and `finalIndex` (findIndex()) lies within a schedule
   * of `numPhases` phases: its lift-off is an event of the schedule (not -1)
   * and its touch-down is followed by a phase in contact.
   */
  static bool isPlannableSwing(int startIndex,
                               int finalIndex,
                               size_t numPhases);

  static scalar_t swingTrajectoryScaling(scalar_t startTime,
                                         scalar_t finalTime,
                                         scalar_t swingTimeScale);

  /** Normalized toe-up pitch profile: 0 at lift-off, 1 over the held section, 0
   * at touch-down. */
  scalar_t swingPitchProfile(scalar_t tau) const;

  /** The swing this mode-schedule phase belongs to, so that the pitch can be
   * evaluated on the swing's own time base. */
  struct SwingWindow {
    bool isSwing = false;
    scalar_t startTime = 0.0;
    scalar_t finalTime = 0.0;  // the planned touch-down: the spline and the
                               // pitch profile end here
    scalar_t scaling = 1.0;
    // Ground search past finalTime (GroundSearch): a straight descent from the
    // planned touch-down height.
    bool searching = false;
    scalar_t descentStartPosFootRtWorldEwrtWorldZ = 0.0;
    scalar_t descentSpeedFootRtWorld = 0.0;
  };

  /** The height and impact-proximity references of one swing. */
  struct SwingSplines {
    SplineCpg desPosFootRtWorldEwrtWorldZ;
    SplineCpg impactProximity;
  };

  /**
   * Builds the references of the swing `window` (its planned lift-off and
   * touch-down times and short-swing scaling) from the lift-off height to the
   * touch-down height, through an apex apexPosFootRtSwingendEwrtWorldZ (scaled)
   * above the higher of the two.
   */
  SwingSplines makeSwingSplines(
      const SwingWindow& window,
      scalar_t liftoffPosFootRtWorldEwrtWorldZ,
      scalar_t touchdownPosFootRtWorldEwrtWorldZ) const;

  /** Appends to contact `contact`'s references one phase in stance at the
   * height `posFootRtWorldEwrtWorldZ` [m]: a flat height and an impact
   * proximity of 1. */
  void appendStancePhase(size_t contact, scalar_t posFootRtWorldEwrtWorldZ);

  /** True while the foot is searching for the ground past its planned
   * touch-down at `time`. */
  static bool isSearching(const SwingWindow& window, scalar_t time) {
    return window.isSwing && window.searching && time > window.finalTime;
  }

  Config config_;
  const size_t numFeet_;

  // One entry per contact (numFeet_ of them), each a per-phase vector over the
  // schedule; every vector is empty until the first update().
  ContactArray<std::vector<SwingWindow>> swingWindows_;
  ContactArray<std::vector<SplineCpg>> impactProximityTrajectories_;
  ContactArray<std::vector<SplineCpg>> desPosFeetRtWorldEwrtWorldZTraj_;
  ContactArray<std::vector<scalar_t>> feetHeightTrajectoriesEvents_;
};

}  // namespace ocs2::humanoid
