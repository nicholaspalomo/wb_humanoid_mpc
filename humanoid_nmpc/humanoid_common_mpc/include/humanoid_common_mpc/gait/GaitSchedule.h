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

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {

/**
 * Returns the gait mode layout of the contact set of `modelSettings`: its
 * number of contacts, and as its stance mode kBipedModeStance (the two feet,
 * the last two contacts) on the biped and on a set without legs that has two
 * or more 6-DoF contacts, the last two of which are the feet, or every contact
 * active otherwise (a legged set that is not the biped, such as a quadruped on
 * four 6-DoF soles, or a set of point contacts); its legs (ModelSettings::legs,
 * each with the OR of its contacts' mode bits) and its contact names. The
 * shipped robots and the DRC Atlas multi-contact set therefore keep STANCE = 3.
 */
GaitModeLayout gaitModeLayoutOf(const ModelSettings& modelSettings);

/**
 * The mode schedule of the MPC: an initial schedule, extended over the horizon
 * by tiling the current mode sequence template (the gait), into which a new
 * gait is inserted from a given time on.
 *
 * The stance phase it inserts between gaits, appends after the last one and
 * starts the schedule with is the stance mode of its layout
 * (gaitModeLayoutOf()).
 *
 * Not thread-safe: SwitchedModelReferenceManager and ProceduralMpcMotionManager
 * share it, and both use it on the solver thread.
 */
class GaitSchedule {
 public:
  /** `modeLayout` is the contact set the modes are of; its stanceMode is the
   * stance phase this schedule inserts. */
  GaitSchedule(ModeSchedule initModeSchedule,
               ModeSequenceTemplate initModeSequenceTemplate,
               scalar_t phaseTransitionStanceTime,
               GaitModeLayout modeLayout);

  // Shared by the reference manager and the gait updaters through one
  // std::shared_ptr; a copy would fork the schedule.
  GaitSchedule(const GaitSchedule&) = delete;
  GaitSchedule& operator=(const GaitSchedule&) = delete;
  ~GaitSchedule() = default;

  /**
   * @param [in] lowerBoundTime: The smallest time for which the ModeSchedule
   * should be defined.
   * @param [in] upperBoundTime: The greatest time for which the ModeSchedule
   * should be defined.
   */
  ModeSchedule getModeSchedule(scalar_t lowerBoundTime,
                               scalar_t upperBoundTime);

  ModeSchedule getCurrentModeSchedule() const { return modeSchedule_; }

  /** The contact set the modes of this schedule are of, for reading gait files
   * against it (parseModeNumber()). */
  const GaitModeLayout& modeLayout() const { return modeLayout_; }

  /**
   * Used to insert a new user defined logic in the given time period.
   *
   * @param [in] startTime: The initial time from which the new mode sequence
   * template should start.
   * @param [in] finalTime: The final time until when the new mode sequence
   * needs to be defined.
   */
  void insertModeSequenceTemplate(
      const ModeSequenceTemplate& modeSequenceTemplate,
      scalar_t startTime,
      scalar_t finalTime);

  /**
   * The schedule of a typed reference file: its initial_mode_schedule
   * (initialModeScheduleFromConfig()), extended with its
   * default_mode_sequence_template (defaultModeSequenceTemplateFromConfig()),
   * both read against gaitModeLayoutOf(modelSettings), and fails like those.
   */
  static absl::StatusOr<std::shared_ptr<GaitSchedule>> Create(
      const mpc_config::ReferenceFile& referenceFile,
      const ModelSettings& modelSettings,
      bool verbose = false);

  /**
   * The schedule above of the reference file at `referenceFile`
   * (loadReferenceFile()): the path form of a root of the MPC's configuration.
   *
   * @return loadReferenceFile()'s error for a file that cannot be read or does
   * not parse, and the typed Create()'s errors, prefixed with the file.
   */
  static absl::StatusOr<std::shared_ptr<GaitSchedule>> Create(
      absl::string_view referenceFile,
      const ModelSettings& modelSettings,
      bool verbose = false);

  void updateModeSchedule(const ModeSchedule& modeSchedule);

  /**
   * Puts the schedule and the template back to the ones this schedule was
   * constructed with, i.e. the reference file's initial_mode_schedule and
   * default_mode_sequence_template, both STANCE on every shipped robot. Every
   * gait inserted since is forgotten, including its events in the future: after
   * a reset of the controller, or a clock that ran backwards, those would hold
   * the robot in whatever the old schedule said until the clock caught up with
   * them. The next getModeSchedule() then answers exactly as it would on a
   * freshly constructed schedule.
   */
  void reset();

 private:
  /**
   * Extends the switch information from lowerBoundTime to upperBoundTime based
   * on the template mode sequence.
   *
   * @param [in] startTime: The initial time from which the mode schedule should
   * be appended with the template.
   * @param [in] finalTime: The final time to which the mode schedule should be
   * appended with the template.
   */
  void tileModeSequenceTemplate(scalar_t startTime, scalar_t finalTime);

  // What the schedule was constructed with, for reset().
  const ModeSchedule initModeSchedule_;
  const ModeSequenceTemplate initModeSequenceTemplate_;

  ModeSchedule modeSchedule_;
  ModeSequenceTemplate modeSequenceTemplate_;
  scalar_t phaseTransitionStanceTime_;
  GaitModeLayout modeLayout_;
};

}  // namespace ocs2::humanoid
