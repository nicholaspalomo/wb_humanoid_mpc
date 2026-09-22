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

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"

#include <boost/property_tree/ptree.hpp>

#include <cmath>
#include <stdexcept>

#include <ocs2_core/misc/LoadData.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

using boost::property_tree::ptree;

/**
 * The entries of a YAML sequence under `key`, or an empty list when the key is absent, null, or an empty sequence.
 *
 * All three spell "no heuristic of this kind" and the task files use all three: a block whose entries are all
 * commented out parses as a null scalar, not as an empty sequence, and that is exactly how these lists ship. Treating
 * them differently would make commenting out the last name of a list a different configuration from deleting it.
 */
/** True for the "[0]", "[1]", ... keys loadData::yamlToPropertyTree() gives the children of a YAML sequence. */
bool isSequenceIndex(absl::string_view key) {
  if (key.size() < 3 || key.front() != '[' || key.back() != ']') return false;
  for (size_t i = 1; i + 1 < key.size(); ++i) {
    if (!absl::ascii_isdigit(key[i])) return false;
  }
  return true;
}

absl::StatusOr<std::vector<std::string>> readList(const ptree& block, absl::string_view key) {
  std::vector<std::string> list;
  const boost::optional<const ptree&> child = block.get_child_optional(std::string(key));
  if (!child) return list;
  // A YAML sequence converts to a node with children and NO data of its own, a scalar to one with data and no
  // children, and a map to one with children whose keys are the map's. All three used to read as "no heuristic",
  // which is right only for the first (and for a null, which is what a list whose entries are all commented out
  // parses as). The other two are a mistake in the file, and a silently inert configuration is exactly the failure
  // this subsystem refuses to ship elsewhere.
  if (!child->data().empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.", key, " must be a list of heuristic names, but is the scalar '",
                     child->data(), "'. Write it as a YAML sequence: `", key, ":\n    - <name>`, or `", key, ": []` for none."));
  }
  for (const std::pair<const std::string, ptree>& item : *child) {
    // loadData::yamlToPropertyTree() keys a sequence's children "[0]", "[1]", ...; a map's children keep their own
    // keys. Anything that is not an index is therefore a map where a list belongs.
    if (!isSequenceIndex(item.first)) {
      return absl::InvalidArgumentError(absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.", key,
                                                     " must be a list of heuristic names, but is a map (it has the key '", item.first,
                                                     "'). Write it as a YAML sequence: `", key, ":\n    - <name>`."));
    }
    const std::string value = item.second.data();
    if (!value.empty()) list.push_back(value);
  }
  return list;
}

/**
 * Reads one scalar of the block into `target`, leaving it untouched when the key is absent.
 *
 * A free function rather than a lambda because a lambda would have to be held in an `auto` variable, which this
 * repository does not use outside make_unique / make_shared.
 */
void loadScalar(const ptree& pt, absl::string_view prefix, scalar_t& target, absl::string_view key, bool verbose) {
  loadData::loadPtreeValue(pt, target, absl::StrCat(prefix, key), verbose);
}

/** Rejects a parameter that is outside the range its formula is defined on. */
absl::Status requirePositive(scalar_t value, absl::string_view key) {
  if (value <= 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.", key, " must be positive, got ", value, "."));
  }
  return absl::OkStatus();
}

absl::Status requireNonNegative(scalar_t value, absl::string_view key) {
  if (value < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.", key, " must not be negative, got ", value, "."));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status LocomotionHeuristicConfig::validate() const {
  RETURN_IF_ERROR(formulation.validate());

  // Only the keys a wrong value would make MEANINGLESS are checked, and only against the range their own formula
  // needs. Every coefficient of Table C.2 is deliberately absent from this list: a fitted number has no admissible
  // range at all and is as legitimately negative as positive, which is the whole point of fitting it rather than
  // designing it.
  RETURN_IF_ERROR(requirePositive(orientationCompensation.maximumTilt, "orientation_compensation.maximumTilt"));
  RETURN_IF_ERROR(requireNonNegative(heightCompensation.maximumHeightOffset, "height_compensation.maximumHeightOffset"));
  RETURN_IF_ERROR(requireNonNegative(capturePoint.comHeightOverride, "capture_point.comHeightOverride"));
  RETURN_IF_ERROR(requirePositive(capturePoint.gravity, "capture_point.gravity"));
  // Positive, not merely non-negative: this is the clamp that stops a bad velocity estimate throwing the landing
  // target out of the leg's reach, and a value of zero would remove it rather than tighten it.
  RETURN_IF_ERROR(requirePositive(capturePoint.maximumOffset, "capture_point.maximumOffset"));
  RETURN_IF_ERROR(requireNonNegative(hipCenteredStepping.lateralScale, "hip_centered_stepping.lateralScale"));
  // A negative blend would invert the heuristic rather than reduce it: the reference would move AWAY from Bledt's
  // value as the knob is turned down. Both scales are therefore floors at zero.
  RETURN_IF_ERROR(requireNonNegative(impulseScaling.scale, "impulse_scaling.scale"));
  RETURN_IF_ERROR(requireNonNegative(centripetalAcceleration.scale, "centripetal_acceleration.scale"));
  RETURN_IF_ERROR(requireNonNegative(centripetalAcceleration.maximumForce, "centripetal_acceleration.maximumForce"));
  RETURN_IF_ERROR(
      requireNonNegative(centripetalAcceleration.maximumForceRatioOfWeight, "centripetal_acceleration.maximumForceRatioOfWeight"));

  // beta is a fraction of a gait cycle, so a clamp outside (0, 1] cannot be reached, and one at 0 lets 1/beta run
  // away at the onset of a flight phase - which is the failure this clamp exists to prevent.
  if (impulseScaling.minimumDutyFactor <= 0.0 || impulseScaling.minimumDutyFactor > 1.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.impulse_scaling.minimumDutyFactor must lie in (0, 1], got ",
                     impulseScaling.minimumDutyFactor,
                     ". It is the floor the stance duty factor is clamped to before W / (F beta) is formed, and that is unbounded "
                     "as a flight phase opens."));
  }
  if (impulseScaling.maximumForceRatio < 1.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicConfig] locomotion_heuristics.impulse_scaling.maximumForceRatio must be at least 1, got ",
                     impulseScaling.maximumForceRatio,
                     ". It clamps the scaled reference as a multiple of weight compensation, and a value below 1 would ask the "
                     "stance feet to carry less than the robot weighs even while standing."));
  }
  // `maximumForce: 0` is the documented way to say "use the ratio instead", so exactly one of the two has to be
  // positive; both at zero would remove the clamp entirely rather than select a default.
  if (centripetalAcceleration.maximumForce <= 0.0 && centripetalAcceleration.maximumForceRatioOfWeight <= 0.0) {
    return absl::InvalidArgumentError(
        "[LocomotionHeuristicConfig] locomotion_heuristics.centripetal_acceleration needs a force clamp: set "
        "maximumForce to a positive value in newtons, or leave it at 0 and set maximumForceRatioOfWeight to a "
        "positive fraction of body weight. With both at zero nothing bounds a horizontal force reference that no "
        "friction cone would admit.");
  }
  return absl::OkStatus();
}

absl::StatusOr<LocomotionHeuristicConfig> loadLocomotionHeuristicConfig(absl::string_view taskFile, bool verbose) {
  LocomotionHeuristicConfig config;
  const std::string taskFilePath(taskFile);
  ptree pt;
  try {
    loadData::readPropertyTree(taskFilePath, pt);
  } catch (const std::exception& exception) {
    return absl::NotFoundError(absl::StrCat("[LocomotionHeuristicConfig] failed to read '", taskFilePath, "': ", exception.what()));
  }

  const ptree& constPt = pt;
  const boost::optional<const ptree&> block = constPt.get_child_optional(kLocomotionHeuristicsBlockKey);
  if (!block) {
    // A task file without the block is a robot that has not been given the layer, and the default configuration is an
    // exact no-op. This is deliberately silent even when verbose: it is the state of every robot in the repository
    // until someone opts one of them in, and a start-up warning that fires on every launch of every robot is noise.
    return config;
  }

  const std::string prefix = absl::StrCat(kLocomotionHeuristicsBlockKey, ".");
  // loadPtreeValue leaves its destination untouched when the key is absent, so every struct member initializer IS the
  // default and a partially written block inherits the rest. Same contract as ModelSettings and ContactPlanningConfig.

  LocomotionHeuristicFormulation& formulation = config.formulation;
  ASSIGN_OR_RETURN(formulation.basePose, readList(*block, "base_pose"));
  ASSIGN_OR_RETURN(formulation.foothold, readList(*block, "foothold"));
  ASSIGN_OR_RETURN(formulation.wrench, readList(*block, "wrench"));

  // loadPtreeValue() catches only ptree_bad_path - the missing-key case that makes every scalar optional - so a key
  // that IS present but does not parse as a number throws ptree_bad_data straight through it. That would escape this
  // StatusOr as an exception from a function whose whole contract is to return a status, and the caller
  // (CentroidalMpcInterface::setupLocomotionHeuristics) would surface it as a crash rather than as the file-and-key
  // message the operator needs. A typo like `rollOffset: 0.0.1` is an ordinary mistake, not an exceptional one.
  try {
    // LINT.IfChange(locomotion_heuristic_keys)
    loadScalar(pt, prefix, config.orientationCompensation.rollPerLateralVelocity, "orientation_compensation.rollPerLateralVelocity",
               verbose);
    loadScalar(pt, prefix, config.orientationCompensation.rollOffset, "orientation_compensation.rollOffset", verbose);
    loadScalar(pt, prefix, config.orientationCompensation.pitchPerForwardVelocity, "orientation_compensation.pitchPerForwardVelocity",
               verbose);
    loadScalar(pt, prefix, config.orientationCompensation.pitchOffset, "orientation_compensation.pitchOffset", verbose);
    loadScalar(pt, prefix, config.orientationCompensation.maximumTilt, "orientation_compensation.maximumTilt", verbose);

    loadScalar(pt, prefix, config.periodicOrientation.rollAmplitude, "periodic_orientation.rollAmplitude", verbose);
    loadScalar(pt, prefix, config.periodicOrientation.rollPhaseRate, "periodic_orientation.rollPhaseRate", verbose);
    loadScalar(pt, prefix, config.periodicOrientation.rollPhaseOffset, "periodic_orientation.rollPhaseOffset", verbose);
    loadScalar(pt, prefix, config.periodicOrientation.pitchAmplitude, "periodic_orientation.pitchAmplitude", verbose);
    loadScalar(pt, prefix, config.periodicOrientation.pitchPhaseRate, "periodic_orientation.pitchPhaseRate", verbose);
    loadScalar(pt, prefix, config.periodicOrientation.pitchPhaseOffset, "periodic_orientation.pitchPhaseOffset", verbose);

    loadScalar(pt, prefix, config.heightCompensation.heightPerSpeedSquared, "height_compensation.heightPerSpeedSquared", verbose);
    loadScalar(pt, prefix, config.heightCompensation.heightPerSpeed, "height_compensation.heightPerSpeed", verbose);
    loadScalar(pt, prefix, config.heightCompensation.heightOffset, "height_compensation.heightOffset", verbose);
    loadScalar(pt, prefix, config.heightCompensation.maximumHeightOffset, "height_compensation.maximumHeightOffset", verbose);

    loadScalar(pt, prefix, config.hipCenteredStepping.lateralScale, "hip_centered_stepping.lateralScale", verbose);
    loadScalar(pt, prefix, config.hipCenteredStepping.longitudinalScale, "hip_centered_stepping.longitudinalScale", verbose);

    loadScalar(pt, prefix, config.capturePoint.gain, "capture_point.gain", verbose);
    loadScalar(pt, prefix, config.capturePoint.comHeightOverride, "capture_point.comHeightOverride", verbose);
    loadScalar(pt, prefix, config.capturePoint.gravity, "capture_point.gravity", verbose);
    loadScalar(pt, prefix, config.capturePoint.maximumOffset, "capture_point.maximumOffset", verbose);

    loadScalar(pt, prefix, config.translationalStepping.forwardPerForwardVelocity, "translational_stepping.forwardPerForwardVelocity",
               verbose);
    loadScalar(pt, prefix, config.translationalStepping.forwardOffset, "translational_stepping.forwardOffset", verbose);
    loadScalar(pt, prefix, config.translationalStepping.lateralPerLateralVelocity, "translational_stepping.lateralPerLateralVelocity",
               verbose);
    loadScalar(pt, prefix, config.translationalStepping.lateralOffset, "translational_stepping.lateralOffset", verbose);

    loadScalar(pt, prefix, config.inPlaceTurning.forwardPerYawRate, "in_place_turning.forwardPerYawRate", verbose);
    loadScalar(pt, prefix, config.inPlaceTurning.forwardOffset, "in_place_turning.forwardOffset", verbose);
    loadScalar(pt, prefix, config.inPlaceTurning.lateralPerYawRate, "in_place_turning.lateralPerYawRate", verbose);
    loadScalar(pt, prefix, config.inPlaceTurning.lateralOffset, "in_place_turning.lateralOffset", verbose);

    loadScalar(pt, prefix, config.highSpeedTurning.forwardPerCrossTerm, "high_speed_turning.forwardPerCrossTerm", verbose);
    loadScalar(pt, prefix, config.highSpeedTurning.forwardOffset, "high_speed_turning.forwardOffset", verbose);
    loadScalar(pt, prefix, config.highSpeedTurning.lateralPerCrossTerm, "high_speed_turning.lateralPerCrossTerm", verbose);
    loadScalar(pt, prefix, config.highSpeedTurning.lateralOffset, "high_speed_turning.lateralOffset", verbose);

    loadScalar(pt, prefix, config.impulseScaling.scale, "impulse_scaling.scale", verbose);
    loadScalar(pt, prefix, config.impulseScaling.minimumDutyFactor, "impulse_scaling.minimumDutyFactor", verbose);
    loadScalar(pt, prefix, config.impulseScaling.maximumForceRatio, "impulse_scaling.maximumForceRatio", verbose);

    loadScalar(pt, prefix, config.centripetalAcceleration.scale, "centripetal_acceleration.scale", verbose);
    loadScalar(pt, prefix, config.centripetalAcceleration.maximumForce, "centripetal_acceleration.maximumForce", verbose);
    loadScalar(pt, prefix, config.centripetalAcceleration.maximumForceRatioOfWeight, "centripetal_acceleration.maximumForceRatioOfWeight",
               verbose);
    // Every robot's task file carries this block, so a key added above has to be documented in each of them - the
    // failure the directive pair exists to prevent is the one the contact planner's own comment records, where one
    // robot's file was left to drift out of sync with the loader unnoticed.
    // clang-format off
    // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:locomotion_heuristics_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:locomotion_heuristics_config, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:locomotion_heuristics_config, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml:locomotion_heuristics_config)
    // clang-format on

  } catch (const std::exception& exception) {
    return absl::InvalidArgumentError(absl::StrCat("[LocomotionHeuristicConfig] a value in the locomotion_heuristics block of '",
                                                   taskFilePath, "' could not be read as a number: ", exception.what()));
  }

  RETURN_IF_ERROR(config.validate());

  if (verbose) {
    LOG(INFO) << "\n #### Locomotion Heuristics (Bledt RPC, Appendix C) loaded from: " << taskFilePath;
    LOG(INFO) << "\n" << config.formulation.summary();
  }
  return config;
}

}  // namespace ocs2::humanoid
