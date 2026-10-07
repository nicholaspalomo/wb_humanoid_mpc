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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "tools/config_dump/SettingsDump.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/integration/Integrator.h"
#include "ocs2_core/integration/SensitivityIntegrator.h"

namespace ocs2::humanoid::config_dump {
namespace {

void dumpOptionalDouble(ValueDump& dump, absl::string_view path, std::optional<double> value) {
  dump.addBool(absl::StrCat(path, ".has_value"), value.has_value());
  if (value.has_value()) {
    dump.addDouble(path, *value);
  }
}

}  // namespace

void dumpModelSettings(ValueDump& dump, absl::string_view path, const ModelSettings& settings) {
  dump.addString(joinPath(path, "robotName"), settings.robotName);
  dump.addBool(joinPath(path, "verboseCppAd"), settings.verboseCppAd);
  dump.addBool(joinPath(path, "recompileLibrariesCppAd"), settings.recompileLibrariesCppAd);
  dump.addString(joinPath(path, "modelFolderCppAd"), settings.modelFolderCppAd);
  dump.addDouble(joinPath(path, "phaseTransitionStanceTime"), settings.phaseTransitionStanceTime);
  dump.addStrings(joinPath(path, "fullJointNames"), settings.fullJointNames);
  dump.addStrings(joinPath(path, "fixedJointNames"), settings.fixedJointNames);
  dump.addStrings(joinPath(path, "contactNames6DoF"), settings.contactNames6DoF);
  dump.addStrings(joinPath(path, "contactNames3DoF"), settings.contactNames3DoF);
  dump.addStrings(joinPath(path, "contactParentJointNames"), settings.contactParentJointNames);
  dump.addStrings(joinPath(path, "mpcModelJointNames"), settings.mpcModelJointNames);
  dump.addSizes(joinPath(path, "mpcModelToFullJointsIndices"), settings.mpcModelToFullJointsIndices);
  std::vector<std::pair<std::string, size_t>> jointIndices(settings.jointIndexMap.begin(), settings.jointIndexMap.end());
  std::sort(jointIndices.begin(), jointIndices.end());
  for (const std::pair<std::string, size_t>& entry : jointIndices) {
    dump.addInteger(absl::StrCat(joinPath(path, "jointIndexMap"), "[", entry.first, "]"), static_cast<int64_t>(entry.second));
  }
  dump.addStrings(joinPath(path, "contactNames"), settings.contactNames);
  dump.addInteger(joinPath(path, "mpc_joint_dim"), static_cast<int64_t>(settings.mpc_joint_dim));
  dump.addInteger(joinPath(path, "full_joint_dim"), static_cast<int64_t>(settings.full_joint_dim));
  dump.addBool(joinPath(path, "hasArmSwingJoints"), settings.hasArmSwingJoints);
  dump.addInteger(joinPath(path, "j_l_shoulder_y_index"), static_cast<int64_t>(settings.j_l_shoulder_y_index));
  dump.addInteger(joinPath(path, "j_r_shoulder_y_index"), static_cast<int64_t>(settings.j_r_shoulder_y_index));
  dump.addInteger(joinPath(path, "j_l_elbow_y_index"), static_cast<int64_t>(settings.j_l_elbow_y_index));
  dump.addInteger(joinPath(path, "j_r_elbow_y_index"), static_cast<int64_t>(settings.j_r_elbow_y_index));
  const ModelSettings::FootConstraintConfig& foot = settings.footConstraintConfig;
  const std::string footPath = joinPath(path, "footConstraintConfig");
  dump.addDouble(joinPath(footPath, "positionErrorGain_z"), foot.positionErrorGain_z);
  dump.addDouble(joinPath(footPath, "orientationErrorGain"), foot.orientationErrorGain);
  dump.addDouble(joinPath(footPath, "linearVelocityErrorGain_z"), foot.linearVelocityErrorGain_z);
  dump.addDouble(joinPath(footPath, "linearVelocityErrorGain_xy"), foot.linearVelocityErrorGain_xy);
  dump.addDouble(joinPath(footPath, "angularVelocityErrorGain"), foot.angularVelocityErrorGain);
  dump.addDouble(joinPath(footPath, "linearAccelerationErrorGain_z"), foot.linearAccelerationErrorGain_z);
  dump.addDouble(joinPath(footPath, "linearAccelerationErrorGain_xy"), foot.linearAccelerationErrorGain_xy);
  dump.addDouble(joinPath(footPath, "angularAccelerationErrorGain"), foot.angularAccelerationErrorGain);
  dump.addDouble(joinPath(footPath, "softConstraintWeight"), foot.softConstraintWeight);
  dump.addDouble(joinPath(footPath, "normalVelocitySoftConstraintWeight"), foot.normalVelocitySoftConstraintWeight);
  dump.addBool(joinPath(footPath, "constrainOrientation"), foot.constrainOrientation);
  dump.addBool(joinPath(footPath, "constrainYawRateAboutContactNormal"), foot.constrainYawRateAboutContactNormal);
  const ModelSettings::ContactImplicitConfig& implicit = settings.contactImplicitConfig;
  const std::string implicitPath = joinPath(path, "contactImplicitConfig");
  dump.addDouble(joinPath(implicitPath, "complementarityWeight"), implicit.complementarityWeight);
  dump.addDouble(joinPath(implicitPath, "slipWeight"), implicit.slipWeight);
  dump.addDouble(joinPath(implicitPath, "heightReference"), implicit.heightReference);
  dump.addDouble(joinPath(implicitPath, "velocityReference"), implicit.velocityReference);
  dump.addDouble(joinPath(implicitPath, "angularVelocityReference"), implicit.angularVelocityReference);
  dump.addDouble(joinPath(implicitPath, "penetrationWeight"), implicit.penetrationWeight);
  dump.addDouble(joinPath(implicitPath, "gapSmoothing"), implicit.gapSmoothing);
  dump.addDouble(joinPath(path, "nominalFootholdConfig.stepWidth"), settings.nominalFootholdConfig.stepWidth);
  dump.addDouble(joinPath(path, "terrainHeight"), settings.terrainHeight);
}

void dumpSqpSettings(ValueDump& dump, absl::string_view path, const sqp::Settings& settings) {
  dump.addInteger(joinPath(path, "sqpIteration"), static_cast<int64_t>(settings.sqpIteration));
  dump.addDouble(joinPath(path, "deltaTol"), settings.deltaTol);
  dump.addDouble(joinPath(path, "costTol"), settings.costTol);
  dump.addDouble(joinPath(path, "alpha_decay"), settings.alpha_decay);
  dump.addDouble(joinPath(path, "alpha_min"), settings.alpha_min);
  dump.addDouble(joinPath(path, "g_max"), settings.g_max);
  dump.addDouble(joinPath(path, "g_min"), settings.g_min);
  dump.addDouble(joinPath(path, "armijoFactor"), settings.armijoFactor);
  dump.addDouble(joinPath(path, "gamma_c"), settings.gamma_c);
  dump.addBool(joinPath(path, "useFeedbackPolicy"), settings.useFeedbackPolicy);
  dump.addBool(joinPath(path, "createValueFunction"), settings.createValueFunction);
  dump.addDouble(joinPath(path, "dt"), settings.dt);
  dump.addString(joinPath(path, "integratorType"), sensitivity_integrator::toString(settings.integratorType));
  dump.addBool(joinPath(path, "projectStateInputEqualityConstraints"), settings.projectStateInputEqualityConstraints);
  dump.addBool(joinPath(path, "extractProjectionMultiplier"), settings.extractProjectionMultiplier);
  dump.addBool(joinPath(path, "printSolverStatus"), settings.printSolverStatus);
  dump.addBool(joinPath(path, "printSolverStatistics"), settings.printSolverStatistics);
  dump.addBool(joinPath(path, "printLinesearch"), settings.printLinesearch);
  dump.addBool(joinPath(path, "enableLogging"), settings.enableLogging);
  dump.addInteger(joinPath(path, "logSize"), static_cast<int64_t>(settings.logSize));
  dump.addString(joinPath(path, "logFilePath"), settings.logFilePath);
  dump.addInteger(joinPath(path, "nThreads"), static_cast<int64_t>(settings.nThreads));
  dump.addInteger(joinPath(path, "threadPriority"), settings.threadPriority);
  const hpipm_interface::Settings& hpipm = settings.hpipmSettings;
  const std::string hpipmPath = joinPath(path, "hpipm");
  dump.addInteger(joinPath(hpipmPath, "hpipmMode"), static_cast<int64_t>(hpipm.hpipmMode));
  dump.addInteger(joinPath(hpipmPath, "iter_max"), hpipm.iter_max);
  dump.addDouble(joinPath(hpipmPath, "alpha_min"), hpipm.alpha_min);
  dump.addDouble(joinPath(hpipmPath, "mu0"), hpipm.mu0);
  dump.addDouble(joinPath(hpipmPath, "tol_stat"), hpipm.tol_stat);
  dump.addDouble(joinPath(hpipmPath, "tol_eq"), hpipm.tol_eq);
  dump.addDouble(joinPath(hpipmPath, "tol_ineq"), hpipm.tol_ineq);
  dump.addDouble(joinPath(hpipmPath, "tol_comp"), hpipm.tol_comp);
  dump.addDouble(joinPath(hpipmPath, "reg_prim"), hpipm.reg_prim);
  dump.addInteger(joinPath(hpipmPath, "warm_start"), hpipm.warm_start);
  dump.addInteger(joinPath(hpipmPath, "pred_corr"), hpipm.pred_corr);
  dump.addInteger(joinPath(hpipmPath, "ric_alg"), hpipm.ric_alg);
}

void dumpRolloutSettings(ValueDump& dump, absl::string_view path, const rollout::Settings& settings) {
  dump.addDouble(joinPath(path, "absTolODE"), settings.absTolODE);
  dump.addDouble(joinPath(path, "relTolODE"), settings.relTolODE);
  dump.addInteger(joinPath(path, "maxNumStepsPerSecond"), static_cast<int64_t>(settings.maxNumStepsPerSecond));
  dump.addDouble(joinPath(path, "timeStep"), settings.timeStep);
  dump.addString(joinPath(path, "integratorType"), integrator_type::toString(settings.integratorType));
  dump.addBool(joinPath(path, "checkNumericalStability"), settings.checkNumericalStability);
  dump.addBool(joinPath(path, "reconstructInputTrajectory"), settings.reconstructInputTrajectory);
  dump.addInteger(joinPath(path, "rootFindingAlgorithm"), static_cast<int64_t>(settings.rootFindingAlgorithm));
  dump.addInteger(joinPath(path, "maxSingleEventIterations"), settings.maxSingleEventIterations);
  dump.addBool(joinPath(path, "useTrajectorySpreadingController"), settings.useTrajectorySpreadingController);
}

void dumpMpcSettings(ValueDump& dump, absl::string_view path, const mpc::Settings& settings) {
  dump.addDouble(joinPath(path, "timeHorizon"), settings.timeHorizon_);
  dump.addDouble(joinPath(path, "solutionTimeWindow"), settings.solutionTimeWindow_);
  dump.addBool(joinPath(path, "debugPrint"), settings.debugPrint_);
  dump.addBool(joinPath(path, "coldStart"), settings.coldStart_);
  dump.addDouble(joinPath(path, "mpcDesiredFrequency"), settings.mpcDesiredFrequency_);
  dump.addDouble(joinPath(path, "mrtDesiredFrequency"), settings.mrtDesiredFrequency_);
}

void dumpSwingSettings(ValueDump& dump, absl::string_view path, const SwingTrajectoryPlanner::Config& settings) {
  dump.addDouble(joinPath(path, "liftOffVelocity"), settings.liftOffVelocity);
  dump.addDouble(joinPath(path, "touchDownVelocity"), settings.touchDownVelocity);
  dump.addDouble(joinPath(path, "swingHeight"), settings.swingHeight);
  dump.addDouble(joinPath(path, "swingTimeScale"), settings.swingTimeScale);
  dump.addDouble(joinPath(path, "touchDownHeightOffset"), settings.touchDownHeightOffset);
  dump.addDouble(joinPath(path, "impactProximityFactorLiftOffVelocity"), settings.impactProximityFactorLiftOffVelocity);
  dump.addDouble(joinPath(path, "impactProximityFactorTouchDownVelocity"), settings.impactProximityFactorTouchDownVelocity);
  dump.addDouble(joinPath(path, "impactProximityFactorMidPointValue"), settings.impactProximityFactorMidPointValue);
  dump.addDouble(joinPath(path, "swingPitchAngle"), settings.swingPitchAngle);
  dump.addDouble(joinPath(path, "swingPitchRiseFraction"), settings.swingPitchRiseFraction);
  dump.addDouble(joinPath(path, "swingPitchFallFraction"), settings.swingPitchFallFraction);
}

void dumpModeSchedule(ValueDump& dump, absl::string_view path, const ModeSchedule& schedule) {
  dump.addDoubles(joinPath(path, "eventTimes"), schedule.eventTimes);
  dump.addSizes(joinPath(path, "modeSequence"), schedule.modeSequence);
}

void dumpGaitMap(ValueDump& dump, absl::string_view path, const std::map<std::string, ModeSequenceTemplate>& gaits) {
  dump.addInteger(joinPath(path, "size"), static_cast<int64_t>(gaits.size()));
  for (const std::pair<const std::string, ModeSequenceTemplate>& gait : gaits) {
    const std::string gaitPath = joinPath(path, gait.first);
    dump.addDoubles(joinPath(gaitPath, "switchingTimes"), gait.second.switchingTimes);
    dump.addSizes(joinPath(gaitPath, "modeSequence"), gait.second.modeSequence);
  }
}

void dumpTargetTrajectories(ValueDump& dump, absl::string_view path, const TargetTrajectories& targets) {
  dump.addDoubles(joinPath(path, "timeTrajectory"), targets.timeTrajectory);
  for (size_t i = 0; i < targets.stateTrajectory.size(); ++i) {
    dump.addMatrix(absl::StrCat(joinPath(path, "stateTrajectory"), "[", i, "]"), targets.stateTrajectory[i]);
  }
  for (size_t i = 0; i < targets.inputTrajectory.size(); ++i) {
    dump.addMatrix(absl::StrCat(joinPath(path, "inputTrajectory"), "[", i, "]"), targets.inputTrajectory[i]);
  }
}

void dumpJointPdGains(ValueDump& dump, absl::string_view path, const JointPdGains& gains) {
  dump.addMatrix(joinPath(path, "mpcJointKp"), gains.mpcJointKp);
  dump.addMatrix(joinPath(path, "mpcJointKd"), gains.mpcJointKd);
  dump.addMatrix(joinPath(path, "mpcJointTorqueLimit"), gains.mpcJointTorqueLimit);
  dump.addMatrix(joinPath(path, "otherJointKp"), gains.otherJointKp);
  dump.addMatrix(joinPath(path, "otherJointKd"), gains.otherJointKd);
  dump.addMatrix(joinPath(path, "otherJointTorqueLimit"), gains.otherJointTorqueLimit);
  dump.addDouble(joinPath(path, "defaults.kp"), gains.defaults.kp);
  dump.addDouble(joinPath(path, "defaults.kd"), gains.defaults.kd);
  dumpOptionalDouble(dump, joinPath(path, "defaults.torqueLimit"), gains.defaults.torqueLimit);
}

void dumpRobotProcessSettings(ValueDump& dump, absl::string_view path, const RobotProcessSettings& settings) {
  dump.addString(joinPath(path, "contactEstimator"), settings.contactEstimator);
  const std::string simulatorPath = joinPath(path, "simulator");
  dump.addDouble(joinPath(simulatorPath, "contactForceThreshold"), settings.simulator.contactForceThreshold);
  dump.addDouble(joinPath(simulatorPath, "contactTimelineWindow"), settings.simulator.contactTimelineWindow);
  dump.addBool(joinPath(simulatorPath, "visualizations.has_value"), settings.simulator.visualizations.has_value());
  if (settings.simulator.visualizations.has_value()) {
    dump.addStrings(joinPath(simulatorPath, "visualizations"), *settings.simulator.visualizations);
  }
  dump.addString(joinPath(simulatorPath, "gantryHold"), settings.simulator.gantryHold);
  dump.addString(joinPath(simulatorPath, "projectile"), settings.simulator.projectile);
  const SimFallRecovery::Config& fall = settings.fallRecovery;
  const std::string fallPath = joinPath(path, "fallRecovery");
  dump.addDouble(joinPath(fallPath, "maxBaseTiltAngle"), fall.maxBaseTiltAngle);
  dump.addDouble(joinPath(fallPath, "catchLift"), fall.catchLift);
  dump.addDouble(joinPath(fallPath, "gantryHeightRate"), fall.gantryHeightRate);
  dump.addDouble(joinPath(fallPath, "settleTilt"), fall.settleTilt);
  dump.addDouble(joinPath(fallPath, "settleLinearSpeed"), fall.settleLinearSpeed);
  dump.addDouble(joinPath(fallPath, "settleAngularSpeed"), fall.settleAngularSpeed);
  dump.addDouble(joinPath(fallPath, "settleJointError"), fall.settleJointError);
  dump.addDouble(joinPath(fallPath, "settleHoldTime"), fall.settleHoldTime);
  dump.addDouble(joinPath(fallPath, "settleTimeout"), fall.settleTimeout);
  dumpOptionalDouble(dump, joinPath(path, "mpcEntryBlendTime"), settings.mpcEntryBlendTime);
  dumpOptionalDouble(dump, joinPath(path, "safetyDecayTimeConstant"), settings.safetyDecayTimeConstant);
  dump.addBool(joinPath(path, "contactWrenchGate.has_value"), settings.contactWrenchGate.has_value());
  if (settings.contactWrenchGate.has_value()) {
    dump.addDouble(joinPath(path, "contactWrenchGate.debounceTime"), settings.contactWrenchGate->debounceTime);
    dump.addDouble(joinPath(path, "contactWrenchGate.rampTime"), settings.contactWrenchGate->rampTime);
  }
  dump.addString(joinPath(path, "wbMpcFeedforward"), wbMpcFeedforwardName(settings.wbMpcFeedforward));
  dump.addStrings(joinPath(path, "telemetrySinks"), settings.telemetrySinks);
  dumpOptionalDouble(dump, joinPath(path, "telemetryFrequency"), settings.telemetryFrequency);
  dump.addDouble(joinPath(path, "mpcLinkPolicyTimeout"), settings.mpcLinkPolicyTimeout);
}

void dumpVisualizationConfig(ValueDump& dump, absl::string_view path, const visualization::VisualizationConfig& config) {
  dump.addDouble(joinPath(path, "sceneFrequency"), config.sceneFrequency);
  dump.addBool(joinPath(path, "sceneFrequencyIsDefault"), config.sceneFrequencyIsDefault);
  dump.addStrings(joinPath(path, "telemetryFrames"), config.telemetryFrames);
  dump.addStrings(joinPath(path, "planFrames"), config.planFrames);
}

void dumpKeyboardCommandLimits(ValueDump& dump, absl::string_view path, const teleop::KeyboardCommandLimits& limits) {
  dump.addMatrix(joinPath(path, "limits"), limits.limits);
  dump.addDouble(joinPath(path, "defaultBaseHeight"), limits.defaultBaseHeight);
}

}  // namespace ocs2::humanoid::config_dump
