/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

#include <iostream>
#include <stdexcept>

#include <boost/property_tree/ptree.hpp>

#include <ocs2_core/misc/LoadData.h>

namespace ocs2::humanoid {

void ContactPlanningConfig::validate() const {
  const auto fail = [](const std::string& what) { throw std::invalid_argument("[ContactPlanningConfig] " + what); };
  if (dt <= 0.0) fail("dt must be positive");
  if (numNodes < 2) fail("numNodes must be at least 2");
  if (comHeight <= 0.0 || gravity <= 0.0) fail("comHeight and gravity must be positive");
  if (minSwingDuration <= 0.0 || maxSwingDuration < minSwingDuration) fail("need 0 < minSwingDuration <= maxSwingDuration");
  if (minContactDuration <= 0.0) fail("minContactDuration must be positive");
  if (maxContactDuration > 0.0 && maxContactDuration < minContactDuration) fail("maxContactDuration must be >= minContactDuration");
  if (zmpHalfWidthX <= 0.0 || zmpHalfWidthY <= 0.0) fail("ZMP half widths must be positive");
  if (minStepWidth <= 0.0 || maxStepWidth < minStepWidth) fail("need 0 < minStepWidth <= maxStepWidth");
  if (nominalStepWidth < minStepWidth || nominalStepWidth > maxStepWidth) fail("nominalStepWidth must lie within the step width bounds");
  if (maxStepLength <= 0.0 || reachX <= 0.0) fail("maxStepLength and reachX must be positive");
  if (reachYOuter <= reachYInner) fail("reachYOuter must exceed reachYInner");
  if (bigM <= maxStepLength) fail("bigM must exceed maxStepLength");
  if (commitTime < 0.0) fail("commitTime must be non-negative");
  if (maxBranchAndBoundNodes < 1 || maxSolveTime <= 0.0 || maxQpIterations < 1) fail("invalid solver limits");
  if (localSearchIterations < 0 || localSearchMaxTime < 0.0) fail("invalid local search limits");
  if (planningFrequency <= 0.0) fail("planningFrequency must be positive");
  if (commitNodes() >= numNodes) fail("commitTime must be shorter than the planning horizon");
}

ContactPlanningConfig loadContactPlanningConfig(const std::string& taskFile, const std::string& prefix, bool verbose) {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile, pt);
  ContactPlanningConfig config;
  if (verbose) {
    std::cerr << "\n #### Contact Planning Config:";
    std::cerr << "\n #### =============================================================================\n";
  }
  // LINT.IfChange(contact_planning_keys)
  loadData::loadPtreeValue(pt, config.dt, prefix + "dt", verbose);
  loadData::loadPtreeValue(pt, config.numNodes, prefix + "numNodes", verbose);
  loadData::loadPtreeValue(pt, config.commitTime, prefix + "commitTime", verbose);
  loadData::loadPtreeValue(pt, config.comHeight, prefix + "comHeight", verbose);
  loadData::loadPtreeValue(pt, config.gravity, prefix + "gravity", verbose);
  loadData::loadPtreeValue(pt, config.minSwingDuration, prefix + "minSwingDuration", verbose);
  loadData::loadPtreeValue(pt, config.maxSwingDuration, prefix + "maxSwingDuration", verbose);
  loadData::loadPtreeValue(pt, config.minContactDuration, prefix + "minContactDuration", verbose);
  loadData::loadPtreeValue(pt, config.maxContactDuration, prefix + "maxContactDuration", verbose);
  loadData::loadPtreeValue(pt, config.enforceAlternatingFeet, prefix + "enforceAlternatingFeet", verbose);
  loadData::loadPtreeValue(pt, config.zmpHalfWidthX, prefix + "zmpHalfWidthX", verbose);
  loadData::loadPtreeValue(pt, config.zmpHalfWidthY, prefix + "zmpHalfWidthY", verbose);
  loadData::loadPtreeValue(pt, config.nominalStepWidth, prefix + "nominalStepWidth", verbose);
  loadData::loadPtreeValue(pt, config.minStepWidth, prefix + "minStepWidth", verbose);
  loadData::loadPtreeValue(pt, config.maxStepWidth, prefix + "maxStepWidth", verbose);
  loadData::loadPtreeValue(pt, config.maxStepLength, prefix + "maxStepLength", verbose);
  loadData::loadPtreeValue(pt, config.reachX, prefix + "reachX", verbose);
  loadData::loadPtreeValue(pt, config.reachYInner, prefix + "reachYInner", verbose);
  loadData::loadPtreeValue(pt, config.reachYOuter, prefix + "reachYOuter", verbose);
  loadData::loadPtreeValue(pt, config.bigM, prefix + "bigM", verbose);
  loadData::loadPtreeValue(pt, config.velocityTrackingWeight, prefix + "velocityTrackingWeight", verbose);
  loadData::loadPtreeValue(pt, config.zmpRegularizationWeight, prefix + "zmpRegularizationWeight", verbose);
  loadData::loadPtreeValue(pt, config.footholdRegularizationWeight, prefix + "footholdRegularizationWeight", verbose);
  loadData::loadPtreeValue(pt, config.stepWidthWeight, prefix + "stepWidthWeight", verbose);
  loadData::loadPtreeValue(pt, config.contactSwitchCost, prefix + "contactSwitchCost", verbose);
  loadData::loadPtreeValue(pt, config.terminalDcmWeight, prefix + "terminalDcmWeight", verbose);
  loadData::loadPtreeValue(pt, config.constraintSlackWeight, prefix + "constraintSlackWeight", verbose);
  loadData::loadPtreeValue(pt, config.constraintSlackLinearWeight, prefix + "constraintSlackLinearWeight", verbose);
  loadData::loadPtreeValue(pt, config.maxBranchAndBoundNodes, prefix + "maxBranchAndBoundNodes", verbose);
  loadData::loadPtreeValue(pt, config.maxSolveTime, prefix + "maxSolveTime", verbose);
  loadData::loadPtreeValue(pt, config.maxQpIterations, prefix + "maxQpIterations", verbose);
  loadData::loadPtreeValue(pt, config.localSearchIterations, prefix + "localSearchIterations", verbose);
  loadData::loadPtreeValue(pt, config.localSearchMaxTime, prefix + "localSearchMaxTime", verbose);
  loadData::loadPtreeValue(pt, config.verbose, prefix + "verbose", verbose);
  loadData::loadPtreeValue(pt, config.runInBackgroundThread, prefix + "runInBackgroundThread", verbose);
  loadData::loadPtreeValue(pt, config.planningFrequency, prefix + "planningFrequency", verbose);
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:contact_planning_config)
  if (verbose) {
    std::cerr << " #### =============================================================================" << std::endl;
  }
  config.validate();
  return config;
}

}  // namespace ocs2::humanoid
