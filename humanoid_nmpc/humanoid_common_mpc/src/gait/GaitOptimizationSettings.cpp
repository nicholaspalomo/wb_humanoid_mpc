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

#include "humanoid_common_mpc/gait/GaitOptimizationSettings.h"

#include <ocs2_core/misc/LoadData.h>
#include <iostream>
#include "absl/status/status.h"

namespace ocs2::humanoid {

absl::StatusOr<GaitOptimizationSettings> loadGaitOptimizationSettings(
    const std::string& filename, const std::string& fieldName, bool verbose) {
  loadData::PropertyTree pt;
  try {
    loadData::readPropertyTree(filename, pt);
  } catch (const std::exception& e) {
    return absl::NotFoundError("Failed to read property tree from file: " +
                               filename + " (" + e.what() + ")");
  }

  GaitOptimizationSettings settings;
  const bool hasField = loadData::containsPtreeValueFind(pt, fieldName);
  if (!hasField) {
    if (verbose) {
      std::cout << " #### GaitOptimizationSettings: [" << fieldName
                << "] not found in " << filename
                << ". Using defaults (enabled=false)." << std::endl;
    }
    return settings;
  }

  const std::string prefix = fieldName + ".";
  loadData::loadPtreeValue(pt, settings.enabled, prefix + "enabled", verbose);
  loadData::loadPtreeValue(pt, settings.enableTrajectorySensitivity,
                           prefix + "enableTrajectorySensitivity", verbose);
  loadData::loadPtreeValue(pt, settings.enableContactFeedback,
                           prefix + "enableContactFeedback", verbose);
  loadData::loadPtreeValue(pt, settings.minSingleSupportDuration,
                           prefix + "minSingleSupportDuration", verbose);
  loadData::loadPtreeValue(pt, settings.maxSingleSupportDuration,
                           prefix + "maxSingleSupportDuration", verbose);
  loadData::loadPtreeValue(pt, settings.minDoubleSupportDuration,
                           prefix + "minDoubleSupportDuration", verbose);
  loadData::loadPtreeValue(pt, settings.maxDoubleSupportDuration,
                           prefix + "maxDoubleSupportDuration", verbose);
  loadData::loadPtreeValue(pt, settings.earlyTouchDownTimeWindow,
                           prefix + "earlyTouchDownTimeWindow", verbose);
  loadData::loadPtreeValue(pt, settings.stepSize, prefix + "stepSize", verbose);
  loadData::loadPtreeValue(pt, settings.maxIterations, prefix + "maxIterations",
                           verbose);

  return settings;
}

}  // namespace ocs2::humanoid
