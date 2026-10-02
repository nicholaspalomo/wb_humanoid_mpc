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

#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/Numerics.h>
#include <ocs2_core/misc/PropertyTree.h>
#include <ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_sqp/SqpSettings.h>
#include <ocs2_sqp/SqpSolver.h>

#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactMomentXYConstraintCppAd.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

namespace {

/**
 * Reads the optional scalar `key` of `pt` into `value`, leaving it untouched when the file does not carry the key. A
 * value that does not parse as a T is an InvalidArgument naming the key, and `value` is left untouched then too.
 */
template <typename T>
absl::Status loadOptionalValue(const PropertyTree& pt, absl::string_view key, T& value) {
  const PropertyTree* child = pt.findChild(key);
  if (child == nullptr) {
    return absl::OkStatus();
  }
  const std::optional<T> parsed = child->getValueOptional<T>();
  if (!parsed.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(key, " is '", child->data(), "', which is not a value of the expected type."));
  }
  value = *parsed;
  return absl::OkStatus();
}

/**
 * Loads a weight matrix out of an already-parsed property tree.
 *
 * loadData::loadEigenMatrix re-reads and re-parses the whole task file on every
 * call. This runs on the solver thread inside preSolverRun, at the rate the
 * operator drags a slider, so the tree is parsed once by the caller and reused.
 *
 * Mirrors loadEigenMatrix's semantics: a `scaling` key multiplies every entry, a
 * `default` key fills the entries that are absent.
 *
 * @return NotFound when the file carries no entry of the matrix at all; InvalidArgument naming the key when `scaling`,
 *         `default` or an entry does not parse as a number - loadEigenMatrix used to read those with a `get` that
 *         silently substituted its default, so a mistyped weight became 1 or 0 on the running problem. `matrix` is then
 *         partly written and must not be applied.
 */
absl::Status loadEigenMatrixFromPtree(const PropertyTree& pt, const std::string& matrixName, matrix_t& matrix) {
  scalar_t scaling = 1.0;
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(matrixName, ".scaling"), scaling));
  scalar_t defaultValue = 0.0;
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(matrixName, ".default"), defaultValue));

  Eigen::Index numFound = 0;
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      const std::string key = absl::StrCat(matrixName, ".(", i, ",", j, ")");
      const bool present = pt.findChild(key) != nullptr;
      scalar_t entry = defaultValue;
      RETURN_IF_ERROR(loadOptionalValue(pt, key, entry));
      matrix(i, j) = scaling * entry;
      numFound += present ? 1 : 0;
    }
  }
  if (numFound == 0) {
    return absl::NotFoundError(absl::StrCat("[MpcParameterUpdaterModule] the task file carries no entry of the matrix ", matrixName, "."));
  }
  return absl::OkStatus();
}

/**
 * Whether the running problem regulates the base pose through ComAndAcomTrackingCost, i.e. whether it was built from a
 * task file listing `com_and_acom_tracking_cost`. That is decided once, when the problem is assembled, and a hot
 * reload cannot change it - so this, and not the file being applied, is what decides whether the base-pose blocks of
 * a reloaded Q and Q_final are zeroed.
 */
bool runsComAndAcomTrackingCost(const OptimalControlProblem& ocp) {
  return ocp.stateCostPtr != nullptr && ocp.stateCostPtr->getTermNameMap().count(std::string(ComAndAcomTrackingCost::kRunningTermName)) > 0;
}

// The terminal cost terms CentroidalMpcInterface::setupOptimalControlProblem adds: one of the two, or neither.
// LINT.IfChange(terminal_cost_term_names)
constexpr const char* kQuadraticTerminalCostTerm = "terminalCost";
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/src/CentroidalMpcInterface.cpp:quadratic_terminal_cost_term)
constexpr const char* kDcmTerminalCostTerm = DcmTerminalCost::kTermName;

/** Whether the RUNNING problem ends on the final-cost term `termName`, the same way for every worker (they are clones). */
bool runsFinalCostTerm(const OptimalControlProblem& ocp, absl::string_view termName) {
  return ocp.finalCostPtr != nullptr && ocp.finalCostPtr->getTermNameMap().count(std::string(termName)) > 0;
}

/**
 * The cost, constraint and formulation lists of a task file are structural: they decide which terms the problem was
 * assembled from, and a hot reload applies none of them. Reports a reloaded file whose lists the start-up loader would
 * refuse (a retired key such as `useComAndAcomTracking` or `useDcmTerminalCost` among them), whose `costs` list
 * disagrees with the running problem about `com_and_acom_tracking_cost`, whose choice of terminal cost
 * (`dcm_terminal_cost` or `terminal_cost` in `costs`) disagrees with the one the running problem ends on, whose contact
 * input parameterization differs from the running one, or whose contactScheduleSource differs from the running one, so
 * that the operator knows the edit waits for a restart.
 */
void warnAboutStructuralFormulationChanges(const std::string& yamlFile,
                                           bool runsComAndAcomTracking,
                                           bool runsBasisVectorInputs,
                                           bool runsDcmTerminalCost,
                                           bool runsContactPlanner) {
  // The contact input parameterization fixes the input dimension, so a reload cannot switch it either; nor the retired
  // boolean it replaced, which the start-up loader refuses.
  const absl::StatusOr<ContactInputParameterization> contactInputs = loadContactInputParameterization(yamlFile);
  if (!contactInputs.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] the contact input parameterization of " << yamlFile
                 << " would be refused at start-up, and it is not hot-reloaded: " << contactInputs.status().message();
  } else if ((*contactInputs == ContactInputParameterization::kBasisVectors) != runsBasisVectorInputs) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] " << kContactInputParameterizationKey << " is "
                 << contactInputParameterizationName(*contactInputs) << " in " << yamlFile << ", but the running problem uses "
                 << (runsBasisVectorInputs ? kBasisVectorsContactInputParameterization : kWrenchContactInputParameterization)
                 << ". It fixes the input dimension, so it takes effect at the next start.";
  }

  // The contact schedule source decides which reference manager the problem was built on; nor can a reload bring back
  // the retired boolean it replaced.
  const absl::StatusOr<ContactScheduleSource> contactScheduleSource = loadContactScheduleSource(yamlFile);
  if (!contactScheduleSource.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] the contact schedule source of " << yamlFile
                 << " would be refused at start-up, and it is not hot-reloaded: " << contactScheduleSource.status().message();
  } else if ((*contactScheduleSource == ContactScheduleSource::kContactPlanner) != runsContactPlanner) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] " << kContactScheduleSourceKey << " is "
                 << contactScheduleSourceName(*contactScheduleSource) << " in " << yamlFile
                 << ", but the running problem takes its contact schedule from "
                 << (runsContactPlanner ? kContactPlannerContactScheduleSource : kGaitScheduleContactScheduleSource)
                 << ". It is structural and takes effect at the next start.";
  }

  const absl::StatusOr<MpcFormulationTasks> formulation = loadMpcFormulationTasks(yamlFile, /*verbose=*/false);
  if (!formulation.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] the task lists of " << yamlFile
                 << " would be refused at start-up, and the lists are not hot-reloaded, so the running problem keeps its own: "
                 << formulation.status().message();
    return;
  }
  const bool listed = formulation->hasCost(MpcCostType::ComAndAcomTrackingCost);
  if (listed != runsComAndAcomTracking) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] com_and_acom_tracking_cost is " << (listed ? "listed" : "not listed")
                 << " in the costs of " << yamlFile << ", but the running problem was built "
                 << (runsComAndAcomTracking ? "with" : "without")
                 << " it. The costs list is structural and takes effect at the next start; until then the base-pose blocks of Q "
                    "and Q_final follow the running problem ("
                 << (runsComAndAcomTracking ? "zeroed" : "live") << ").";
  }
  // The same selection CentroidalMpcInterface::setupOptimalControlProblem makes.
  const bool fileSelectsDcmTerminalCost = formulation->hasCost(MpcCostType::DcmTerminalCost);
  if (fileSelectsDcmTerminalCost != runsDcmTerminalCost) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] " << yamlFile << " ends the horizon on "
                 << (fileSelectsDcmTerminalCost ? "the DCM terminal cost (dcm_terminal_cost in the costs)"
                                                : "the quadratic terminal cost (Q_final), not on dcm_terminal_cost")
                 << ", but the running problem ends on "
                 << (runsDcmTerminalCost ? "the DCM terminal cost" : "the quadratic terminal cost (Q_final), or none")
                 << ". The terminal cost is structural and changes at the next start; until then a reload applies Q_final and "
                    "dcm_terminal_cost to whichever of the two the running problem carries.";
  }
}

/**
 * Reads the hot-reloadable `mu` and `delta` of a barrier section. Returns false, leaving both untouched, when the file
 * does not carry the section, so that the apply step can leave the running value alone rather than apply a default.
 * A value that is not a number is logged as a warning naming its key and returns false as well: the running barrier is
 * kept, and the exception it used to throw no longer escapes the reload (out of preSolverRun, on a file-watch reload).
 */
bool loadBarrierSection(const PropertyTree& pt, const std::string& section, scalar_t& mu, scalar_t& delta) {
  if (pt.findChild(section) == nullptr) {
    return false;
  }
  scalar_t reloadedMu = mu;
  scalar_t reloadedDelta = delta;
  absl::Status status = loadOptionalValue(pt, absl::StrCat(section, ".mu"), reloadedMu);
  if (status.ok()) {
    status = loadOptionalValue(pt, absl::StrCat(section, ".delta"), reloadedDelta);
  }
  if (!status.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] the " << section
                 << " barrier was not applied, the running one is kept: " << status.message();
    return false;
  }
  mu = reloadedMu;
  delta = reloadedDelta;
  return true;
}

/**
 * The first leaf under `section` of a reloaded file whose value is neither a number nor a bool, as `section.key is
 * 'value'`, or an empty string. Sequence entries (keyed `[i]`, a list of joint names for instance) are not values and are
 * skipped. A diagnostic only: a library loader that reads a value by its path throws a PropertyTreeBadData naming the
 * key, but one that converts a node it reached by walking the tree (getValue), or throws an exception of its own, names
 * none, and this is what lets the warning name the key the operator has to fix whichever loader refused the section.
 */
std::string firstUnparsableLeaf(const PropertyTree& block, const std::string& path) {
  for (const PropertyTree::value_type& child : block) {
    if (!child.first.empty() && child.first.front() == '[') {
      continue;
    }
    const std::string childPath = absl::StrCat(path, ".", child.first);
    if (!child.second.empty()) {
      const std::string nested = firstUnparsableLeaf(child.second, childPath);
      if (!nested.empty()) {
        return nested;
      }
      continue;
    }
    const std::string& value = child.second.data();
    if (value.empty() || child.second.getValueOptional<scalar_t>().has_value() || child.second.getValueOptional<bool>().has_value()) {
      continue;
    }
    return absl::StrCat(childPath, " is '", value, "'");
  }
  return std::string();
}

/**
 * Runs `load`, which reads the section `section` of the reloaded file through a loader of the library (the one the start-up
 * path uses, so that the two cannot read different keys), and says whether there is something to apply.
 *
 * Returns false without a word when the file does not carry the section: those loaders give every key they do not find
 * its library default rather than throwing, so a file without the section - a hand-written partial file, say - used to
 * apply zero foot weights, a zero ICP weight or a default swing trajectory to the running problem. Returns false with a
 * WARNING naming the section and, where firstUnparsableLeaf() can tell, the key, when the loader throws: that used to be
 * swallowed by an empty catch, so a mistyped weight was simply not applied and nothing said so.
 */
bool loadSection(const PropertyTree& pt, const std::string& section, const std::function<void()>& load) {
  const PropertyTree* block = pt.findChild(section);
  if (block == nullptr) {
    return false;
  }
  try {
    load();
    return true;
  } catch (const std::exception& e) {
    const std::string unparsable = firstUnparsableLeaf(*block, section);
    LOG(WARNING) << "[MpcParameterUpdaterModule] " << section << " was not applied, the running values are kept: "
                 << (unparsable.empty() ? std::string() : absl::StrCat(unparsable, ", which is not a number: ")) << e.what();
    return false;
  }
}

/**
 * The `contact_implicit` block of a reloaded file: its values, over ModelSettings' defaults for the keys it does not
 * carry, and which fields it does carry - only those are applied.
 */
struct ContactImplicitReload {
  ModelSettings::ContactImplicitConfig config;
  std::vector<scalar_t ModelSettings::ContactImplicitConfig::*> carried;

  bool carries(scalar_t ModelSettings::ContactImplicitConfig::*field) const {
    return std::find(carried.begin(), carried.end(), field) != carried.end();
  }
};

/**
 * Reads the `contact_implicit` block of a reloaded file through the key list ModelSettings loads it with, and checks it
 * the way the start-up path does: a key that list does not know (a renamed or misspelled one) is refused by
 * checkContactImplicitBlockKeys(), and the values by validateContactImplicitConfig(), each with a message that names the
 * key. Returns nullopt when the file carries no such block.
 */
absl::StatusOr<std::optional<ContactImplicitReload>> loadContactImplicitReload(const PropertyTree& pt) {
  if (pt.findChild(ModelSettings::kContactImplicitBlock) == nullptr) {
    return std::optional<ContactImplicitReload>();
  }
  RETURN_IF_ERROR(checkContactImplicitBlockKeys(pt));
  ContactImplicitReload reload;
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    const std::string path = absl::StrCat(ModelSettings::kContactImplicitBlock, ".", key.name);
    if (pt.findChild(path) == nullptr) {
      continue;
    }
    RETURN_IF_ERROR(loadOptionalValue(pt, path, reload.config.*key.field));
    reload.carried.push_back(key.field);
  }
  RETURN_IF_ERROR(validateContactImplicitConfig(reload.config));
  return std::optional<ContactImplicitReload>(std::move(reload));
}

/**
 * The foot-constraint gains of a reloaded file (the `model_settings.foot_constraint` section) into `gains`, which keeps
 * its value for every key the file does not carry. A value that does not parse is an InvalidArgument naming its key;
 * `gains` is then partly written and must not be applied. It used to be read with loadPtreeValue, which throws on such
 * a value - out of preSolverRun on a file-watch reload.
 */
absl::Status loadFootConstraintGains(const PropertyTree& pt, ModelSettings::FootConstraintConfig& gains) {
  const std::string prefix = "model_settings.foot_constraint.";
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "positionErrorGain_z"), gains.positionErrorGain_z));
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "orientationErrorGain"), gains.orientationErrorGain));
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "linearVelocityErrorGain_z"), gains.linearVelocityErrorGain_z));
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "linearVelocityErrorGain_xy"), gains.linearVelocityErrorGain_xy));
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "angularVelocityErrorGain"), gains.angularVelocityErrorGain));
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "linearAccelerationErrorGain_z"), gains.linearAccelerationErrorGain_z));
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "linearAccelerationErrorGain_xy"), gains.linearAccelerationErrorGain_xy));
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "angularAccelerationErrorGain"), gains.angularAccelerationErrorGain));
  RETURN_IF_ERROR(loadOptionalValue(pt, absl::StrCat(prefix, "constrainOrientation"), gains.constrainOrientation));
  RETURN_IF_ERROR(
      loadOptionalValue(pt, absl::StrCat(prefix, "constrainYawRateAboutContactNormal"), gains.constrainYawRateAboutContactNormal));
  return absl::OkStatus();
}

/**
 * The hot-reloadable subset of the SQP settings (the `multiple_shooting` section) into `settings`, which keeps its value
 * for every key the file does not carry. The same refusal as loadFootConstraintGains().
 */
absl::Status loadSqpSettingsUpdates(const PropertyTree& pt, sqp::Settings& settings) {
  size_t sqpIteration = settings.sqpIteration;
  RETURN_IF_ERROR(loadOptionalValue(pt, "multiple_shooting.sqpIteration", sqpIteration));
  settings.sqpIteration = sqpIteration;
  RETURN_IF_ERROR(loadOptionalValue(pt, "multiple_shooting.deltaTol", settings.deltaTol));
  RETURN_IF_ERROR(loadOptionalValue(pt, "multiple_shooting.g_max", settings.g_max));
  RETURN_IF_ERROR(loadOptionalValue(pt, "multiple_shooting.g_min", settings.g_min));
  RETURN_IF_ERROR(loadOptionalValue(pt, "multiple_shooting.inequalityConstraintMu", settings.inequalityConstraintMu));
  RETURN_IF_ERROR(loadOptionalValue(pt, "multiple_shooting.inequalityConstraintDelta", settings.inequalityConstraintDelta));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> MpcParameterUpdaterModule::Create(
    MPC_BASE* mpcPtr,
    const std::string& taskFile,
    const std::string& urdfFile,
    const std::string& referenceFile,
    size_t stateDim,
    size_t inputDim,
    const std::vector<std::string>& contactNames,
    SwitchedModelReferenceManager* referenceManager,
    std::optional<BasisInputsCostTransformConfig> basisCostTransform) {
  if (basisCostTransform.has_value()) {
    // The transformed R is written straight into the OCP with setGains(), which performs no size check. Catch a
    // wrench-vs-basis dimension mix-up here, at construction, instead of silently corrupting the input cost online.
    const BasisInputsCostTransformConfig& cfg = *basisCostTransform;
    if (inputDim != cfg.basisInputDim()) {
      return absl::InvalidArgumentError(absl::StrCat("[MpcParameterUpdaterModule] inputDim (", inputDim,
                                                     ") must equal the basis-space input dimension of the cost transform (",
                                                     cfg.basisInputDim(), ")."));
    }
    if (static_cast<size_t>(cfg.basisToWrenchMap.rows()) != cfg.wrenchInputDim) {
      return absl::InvalidArgumentError(absl::StrCat("[MpcParameterUpdaterModule] basisToWrenchMap has ", cfg.basisToWrenchMap.rows(),
                                                     " rows but wrenchInputDim is ", cfg.wrenchInputDim, "."));
    }
    if (cfg.numBasisInputs > inputDim) {
      return absl::InvalidArgumentError(
          absl::StrCat("[MpcParameterUpdaterModule] numBasisInputs (", cfg.numBasisInputs, ") exceeds inputDim (", inputDim, ")."));
    }
    RETURN_IF_ERROR(validateBasisInputsCostTransformConfig(cfg));
  }
  return std::unique_ptr<MpcParameterUpdaterModule>(new MpcParameterUpdaterModule(
      mpcPtr, taskFile, urdfFile, referenceFile, stateDim, inputDim, contactNames, referenceManager, std::move(basisCostTransform)));
}

MpcParameterUpdaterModule::MpcParameterUpdaterModule(MPC_BASE* mpcPtr,
                                                     const std::string& taskFile,
                                                     const std::string& urdfFile,
                                                     const std::string& referenceFile,
                                                     size_t stateDim,
                                                     size_t inputDim,
                                                     const std::vector<std::string>& contactNames,
                                                     SwitchedModelReferenceManager* referenceManager,
                                                     std::optional<BasisInputsCostTransformConfig> basisCostTransform)
    : mpcPtr_(mpcPtr),
      taskFile_(taskFile),
      urdfFile_(urdfFile),
      referenceFile_(referenceFile),
      contactPlanningFile_(taskFile.empty() ? std::string() : resolveContactPlanningConfigFile(taskFile)),
      stateDim_(stateDim),
      inputDim_(inputDim),
      contactNames_(contactNames),
      referenceManagerPtr_(referenceManager),
      basisCostTransform_(std::move(basisCostTransform)) {
  if (!taskFile_.empty() && std::filesystem::exists(taskFile_)) {
    std::error_code ec;
    taskFileLastWriteTime_ = std::filesystem::last_write_time(taskFile_, ec);
  }
  if (!referenceFile_.empty() && std::filesystem::exists(referenceFile_)) {
    std::error_code ec;
    referenceFileLastWriteTime_ = std::filesystem::last_write_time(referenceFile_, ec);
  }
  if (contactPlanningFile_ != taskFile_ && std::filesystem::exists(contactPlanningFile_)) {
    std::error_code ec;
    contactPlanningFileLastWriteTime_ = std::filesystem::last_write_time(contactPlanningFile_, ec);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MpcParameterUpdaterModule::preSolverRun(scalar_t initTime,
                                             scalar_t finalTime,
                                             const vector_t& currentState,
                                             const ReferenceManagerInterface& referenceManager) {
  // First, before any reload: the reference manager has just rebuilt this solve's references, so this is the moment the
  // contact-implicit terms can take the ground those references were built on.
  followAppliedTerrainHeight();

  // Pathway 1: an update enqueued by enqueueParameterUpdate() (takes priority — no file I/O needed for the source YAML)
  if (hasPendingUpdate_.load(std::memory_order_acquire)) {
    std::string yamlContent;
    {
      absl::MutexLock lock(&pendingMutex_);
      yamlContent = std::move(pendingYamlText_);
      hasPendingUpdate_.store(false, std::memory_order_release);  // NOLINT(argument-comment): libstdc++ names the value __i.
    }
    // Write to a temp file so we can reuse the existing loadData parsing pipeline: the loaders of the cost, constraint,
    // swing and planner sections the update reaches read their sections from a file path, so parsing the text with
    // loadData::readPropertyTreeFromString() would serve only the tree read here, not them.
    // Keep the .yaml extension: readPropertyTree reads only .yaml and .yml files.
    const std::string tempFile = absl::StrCat(taskFile_, ".live.yaml");
    try {
      std::ofstream ofs(tempFile, std::ios::trunc);
      if (ofs.is_open()) {
        ofs << yamlContent;
        ofs.close();
        applyParameterUpdates(tempFile);
      } else {
        LOG(ERROR) << "[MpcParameterUpdaterModule] Failed to write temp file: " << tempFile;
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "[MpcParameterUpdaterModule] Error writing temp file: " << e.what();
    }
  }

  // Pathway 2: Check the modification times of task.yaml and of the planner's own file at roughly 1Hz (assuming the
  // solver runs around 100Hz). The planner's file is watched separately only when it is a file of its own; a
  // `contact_planning` block inside the task file is applied by applyParameterUpdates() with the rest.
  if (!taskFile_.empty() && checkCounter_++ % 100 == 0) {
    std::error_code ec;
    const std::filesystem::file_time_type lastWrite = std::filesystem::last_write_time(taskFile_, ec);
    // Every read of a reloaded value reports a value it cannot use by its key and keeps the running one; this is the
    // last line of defense, as on the enqueued path above, so that nothing a saved file contains can throw out of the
    // solver's pre-solve hook.
    if (!ec && lastWrite != taskFileLastWriteTime_) {
      taskFileLastWriteTime_ = lastWrite;
      try {
        applyParameterUpdates(taskFile_);
      } catch (const std::exception& e) {
        LOG(ERROR) << "[MpcParameterUpdaterModule] the reload of " << taskFile_ << " stopped part-way: " << e.what();
      }
    }
    if (contactPlanningFile_ != taskFile_) {
      const std::filesystem::file_time_type planningLastWrite = std::filesystem::last_write_time(contactPlanningFile_, ec);
      if (!ec && planningLastWrite != contactPlanningFileLastWriteTime_) {
        contactPlanningFileLastWriteTime_ = planningLastWrite;
        try {
          applyContactPlanningUpdates(contactPlanningFile_);
        } catch (const std::exception& e) {
          LOG(ERROR) << "[MpcParameterUpdaterModule] the reload of " << contactPlanningFile_ << " stopped part-way: " << e.what();
        }
      }
    }
    // The command limits and ramps live in reference.yaml, which nothing else watches: without this a change to it
    // needed a restart of the controller (the Command Limits tab of the remote control writes exactly this file).
    if (!referenceFile_.empty() && !referenceFileReloaders_.empty()) {
      const std::filesystem::file_time_type referenceLastWrite = std::filesystem::last_write_time(referenceFile_, ec);
      if (!ec && referenceLastWrite != referenceFileLastWriteTime_) {
        referenceFileLastWriteTime_ = referenceLastWrite;
        for (const std::function<void(const std::string&)>& reload : referenceFileReloaders_) {
          try {
            reload(referenceFile_);
          } catch (const std::exception& e) {
            LOG(WARNING) << "[MpcParameterUpdaterModule] Failed to reload " << referenceFile_ << ": " << e.what();
          }
        }
        LOG(INFO) << "[MpcParameterUpdaterModule] Reloaded command limits from " << referenceFile_;
      }
    }
  }
}

void MpcParameterUpdaterModule::enqueueParameterUpdate(std::string yamlText) {
  absl::MutexLock lock(&pendingMutex_);
  pendingYamlText_ = std::move(yamlText);
  hasPendingUpdate_.store(true, std::memory_order_release);  // NOLINT(argument-comment): libstdc++ names the value __i.
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::optional<std::string> MpcParameterUpdaterModule::takeContactEstimatorUpdate() {
  std::lock_guard<std::mutex> lock(controllerSettingsMutex_);
  std::optional<std::string> update = std::move(pendingContactEstimator_);
  pendingContactEstimator_.reset();
  return update;
}

std::optional<ContactWrenchGate::Config> MpcParameterUpdaterModule::takeContactWrenchGateUpdate() {
  std::lock_guard<std::mutex> lock(controllerSettingsMutex_);
  std::optional<ContactWrenchGate::Config> update = pendingContactWrenchGate_;
  pendingContactWrenchGate_.reset();
  return update;
}

void MpcParameterUpdaterModule::recordControllerSettings(const std::string& yamlFile) {
  PropertyTree pt;
  try {
    loadData::readPropertyTree(yamlFile, pt);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Could not parse " << yamlFile << " for the controller settings: " << e.what();
    return;
  }
  // The robot process reads the same keys of the operator's documents and of the task file for a remote MPC.
  // LINT.IfChange(controller_side_keys)
  const std::optional<std::string> name = pt.getOptional<std::string>("contactEstimator");
  std::optional<ContactWrenchGate::Config> gate;
  if (pt.findChild("contact_wrench_gate") != nullptr) {
    // A key the block does not carry keeps the library default, as at start-up; one that does not parse refuses the
    // block, which the controller then keeps running as it was (a `get` with a default used to swallow it silently).
    ContactWrenchGate::Config config;
    absl::Status status = loadOptionalValue(pt, "contact_wrench_gate.debounceTime", config.debounceTime);
    if (status.ok()) {
      status = loadOptionalValue(pt, "contact_wrench_gate.rampTime", config.rampTime);
    }
    if (!status.ok()) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] contact_wrench_gate was not applied, the running gate is kept: " << status.message();
    } else if (config.debounceTime >= 0.0 && config.rampTime >= 0.0) {
      gate = config;
    } else {
      LOG(WARNING) << "[MpcParameterUpdaterModule] contact_wrench_gate.debounceTime and contact_wrench_gate.rampTime must be "
                      "non-negative; the block was not applied, the running gate is kept.";
    }
  }
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/ControllerSideSettings.cpp:controller_side_keys)
  std::lock_guard<std::mutex> lock(controllerSettingsMutex_);
  if (name) pendingContactEstimator_ = *name;
  if (gate) pendingContactWrenchGate_ = gate;
}

void MpcParameterUpdaterModule::applyParameterUpdates(const std::string& yamlFile) {
  LOG(INFO) << "[MpcParameterUpdaterModule] Applying in-place parameter updates from " << yamlFile << "...";

  // The controller-side settings do not touch the solver; they are recorded even when there is no solver to update.
  recordControllerSettings(yamlFile);

  if (mpcPtr_ == nullptr) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] mpcPtr_ is null.";
    return;
  }
  SolverBase* solverBasePtr = mpcPtr_->getSolverPtr();
  if (solverBasePtr == nullptr) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] getSolverPtr() returned null.";
    return;
  }
  SqpSolver* sqpSolverPtr = dynamic_cast<SqpSolver*>(solverBasePtr);
  if (sqpSolverPtr == nullptr) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Underlying solver is not SqpSolver. Cannot update parameters.";
    return;
  }

  // ────────────────────────────────────────────────────────────────
  // 1. Parse property tree and weight matrices from task.yaml
  // ────────────────────────────────────────────────────────────────
  PropertyTree pt;
  try {
    loadData::readPropertyTree(yamlFile, pt);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Could not parse " << yamlFile << ": " << e.what() << ". No parameters were updated.";
    return;
  }

  // From the RUNNING problem, never from the file: the file's `costs` list is not hot-reloadable, and reading it here
  // would zero the base-pose weights of a problem that has no ACoM cost to replace them, or restore them next to one
  // that does. Every per-thread problem is a clone of the same one, so the first answers for all of them.
  std::vector<OptimalControlProblem>& ocpDefinitions = sqpSolverPtr->getOcpDefinitions();
  const bool runsComAndAcomTracking = !ocpDefinitions.empty() && runsComAndAcomTrackingCost(ocpDefinitions.front());
  // Likewise the terminal cost: whether a reloaded Q_final has a term to go to is the running problem's answer, not
  // the file's `costs` list, which is structural and would otherwise leave a running quadratic terminal cost frozen after
  // the file switched to dcm_terminal_cost (or go looking for a term that is not there after it switched back).
  const bool runsQuadraticTerminalCost = !ocpDefinitions.empty() && runsFinalCostTerm(ocpDefinitions.front(), kQuadraticTerminalCostTerm);
  const bool runsDcmTerminalCost = !ocpDefinitions.empty() && runsFinalCostTerm(ocpDefinitions.front(), kDcmTerminalCostTerm);
  warnAboutStructuralFormulationChanges(yamlFile, runsComAndAcomTracking, basisCostTransform_.has_value(), runsDcmTerminalCost,
                                        contactPlannerModulePtr_ != nullptr);

  // Every matrix below is applied only when it loaded: a key that does not parse refuses that matrix, named by the key,
  // and the running problem keeps its own, while the rest of the file still applies.
  matrix_t Q = matrix_t::Zero(stateDim_, stateDim_);
  bool applyQ = true;
  if (const absl::Status status = loadEigenMatrixFromPtree(pt, "Q", Q); !status.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] Q was not applied, the running state cost is kept: " << status.message();
    applyQ = false;
  }

  matrix_t R = matrix_t::Zero(inputDim_, inputDim_);
  bool applyR = true;
  if (basisCostTransform_.has_value()) {
    // The R indices in task.yaml refer to wrench-space inputs (forces/moments/joint velocities). In basis-vector mode
    // the OCP input is [λ, joint velocities], so loading R directly at inputDim_ would put force weights on λ entries.
    // Load at the wrench dimension and apply the transform the OCP factory applied at start-up,
    // R_basis = Mᵀ R_wrench M + reg * blkdiag(S, 0), with the regularization weight `reg` and its shape S reloaded
    // too. The generator set (contacts.basisGeneratorSet) is NOT reloaded: it fixes M and the input dimension.
    BasisInputsCostTransformConfig candidate = *basisCostTransform_;
    // LINT.IfChange(basis_regularization_updater_yaml_path)
    absl::Status status = loadOptionalValue(pt, kBasisScalingRegularizationKey, candidate.lambdaRegularization);
    if (status.ok()) {
      status = loadOptionalValue(pt, kBasisRegularizationKey, candidate.regularization);
    }
    // clang-format off
    // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:basis_regularization_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:basis_regularization_config)
    // clang-format on
    matrix_t R_wrench = matrix_t::Zero(candidate.wrenchInputDim, candidate.wrenchInputDim);
    if (status.ok()) {
      status = loadEigenMatrixFromPtree(pt, "R", R_wrench);
    }
    // A weight that is not a number, an unknown regularization name, a negative weight, or a λ block that is not
    // positive definite (a zero weight, or null_space with a contact wrench direction R does not weigh) would install
    // an R the QP cannot solve with, or none at all. The whole basis-space R is refused then, with a message naming
    // the key, and the running R and regularization are kept; the rest of the file still applies.
    if (status.ok()) {
      status = validateBasisInputsCostTransformConfig(candidate);
    }
    if (status.ok()) {
      R = transformWrenchInputCostToBasisSpace(R_wrench, candidate);
      status = checkLambdaBlockPositiveDefinite(R, candidate.numBasisInputs);
    }
    if (status.ok()) {
      basisCostTransform_ = std::move(candidate);
    } else {
      LOG(WARNING) << "[MpcParameterUpdaterModule] R was not applied, the running input cost is kept: " << status.message();
      applyR = false;
    }
  } else if (const absl::Status status = loadEigenMatrixFromPtree(pt, "R", R); !status.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] R was not applied, the running input cost is kept: " << status.message();
    applyR = false;
  }

  // terminalCostScaling weighs both Q_final and the terminal CoM + ACoM instance. Empty when the file does not carry it
  // or carries a value that does not parse; the second case refuses both terminal weights rather than applying them at
  // a scaling of 1.
  std::optional<scalar_t> terminalCostScaling;
  bool terminalCostScalingRefused = false;
  if (pt.findChild("terminalCostScaling") != nullptr) {
    scalar_t scaling = 1.0;
    if (const absl::Status status = loadOptionalValue(pt, "terminalCostScaling", scaling); status.ok()) {
      terminalCostScaling = scaling;
    } else {
      LOG(WARNING) << "[MpcParameterUpdaterModule] the terminal weights were not applied, the running ones are kept: " << status.message();
      terminalCostScalingRefused = true;
    }
  }

  // Q_final is optional, and applied only to a running quadratic terminal cost: a problem that ends on the DCM cost
  // has none, whatever the reloaded file's `costs` list now says.
  matrix_t Q_final = matrix_t::Zero(stateDim_, stateDim_);
  bool hasQFinal = runsQuadraticTerminalCost && !terminalCostScalingRefused && pt.findChild("Q_final") != nullptr;
  if (hasQFinal) {
    if (const absl::Status status = loadEigenMatrixFromPtree(pt, "Q_final", Q_final); !status.ok()) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] Q_final was not applied, the running terminal cost is kept: " << status.message();
      hasQFinal = false;
    }
  }

  // The factory zeroes the base pose block of both the running and the terminal
  // state cost when the problem carries the CoM + aCOM cost, so live updates must
  // do the same or a slider drag silently re-introduces base pose tracking.
  if (runsComAndAcomTracking) {
    ComAndAcomTrackingCost::zeroBasePoseWeights(Q);
    ComAndAcomTrackingCost::zeroBasePoseWeights(Q_final);
  }
  Q_final *= terminalCostScaling.value_or(1.0);

  // CoM and ACoM tracking weights. A file without them is the normal case for a problem without the cost.
  matrix_t Q_com = matrix_t::Zero(3, 3);
  matrix_t Q_acom = matrix_t::Zero(3, 3);
  bool hasComAcom = false;
  {
    absl::Status status = loadEigenMatrixFromPtree(pt, "Q_com", Q_com);
    if (status.ok()) {
      status = loadEigenMatrixFromPtree(pt, "Q_acom", Q_acom);
    }
    if (status.ok()) {
      hasComAcom = true;
    } else if (runsComAndAcomTracking || !absl::IsNotFound(status)) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] the CoM + aCOM tracking weights were not applied, the running ones are kept: "
                   << status.message();
    }
  }

  // ────────────────────────────────────────────────────────────────
  // 2. Parse task-space tracking cost weights
  // ────────────────────────────────────────────────────────────────

  // Each section below is read by the loader the start-up path uses, and only when the file carries it (loadSection):
  // those loaders fill every key they do not find with a library default, so an absent section applied that way would
  // replace the running values with defaults. A section they cannot read is reported by its key and not applied.
  vector12_t footTrackingWeightsVec = vector12_t::Zero();
  const bool hasFootTrackingWeights = loadSection(pt, "task_space_foot_cost_weights", [&]() {
    footTrackingWeightsVec =
        EndEffectorKinematicsWeights::getWeights(yamlFile, "task_space_foot_cost_weights.", /*verbose=*/false).toVector();
  });

  // Applied only when the file carries it: an absent key applied as its default would silently switch the flag off.
  bool footActiveInStance = false;
  bool hasFootActiveInStance = pt.findChild("task_space_foot_cost_weights.activeInStance") != nullptr;
  if (hasFootActiveInStance) {
    const absl::Status status = loadOptionalValue(pt, "task_space_foot_cost_weights.activeInStance", footActiveInStance);
    if (!status.ok()) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] activeInStance was not applied, the running value is kept: " << status.message();
      hasFootActiveInStance = false;
    }
  }

  vector2_t icpWeights = vector2_t::Zero();
  const bool hasIcpWeights =
      loadSection(pt, "icp_cost_weights", [&]() { icpWeights = ICPCost::getWeights(yamlFile, "icp_cost_weights.", /*verbose=*/false); });

  // DCM terminal cost (weights, CoM height, velocity offset); the CppAD model is parameterized, no recompilation needed.
  // The block is read as the file writes it: a comHeight of 0 is resolved to the model's pendulum length by each
  // running cost's setConfig(), exactly as at start-up.
  std::optional<DcmTerminalCost::Config> dcmTerminalConfig;
  if (pt.findChild("dcm_terminal_cost") != nullptr) {
    absl::StatusOr<DcmTerminalCost::Config> loaded =
        DcmTerminalCost::loadConfig(yamlFile, DcmTerminalCost::kConfigPrefix, /*verbose=*/false);
    if (loaded.ok()) {
      dcmTerminalConfig = *std::move(loaded);
    } else {
      LOG(WARNING) << "[MpcParameterUpdaterModule] dcm_terminal_cost not applied, the running cost is kept: " << loaded.status().message();
    }
  }
  bool dcmTerminalConfigRefusalLogged = false;

  // Task-space torso/body tracking cost weights, one section per tracked body.
  std::vector<std::pair<std::string, vector12_t>> taskSpaceCostUpdates;
  if (const PropertyTree* taskSpaceCosts = pt.findChild("task_space_costs")) {
    for (const PropertyTree::value_type& taskSpaceCost : *taskSpaceCosts) {
      const std::string& costName = taskSpaceCost.first;
      const std::string section = absl::StrCat("task_space_costs.", costName, ".weights");
      vector12_t weights = vector12_t::Zero();
      if (loadSection(pt, section, [&]() {
            weights = EndEffectorKinematicsWeights::getWeights(yamlFile, section + ".", /*verbose=*/false).toVector();
          })) {
        taskSpaceCostUpdates.emplace_back(absl::StrCat(costName, "_TaskSpaceKinematicsCost"), weights);
      }
    }
  }

  // External torque cost weights, one section per leg.
  std::vector<std::pair<std::string, ExternalTorqueQuadraticCostAD::Config>> extTorqueConfigs;
  for (size_t i = 0; i < contactNames_.size(); ++i) {
    const std::string section = (i == 0) ? "left_leg_torque_cost" : "right_leg_torque_cost";
    ExternalTorqueQuadraticCostAD::Config config;
    if (loadSection(pt, section,
                    [&]() { config = ExternalTorqueQuadraticCostAD::loadConfigFromFile(yamlFile, section + ".", /*verbose=*/false); })) {
      extTorqueConfigs.emplace_back(absl::StrCat(contactNames_[i], "_ExternalTorqueQuadraticCost"), std::move(config));
    }
  }

  // ────────────────────────────────────────────────────────────────
  // 3. Parse barrier penalty configs
  // ────────────────────────────────────────────────────────────────
  // Every barrier below is default-constructed, so an absent section must leave
  // the running value alone rather than be applied as a library default.
  // loadPtreeValue does not throw for a missing key, so presence is probed on the
  // section itself and the corresponding apply step is skipped when it is absent.
  RelaxedBarrierPenalty::Config wrenchConeBarrier, frictionConeBarrier, contactMomentBarrier;
  PieceWisePolynomialBarrierPenalty::Config jointLimitsBarrier, collisionBarrier, basisNonNegativityBarrier;

  // Only `mu` and `delta` are hot-reloadable. The geometric coefficients in the same yaml sections
  // (frictionCoefficient, minNormalForce, numBasisVectors, ...) are baked in when the problem is built - into the
  // constraints, and under basis-vector contact inputs into the generators and the input dimension - and cannot be
  // updated here; the tuning GUI labels them "(restart)".
  // LINT.IfChange(hot_reloadable_barrier_keys)
  const bool hasWrenchConeBarrier =
      loadBarrierSection(pt, "contacts.contactWrenchConeSoftConstraint", wrenchConeBarrier.mu, wrenchConeBarrier.delta);
  const bool hasFrictionConeBarrier =
      loadBarrierSection(pt, "contacts.frictionForceConeSoftConstraint", frictionConeBarrier.mu, frictionConeBarrier.delta);
  const bool hasContactMomentBarrier =
      loadBarrierSection(pt, "contacts.contactMomentXYSoftConstraint", contactMomentBarrier.mu, contactMomentBarrier.delta);
  const bool hasJointLimitsBarrier = loadBarrierSection(pt, "jointLimits", jointLimitsBarrier.mu, jointLimitsBarrier.delta);
  const bool hasCollisionBarrier = loadBarrierSection(pt, "collision_constraint", collisionBarrier.mu, collisionBarrier.delta);
  // LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/mpc_params_tab.py:build_time_contact_keys)
  // The λ ≥ 0 barrier of the basis-vector contact inputs, which CentroidalMpcInterface builds from this section.
  // LINT.IfChange(basis_barrier_updater_yaml_path)
  const bool hasBasisNonNegativityBarrier =
      loadBarrierSection(pt, "contacts.basisNonNegativityBarrier", basisNonNegativityBarrier.mu, basisNonNegativityBarrier.delta);
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:basis_barrier_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:basis_barrier_config)
  // clang-format on

  // ────────────────────────────────────────────────────────────────
  // 3a. Parse contact-implicit config
  // ────────────────────────────────────────────────────────────────
  // Where the ground is: one top-level key, shared with the swing trajectories, rather than a second definition inside
  // the contact_implicit block. NaN marks a file that does not carry it, or carries a value that cannot be applied.
  // LINT.IfChange(terrain_height_updater_yaml_path)
  scalar_t terrainHeight = std::numeric_limits<scalar_t>::quiet_NaN();
  const absl::Status terrainHeightStatus = loadOptionalValue(pt, "terrainHeight", terrainHeight);
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:terrain_height_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:terrain_height_config)
  // clang-format on
  if (!terrainHeightStatus.ok() || (pt.findChild("terrainHeight") != nullptr && !std::isfinite(terrainHeight))) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] terrainHeight not applied, the running ground is kept: "
                 << (terrainHeightStatus.ok() ? absl::StrCat("terrainHeight is ", terrainHeight, ", which is not finite.")
                                              : std::string(terrainHeightStatus.message()));
    terrainHeight = std::numeric_limits<scalar_t>::quiet_NaN();
  }
  // The residual references and the weights, retuned live. Every key of this block becomes a slider in the tuning
  // dashboard, so one that quietly did nothing until the next launch would be a trap rather than a parameter. The block
  // is refused as a whole - a key the start-up path would not read, or a value it would refuse - so that the terms never
  // run on half of an edit.
  std::optional<ContactImplicitReload> contactImplicitReload;
  {
    absl::StatusOr<std::optional<ContactImplicitReload>> reload = loadContactImplicitReload(pt);
    if (reload.ok()) {
      contactImplicitReload = *std::move(reload);
    } else {
      LOG(WARNING) << "[MpcParameterUpdaterModule] contact_implicit not applied, the running terms are kept: " << reload.status().message();
    }
  }

  // LINT.IfChange(softConstraintWeight_yaml_path)
  // The negative sentinel is what marks the weight absent: loadOptionalValue leaves it untouched for a missing key, and
  // for a value that does not parse (reported by its key), and the apply step below only runs for a positive value.
  scalar_t zeroVelWeight = -1.0;
  if (const absl::Status status = loadOptionalValue(pt, "model_settings.foot_constraint.softConstraintWeight", zeroVelWeight);
      !status.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] the zero_velocity soft weight was not applied, the running one is kept: "
                 << status.message();
  }
  // The weight of the SOFT normal-velocity term, which is what shapes the swing under the contact-implicit
  // formulation and is therefore the knob an operator reaches for first. It was missing here while its neighbor
  // above was present, so the task file told the operator to tune a key that took effect only on the next launch.
  scalar_t normalVelSoftWeight = -1.0;
  if (const absl::Status status =
          loadOptionalValue(pt, "model_settings.foot_constraint.normalVelocitySoftConstraintWeight", normalVelSoftWeight);
      !status.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] the normal_velocity soft weight was not applied, the running one is kept: "
                 << status.message();
  }
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:foot_constraint_section, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:foot_constraint_section, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:foot_constraint_section)
  // clang-format on
  // ────────────────────────────────────────────────────────────────
  // 3b. Parse foot constraint error gains
  // ────────────────────────────────────────────────────────────────
  ModelSettings::FootConstraintConfig footCfg;
  bool hasFootConstraintGains = false;
  if (pt.findChild("model_settings.foot_constraint") != nullptr) {
    // All or nothing: a gain that does not parse refuses the group, so the constraint never runs on half of an edit.
    const absl::Status status = loadFootConstraintGains(pt, footCfg);
    if (status.ok()) {
      hasFootConstraintGains = true;
    } else {
      LOG(WARNING) << "[MpcParameterUpdaterModule] the foot constraint gains were not applied, the running ones are kept: "
                   << status.message();
    }
  }

  // Build the Ax/Av config from foot constraint gains (mirrors CentroidalMpcInterface::getStanceFootConstraint)
  EndEffectorKinematicsTwistConstraint::Config footTwistConfig;
  if (hasFootConstraintGains) {
    footTwistConfig.b.setZero(6);
    footTwistConfig.Ax.setZero(6, 6);
    footTwistConfig.Av.setZero(6, 6);
    if (!numerics::almost_eq(footCfg.positionErrorGain_z, /*y=*/0.0)) {
      footTwistConfig.Ax(2, 2) = footCfg.positionErrorGain_z;
    }
    if (!numerics::almost_eq(footCfg.orientationErrorGain, /*y=*/0.0)) {
      footTwistConfig.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footCfg.orientationErrorGain;
    }
    footTwistConfig.Av(0, 0) = footCfg.linearVelocityErrorGain_xy;
    footTwistConfig.Av(1, 1) = footCfg.linearVelocityErrorGain_xy;
    footTwistConfig.Av(2, 2) = footCfg.linearVelocityErrorGain_z;
    footTwistConfig.Av(3, 3) = footCfg.angularVelocityErrorGain;
    footTwistConfig.Av(4, 4) = footCfg.angularVelocityErrorGain;
    footTwistConfig.Av(5, 5) = footCfg.angularVelocityErrorGain;
  }

  // ────────────────────────────────────────────────────────────────
  // 3c. Parse SQP solver settings (safe subset)
  // ────────────────────────────────────────────────────────────────
  sqp::Settings sqpUpdates = sqpSolverPtr->getSettings();
  bool hasSqpUpdates = false;
  if (pt.findChild("multiple_shooting") != nullptr) {
    const absl::Status status = loadSqpSettingsUpdates(pt, sqpUpdates);
    if (status.ok()) {
      hasSqpUpdates = true;
    } else {
      LOG(WARNING) << "[MpcParameterUpdaterModule] the multiple_shooting settings were not applied, the running ones are kept: "
                   << status.message();
    }
  }

  // ────────────────────────────────────────────────────────────────
  // 3d. Parse swing trajectory config
  // ────────────────────────────────────────────────────────────────
  SwingTrajectoryPlanner::Config swingConfig;
  const bool hasSwingConfig = referenceManagerPtr_ != nullptr && loadSection(pt, "swing_trajectory_config", [&]() {
                                swingConfig = loadSwingTrajectorySettings(yamlFile, "swing_trajectory_config", /*verbose=*/false);
                              });

  // ────────────────────────────────────────────────────────────────
  // 4. Apply updates in-place to every thread-local OCP
  // ────────────────────────────────────────────────────────────────
  const matrix_t zeroQ = matrix_t::Zero(stateDim_, stateDim_);
  const matrix_t zeroR = matrix_t::Zero(inputDim_, inputDim_);

  for (OptimalControlProblem& ocp : ocpDefinitions) {
    // ── Quadratic costs ──
    try {
      QuadraticStateInputCost& stateInputCost = ocp.costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost");
      // Whichever of Q and R was refused above keeps its running value.
      matrix_t runningQ;
      matrix_t runningR;
      matrix_t runningP;
      stateInputCost.getGains(runningQ, runningR, runningP);
      stateInputCost.setGains(applyQ ? Q : runningQ, applyR ? R : runningR, runningP);
    } catch (const std::out_of_range&) {
      // Expected if not used in task.yaml
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update stateInputQuadraticCost: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update stateInputQuadraticCost: unknown exception";
    }

    try {
      if (applyQ) {
        ocp.costPtr->get<QuadraticStateInputCost>("stateQuadraticCost").setGains(Q, zeroR);
      }
    } catch (const std::out_of_range&) {
      // Expected if not used in task.yaml
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update stateQuadraticCost: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update stateQuadraticCost: unknown exception";
    }

    try {
      if (applyR) {
        ocp.costPtr->get<QuadraticStateInputCost>("inputQuadraticCost").setGains(zeroQ, R);
      }
    } catch (const std::out_of_range&) {
      // Expected if not used in task.yaml
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to update inputQuadraticCost: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to update inputQuadraticCost: unknown exception";
    }

    if (hasQFinal) {
      try {
        ocp.finalCostPtr->get<QuadraticStateCost>(kQuadraticTerminalCostTerm).setGains(Q_final);
      } catch (const std::out_of_range&) {
        // Expected if not used in task.yaml
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update terminalCost: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update terminalCost: unknown exception";
      }
    }

    // ── DCM terminal cost ──
    if (dcmTerminalConfig.has_value()) {
      try {
        const absl::Status applied = ocp.finalCostPtr->get<DcmTerminalCost>(kDcmTerminalCostTerm).setConfig(*dcmTerminalConfig);
        if (!applied.ok() && !dcmTerminalConfigRefusalLogged) {
          LOG(WARNING) << "[MpcParameterUpdaterModule] dcm_terminal_cost not applied, the running cost is kept: " << applied.message();
          dcmTerminalConfigRefusalLogged = true;
        }
      } catch (const std::out_of_range&) {
        // Expected if dcm_terminal_cost is not in the cost list
      }
    }

    // ── CoM and ACoM tracking cost, running and terminal ──
    if (hasComAcom && runsComAndAcomTracking) {
      try {
        ocp.stateCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kRunningTermName)).setWeights(Q_com, Q_acom);
        if (ocp.finalCostPtr != nullptr &&
            ocp.finalCostPtr->getTermNameMap().count(std::string(ComAndAcomTrackingCost::kTerminalTermName)) > 0) {
          // Weighted like Q_final, by terminalCostScaling, which a refused value (reported above) or an absent key leaves
          // as it was.
          if (terminalCostScaling.has_value()) {
            ocp.finalCostPtr->get<ComAndAcomTrackingCost>(std::string(ComAndAcomTrackingCost::kTerminalTermName))
                .setWeights(*terminalCostScaling * Q_com, *terminalCostScaling * Q_acom);
          } else if (!terminalCostScalingRefused) {
            LOG(WARNING) << "[MpcParameterUpdaterModule] terminalCostScaling is missing from " << yamlFile
                         << "; the terminal CoM + aCOM tracking weights are left as they were.";
          }
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update the CoM + aCOM tracking cost: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update the CoM + aCOM tracking cost: unknown exception";
      }
    }

    // ── Foot tracking costs ──
    if (hasFootTrackingWeights || hasFootActiveInStance) {
      for (const std::string& footName : contactNames_) {
        try {
          CentroidalMpcEndEffectorFootCost& footCost =
              ocp.costPtr->get<CentroidalMpcEndEffectorFootCost>(absl::StrCat(footName, "_TaskSpaceKinematicsCost"));
          if (hasFootTrackingWeights) {
            footCost.setWeights(footTrackingWeightsVec);
          }
          if (hasFootActiveInStance) {
            footCost.setActiveInStance(footActiveInStance);
          }
        } catch (const std::out_of_range&) {
          // The term is absent when its cost or constraint is not listed; the file's section is then read for nothing.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_TaskSpaceKinematicsCost: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_TaskSpaceKinematicsCost: unknown exception";
        }
      }
    }

    // ── Task-space body tracking costs (torso, etc.) ──
    for (const std::pair<std::string, vector12_t>& update : taskSpaceCostUpdates) {
      const std::string& costName = update.first;
      try {
        ocp.costPtr->get<EndEffectorKinematicsQuadraticCost>(costName).setWeights(update.second);
      } catch (const std::out_of_range&) {
        // The term is absent when its cost is not listed; the file's section is then read for nothing.
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update " << costName << ": " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update " << costName << ": unknown exception";
      }
    }

    // ── ICP cost ──
    if (hasIcpWeights) {
      try {
        ocp.costPtr->get<ICPCost>("icp_Cost").setWeights(icpWeights);
      } catch (const std::out_of_range&) {
        // The term is absent when icp_cost is not listed; the file's section is then read for nothing.
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update icp_Cost: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update icp_Cost: unknown exception";
      }
    }

    // ── External torque costs ──
    for (const std::pair<std::string, ExternalTorqueQuadraticCostAD::Config>& update : extTorqueConfigs) {
      const std::string& costName = update.first;
      try {
        ocp.costPtr->get<ExternalTorqueQuadraticCostAD>(costName).setWeights(update.second.weights);
      } catch (const std::out_of_range&) {
        // The term is absent when external_torque_cost is not listed; the file's section is then read for nothing.
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update " << costName << ": " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update " << costName << ": unknown exception";
      }
    }

    // ── Soft constraints: wrench cone, friction cone, contact moment XY ──
    // Penalties are wrapped in PenaltyBaseWrapper (AugmentedPenaltyBase), so we use
    // setParameters(vector_t{mu, delta}) which delegates through to the inner PenaltyBase.
    //
    // WHICH delta is written depends on the penalty that is actually installed, not on the YAML, and the factory's own
    // contactConePenaltyParameters() - the function it built the penalty with - decides it. A schedule-gated cone is
    // wrapped in a RelaxedBarrierPenalty, whose `delta` is the width of its quadratic relaxation and is read straight
    // from the file. An UN-gated cone - the contact-implicit formulation - is wrapped in a SquaredHingePenalty built with
    // delta = 0, because a hinge's delta is the OFFSET of its zero, and the whole point of the hinge there is that its
    // zero sits exactly on the cone that a foot at zero wrench lies on. Writing the barrier's delta into it would move
    // that zero into the interior and reinstate the very force floor that dropping `minNormalForce` and the friction
    // cone's parabolic margin exists to remove: with the shipped friction settings (mu 0.2, delta 5) a foot in flight
    // would be charged 2.5 with a gradient of -1.0 pushing its normal force up.
    for (const std::string& footName : contactNames_) {
      // Contact wrench cone
      if (hasWrenchConeBarrier) {
        try {
          StateInputSoftConstraint& softCon =
              ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kContactWrenchCone));
          const vector_t parameters =
              contactConePenaltyParameters(wrenchConeBarrier, softCon.get<ContactWrenchConeConstraint>().isScheduleGated());
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(parameters);
          }
        } catch (const std::out_of_range&) {
          // The term is absent whenever its constraint is not listed, and the wrench cone is never built under
          // basis-vector contact inputs. Its section still exists in the file, so this is the normal case and must not
          // warn, or every reload logs one failure per foot per worker.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_contactWrenchCone: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_contactWrenchCone: unknown exception";
        }
      }

      // Friction force cone
      if (hasFrictionConeBarrier) {
        try {
          StateInputSoftConstraint& softCon =
              ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kFrictionForceCone));
          const vector_t parameters =
              contactConePenaltyParameters(frictionConeBarrier, softCon.get<FrictionForceConeConstraint>().isScheduleGated());
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(parameters);
          }
        } catch (const std::out_of_range&) {
          // The term is absent whenever its constraint is not listed, and the wrench cone is never built under
          // basis-vector contact inputs. Its section still exists in the file, so this is the normal case and must not
          // warn, or every reload logs one failure per foot per worker.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_frictionForceCone: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_frictionForceCone: unknown exception";
        }
      }

      // Contact moment XY
      if (hasContactMomentBarrier) {
        try {
          StateInputSoftConstraint& softCon =
              ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kContactMomentXY));
          const vector_t parameters =
              contactConePenaltyParameters(contactMomentBarrier, softCon.get<ContactMomentXYConstraintCppAd>().isScheduleGated());
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(parameters);
          }
        } catch (const std::out_of_range&) {
          // The term is absent whenever its constraint is not listed, and the wrench cone is never built under
          // basis-vector contact inputs. Its section still exists in the file, so this is the normal case and must not
          // warn, or every reload logs one failure per foot per worker.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_contactMomentXY: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_contactMomentXY: unknown exception";
        }
      }

      // Basis scaling non-negativity barrier (λ ≥ 0)
      if (hasBasisNonNegativityBarrier) {
        try {
          ocp.costPtr->get<BasisScalingNonNegativityConstraint>(absl::StrCat(footName, "_basisNonNegativity"))
              .setBarrierPenalty(basisNonNegativityBarrier);
        } catch (const std::out_of_range&) {
          // Expected if not in basis-vector mode
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_basisNonNegativity: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_basisNonNegativity: unknown exception";
        }
      }
    }

    // ── Joint limits ──
    if (hasJointLimitsBarrier) {
      try {
        ocp.stateSoftConstraintPtr->get<JointLimitsSoftConstraint>("jointLimits").setGains(jointLimitsBarrier.mu, jointLimitsBarrier.delta);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update jointLimits: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update jointLimits: unknown exception";
      }
    }

    // ── Foot collision ──
    if (hasCollisionBarrier) {
      try {
        StateSoftConstraint& softCon = ocp.stateSoftConstraintPtr->get<StateSoftConstraint>("FootCollisionSoftConstraint");
        const vector_t collisionParams = (vector_t(2) << collisionBarrier.mu, collisionBarrier.delta).finished();
        for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
          penalty->setParameters(collisionParams);
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to update FootCollisionSoftConstraint: " << e.what();
      } catch (...) {
        LOG(WARNING) << "Failed to update FootCollisionSoftConstraint: unknown exception";
      }
    }

    // ── Contact implicit soft constraints ──
    // Only what the reloaded block carries is applied; the terms are absent unless the formulation is listed, which is
    // the normal case. The ground is not applied here: see followAppliedTerrainHeight().
    // LINT.IfChange(contact_implicit_updater_keys)
    if (contactImplicitReload.has_value()) {
      const ModelSettings::ContactImplicitConfig& reloaded = contactImplicitReload->config;
      const bool carriesComplementarityWeight =
          contactImplicitReload->carries(&ModelSettings::ContactImplicitConfig::complementarityWeight);
      const bool carriesHeightReference = contactImplicitReload->carries(&ModelSettings::ContactImplicitConfig::heightReference);
      const bool carriesGapSmoothing = contactImplicitReload->carries(&ModelSettings::ContactImplicitConfig::gapSmoothing);
      const bool carriesSlipWeight = contactImplicitReload->carries(&ModelSettings::ContactImplicitConfig::slipWeight);
      const bool carriesVelocityReference = contactImplicitReload->carries(&ModelSettings::ContactImplicitConfig::velocityReference);
      const bool carriesAngularVelocityReference =
          contactImplicitReload->carries(&ModelSettings::ContactImplicitConfig::angularVelocityReference);
      const bool carriesPenetrationWeight = contactImplicitReload->carries(&ModelSettings::ContactImplicitConfig::penetrationWeight);
      for (const std::string& footName : contactNames_) {
        try {
          StateInputSoftConstraint& softCon =
              ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kContactComplementarity));
          if (carriesComplementarityWeight) {
            const vector_t weight = (vector_t(1) << reloaded.complementarityWeight).finished();
            for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
              penalty->setParameters(weight);
            }
          }
          ContactComplementarityConstraint& constraint = softCon.get<ContactComplementarityConstraint>();
          if (carriesHeightReference) {
            constraint.setHeightReference(reloaded.heightReference);
          }
          if (carriesGapSmoothing) {
            constraint.setGapSmoothing(reloaded.gapSmoothing);
          }
        } catch (const std::out_of_range&) {
          // Not listed: the formulation is off.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << contact_term::name(footName, contact_term::kContactComplementarity) << ": " << e.what();
        }

        try {
          StateInputSoftConstraint& softCon =
              ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kForceWeightedSlip));
          if (carriesSlipWeight) {
            const vector_t weight = (vector_t(1) << reloaded.slipWeight).finished();
            for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
              penalty->setParameters(weight);
            }
          }
          // The setter takes both references at once, so a file carrying only one of them keeps the other at the value
          // the term already holds rather than dropping the update on the floor.
          if (carriesVelocityReference || carriesAngularVelocityReference) {
            ForceWeightedSlipConstraint& constraint = softCon.get<ForceWeightedSlipConstraint>();
            const vector3_t inverseCurrent = constraint.getInverseTwistReference();
            constraint.setTwistReferences(carriesVelocityReference ? reloaded.velocityReference : 1.0 / inverseCurrent(0),
                                          carriesAngularVelocityReference ? reloaded.angularVelocityReference : 1.0 / inverseCurrent(2));
          }
        } catch (const std::out_of_range&) {
          // Not listed: the formulation is off.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << contact_term::name(footName, contact_term::kForceWeightedSlip) << ": " << e.what();
        }

        if (carriesPenetrationWeight) {
          try {
            StateSoftConstraint& softCon =
                ocp.stateSoftConstraintPtr->get<StateSoftConstraint>(contact_term::name(footName, contact_term::kGroundPenetration));
            // SquaredHingePenalty::setParameters takes (mu, delta); delta stays 0 so the hinge's zero stays on the ground.
            const vector_t penetrationParams = (vector_t(2) << reloaded.penetrationWeight, 0.0).finished();
            for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
              penalty->setParameters(penetrationParams);
            }
          } catch (const std::out_of_range&) {
            // Not listed: the formulation is off.
          } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to update " << contact_term::name(footName, contact_term::kGroundPenetration) << ": " << e.what();
          }
        }
      }
    }
    // clang-format off
    // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/common/ModelSettings.cpp:contact_implicit_yaml_path, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:contact_implicit_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:contact_implicit_config)
    // clang-format on

    // ── Zero velocity soft constraint weight ──
    if (zeroVelWeight > 0.0) {
      // The QuadraticPenalty is wrapped inside a PenaltyBaseWrapper (AugmentedPenaltyBase).
      // Use setParameters() which delegates through the wrapper to QuadraticPenalty::setParameters().
      vector_t scaleParam(1);
      scaleParam[0] = zeroVelWeight;
      for (const std::string& footName : contactNames_) {
        try {
          StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_zeroVelocity"));
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(scaleParam);
          }
        } catch (const std::out_of_range&) {
          // No soft zero_velocity: it is listed as a hard constraint (the shipped Atlas), or not at all (the
          // contact-implicit formulation). The weight then has no penalty to scale, which is not a failure.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (soft): " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (soft): unknown exception";
        }
      }
    }

    // ── Normal velocity soft constraint weight ──
    if (normalVelSoftWeight > 0.0) {
      vector_t scaleParam(1);
      scaleParam[0] = normalVelSoftWeight;
      for (const std::string& footName : contactNames_) {
        try {
          StateInputSoftConstraint& softCon =
              ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kNormalVelocitySoft));
          for (std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softCon.getPenalty().getPenaltyPtrArray()) {
            penalty->setParameters(scaleParam);
          }
        } catch (const std::out_of_range&) {
          // The term is absent whenever normal_velocity is not listed as a soft constraint, which is every
          // schedule-gated configuration. That is not a failure and must not warn, or the log fills up once per foot
          // per slider drag.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_normalVelocitySoft: " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_normalVelocitySoft: unknown exception";
        }
      }
    }

    // ── Foot constraint error gains ──
    if (hasFootConstraintGains) {
      // positionErrorGain_z has to reach the PRE-COMPUTATION as well as the zeroVelocity twist config below, and
      // that is not a refinement: under the contact-implicit formulation zeroVelocity is not built at all, so the
      // block below writes the gain into a term that does not exist while the term that does - the soft
      // normal-velocity servo - reads it from here. Without this the task-file key, and the slider the dashboard
      // renders for it, silently did nothing until the next launch.
      //
      // Each worker thread owns its own PreComputation, so writing it per OCP is also what keeps this race-free;
      // mutating the shared ModelSettings would not be.
      HumanoidPreComputation* preComputationPtr = dynamic_cast<HumanoidPreComputation*>(ocp.preComputationPtr.get());
      if (preComputationPtr != nullptr) {
        preComputationPtr->setNormalVelocityPositionErrorGain(footCfg.positionErrorGain_z);
      }

      for (const std::string& footName : contactNames_) {
        // Hard constraint path
        try {
          ZeroVelocityConstraintCppAd& con =
              ocp.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(absl::StrCat(footName, "_zeroVelocity"));
          con.getTwistConstraint().setNumConstraints(footCfg.constrainOrientation ? 6 : 3);
          con.getTwistConstraint().setConstrainYawRateAboutNormal(footCfg.constrainYawRateAboutContactNormal);
          con.getTwistConstraint().configure(EndEffectorKinematicsTwistConstraint::Config(footTwistConfig));
        } catch (const std::out_of_range&) {
          // No hard zero_velocity: it is soft (handled below) or not listed.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (hard): " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (hard): unknown exception";
        }
        // Soft constraint path: the inner constraint is wrapped in StateInputSoftConstraint.
        // ZeroVelocityConstraintCppAd is reached via dynamic_cast through the soft constraint wrapper.
        try {
          StateInputSoftConstraint& softCon = ocp.softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_zeroVelocity"));
          ZeroVelocityConstraintCppAd* zeroVelCon = dynamic_cast<ZeroVelocityConstraintCppAd*>(softCon.getConstraintPtr().get());
          if (zeroVelCon != nullptr) {
            zeroVelCon->getTwistConstraint().setNumConstraints(footCfg.constrainOrientation ? 6 : 3);
            zeroVelCon->getTwistConstraint().setConstrainYawRateAboutNormal(footCfg.constrainYawRateAboutContactNormal);
            zeroVelCon->getTwistConstraint().configure(EndEffectorKinematicsTwistConstraint::Config(footTwistConfig));
          }
        } catch (const std::out_of_range&) {
          // No soft zero_velocity: it is hard (handled above) or not listed.
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (soft config): " << e.what();
        } catch (...) {
          LOG(WARNING) << "Failed to update " << footName << "_zeroVelocity (soft config): unknown exception";
        }
      }
    }
  }

  // ── The ground ──
  // It goes to the reference manager, which owns it: the swing trajectories and the landing targets are rebuilt on it
  // at the next solve, and followAppliedTerrainHeight() moves the complementarity and penetration terms onto it in that
  // same solve - not in this one, whose references were built before this module ran. Writing the terms here, as this
  // used to, gave them a ground the references would not have for a whole solve, and left the swing trajectories and
  // the landing targets on the launch value for good. Without a reference manager there are no references to agree
  // with, and the terms take the height at once.
  if (std::isfinite(terrainHeight)) {
    if (referenceManagerPtr_ != nullptr) {
      referenceManagerPtr_->setTerrainHeight(terrainHeight);
    } else {
      setContactImplicitTermsTerrainHeight(ocpDefinitions, terrainHeight);
    }
  }

  // ── SQP solver settings (applied once, not per-thread OCP) ──
  if (hasSqpUpdates) {
    sqpSolverPtr->getSettings() = sqpUpdates;
  }

  // ── Swing trajectory config ──
  if (hasSwingConfig && referenceManagerPtr_ != nullptr) {
    const std::shared_ptr<SwingTrajectoryPlanner>& swingPlanner = referenceManagerPtr_->getSwingTrajectoryPlanner();
    if (swingPlanner) {
      swingPlanner->setConfig(swingConfig);
    }
  }

  // ── Contact planning config, when the file carries the block (a task file with it inline, or an enqueued update: the
  // YAML the GUI publishes, which it assembles from both files) ──
  if (pt.findChild("contact_planning") != nullptr) {
    applyContactPlanningUpdates(yamlFile);
  }

  // ── Locomotion heuristics: the coefficients of Bledt's RPC reference-shaping layer
  // (humanoid_nmpc/docs/locomotion_heuristics/README.md) ──
  //
  // The layer lives on the reference manager, which is shared rather than cloned per worker, so there is nothing to
  // walk here - one reconfigure() reaches every thread's view of it. It is safe from this thread because
  // preSolverRun() runs before any worker exists for the solve that follows.
  //
  // LINT.IfChange(locomotion_heuristics_updater_yaml_path)
  if (locomotionHeuristicLayerPtr_ != nullptr && pt.findChild(kLocomotionHeuristicsBlockKey) != nullptr) {
    const absl::StatusOr<LocomotionHeuristicConfig> heuristicConfig = loadLocomotionHeuristicConfig(yamlFile, /*verbose=*/false);
    if (!heuristicConfig.ok()) {
      // Reported and skipped rather than thrown: a half-typed coefficient in the tuning GUI must not take the
      // controller down, and the layer keeps running on the values it already has.
      LOG(WARNING) << "[MpcParameterUpdaterModule] locomotion_heuristics not applied: " << heuristicConfig.status().message();
    } else if (const absl::Status status = locomotionHeuristicLayerPtr_->reconfigure(*heuristicConfig); !status.ok()) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] locomotion_heuristics not applied: " << status.message();
    } else if (!locomotionHeuristicLayerPtr_->empty()) {
      LOG(INFO) << "[MpcParameterUpdaterModule] Applied the locomotion_heuristics coefficients from " << yamlFile << ":\n"
                << locomotionHeuristicLayerPtr_->summary();
    }
  }
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/locomotion_heuristics/LocomotionHeuristicConfig.cpp:locomotion_heuristic_keys)
  // clang-format on

  LOG(INFO) << "[MpcParameterUpdaterModule] Successfully applied in-place parameter updates to SqpSolver.";
}

void MpcParameterUpdaterModule::followAppliedTerrainHeight() {
  if (referenceManagerPtr_ == nullptr || mpcPtr_ == nullptr) {
    return;
  }
  const scalar_t appliedTerrainHeight = referenceManagerPtr_->getAppliedTerrainHeight();
  if (contactImplicitTermsTerrainHeight_.has_value() && *contactImplicitTermsTerrainHeight_ == appliedTerrainHeight) {
    return;
  }
  SqpSolver* sqpSolverPtr = dynamic_cast<SqpSolver*>(mpcPtr_->getSolverPtr());
  if (sqpSolverPtr == nullptr) {
    return;
  }
  setContactImplicitTermsTerrainHeight(sqpSolverPtr->getOcpDefinitions(), appliedTerrainHeight);
}

void MpcParameterUpdaterModule::setContactImplicitTermsTerrainHeight(std::vector<OptimalControlProblem>& ocpDefinitions,
                                                                     scalar_t terrainHeight) {
  // The two terms are a pair - one says a foot may not carry load above the ground, the other that it may not go below
  // it - so they always move together. Absent unless the contact-implicit formulation is listed, which is the normal case.
  for (OptimalControlProblem& ocp : ocpDefinitions) {
    for (const std::string& footName : contactNames_) {
      try {
        ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, contact_term::kContactComplementarity))
            .get<ContactComplementarityConstraint>()
            .setTerrainHeight(terrainHeight);
      } catch (const std::out_of_range&) {
      }
      try {
        ocp.stateSoftConstraintPtr->get<StateSoftConstraint>(contact_term::name(footName, contact_term::kGroundPenetration))
            .get<GroundPenetrationConstraint>()
            .setTerrainHeight(terrainHeight);
      } catch (const std::out_of_range&) {
      }
    }
  }
  contactImplicitTermsTerrainHeight_ = terrainHeight;
}

void MpcParameterUpdaterModule::applyContactPlanningUpdates(const std::string& yamlFile) {
  if (contactPlannerModulePtr_ == nullptr) return;
  // Applied before the planner's next run.
  //
  // The file is parsed WITHOUT the loader's own validation, which is exactly what CentroidalMpcInterface does at
  // start-up: the parameters a robot is allowed to leave at 0 in contact_planning.yaml to mean "derive this one from
  // the model" - shared.comHeight (both shipped robots) and the two zmp_support_region half widths - are only filled in
  // afterwards, by ContactPlanningModelParameters::applyTo(), which ContactPlannerModule::setConfig() runs before it
  // validates. This call used to validate the freshly parsed configuration before applyTo() could ever see it, the
  // exact inverse of the start-up order. The fields applyTo() fills are precisely the ones validation rejects at 0, so
  // on a robot that takes the documented option every reload was refused with "shared.comHeight must be positive", and
  // because the file watcher in preSolverRun() had already stored the new modification time, the edit was gone: every
  // later save of that file was discarded for the rest of the run while the tuning GUI reported each change as applied.
  // Nothing is left unvalidated by passing false here, since setConfig() applies the model parameters and then
  // validates, returning the rejection that names the key - so a genuinely inconsistent edit is still refused and still
  // reported.
  absl::StatusOr<ContactPlanningConfig> loaded =
      loadContactPlanningConfigStatus(yamlFile, "contact_planning.", /*verbose=*/false, /*validate=*/false);
  absl::Status applied = loaded.status();
  if (applied.ok()) applied = contactPlannerModulePtr_->setConfig(*loaded);
  if (applied.ok()) {
    LOG(INFO) << "[MpcParameterUpdaterModule] Applied the contact_planning configuration from " << yamlFile << ".";
  } else {
    LOG(WARNING) << "[MpcParameterUpdaterModule] contact_planning configuration of " << yamlFile
                 << " not applied, the running one is kept: " << applied.message();
  }
}

}  // namespace ocs2::humanoid
