/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
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

#include <cmath>
#include <string>
#include <utility>

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/LoadStdVectorOfPair.h>

#include <boost/optional.hpp>
#include <boost/property_tree/ptree.hpp>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/cost/QuadraticStateCost.h>

#include <ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/SquaredHingePenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include "humanoid_common_mpc/cost/BasePoseShapedQuadraticStateCost.h"

#include <humanoid_common_mpc/common/BasisInputsCostTransform.h>
#include <humanoid_common_mpc/constraint/FrictionForceConeConstraint.h>
#include <humanoid_common_mpc/constraint/ZeroWrenchConstraint.h>
#include <humanoid_common_mpc/contact/ContactRectangle.h>
#include <humanoid_common_mpc/cost/InputQuadraticCost.h>
#include <humanoid_common_mpc/cost/StateInputQuadraticCost.h>
#include <humanoid_common_mpc/cost/StateQuadraticCost.h>
#include <humanoid_common_mpc/pinocchio_model/pinocchioUtils.h>
#include "humanoid_common_mpc/constraint/ContactMomentXYConstraintCppAd.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"

namespace ocs2::humanoid {

namespace {

/**
 * The error for an input of com_and_acom_tracking_cost - the 3x3 matrix Q_com or Q_acom, or the terminal instance's
 * terminalCostScaling - that the task file does not carry. The loadData functions report a missing key by throwing;
 * here it is an InvalidArgument naming the key, because listing the cost without writing what it reads - the state the
 * G1 and SA01 task files are in, which carry no Q_com / Q_acom - is a configuration error fixed in the task file.
 */
absl::Status comAndAcomTrackingInputError(absl::string_view taskFile, absl::string_view key, absl::string_view what) {
  return absl::InvalidArgumentError(absl::StrCat("[HumanoidCostConstraintFactory] costs lists com_and_acom_tracking_cost, but '", key,
                                                 "' could not be loaded from ", taskFile, " (", what,
                                                 "). Write the 3x3 matrices Q_com and Q_acom in the task file (terminalCostScaling too "
                                                 "beside terminal_cost), or remove com_and_acom_tracking_cost from costs."));
}

/** Loads the 3x3 weight `key` (Q_com or Q_acom) of com_and_acom_tracking_cost, or returns comAndAcomTrackingInputError. */
absl::Status loadComAndAcomTrackingWeight(const std::string& taskFile, absl::string_view key, matrix_t& weight) {
  try {
    loadData::loadEigenMatrix(taskFile, std::string(key), weight);
  } catch (const std::exception& error) {
    return comAndAcomTrackingInputError(taskFile, key, error.what());
  }
  return absl::OkStatus();
}

/**
 * Reads the optional parameter `key` of a relaxed barrier into `value`, which keeps what it holds when the task file does
 * not carry the key. The value must parse as a number, be finite and be positive: a negative `mu` turns the barrier into
 * a reward for violating the constraint, a zero `mu` switches it off without saying so, and the relaxed log barrier is
 * undefined at a `delta` of zero or below. InvalidArgument naming the key otherwise.
 */
absl::Status loadPositiveBarrierParameter(const boost::property_tree::ptree& pt, const std::string& key, scalar_t& value, bool verbose) {
  const boost::optional<const boost::property_tree::ptree&> child = pt.get_child_optional(key);
  if (child) {
    const boost::optional<scalar_t> parsed = child->get_value_optional<scalar_t>();
    if (!parsed) {
      return absl::InvalidArgumentError(
          absl::StrCat("[HumanoidCostConstraintFactory] ", key, " is '", child->data(), "', which is not a number."));
    }
    value = *parsed;
  }
  if (!std::isfinite(value) || value <= 0.0) {
    return absl::InvalidArgumentError(absl::StrCat("[HumanoidCostConstraintFactory] ", key, " must be finite and positive, got ", value,
                                                   ": a negative barrier weight rewards violating the constraint, and the "
                                                   "relaxed barrier is undefined at a non-positive delta."));
  }
  if (verbose) {
    LOG(INFO) << " #### " << key << " = " << value << (child ? "" : " (default)");
  }
  return absl::OkStatus();
}

}  // namespace

HumanoidCostConstraintFactory::HumanoidCostConstraintFactory(const std::string& taskFile,
                                                             const std::string& referenceFile,
                                                             const SwitchedModelReferenceManager& referenceManager,
                                                             const PinocchioInterface& pinocchioInterface,
                                                             const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                             const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                                             const ModelSettings& modelSettings,
                                                             bool verbose,
                                                             bool scheduleGatedContactConstraints)
    : taskFile_(taskFile),
      referenceFile_(referenceFile),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfacePtr_(&pinocchioInterface),
      mpcRobotModelPtr_(&mpcRobotModel),
      mpcRobotModelADPtr_(&mpcRobotModelAD),
      modelSettings_(modelSettings),
      verbose_(verbose),
      scheduleGatedContactConstraints_(scheduleGatedContactConstraints) {}

/**
 * The penalty a contact cone is wrapped in.
 *
 * A schedule-gated cone is only ever evaluated on a foot the schedule has declared loaded, where the slack is
 * comfortably positive and a relaxed log barrier is the right object. An always-active cone is also evaluated on a
 * foot in flight, which sits exactly ON the boundary of the homogeneous cone. A relaxed log barrier has value
 * -mu*ln(delta) - mu/2 and derivative -2*mu/delta there - with the shipped center-of-pressure settings, mu = 0.6 and
 * delta = 0.03, that is a derivative of -40 per row - so it would pay the solver to leave the origin, i.e. to invent a
 * normal force on a foot in the air. That is precisely the failure that dropping `minNormalForce` and the friction
 * cone's parabolic margin exists to remove, and a log barrier would put it straight back.
 *
 * The squared hinge is zero in value AND zero in derivative at zero slack, and quadratic below it, which is what a
 * unilateral condition whose solution lies on the boundary needs. `delta` is 0 for exactly that reason: a positive
 * delta shifts the hinge's zero into the interior and reintroduces the bias. It is DROPPED, not converted - the
 * barrier's `delta` is a smoothing width in the units of the constraint row, while the hinge's is the offset of its
 * zero, and the two have nothing to do with one another beyond the name.
 *
 * `mu` IS REUSED, AND IT MEANS SOMETHING DIFFERENT IN THE TWO PENALTIES, so the substitution is not scale-preserving
 * and that has to be stated rather than discovered. In the barrier `mu` is a log coefficient whose gradient near the
 * boundary goes as mu/delta; in the hinge it is the quadratic stiffness of `0.5 * mu * h^2`. With the shipped cone
 * settings (mu = 0.2, delta = 5, rows in newtons) the hinge is the STRONGER of the two everywhere a violation is
 * worth caring about - equal at 0.42 N, 2.3x the barrier's restoring gradient at 1 N, 12.5x at 10 N, 22.7x at 100 N,
 * and 25x its curvature throughout. Carrying `mu` across therefore tightens the cone rather than loosening it, which
 * is the safe direction for a bound that is the ONLY thing holding a wrench inside the cone once the schedule gate is
 * gone. Matching the barrier's curvature instead would mean mu/delta^2 = 0.008, an eightyfold softening, and is not
 * what is wanted here.
 */
std::unique_ptr<PenaltyBase> makeContactConePenalty(const RelaxedBarrierPenalty::Config& barrierConfig, bool scheduleGated) {
  const vector_t parameters = contactConePenaltyParameters(barrierConfig, scheduleGated);
  if (scheduleGated) {
    return std::unique_ptr<PenaltyBase>(new RelaxedBarrierPenalty(RelaxedBarrierPenalty::Config(parameters(0), parameters(1))));
  }
  LOG(INFO) << "[HumanoidCostConstraintFactory] Un-gated cone: squared hinge with stiffness mu = " << parameters(0)
            << " (cost 0.5*mu*h^2 below the cone, zero on and above it). The configured barrier delta of " << barrierConfig.delta
            << " is deliberately not carried over; see makeContactConePenalty().";
  return std::unique_ptr<PenaltyBase>(new SquaredHingePenalty(SquaredHingePenalty::Config(parameters(0), parameters(1))));
}

vector_t contactConePenaltyParameters(const RelaxedBarrierPenalty::Config& barrierConfig, bool scheduleGated) {
  // An un-gated cone's hinge keeps its zero on the cone, which a foot at zero wrench lies on: its delta is the OFFSET of
  // that zero, not the barrier's smoothing width, and is 0 whatever the file's barrier delta says.
  return (vector_t(2) << barrierConfig.mu, scheduleGated ? barrierConfig.delta : 0.0).finished();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void HumanoidCostConstraintFactory::setBasisInputsCostTransform(BasisInputsCostTransformConfig config) {
  basisCostTransform_ = std::move(config);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

matrix_t HumanoidCostConstraintFactory::loadAndTransformR() const {
  if (basisCostTransform_.has_value()) {
    // R is written in wrench dimensions; the basis-vector input needs R_basis = M^T R_wrench M + reg * blkdiag(S, 0).
    const size_t wrenchInputDim = basisCostTransform_->wrenchInputDim;
    matrix_t R_wrench(wrenchInputDim, wrenchInputDim);
    loadData::loadEigenMatrix(taskFile_, "R", R_wrench);
    matrix_t R_basis = transformWrenchInputCostToBasisSpace(R_wrench, *basisCostTransform_);
    if (verbose_) {
      LOG(INFO) << "\n #### R cost loaded in wrench space (" << wrenchInputDim << "x" << wrenchInputDim
                << ") and transformed to basis-vector space (" << R_basis.rows() << "x" << R_basis.cols() << ", " << kBasisRegularizationKey
                << " = " << basisCostTransform_->regularization << ", " << kBasisScalingRegularizationKey << " = "
                << basisCostTransform_->lambdaRegularization << ")";
    }
    return R_basis;
  } else {
    matrix_t R(mpcRobotModelADPtr_->getInputDim(), mpcRobotModelADPtr_->getInputDim());
    loadData::loadEigenMatrix(taskFile_, "R", R);
    return R;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputCost> HumanoidCostConstraintFactory::getStateInputQuadraticCost() const {
  matrix_t Q(mpcRobotModelADPtr_->getStateDim(), mpcRobotModelADPtr_->getStateDim());
  loadData::loadEigenMatrix(taskFile_, "Q", Q);
  // The same replacement as in getStateQuadraticCost(): with com_and_acom_tracking_cost listed the base pose is
  // regulated in CoM and ACoM coordinates, whichever of the two quadratic state costs carries the rest of Q.
  if (comAndAcomTrackingCostListed_) {
    ComAndAcomTrackingCost::zeroBasePoseWeights(Q);
    if (verbose_) {
      LOG(INFO) << "[HumanoidCostConstraintFactory] com_and_acom_tracking_cost is listed. Zeroing out base pose weights in Q.";
    }
  }
  matrix_t R = loadAndTransformR();

  if (verbose_) {
    LOG(INFO) << "\n #### Base Tracking Cost Coefficients: \n"
              << " #### =============================================================================\n"
              << "Q:\n"
              << Q << "\n"
              << "R:\n"
              << R << "\n"
              << " #### =============================================================================";
  }

  return std::unique_ptr<StateInputCost>(
      new StateInputQuadraticCost(std::move(Q), std::move(R), *referenceManagerPtr_, *mpcRobotModelPtr_));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputCost> HumanoidCostConstraintFactory::getStateQuadraticCost() const {
  matrix_t Q(mpcRobotModelADPtr_->getStateDim(), mpcRobotModelADPtr_->getStateDim());
  loadData::loadEigenMatrix(taskFile_, "Q", Q);

  if (comAndAcomTrackingCostListed_) {
    ComAndAcomTrackingCost::zeroBasePoseWeights(Q);
    if (verbose_) {
      LOG(INFO) << "[HumanoidCostConstraintFactory] com_and_acom_tracking_cost is listed. Zeroing out base pose weights in Q.";
    }
  }

  if (verbose_) {
    LOG(INFO) << "\n #### Base Tracking State Cost Coefficients: \n"
              << " #### =============================================================================\n"
              << "Q:\n"
              << Q << "\n"
              << " #### =============================================================================";
  }

  return std::unique_ptr<StateInputCost>(new StateQuadraticCost(std::move(Q), mpcRobotModelADPtr_->getInputDim(), *referenceManagerPtr_));
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::makeComAndAcomTrackingCost(const CentroidalModelInfo& info,
                                                                                                     scalar_t scaling) const {
  matrix_t Q_com(3, 3);
  matrix_t Q_acom(3, 3);
  RETURN_IF_ERROR(loadComAndAcomTrackingWeight(taskFile_, "Q_com", Q_com));
  RETURN_IF_ERROR(loadComAndAcomTrackingWeight(taskFile_, "Q_acom", Q_acom));
  Q_com *= scaling;
  Q_acom *= scaling;

  if (verbose_) {
    LOG(INFO) << "\n #### CoM + ACoM Tracking Cost Coefficients (scaled by " << scaling << "): \n"
              << " #### =============================================================================\n"
              << "Q_com:\n"
              << Q_com << "\n"
              << "Q_acom (rows are ZYX Euler: yaw, pitch, roll):\n"
              << Q_acom << "\n"
              << " #### =============================================================================";
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<ComAndAcomTrackingCost> cost,
      ComAndAcomTrackingCost::Create(std::move(Q_com), std::move(Q_acom), *pinocchioInterfacePtr_, info, modelSettings_.robotName));
  return std::unique_ptr<StateCost>(std::move(cost));
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::getComAndAcomTrackingCost(const CentroidalModelInfo& info) const {
  return makeComAndAcomTrackingCost(info, /*scaling=*/1.0);
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::getTerminalComAndAcomTrackingCost(
    const CentroidalModelInfo& info) const {
  // The factor getTerminalCost() applies to Q_final, so that the terminal node weighs the CoM and the ACoM against the
  // rest of the terminal state exactly as the running nodes weigh them against the rest of Q.
  scalar_t terminalCostScaling = 1.0;
  try {
    loadData::loadCppDataType<scalar_t>(taskFile_, "terminalCostScaling", terminalCostScaling);
  } catch (const std::exception& error) {
    return comAndAcomTrackingInputError(taskFile_, "terminalCostScaling", error.what());
  }
  return makeComAndAcomTrackingCost(info, terminalCostScaling);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputCost> HumanoidCostConstraintFactory::getInputQuadraticCost() const {
  matrix_t R = loadAndTransformR();

  if (verbose_) {
    LOG(INFO) << "\n #### Base Tracking Input Cost Coefficients: \n"
              << " #### =============================================================================\n"
              << "R:\n"
              << R << "\n"
              << " #### =============================================================================";
  }

  return std::unique_ptr<StateInputCost>(
      new InputQuadraticCost(std::move(R), mpcRobotModelADPtr_->getStateDim(), *referenceManagerPtr_, *mpcRobotModelPtr_));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateCost> HumanoidCostConstraintFactory::getFootCollisionConstraint() const {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile_, pt);
  const std::string prefix = "collision_constraint.";

  FootCollisionConstraint::Config collisionConfig = FootCollisionConstraint::loadFootCollisionConstraintConfig(taskFile_, verbose_);
  PieceWisePolynomialBarrierPenalty::Config barrierPenaltyConfig;

  loadData::loadPtreeValue(pt, barrierPenaltyConfig.mu, absl::StrCat(prefix, "mu"), verbose_);
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.delta, absl::StrCat(prefix, "delta"), verbose_);

  std::unique_ptr<PieceWisePolynomialBarrierPenalty> penalty(new PieceWisePolynomialBarrierPenalty(barrierPenaltyConfig));

  std::unique_ptr<FootCollisionConstraint> footCollisionConstraintPtr(
      new FootCollisionConstraint(*referenceManagerPtr_, *pinocchioInterfacePtr_, *mpcRobotModelADPtr_, std::move(collisionConfig),
                                  "FootCollisionConstraint", modelSettings_));

  return std::unique_ptr<StateCost>(new StateSoftConstraint(std::move(footCollisionConstraintPtr), std::move(penalty)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateCost> HumanoidCostConstraintFactory::getJointLimitsConstraint() const {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile_, pt);
  const std::string prefix = "jointLimits.";

  PieceWisePolynomialBarrierPenalty::Config barrierPenaltyConfig;

  if (verbose_) {
    LOG(INFO) << "\n #### Joint Limit Barrier Penalty Config: \n"
              << " #### =============================================================================";
  }
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.mu, absl::StrCat(prefix, "mu"), verbose_);
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.delta, absl::StrCat(prefix, "delta"), verbose_);
  if (verbose_) {
    LOG(INFO) << " #### =============================================================================";
  }

  LOG(INFO) << "Initialized joint limits constraint with zero crossing cost " << barrierPenaltyConfig.getZeroCrossingValue() << ".";

  std::pair<vector_t, vector_t> jointLimits = readPinocchioJointLimits(*pinocchioInterfacePtr_, mpcRobotModelPtr_->modelSettings);

  return std::unique_ptr<StateCost>(new JointLimitsSoftConstraint(jointLimits, barrierPenaltyConfig, *mpcRobotModelPtr_));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputCost> HumanoidCostConstraintFactory::getContactMomentXYConstraint(size_t contactPointIndex,
                                                                                            const std::string& name) const {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile_, pt);
  const std::string prefix = "contacts.contactMomentXYSoftConstraint.";

  RelaxedBarrierPenalty::Config barrierPenaltyConfig;
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.mu, absl::StrCat(prefix, "mu"), verbose_);
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.delta, absl::StrCat(prefix, "delta"), verbose_);

  std::unique_ptr<ContactMomentXYConstraintCppAd> contactMomentXYConstraintPtr(new ContactMomentXYConstraintCppAd(
      *referenceManagerPtr_, ContactRectangle::loadContactRectangle(taskFile_, mpcRobotModelPtr_->modelSettings, contactPointIndex),
      contactPointIndex, *pinocchioInterfacePtr_, *mpcRobotModelADPtr_, name, modelSettings_, scheduleGatedContactConstraints_));

  std::unique_ptr<PenaltyBase> penalty = makeContactConePenalty(barrierPenaltyConfig, scheduleGatedContactConstraints_);

  return std::unique_ptr<StateInputCost>(new StateInputSoftConstraint(std::move(contactMomentXYConstraintPtr), std::move(penalty)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<StateInputCost>> HumanoidCostConstraintFactory::getContactWrenchConeConstraint(
    size_t contactPointIndex) const {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile_, pt);
  const std::string prefix = absl::StrCat(ContactWrenchConeConstraint::kConfigBlock, ".");

  // The barrier's weight and relaxation, range-checked by their keys like the ground below: a negative mu would reward
  // leaving the cone. A key the file does not carry keeps RelaxedBarrierPenalty's default.
  RelaxedBarrierPenalty::Config barrierPenaltyConfig;
  RETURN_IF_ERROR(loadPositiveBarrierParameter(pt, absl::StrCat(prefix, "mu"), barrierPenaltyConfig.mu, verbose_));
  RETURN_IF_ERROR(loadPositiveBarrierParameter(pt, absl::StrCat(prefix, "delta"), barrierPenaltyConfig.delta, verbose_));
  if (verbose_) {
    LOG(INFO) << " #### =============================================================================";
  }

  // The ground itself - friction, torsion, the affine offsets and the pyramid's facets - is read by the loader every
  // consumer of this block shares, which refuses a missing key instead of keeping a library default.
  ASSIGN_OR_RETURN(ContactWrenchConeConstraint::Config config, ContactWrenchConeConstraint::loadConfig(taskFile_, verbose_));

  ASSIGN_OR_RETURN(
      std::unique_ptr<ContactWrenchConeConstraint> contactWrenchConeConstraintPtr,
      ContactWrenchConeConstraint::Create(
          *referenceManagerPtr_, ContactRectangle::loadContactRectangle(taskFile_, mpcRobotModelPtr_->modelSettings, contactPointIndex),
          contactPointIndex, *pinocchioInterfacePtr_, *mpcRobotModelPtr_, std::move(config), scheduleGatedContactConstraints_));

  std::unique_ptr<PenaltyBase> penalty = makeContactConePenalty(barrierPenaltyConfig, scheduleGatedContactConstraints_);

  return std::unique_ptr<StateInputCost>(new StateInputSoftConstraint(std::move(contactWrenchConeConstraintPtr), std::move(penalty)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputConstraint> HumanoidCostConstraintFactory::getZeroWrenchConstraint(size_t contactPointIndex) const {
  return std::unique_ptr<StateInputConstraint>(new ZeroWrenchConstraint(*referenceManagerPtr_, contactPointIndex, *mpcRobotModelPtr_));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputCost> HumanoidCostConstraintFactory::getFrictionForceConeConstraint(size_t contactPointIndex) const {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile_, pt);
  const std::string prefix = "contacts.frictionForceConeSoftConstraint.";

  scalar_t frictionCoefficient = 1.0;
  RelaxedBarrierPenalty::Config barrierPenaltyConfig;
  if (verbose_) {
    LOG(INFO) << "\n #### Friction Cone Settings: \n"
              << " #### =============================================================================";
  }
  loadData::loadPtreeValue(pt, frictionCoefficient, absl::StrCat(prefix, "frictionCoefficient"), verbose_);
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.mu, absl::StrCat(prefix, "mu"), verbose_);
  loadData::loadPtreeValue(pt, barrierPenaltyConfig.delta, absl::StrCat(prefix, "delta"), verbose_);
  if (verbose_) {
    LOG(INFO) << " #### =============================================================================";
  }

  FrictionForceConeConstraint::Config frictionConeConConfig(frictionCoefficient);
  std::unique_ptr<FrictionForceConeConstraint> frictionForceConeConstraintPtr(new FrictionForceConeConstraint(
      *referenceManagerPtr_, std::move(frictionConeConConfig), contactPointIndex, *mpcRobotModelPtr_, scheduleGatedContactConstraints_));

  std::unique_ptr<PenaltyBase> penalty = makeContactConePenalty(barrierPenaltyConfig, scheduleGatedContactConstraints_);

  return std::unique_ptr<StateInputCost>(new StateInputSoftConstraint(std::move(frictionForceConeConstraintPtr), std::move(penalty)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateCost> HumanoidCostConstraintFactory::getTerminalCost() const {
  scalar_t terminalCostScaling;
  loadData::loadCppDataType<scalar_t>(taskFile_, "terminalCostScaling", terminalCostScaling);
  LOG(INFO) << "terminalCostScaling: " << terminalCostScaling;

  matrix_t Qf(mpcRobotModelPtr_->getStateDim(), mpcRobotModelPtr_->getStateDim());
  loadData::loadEigenMatrix(taskFile_, "Q_final", Qf);

  if (comAndAcomTrackingCostListed_) {
    // The running cost's base pose block is zeroed for the same reason. Leaving it
    // in the terminal cost would regulate the end of the horizon in base
    // coordinates while every other node is regulated in aCOM coordinates, and
    // terminalCostScaling makes that mismatch the dominant term at the horizon end.
    // What regulates the terminal base pose instead is getTerminalComAndAcomTrackingCost(), which the interface adds
    // beside this cost; without it the last node would carry no CoM, height or orientation weight at all.
    ComAndAcomTrackingCost::zeroBasePoseWeights(Qf);
    if (verbose_) {
      LOG(INFO) << "[HumanoidCostConstraintFactory] com_and_acom_tracking_cost is listed. Zeroing out base pose weights in Q_final.";
    }
  }

  Qf *= terminalCostScaling;
  if (verbose_) LOG(INFO) << "Q_final:\n" << Qf;
  // Shaped by the locomotion heuristics' base-pose offsets like the running costs, so the horizon's last node does not
  // pull back to the unshaped pose; identical to QuadraticStateCost when none is listed.
  return std::unique_ptr<StateCost>(new BasePoseShapedQuadraticStateCost(Qf, *referenceManagerPtr_));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputCost> HumanoidCostConstraintFactory::getExternalTorqueQuadraticCost(size_t contactPointIndex) const {
  std::string fieldName;
  if (contactPointIndex == 0) {
    fieldName = "left_leg_torque_cost.";
  }
  if (contactPointIndex == 1) {
    fieldName = "right_leg_torque_cost.";
  }
  ExternalTorqueQuadraticCostAD::Config config = ExternalTorqueQuadraticCostAD::loadConfigFromFile(taskFile_, fieldName, verbose_);
  return std::make_unique<ExternalTorqueQuadraticCostAD>(contactPointIndex, config, *referenceManagerPtr_, *pinocchioInterfacePtr_,
                                                         *mpcRobotModelADPtr_, modelSettings_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

}  // namespace ocs2::humanoid
