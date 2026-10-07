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

#include "support/ProblemFingerprint.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_oc/approximate_model/LinearQuadraticApproximator.h"
#include "ocs2_oc/oc_problem/OptimalControlProblemHelperFunction.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid::test {
namespace {

// The base position in the centroidal state x = [h_norm(6), p_base(3), euler_zyx(3), q_j], written out here rather
// than taken from the model, so that a drift of the one definition shows up as a failure instead of moving the
// reference with it.
constexpr Eigen::Index kBasePositionIndex = 6;

template <typename CollectionT>
void appendTermNames(Fingerprint& fingerprint, absl::string_view label, const CollectionT& collection) {
  std::vector<std::string> names;
  for (const std::pair<const std::string, size_t>& entry : collection.getTermNameMap()) {
    names.push_back(entry.first);
  }
  std::sort(names.begin(), names.end());
  std::string line = absl::StrCat(label, ":");
  for (const std::string& name : names) absl::StrAppend(&line, " ", name);
  fingerprint.emplace_back(line, matrix_t());
}

void append(Fingerprint& fingerprint, const std::string& label, const matrix_t& value) {
  fingerprint.emplace_back(label, value);
}

}  // namespace

void setWalkingReferences(CentroidalMpcInterface& interface) {
  const vector_t& x0 = interface.getInitialState();
  const Eigen::Index nu = static_cast<Eigen::Index>(interface.getEffectiveMpcRobotModel().getInputDim());
  const ModeSchedule schedule({0.2, 0.55, 0.65, 1.0, 1.1, 1.45},
                              {ModeNumber::kStance, ModeNumber::kLf, ModeNumber::kStance, ModeNumber::kRf, ModeNumber::kStance,
                               ModeNumber::kLf, ModeNumber::kStance});
  vector_t xStart = x0;
  xStart(0) = 0.4;  // forward normalized momentum: a walking command
  vector_t xEnd = xStart;
  xEnd(kBasePositionIndex) += 0.8;
  const vector_t u0 = vector_t::Zero(nu);
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = interface.getSwitchedModelReferenceManagerPtr();
  referenceManager->getGaitSchedule()->updateModeSchedule(schedule);
  referenceManager->setTargetTrajectories(TargetTrajectories({0.0, 2.0}, {xStart, xEnd}, {u0, u0}));
  referenceManager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.5, x0, ModeNumber::kStance);
  interface.getOptimalControlProblemRef().targetTrajectoriesPtr = &referenceManager->getTargetTrajectories();
}

Fingerprint fingerprintOf(CentroidalMpcInterface& interface) {
  Fingerprint fingerprint;
  OptimalControlProblem& problem = interface.getOptimalControlProblemRef();
  appendTermNames(fingerprint, "costPtr", *problem.costPtr);
  appendTermNames(fingerprint, "stateCostPtr", *problem.stateCostPtr);
  appendTermNames(fingerprint, "finalCostPtr", *problem.finalCostPtr);
  appendTermNames(fingerprint, "softConstraintPtr", *problem.softConstraintPtr);
  appendTermNames(fingerprint, "stateSoftConstraintPtr", *problem.stateSoftConstraintPtr);
  appendTermNames(fingerprint, "finalSoftConstraintPtr", *problem.finalSoftConstraintPtr);
  appendTermNames(fingerprint, "equalityConstraintPtr", *problem.equalityConstraintPtr);

  setWalkingReferences(interface);
  const vector_t x0 = interface.getInitialState();
  const Eigen::Index nx = x0.size();
  const Eigen::Index nu = static_cast<Eigen::Index>(interface.getEffectiveMpcRobotModel().getInputDim());
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = interface.getSwitchedModelReferenceManagerPtr();

  const std::vector<scalar_t> times = {0.05, 0.3, 0.6, 0.8, 1.2};
  for (size_t k = 0; k < times.size(); ++k) {
    const scalar_t t = times[k];
    vector_t x = x0;
    for (Eigen::Index i = 0; i < nx; ++i) x(i) += 0.01 * std::sin(0.7 * static_cast<scalar_t>(i) + 1.3 * static_cast<scalar_t>(k));
    vector_t u(nu);
    for (Eigen::Index i = 0; i < nu; ++i) u(i) = 5.0 + 3.0 * std::cos(0.4 * static_cast<scalar_t>(i) + static_cast<scalar_t>(k));
    const std::string at = absl::StrCat("@", t);
    append(fingerprint, absl::StrCat("desiredState", at), referenceManager->getDesiredState(*problem.targetTrajectoriesPtr, x, t));
    MultiplierCollection multipliers;
    initializeIntermediateMultiplierCollection(problem, t, multipliers);
    const ModelData model = approximateIntermediateLQ(problem, t, x, u, multipliers);
    append(fingerprint, absl::StrCat("cost.f", at), matrix_t::Constant(1, 1, model.cost.f));
    append(fingerprint, absl::StrCat("cost.dfdx", at), model.cost.dfdx);
    append(fingerprint, absl::StrCat("cost.dfdu", at), model.cost.dfdu);
    append(fingerprint, absl::StrCat("cost.dfdxx", at), model.cost.dfdxx);
    append(fingerprint, absl::StrCat("cost.dfdux", at), model.cost.dfdux);
    append(fingerprint, absl::StrCat("cost.dfduu", at), model.cost.dfduu);
    append(fingerprint, absl::StrCat("dynamics.f", at), model.dynamics.f);
    append(fingerprint, absl::StrCat("dynamics.dfdx", at), model.dynamics.dfdx);
    append(fingerprint, absl::StrCat("dynamics.dfdu", at), model.dynamics.dfdu);
    append(fingerprint, absl::StrCat("equality.f", at), model.stateInputEqConstraint.f);
    append(fingerprint, absl::StrCat("equality.dfdx", at), model.stateInputEqConstraint.dfdx);
    append(fingerprint, absl::StrCat("equality.dfdu", at), model.stateInputEqConstraint.dfdu);
  }
  const scalar_t finalTime = 1.5;
  vector_t xFinal = x0;
  for (Eigen::Index i = 0; i < nx; ++i) xFinal(i) += 0.02 * std::cos(0.3 * static_cast<scalar_t>(i));
  MultiplierCollection multipliers;
  initializeFinalMultiplierCollection(problem, finalTime, multipliers);
  const ModelData finalModel = approximateFinalLQ(problem, finalTime, xFinal, multipliers);
  append(fingerprint, "final.f", matrix_t::Constant(1, 1, finalModel.cost.f));
  append(fingerprint, "final.dfdx", finalModel.cost.dfdx);
  append(fingerprint, "final.dfdxx", finalModel.cost.dfdxx);
  return fingerprint;
}

::testing::AssertionResult identical(const Fingerprint& a, const Fingerprint& b) {
  if (a.size() != b.size()) {
    return ::testing::AssertionFailure() << "the fingerprints have " << a.size() << " and " << b.size() << " entries";
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].first != b[i].first) {
      return ::testing::AssertionFailure() << "entry " << i << " is '" << a[i].first << "' against '" << b[i].first << "'";
    }
    const matrix_t& x = a[i].second;
    const matrix_t& y = b[i].second;
    if (x.rows() != y.rows() || x.cols() != y.cols()) {
      return ::testing::AssertionFailure() << a[i].first << " is " << x.rows() << "x" << x.cols() << " against " << y.rows() << "x"
                                           << y.cols();
    }
    if (!(x.array() == y.array()).all()) {
      return ::testing::AssertionFailure() << a[i].first << " differs, by up to " << (x - y).cwiseAbs().maxCoeff();
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace ocs2::humanoid::test
