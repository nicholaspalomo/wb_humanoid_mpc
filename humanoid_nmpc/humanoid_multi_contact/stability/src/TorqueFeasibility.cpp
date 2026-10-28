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

#include "humanoid_multi_contact/stability/TorqueFeasibility.h"

#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::multi_contact {
namespace {

/** The generalized forces of each generator at unit scaling (one weight along
 * its direction): weight * J_g^T e_g, one column per generator. */
Eigen::MatrixXd generatorForces(const RobotStanceModel& model,
                                const StaticEquilibrium& equilibrium) {
  const WrenchGenerators& generators = equilibrium.generators();
  const double weight = model.mass() * kGravitationalAcceleration;
  Eigen::MatrixXd forces(model.model().nv, generators.size());
  for (int g = 0; g < generators.size(); ++g) {
    const int candidate =
        equilibrium.stance()[generators.contactIndices[g]].candidateIndex;
    forces.col(g) =
        weight *
        model.pointJacobian(candidate, generators.points[g]).transpose() *
        generators.directions[g];
  }
  return forces;
}

}  // namespace

absl::StatusOr<TorqueFeasibilityResult> minimumTorqueUtilization(
    const RobotStanceModel& model, const StaticEquilibrium& equilibrium) {
  for (const StanceContact& contact : equilibrium.stance()) {
    if (contact.candidateIndex < 0 ||
        contact.candidateIndex >= model.contactSet().size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("stance contact '", contact.name,
                       "' names no candidate of the model"));
    }
  }
  const Eigen::MatrixXd forces = generatorForces(model, equilibrium);
  const Eigen::VectorXd& gravity = model.generalizedGravity();
  const int generators = equilibrium.generators().size();

  // The joints with a limit: tau_j / limit_j = gamma_j - T_j lambda.
  std::vector<int> checked;
  for (int j = 0; j < static_cast<int>(model.actuatedJoints().size()); ++j) {
    if (model.actuatedJoints()[j].effortLimit > 0.0) checked.push_back(j);
  }
  const int rows = static_cast<int>(checked.size());
  // Rows [-T, -1] <= -gamma and [T, -1] <= gamma, over [lambda; u].
  Eigen::MatrixXd constraints =
      Eigen::MatrixXd::Zero(rows + rows, generators + 1);
  Eigen::VectorXd upper(rows + rows);
  for (int r = 0; r < rows; ++r) {
    const ActuatedJoint& joint = model.actuatedJoints()[checked[r]];
    const Eigen::RowVectorXd scaled =
        forces.row(joint.velocityIndex) / joint.effortLimit;
    const double gamma = gravity(joint.velocityIndex) / joint.effortLimit;
    constraints.row(r).head(generators) = -scaled;
    constraints(r, generators) = -1.0;
    upper(r) = -gamma;
    constraints.row(rows + r).head(generators) = scaled;
    constraints(rows + r, generators) = -1.0;
    upper(rows + r) = gamma;
  }
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(
      rows + rows, -std::numeric_limits<double>::infinity());
  const Eigen::Vector2d comXy =
      model.centerOfMass().head<kHorizontalPositionDim>();
  ASSIGN_OR_RETURN(const Eigen::VectorXd solution,
                   equilibrium.solveWithExtraVariables(
                       equilibrium.staticWrench(comXy),
                       Eigen::VectorXd::Ones(1), constraints, lower, upper));

  TorqueFeasibilityResult result;
  result.scalings = solution.head(generators);
  const Eigen::VectorXd generalized = gravity - forces * result.scalings;
  result.jointTorques.resize(model.actuatedJoints().size());
  for (int j = 0; j < static_cast<int>(model.actuatedJoints().size()); ++j) {
    const ActuatedJoint& joint = model.actuatedJoints()[j];
    result.jointTorques(j) = generalized(joint.velocityIndex);
    if (joint.effortLimit > 0.0) {
      const double utilization =
          std::abs(result.jointTorques(j)) / joint.effortLimit;
      if (result.maxTorqueJoint.empty() || utilization > result.utilization) {
        result.utilization = utilization;
        result.maxTorqueJoint = joint.name;
      }
    }
  }
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
