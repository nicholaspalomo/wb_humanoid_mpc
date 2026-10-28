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

#include "humanoid_multi_contact/stability/Stance.h"

#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// How far a stance rotation may be from orthonormal.
constexpr double kRotationTolerance = 1.0e-6;

absl::Status checkContact(const StanceContact& contact) {
  if (!(contact.friction > 0.0) || !std::isfinite(contact.friction)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "stance contact '", contact.name, "': friction must be positive"));
  }
  if (!contact.position.allFinite() ||
      !(contact.rotation.transpose() * contact.rotation)
           .isIdentity(kRotationTolerance) ||
      !(contact.rotation.determinant() > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("stance contact '", contact.name,
                     "': needs a finite position and a rotation matrix"));
  }
  if (contact.kind == ContactKind::kSurface && !contact.rectangle.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "stance contact '", contact.name, "': a surface needs a rectangle"));
  }
  return absl::OkStatus();
}

/** The points a contact applies its pyramids at, in the world frame. */
std::vector<Eigen::Vector3d> applicationPoints(const StanceContact& contact) {
  if (contact.kind == ContactKind::kPoint) return {contact.position};
  const ContactRectangle& rectangle = *contact.rectangle;
  const std::array<Eigen::Vector3d, kRectangleCorners> corners = {
      Eigen::Vector3d(rectangle.xMax, rectangle.yMax, 0.0),
      Eigen::Vector3d(rectangle.xMin, rectangle.yMax, 0.0),
      Eigen::Vector3d(rectangle.xMin, rectangle.yMin, 0.0),
      Eigen::Vector3d(rectangle.xMax, rectangle.yMin, 0.0)};
  std::vector<Eigen::Vector3d> points;
  points.reserve(corners.size());
  for (const Eigen::Vector3d& corner : corners) {
    points.push_back(contact.position + contact.rotation * corner);
  }
  return points;
}

}  // namespace

absl::StatusOr<WrenchGenerators> buildWrenchGenerators(const Stance& stance,
                                                       int edges) {
  if (edges < kMinFrictionConeEdges) {
    return absl::InvalidArgumentError(
        absl::StrCat("a friction cone needs at least ", kMinFrictionConeEdges,
                     " edges, not ", edges));
  }
  if (stance.empty()) {
    return absl::InvalidArgumentError("the stance has no contact");
  }
  WrenchGenerators generators;
  for (const StanceContact& contact : stance) {
    if (absl::Status status = checkContact(contact); !status.ok()) {
      return status;
    }
    generators.reference += contact.position;
  }
  generators.reference /= static_cast<double>(stance.size());

  const double halfAngle = std::numbers::pi / edges;
  for (int index = 0; index < static_cast<int>(stance.size()); ++index) {
    const StanceContact& contact = stance[index];
    // The inscribed pyramid's tangential reach per unit normal force.
    const double inscribedFriction = contact.friction * std::cos(halfAngle);
    const Eigen::Vector3d normal = contact.rotation.col(kZAxisIndex);
    for (const Eigen::Vector3d& point : applicationPoints(contact)) {
      for (int edge = 0; edge < edges; ++edge) {
        const double angle = 2.0 * halfAngle * edge;
        const Eigen::Vector3d direction =
            normal + inscribedFriction *
                         (std::cos(angle) * contact.rotation.col(kXAxisIndex) +
                          std::sin(angle) * contact.rotation.col(kYAxisIndex));
        generators.contactIndices.push_back(index);
        generators.points.push_back(point);
        generators.directions.push_back(direction);
      }
    }
  }
  generators.wrenches.resize(kWrenchDim, generators.points.size());
  for (int column = 0; column < generators.size(); ++column) {
    const Eigen::Vector3d& direction = generators.directions[column];
    generators.wrenches.col(column).head<kForceDim>() = direction;
    generators.wrenches.col(column).tail<kTorqueDim>() =
        (generators.points[column] - generators.reference).cross(direction);
  }
  return generators;
}

}  // namespace ocs2::humanoid::multi_contact
