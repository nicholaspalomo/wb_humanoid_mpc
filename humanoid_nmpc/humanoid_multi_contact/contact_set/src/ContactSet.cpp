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

#include "humanoid_multi_contact/contact_set/ContactSet.h"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/contact_set/contact_set.nproto.pb.h"
#include "humanoid_multi_contact/contact_set/contact_set.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// How far from 1 the norm of a pad's normal or x axis, and how far from 0 their
// dot product, may be; they are then normalized and orthogonalized.
constexpr double kUnitTolerance = 1.0e-3;

using Candidate = config::ContactSet::Candidate;

absl::Status candidateError(absl::string_view robot,
                            absl::string_view name,
                            absl::string_view what) {
  return absl::InvalidArgumentError(absl::StrCat(
      "contact set of '", robot, "': candidate '", name, "': ", what));
}

absl::StatusOr<Eigen::Vector3d> vector3(const Eigen::VectorXd& values,
                                        absl::string_view robot,
                                        absl::string_view name,
                                        absl::string_view field) {
  if (values.size() != kPositionDim || !values.allFinite()) {
    return candidateError(robot, name,
                          absl::StrCat(field, " must hold ", kPositionDim,
                                       " finite values, not ", values.size()));
  }
  return Eigen::Vector3d(values);
}

absl::StatusOr<Eigen::Vector3d> unitVector3(const Eigen::VectorXd& values,
                                            absl::string_view robot,
                                            absl::string_view name,
                                            absl::string_view field) {
  ASSIGN_OR_RETURN(Eigen::Vector3d vector, vector3(values, robot, name, field));
  if (!(std::abs(vector.norm() - 1.0) <= kUnitTolerance)) {
    return candidateError(robot, name,
                          absl::StrCat(field, " must be a unit vector"));
  }
  return vector.normalized();
}

/** parent_R_pad from the outward normal and, when given, the pad's x axis. */
absl::StatusOr<Eigen::Matrix3d> padRotation(const Candidate& candidate,
                                            const Eigen::Vector3d& padNormal,
                                            absl::string_view robot) {
  const Eigen::Vector3d zAxis = -padNormal;
  Eigen::Vector3d xAxis = zAxis.unitOrthogonal();
  if (candidate.pad_x_axis.size() != 0) {
    ASSIGN_OR_RETURN(xAxis, unitVector3(candidate.pad_x_axis, robot,
                                        candidate.name, "pad_x_axis"));
    if (!(std::abs(xAxis.dot(zAxis)) <= kUnitTolerance)) {
      return candidateError(robot, candidate.name,
                            "pad_x_axis must be orthogonal to pad_normal");
    }
    xAxis = (xAxis - xAxis.dot(zAxis) * zAxis).normalized();
  }
  Eigen::Matrix3d rotation;
  rotation.col(kXAxisIndex) = xAxis;
  rotation.col(kYAxisIndex) = zAxis.cross(xAxis);
  rotation.col(kZAxisIndex) = zAxis;
  return rotation;
}

absl::StatusOr<ContactKind> contactKind(const Candidate& candidate,
                                        absl::string_view robot) {
  if (candidate.kind == config::ContactSet::Kind::kPoint) {
    return ContactKind::kPoint;
  }
  if (candidate.kind == config::ContactSet::Kind::kSurface) {
    return ContactKind::kSurface;
  }
  return candidateError(robot, candidate.name,
                        "kind must be KIND_POINT or KIND_SURFACE");
}

absl::StatusOr<std::optional<ContactRectangle>> contactRectangle(
    const Candidate& candidate, ContactKind kind, absl::string_view robot) {
  if (kind == ContactKind::kPoint) {
    if (candidate.rectangle.has_value()) {
      return candidateError(robot, candidate.name,
                            "a point candidate has no rectangle");
    }
    return std::optional<ContactRectangle>();
  }
  if (!candidate.rectangle.has_value()) {
    return candidateError(robot, candidate.name,
                          "a surface candidate needs a rectangle");
  }
  if (candidate.pad_x_axis.size() == 0) {
    return candidateError(robot, candidate.name,
                          "a surface candidate needs a pad_x_axis");
  }
  const config::ContactSet::Rectangle& rectangle = *candidate.rectangle;
  if (!(rectangle.x_min < rectangle.x_max) ||
      !(rectangle.y_min < rectangle.y_max)) {
    return candidateError(
        robot, candidate.name,
        "the rectangle needs x_min < x_max and y_min < y_max");
  }
  return std::optional<ContactRectangle>(
      ContactRectangle{.xMin = rectangle.x_min,
                       .xMax = rectangle.x_max,
                       .yMin = rectangle.y_min,
                       .yMax = rectangle.y_max});
}

absl::StatusOr<ContactCandidate> contactCandidate(const Candidate& candidate,
                                                  absl::string_view robot) {
  if (candidate.name.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("contact set of '", robot, "': a candidate has no name"));
  }
  ContactCandidate result;
  result.name = candidate.name;
  ASSIGN_OR_RETURN(result.kind, contactKind(candidate, robot));
  if (candidate.parent_joint.empty()) {
    return candidateError(robot, candidate.name, "has no parent_joint");
  }
  result.parentJoint = candidate.parent_joint;
  ASSIGN_OR_RETURN(result.offset,
                   vector3(candidate.offset, robot, candidate.name, "offset"));
  ASSIGN_OR_RETURN(result.padNormal, unitVector3(candidate.pad_normal, robot,
                                                 candidate.name, "pad_normal"));
  ASSIGN_OR_RETURN(result.padRotation,
                   padRotation(candidate, result.padNormal, robot));
  if (!(candidate.geom_radius >= 0.0) ||
      !std::isfinite(candidate.geom_radius)) {
    return candidateError(robot, candidate.name,
                          "geom_radius must be finite and not negative");
  }
  result.geomRadius = candidate.geom_radius;
  ASSIGN_OR_RETURN(result.rectangle,
                   contactRectangle(candidate, result.kind, robot));
  return result;
}

}  // namespace

absl::StatusOr<ContactSet> ContactSet::Create(
    const config::ContactSet& config) {
  if (config.candidates.empty() ||
      static_cast<int>(config.candidates.size()) > kMaxContactCandidates) {
    return absl::InvalidArgumentError(absl::StrCat(
        "contact set of '", config.robot, "' needs 1 to ",
        kMaxContactCandidates, " candidates, not ", config.candidates.size()));
  }
  ContactSet set;
  set.robot_ = config.robot;
  for (const Candidate& candidate : config.candidates) {
    ASSIGN_OR_RETURN(ContactCandidate validated,
                     contactCandidate(candidate, config.robot));
    if (!set.indexByName_.try_emplace(validated.name, set.size()).second) {
      return candidateError(config.robot, validated.name, "is listed twice");
    }
    set.candidates_.push_back(std::move(validated));
  }
  return set;
}

absl::StatusOr<int> ContactSet::indexOf(absl::string_view name) const {
  const absl::flat_hash_map<std::string, int>::const_iterator found =
      indexByName_.find(name);
  if (found == indexByName_.end()) {
    return absl::NotFoundError(absl::StrCat("the contact set of '", robot_,
                                            "' has no candidate '", name, "'"));
  }
  return found->second;
}

absl::StatusOr<ContactMask> ContactSet::maskOf(
    absl::Span<const std::string> names) const {
  ContactMask mask = 0;
  for (const std::string& name : names) {
    ASSIGN_OR_RETURN(const int index, indexOf(name));
    mask |= contactBit(index);
  }
  return mask;
}

absl::StatusOr<std::vector<int>> ContactSet::activeIndices(
    ContactMask mask) const {
  if (size() < kMaxContactCandidates && (mask >> size()) != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "contact mask ", mask, " activates a candidate beyond the ", size(),
        " of the contact set of '", robot_, "'"));
  }
  std::vector<int> indices;
  for (int index = 0; index < size(); ++index) {
    if (isContactActive(mask, index)) indices.push_back(index);
  }
  return indices;
}

absl::StatusOr<std::vector<std::string>> ContactSet::namesOf(
    ContactMask mask) const {
  ASSIGN_OR_RETURN(const std::vector<int> indices, activeIndices(mask));
  std::vector<std::string> names;
  names.reserve(indices.size());
  for (const int index : indices) {
    names.push_back(candidates_[index].name);
  }
  return names;
}

absl::StatusOr<config::ContactSet> loadContactSetFile(absl::string_view path) {
  return nproto::LoadTextprotoFile<config::ContactSet,
                                   humanoid_multi_contact::ContactSet>(path);
}

}  // namespace ocs2::humanoid::multi_contact
