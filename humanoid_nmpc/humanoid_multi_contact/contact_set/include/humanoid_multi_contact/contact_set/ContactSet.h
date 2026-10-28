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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_multi_contact/contact_set/contact_set.nproto.h"

namespace ocs2::humanoid::multi_contact {

/** A contact mode: bit i is set when candidate i of the ContactSet is active.
 * A mask of bits, hence unsigned. */
using ContactMask = uint64_t;

/** The most candidates a ContactSet holds: the bits of a ContactMask. */
inline constexpr int kMaxContactCandidates = 64;

/** How a candidate transmits load (design section 3.1). */
enum class ContactKind {
  kPoint,    // a force at one point, in a friction cone
  kSurface,  // a wrench of a rectangular pad, in its contact wrench cone
};

/** [m] The pad of a surface candidate in its pad frame, with xMin < xMax and
 * yMin < yMax. */
struct ContactRectangle {
  double xMin = 0.0;
  double xMax = 0.0;
  double yMin = 0.0;
  double yMax = 0.0;
};

/** One validated candidate. Passive data. */
struct ContactCandidate {
  std::string name;
  ContactKind kind = ContactKind::kPoint;
  std::string parentJoint;
  // [m] The pad's center in the parent joint's frame.
  Eigen::Vector3d offset = Eigen::Vector3d::Zero();
  // The outward unit normal of the pad in the parent joint's frame.
  Eigen::Vector3d padNormal = -Eigen::Vector3d::UnitZ();
  // parent_R_pad: z = -padNormal (the terrain normal when the pad rests on a
  // surface), x = the pad's x axis, y = z x x.
  Eigen::Matrix3d padRotation = Eigen::Matrix3d::Identity();
  // [m] The radius of the collision geometry the pad sits on; 0 when flat.
  double geomRadius = 0.0;
  // Set exactly for a surface candidate.
  std::optional<ContactRectangle> rectangle;
};

/**
 * The runtime list of contact candidates of a robot (design section 3.1), in
 * the order that numbers the bits of a ContactMask.
 *
 * Built and validated from the ContactSet schema by Create(); it knows nothing
 * of the robot model, so a parent joint is checked against the URDF by the code
 * that builds the kinematics (RobotStanceModel). A value type, immutable after
 * Create(): every method is const and safe to call concurrently.
 */
class ContactSet {
 public:
  /**
   * Validates `config`. InvalidArgument names the candidate and what is wrong:
   * no candidate or more than kMaxContactCandidates, an empty or duplicate
   * name, no kind, no parent joint, a vector of the wrong size, a pad normal
   * that is not unit (within 1e-3, then normalized), a pad x axis that is not
   * unit or not orthogonal to the normal, a rectangle missing from a surface or
   * given to a point, an empty rectangle, or a negative geom radius.
   */
  static absl::StatusOr<ContactSet> Create(const config::ContactSet& config);

  ContactSet(const ContactSet&) = default;
  ContactSet& operator=(const ContactSet&) = default;
  ContactSet(ContactSet&&) noexcept = default;
  ContactSet& operator=(ContactSet&&) noexcept = default;
  ~ContactSet() = default;

  const std::string& robot() const { return robot_; }
  int size() const { return static_cast<int>(candidates_.size()); }
  const std::vector<ContactCandidate>& candidates() const {
    return candidates_;
  }
  /** The candidate at `index`; requires 0 <= index < size(). */
  const ContactCandidate& candidate(int index) const {
    return candidates_[index];
  }

  /** The index of the candidate `name`; NotFound names the set. */
  absl::StatusOr<int> indexOf(absl::string_view name) const;

  /** The mask of the named candidates; NotFound for an unknown name. A name
   * given twice is set once. */
  absl::StatusOr<ContactMask> maskOf(absl::Span<const std::string> names) const;

  /** The indices of the active candidates of `mask`, ascending;
   * InvalidArgument when it sets a bit at or beyond size(). */
  absl::StatusOr<std::vector<int>> activeIndices(ContactMask mask) const;

  /** The names of the active candidates of `mask`, in set order; as
   * activeIndices(). */
  absl::StatusOr<std::vector<std::string>> namesOf(ContactMask mask) const;

 private:
  ContactSet() = default;

  std::string robot_;
  std::vector<ContactCandidate> candidates_;  // 1 to kMaxContactCandidates
  absl::flat_hash_map<std::string, int> indexByName_;
};

/** Whether candidate `index` is active in `mask`; requires 0 <= index <
 * kMaxContactCandidates. */
inline bool isContactActive(ContactMask mask, int index) {
  return ((mask >> index) & ContactMask{1}) != 0;
}

/** The mask with only candidate `index` active; requires 0 <= index <
 * kMaxContactCandidates. */
inline ContactMask contactBit(int index) {
  return ContactMask{1} << index;
}

/** Reads and validates the contact-set textproto at `path` (its header names
 * humanoid_multi_contact.ContactSet). */
absl::StatusOr<config::ContactSet> loadContactSetFile(absl::string_view path);

}  // namespace ocs2::humanoid::multi_contact
