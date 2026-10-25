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

#include <array>
#include <cstddef>
#include <string>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The spheres around the feet and the knees that must not overlap: the frames
 * they are centered on and their radii.
 *
 * FootCollisionConstraint keeps them apart by their distances and
 * FootCollisionCbfConstraint by a control barrier function of the same
 * distances; both read the task file's collision_constraint block
 * (footCollisionConstraintConfigFromConfig()). Passive data.
 */
struct FootCollisionSpheres {
  // Foot and ankle
  std::string leftAnkleFrame;
  std::string rightAnkleFrame;

  std::string leftFootCenterFrame = "foot_l_contact";
  std::string rightFootCenterFrame = "foot_r_contact";

  // The collision points in front of (1) and behind (2) the center of each
  // foot, frames createPinocchioModel() adds.
  std::string leftFootFrame1 = "foot_l_contact_collision_p_1";
  std::string rightFootFrame1 = "foot_r_contact_collision_p_1";

  std::string leftFootFrame2 = "foot_l_contact_collision_p_2";
  std::string rightFootFrame2 = "foot_r_contact_collision_p_2";

  // [m] 0 when the task file does not set it (the loader leaves a missing key
  // at its default).
  scalar_t footCollisionSphereRadius = 0.0;

  // Knee
  std::string leftKneeFrame;
  std::string rightKneeFrame;
  // [m]
  scalar_t kneeCollisionSphereRadius = 0.0;
};

/** A point of the legs that carries a collision sphere: an index into
 * footCollisionPointFrames(). */
enum class FootCollisionPoint {
  kLeftAnkle,
  kRightAnkle,
  kLeftCenter,
  kRightCenter,
  kLeftPoint1,
  kRightPoint1,
  kLeftPoint2,
  kRightPoint2,
  kLeftKnee,
  kRightKnee,
};
inline constexpr size_t kNumFootCollisionPoints = 10;

/** The radius a pair of points is kept apart by: twice the foot's or twice the
 * knee's. */
enum class FootCollisionRadius { kFoot, kKnee };

/** Two points of different legs whose spheres must not overlap, and the radius
 * of their spheres. */
struct FootCollisionPair {
  FootCollisionPoint first = FootCollisionPoint::kLeftCenter;
  FootCollisionPoint second = FootCollisionPoint::kRightCenter;
  FootCollisionRadius radius = FootCollisionRadius::kFoot;
};

inline constexpr size_t kNumFootCollisionPairs = 16;

/**
 * The pairs kept apart, one constraint row each, in the order the generated
 * libraries were taped in: nine between the two feet's collision points and
 * centers, one between the knees and six between a foot and the other leg's
 * ankle.
 */
inline constexpr std::array<FootCollisionPair, kNumFootCollisionPairs>
    kFootCollisionPairs = {{
        {.first = FootCollisionPoint::kLeftPoint1,
         .second = FootCollisionPoint::kRightPoint1,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftPoint1,
         .second = FootCollisionPoint::kRightPoint2,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftPoint2,
         .second = FootCollisionPoint::kRightPoint1,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftPoint2,
         .second = FootCollisionPoint::kRightPoint2,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftCenter,
         .second = FootCollisionPoint::kRightPoint1,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftCenter,
         .second = FootCollisionPoint::kRightPoint2,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kRightCenter,
         .second = FootCollisionPoint::kLeftPoint1,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kRightCenter,
         .second = FootCollisionPoint::kLeftPoint2,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftCenter,
         .second = FootCollisionPoint::kRightCenter,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftKnee,
         .second = FootCollisionPoint::kRightKnee,
         .radius = FootCollisionRadius::kKnee},
        {.first = FootCollisionPoint::kLeftCenter,
         .second = FootCollisionPoint::kRightAnkle,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftPoint1,
         .second = FootCollisionPoint::kRightAnkle,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kLeftPoint2,
         .second = FootCollisionPoint::kRightAnkle,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kRightCenter,
         .second = FootCollisionPoint::kLeftAnkle,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kRightPoint1,
         .second = FootCollisionPoint::kLeftAnkle,
         .radius = FootCollisionRadius::kFoot},
        {.first = FootCollisionPoint::kRightPoint2,
         .second = FootCollisionPoint::kLeftAnkle,
         .radius = FootCollisionRadius::kFoot},
    }};

/** Returns the frame of every point of `spheres`, indexed by
 * FootCollisionPoint. */
inline std::array<std::string, kNumFootCollisionPoints>
footCollisionPointFrames(const FootCollisionSpheres& spheres) {
  std::array<std::string, kNumFootCollisionPoints> frames;
  frames[static_cast<size_t>(FootCollisionPoint::kLeftAnkle)] =
      spheres.leftAnkleFrame;
  frames[static_cast<size_t>(FootCollisionPoint::kRightAnkle)] =
      spheres.rightAnkleFrame;
  frames[static_cast<size_t>(FootCollisionPoint::kLeftCenter)] =
      spheres.leftFootCenterFrame;
  frames[static_cast<size_t>(FootCollisionPoint::kRightCenter)] =
      spheres.rightFootCenterFrame;
  frames[static_cast<size_t>(FootCollisionPoint::kLeftPoint1)] =
      spheres.leftFootFrame1;
  frames[static_cast<size_t>(FootCollisionPoint::kRightPoint1)] =
      spheres.rightFootFrame1;
  frames[static_cast<size_t>(FootCollisionPoint::kLeftPoint2)] =
      spheres.leftFootFrame2;
  frames[static_cast<size_t>(FootCollisionPoint::kRightPoint2)] =
      spheres.rightFootFrame2;
  frames[static_cast<size_t>(FootCollisionPoint::kLeftKnee)] =
      spheres.leftKneeFrame;
  frames[static_cast<size_t>(FootCollisionPoint::kRightKnee)] =
      spheres.rightKneeFrame;
  return frames;
}

}  // namespace ocs2::humanoid
