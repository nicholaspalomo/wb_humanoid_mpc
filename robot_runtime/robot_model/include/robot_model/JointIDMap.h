/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <functional>
#include <limits>
#include <vector>

#include "robot_model/IDMapBase.h"
#include "robot_model/RobotDescription.h"

namespace robot::model {

/**
 * An IDMapBase over the joint indices of a robot description: one slot per joint, read into Eigen vectors by lists of
 * joint indices. Not thread-safe.
 */
template <typename T>
class JointIdMap : public IDMapBase<T> {
 public:
  explicit JointIdMap(const RobotDescription& robotDescription) : IDMapBase<T>(robotDescription.getNumJoints()) {}

  JointIdMap() = delete;

  // Every RobotDescription::Create() accepts makes a valid map.
  static_assert(RobotDescription::kMinJoints >= IDMapBase<T>::kMinSize && RobotDescription::kMaxJoints <= IDMapBase<T>::kMaxSize);

  //  Get a vector of joint properties given a vector of joint IDs
  template <IDMapExtractor<T, scalar_t> Extractor>
  vector_t toVector(const std::vector<joint_index_t>& jointIds,
                    Extractor valueExtractor,
                    scalar_t defaultValue = std::numeric_limits<scalar_t>::quiet_NaN()) const {
    return this->toEigenVector(jointIds.begin(), jointIds.end(), valueExtractor, defaultValue);
  }

  //  The same into `vector`, resized to jointIds.size(): no allocation once it has that size.
  template <IDMapExtractor<T, scalar_t> Extractor>
  void writeVector(const std::vector<joint_index_t>& jointIds,
                   Extractor valueExtractor,
                   vector_t& vector,
                   scalar_t defaultValue = std::numeric_limits<scalar_t>::quiet_NaN()) const {
    this->writeEigenVector(jointIds.begin(), jointIds.end(), valueExtractor, defaultValue, vector);
  }
};

}  // namespace robot::model
