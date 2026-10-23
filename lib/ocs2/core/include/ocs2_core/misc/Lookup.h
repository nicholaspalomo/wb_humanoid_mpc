/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <algorithm>
#include <vector>

namespace ocs2 {
namespace lookup {

/**
 *  Find index into a sorted time Array
 *
 *  Indices are counted as follows:
 *          ------ | ----- | ---  ... ---    | -----
 *                t0     t1              t(n-1)
 *  Index     0        1      2   ...  (n-1)    n
 *
 *  Corner cases:
 *     - If time equal to a time in the timeArray is requested, the lower index is taken (e.g. t = t1 -> index = 1)
 *     - If multiple times in the timeArray are equal, the index before the first occurrence is taken.
 *       for example: if t1 = t2 = t3  and the requested time t <= t3 -> index = 1
 *
 *
 * @tparam SCALAR : numerical type of time
 * @param timeArray : sorted time array to perform the lookup in
 * @param time : enquiry time
 * @return index between [0, size(timeArray)]
 */
template <typename SCALAR = double>
int findIndexInTimeArray(const std::vector<SCALAR>& timeArray, SCALAR time) {
  auto firstLargerValueIterator = std::lower_bound(timeArray.begin(), timeArray.end(), time);
  return static_cast<int>(firstLargerValueIterator - timeArray.begin());
}

/**
 *  Find interval into a sorted time Array
 *
 *  Intervals are counted as follows:
 *           ------ | ----- | ---  ... ---    | -----
 *                 t0     t1              t(n-1)
 *  Interval  -1       0      1   ...  (n-2)    (n-1)
 *
 *  Corner cases are handled as in findIndexInTimeArray
 *
 * @tparam SCALAR : numerical type of time
 * @param timeArray : sorted time array to perform the lookup in
 * @param time : enquiry time
 * @return interval between [-1, size(timeArray)-1]
 */
template <typename SCALAR = double>
int findIntervalInTimeArray(const std::vector<SCALAR>& timeArray, SCALAR time) {
  if (!timeArray.empty()) {
    return findIndexInTimeArray(timeArray, time) - 1;
  } else {
    return 0;
  }
}

}  // namespace lookup
}  // namespace ocs2
