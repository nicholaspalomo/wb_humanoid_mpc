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

#include <ocs2_core/Types.h>

#include "ocs2_oc/multiple_shooting/Transcription.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"

namespace ocs2 {
namespace multiple_shooting {

/**
 * Compute the performance index from the transcription for an intermediate node.
 * @param transcription: multiple shooting transcription for an intermediate node.
 * @param dt : Duration of the interval
 * @return Performance index for a single intermediate node.
 */
PerformanceIndex computePerformanceIndex(const Transcription& transcription, scalar_t dt);

/**
 * Compute the performance index from the transcription for the event node.
 * @param transcription: multiple shooting transcription for this node.
 * @return Performance index for the event node.
 */
PerformanceIndex computePerformanceIndex(const EventTranscription& transcription);

/**
 * Compute the performance index from the transcription for the terminal node.
 * @param transcription: multiple shooting transcription for this node.
 * @return Performance index for the terminal node.
 */
PerformanceIndex computePerformanceIndex(const TerminalTranscription& transcription);

}  // namespace multiple_shooting
}  // namespace ocs2
