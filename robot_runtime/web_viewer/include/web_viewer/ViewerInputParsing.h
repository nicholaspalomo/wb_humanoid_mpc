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

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "web_viewer/ViewerInput.h"

namespace robot::web_viewer {

// The parsers of the query parameters of POST /input and POST /settings (README.md, "Routes"). Each takes the
// parameters of one request by name, refuses an unknown or missing parameter and a malformed value with
// InvalidArgument naming the parameter (the server answers 400 with the message), and clamps a value that is merely
// out of range. They are pure functions, safe to call from any thread.

/**
 * Parses POST /input?type=...: drag (button=left|middle|right, shift=0|1, dx, dy, finite numbers clamped to [-2, 2]),
 * scroll (notches, finite, clamped to [-10, 10]), key (key, one character of [0-5a-z]), reset_camera,
 * camera (mode=free|tracking) and visualization (name, one of `visualizationNames`).
 */
absl::StatusOr<ViewerInput> parseViewerInput(const absl::flat_hash_map<std::string, std::string>& parameters,
                                             absl::Span<const std::string> visualizationNames);

/**
 * Parses POST /settings?width=&height=&fps=&quality= into a kSettings input: at least one of the four, each an
 * integer, clamped to width 160-3840, height 120-2160, fps 1-60 and quality 10-95.
 */
absl::StatusOr<ViewerInput> parseViewerSettings(const absl::flat_hash_map<std::string, std::string>& parameters);

}  // namespace robot::web_viewer
