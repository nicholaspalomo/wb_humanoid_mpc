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

#include <functional>
#include <optional>
#include <string>

#include "absl/time/time.h"

namespace ocs2::humanoid::teleop {

/**
 * Reads lines from a file descriptor (the terminal's standard input) without blocking a shutdown: it waits for input
 * in slices of `pollPeriod` and gives up when the caller's stop predicate holds, so that Ctrl-C ends a program waiting
 * for a line. (The ROS keyboard node read with std::getline() on a thread of its own and ended the process with
 * std::terminate() on shutdown.)
 */
class LineReader {
 public:
  explicit LineReader(int fileDescriptor, absl::Duration pollPeriod = absl::Milliseconds(100));

  /**
   * The next line, without its line break; nullopt at the end of the input (a last line without a line break is still
   * returned first), on a read error, or once `shouldStop()` holds.
   */
  std::optional<std::string> readLine(const std::function<bool()>& shouldStop);

 private:
  int fileDescriptor_;
  absl::Duration pollPeriod_;
  std::string buffer_;
  bool endOfInput_ = false;
};

}  // namespace ocs2::humanoid::teleop
