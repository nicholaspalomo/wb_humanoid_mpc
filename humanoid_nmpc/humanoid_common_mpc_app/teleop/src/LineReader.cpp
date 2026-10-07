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

#include "humanoid_common_mpc_app/teleop/LineReader.h"

#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <string>
#include <utility>

namespace ocs2::humanoid::teleop {

LineReader::LineReader(int fileDescriptor, absl::Duration pollPeriod) : fileDescriptor_(fileDescriptor), pollPeriod_(pollPeriod) {}

std::optional<std::string> LineReader::readLine(const std::function<bool()>& shouldStop) {
  while (true) {
    const size_t lineBreak = buffer_.find('\n');
    if (lineBreak != std::string::npos) {
      std::string line = buffer_.substr(0, lineBreak);
      buffer_.erase(0, lineBreak + 1);
      return line;
    }
    if (endOfInput_) {
      if (buffer_.empty()) return std::nullopt;
      std::string line = std::move(buffer_);
      buffer_.clear();
      return line;
    }
    if (shouldStop()) return std::nullopt;

    pollfd descriptor{};
    descriptor.fd = fileDescriptor_;
    descriptor.events = POLLIN;
    const int ready = ::poll(&descriptor, /*nfds=*/1, static_cast<int>(absl::ToInt64Milliseconds(pollPeriod_)));
    if (ready < 0) {
      if (errno == EINTR) continue;  // a signal: look at the stop predicate again
      return std::nullopt;
    }
    if (ready == 0) continue;
    char chunk[256];
    const ssize_t count = ::read(fileDescriptor_, chunk, sizeof(chunk));
    if (count < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      return std::nullopt;
    }
    if (count == 0) {
      endOfInput_ = true;
      continue;
    }
    buffer_.append(chunk, static_cast<size_t>(count));
  }
}

}  // namespace ocs2::humanoid::teleop
