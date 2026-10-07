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

// Strict parsing of a textproto configuration file. Private to the library: not in include/.

#pragma once

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"

namespace robot::ipc {

/**
 * Parses `text` into `message` with google::protobuf::TextFormat, strictly: a syntax error, an unknown field, a value
 * of the wrong type or a non-repeated field given twice is an InvalidArgument error "<source>:<line>:<column>:
 * <problem>", with 1-based line and column. What the parser would only warn about (a field marked deprecated) is an
 * error too, so that a file still naming a field on its way out fails at once. `message` is cleared first; on error its
 * contents are unspecified.
 */
absl::Status parseTextproto(absl::string_view text, absl::string_view source, google::protobuf::Message& message);

/**
 * parseTextproto() on the file at `path`, which also names it in the error messages. A file that cannot be read is
 * NotFound.
 */
absl::Status loadTextproto(const std::string& path, google::protobuf::Message& message);

}  // namespace robot::ipc
