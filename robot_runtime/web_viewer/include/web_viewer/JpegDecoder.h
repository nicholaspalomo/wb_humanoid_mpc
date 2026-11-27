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
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace robot::web_viewer {

/**
 * Decodes JPEG images to RGB with libjpeg-turbo, for the tests that check the
 * frames the viewer serves.
 *
 * Not thread-safe: one decoder per thread.
 */
class JpegDecoder {
 public:
  // A decoded image: `height` rows of `width` RGB pixels, top row first, no row
  // padding.
  struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb = {};
  };

  /** Creates a decoder; Internal when libjpeg-turbo cannot allocate its
   * decompressor. */
  static absl::StatusOr<std::unique_ptr<JpegDecoder>> Create();

  ~JpegDecoder();
  JpegDecoder(const JpegDecoder&) = delete;
  JpegDecoder& operator=(const JpegDecoder&) = delete;

  /** Decodes `jpeg`; InvalidArgument when it is not a JPEG image libjpeg-turbo
   * can read. */
  absl::StatusOr<Image> decode(absl::string_view jpeg);

 private:
  explicit JpegDecoder(void* absl_nonnull decompressor);

  // The tjhandle of tj3Init(TJINIT_DECOMPRESS), destroyed with the decoder.
  void* absl_nonnull decompressor_;
};

}  // namespace robot::web_viewer
