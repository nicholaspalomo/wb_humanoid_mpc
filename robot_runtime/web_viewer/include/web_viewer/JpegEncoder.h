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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace robot::web_viewer {

// The order of the rows of an image in memory: top row first (a browser's), or bottom row first (OpenGL's, which
// mjr_readPixels returns).
enum class RowOrder {
  kTopDown,
  kBottomUp,
};

/**
 * Encodes RGB images as baseline JPEG with libjpeg-turbo (TurboJPEG 3, SIMD, 4:2:0 chroma subsampling). A
 * kBottomUp image is encoded as it lies in memory, with no copy to flip it.
 *
 * About 1.2 ms for a 1280x720 frame. It keeps libjpeg-turbo's output buffer between frames, so after the first frame
 * of a size it allocates only the returned string.
 *
 * Not thread-safe: one encoder per thread (the viewer's render thread owns one).
 */
class JpegEncoder {
 public:
  /** Creates an encoder; Internal when libjpeg-turbo cannot allocate its compressor. */
  static absl::StatusOr<std::unique_ptr<JpegEncoder>> Create();

  ~JpegEncoder();
  JpegEncoder(const JpegEncoder&) = delete;
  JpegEncoder& operator=(const JpegEncoder&) = delete;

  /**
   * Encodes `rgb`, `height` rows of `width` pixels of 3 bytes each (no row padding), in `rowOrder`, at `quality` (1 to
   * 100), and returns the JFIF bytes. InvalidArgument when a size is not positive, `rgb` does not hold exactly width x
   * height x 3 bytes, or `quality` is out of range; Internal when libjpeg-turbo fails.
   */
  absl::StatusOr<std::string> encode(absl::Span<const uint8_t> rgb, int width, int height, int quality, RowOrder rowOrder);

 private:
  explicit JpegEncoder(void* absl_nonnull compressor);

  // The tjhandle of tj3Init(TJINIT_COMPRESS), destroyed with the encoder.
  void* absl_nonnull compressor_;
  // libjpeg-turbo's output buffer (tj3Alloc), grown by it as needed and freed with the encoder; null before the first
  // frame. outputCapacity_ is its size in bytes.
  unsigned char* absl_nullable output_ = nullptr;
  size_t outputCapacity_ = 0;
};

}  // namespace robot::web_viewer
