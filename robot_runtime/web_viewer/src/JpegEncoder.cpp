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

#include "web_viewer/JpegEncoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "turbojpeg.h"  // NOLINT(build/include_subdir): libjpeg-turbo installs turbojpeg.h without a directory

namespace robot::web_viewer {
namespace {

// The highest JPEG quality.
constexpr int kMaxJpegQuality = 100;

}  // namespace

absl::StatusOr<std::unique_ptr<JpegEncoder>> JpegEncoder::Create() {
  void* absl_nullable compressor = tj3Init(TJINIT_COMPRESS);
  if (compressor == nullptr) {
    return absl::InternalError(absl::StrCat("libjpeg-turbo could not create a compressor: ", tj3GetErrorStr(/*handle=*/nullptr)));
  }
  return absl::WrapUnique(new JpegEncoder(compressor));
}

JpegEncoder::JpegEncoder(void* absl_nonnull compressor) : compressor_(compressor) {}

JpegEncoder::~JpegEncoder() {
  tj3Free(output_);
  tj3Destroy(compressor_);
}

absl::StatusOr<std::string> JpegEncoder::encode(absl::Span<const uint8_t> rgb, int width, int height, int quality, RowOrder rowOrder) {
  if (width <= 0 || height <= 0) {
    return absl::InvalidArgumentError(absl::StrCat("JPEG encoding needs a positive size, not ", width, "x", height));
  }
  const int64_t expectedBytes = int64_t{width} * height * kRgbBytesPerPixel;
  if (static_cast<int64_t>(rgb.size()) != expectedBytes) {
    return absl::InvalidArgumentError(absl::StrCat("a ", width, "x", height, " RGB image has ", expectedBytes, " bytes, not ", rgb.size()));
  }
  if (quality < 1 || quality > kMaxJpegQuality) {
    return absl::InvalidArgumentError(absl::StrCat("JPEG quality is 1 to ", kMaxJpegQuality, ", not ", quality));
  }
  const bool configured = tj3Set(compressor_, TJPARAM_QUALITY, quality) == 0 && tj3Set(compressor_, TJPARAM_SUBSAMP, TJSAMP_420) == 0 &&
                          tj3Set(compressor_, TJPARAM_BOTTOMUP, rowOrder == RowOrder::kBottomUp ? 1 : 0) == 0;
  if (!configured) return absl::InternalError(absl::StrCat("libjpeg-turbo refused the settings: ", tj3GetErrorStr(compressor_)));

  // On input the capacity of the buffer that libjpeg-turbo may grow; on output the size of the image.
  size_t size = outputCapacity_;
  unsigned char* absl_nullable const previousOutput = output_;
  if (tj3Compress8(compressor_, rgb.data(), width, /*pitch=*/width * kRgbBytesPerPixel, height, TJPF_RGB, &output_, &size) != 0) {
    return absl::InternalError(absl::StrCat("libjpeg-turbo could not encode the frame: ", tj3GetErrorStr(compressor_)));
  }
  // A grown buffer holds at least the image; the next frame lets libjpeg-turbo grow it again if needed.
  if (output_ != previousOutput) outputCapacity_ = size;
  return std::string(reinterpret_cast<const char*>(output_), size);
}

}  // namespace robot::web_viewer
