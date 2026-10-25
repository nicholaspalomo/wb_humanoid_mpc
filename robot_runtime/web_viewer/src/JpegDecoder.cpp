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

#include "web_viewer/JpegDecoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "turbojpeg.h"  // NOLINT(build/include_subdir): libjpeg-turbo installs turbojpeg.h without a directory

namespace robot::web_viewer {

absl::StatusOr<std::unique_ptr<JpegDecoder>> JpegDecoder::Create() {
  void* absl_nullable decompressor = tj3Init(TJINIT_DECOMPRESS);
  if (decompressor == nullptr) {
    return absl::InternalError(
        absl::StrCat("libjpeg-turbo could not create a decompressor: ",
                     tj3GetErrorStr(/*handle=*/nullptr)));
  }
  return absl::WrapUnique(new JpegDecoder(decompressor));
}

JpegDecoder::JpegDecoder(void* absl_nonnull decompressor)
    : decompressor_(decompressor) {}

JpegDecoder::~JpegDecoder() {
  tj3Destroy(decompressor_);
}

absl::StatusOr<JpegDecoder::Image> JpegDecoder::decode(absl::string_view jpeg) {
  if (jpeg.empty())
    return absl::InvalidArgumentError("not a JPEG image: no bytes");
  const unsigned char* absl_nonnull data =
      reinterpret_cast<const unsigned char*>(jpeg.data());
  if (tj3DecompressHeader(decompressor_, data, jpeg.size()) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("not a JPEG image (", jpeg.size(),
                     " bytes): ", tj3GetErrorStr(decompressor_)));
  }
  Image image = {.width = tj3Get(decompressor_, TJPARAM_JPEGWIDTH),
                 .height = tj3Get(decompressor_, TJPARAM_JPEGHEIGHT),
                 .rgb = {}};
  if (image.width <= 0 || image.height <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the JPEG image has no size: ", image.width, "x", image.height));
  }
  image.rgb.resize(static_cast<size_t>(image.width) *
                   static_cast<size_t>(image.height) * 3);
  if (tj3Set(decompressor_, TJPARAM_BOTTOMUP, /*value=*/0) != 0 ||
      tj3Decompress8(decompressor_, data, jpeg.size(), image.rgb.data(),
                     /*pitch=*/image.width * 3, TJPF_RGB) != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the JPEG image does not decode: ", tj3GetErrorStr(decompressor_)));
  }
  return image;
}

}  // namespace robot::web_viewer
