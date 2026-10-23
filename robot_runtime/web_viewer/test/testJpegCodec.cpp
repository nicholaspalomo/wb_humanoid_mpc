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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "robot_runtime/web_viewer/test/WebViewerTestSupport.h"
#include "web_viewer/JpegDecoder.h"
#include "web_viewer/JpegEncoder.h"

namespace robot::web_viewer {
namespace {

double psnr(const std::vector<uint8_t>& expected, const std::vector<uint8_t>& actual) {
  double squaredError = 0.0;
  for (size_t index = 0; index < expected.size(); ++index) {
    const double difference = static_cast<double>(expected[index]) - static_cast<double>(actual[index]);
    squaredError += difference * difference;
  }
  const double meanSquaredError = squaredError / static_cast<double>(expected.size());
  return 10.0 * std::log10(255.0 * 255.0 / std::max(meanSquaredError, 1.0e-12));
}

std::unique_ptr<JpegEncoder> makeEncoder() {
  absl::StatusOr<std::unique_ptr<JpegEncoder>> encoder = JpegEncoder::Create();
  EXPECT_TRUE(encoder.ok()) << encoder.status();
  return encoder.ok() ? *std::move(encoder) : nullptr;
}

std::unique_ptr<JpegDecoder> makeDecoder() {
  absl::StatusOr<std::unique_ptr<JpegDecoder>> decoder = JpegDecoder::Create();
  EXPECT_TRUE(decoder.ok()) << decoder.status();
  return decoder.ok() ? *std::move(decoder) : nullptr;
}

TEST(JpegCodecTest, RoundTripKeepsTheSizeAndTheImage) {
  std::unique_ptr<JpegEncoder> encoder = makeEncoder();
  std::unique_ptr<JpegDecoder> decoder = makeDecoder();
  ASSERT_NE(encoder, nullptr);
  ASSERT_NE(decoder, nullptr);
  const std::vector<uint8_t> image = test_support::gradientImage(/*width=*/320, /*height=*/240, /*phase=*/7);
  absl::StatusOr<std::string> jpeg = encoder->encode(image, /*width=*/320, /*height=*/240, /*quality=*/80, RowOrder::kTopDown);
  ASSERT_TRUE(jpeg.ok()) << jpeg.status();
  EXPECT_EQ(jpeg->substr(0, 2), "\xFF\xD8");  // the JPEG start-of-image marker
  absl::StatusOr<JpegDecoder::Image> decoded = decoder->decode(*jpeg);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->width, 320);
  EXPECT_EQ(decoded->height, 240);
  ASSERT_EQ(decoded->rgb.size(), image.size());
  EXPECT_GT(psnr(image, decoded->rgb), 30.0);
}

TEST(JpegCodecTest, BottomUpRowsAreFlippedWithoutACopy) {
  std::unique_ptr<JpegEncoder> encoder = makeEncoder();
  std::unique_ptr<JpegDecoder> decoder = makeDecoder();
  ASSERT_NE(encoder, nullptr);
  ASSERT_NE(decoder, nullptr);
  // In memory, the first 16 rows are red and the rest blue: OpenGL's bottom rows.
  constexpr int kWidth = 64;
  constexpr int kHeight = 64;
  std::vector<uint8_t> image(static_cast<size_t>(kWidth * kHeight * 3), 0);
  for (int row = 0; row < kHeight; ++row) {
    for (int column = 0; column < kWidth; ++column) {
      const size_t pixel = static_cast<size_t>((row * kWidth + column) * 3);
      image[pixel] = row < 16 ? 255 : 0;
      image[pixel + 2] = row < 16 ? 0 : 255;
    }
  }
  absl::StatusOr<std::string> jpeg = encoder->encode(image, kWidth, kHeight, /*quality=*/90, RowOrder::kBottomUp);
  ASSERT_TRUE(jpeg.ok()) << jpeg.status();
  absl::StatusOr<JpegDecoder::Image> decoded = decoder->decode(*jpeg);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  // The decoded image is top-down: its last row is the first row in memory (red), its first row the last (blue).
  const size_t lastRow = static_cast<size_t>((kHeight - 1) * kWidth * 3);
  EXPECT_GT(decoded->rgb[lastRow], 200);
  EXPECT_LT(decoded->rgb[lastRow + 2], 60);
  EXPECT_LT(decoded->rgb[0], 60);
  EXPECT_GT(decoded->rgb[2], 200);
}

TEST(JpegCodecTest, OneEncoderServesSeveralSizes) {
  std::unique_ptr<JpegEncoder> encoder = makeEncoder();
  std::unique_ptr<JpegDecoder> decoder = makeDecoder();
  ASSERT_NE(encoder, nullptr);
  ASSERT_NE(decoder, nullptr);
  for (const int width : {640, 160, 1280, 320}) {
    const int height = width * 9 / 16;
    absl::StatusOr<std::string> jpeg =
        encoder->encode(test_support::gradientImage(width, height, /*phase=*/width), width, height, /*quality=*/70, RowOrder::kTopDown);
    ASSERT_TRUE(jpeg.ok()) << jpeg.status();
    absl::StatusOr<JpegDecoder::Image> decoded = decoder->decode(*jpeg);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(decoded->width, width);
    EXPECT_EQ(decoded->height, height);
  }
}

TEST(JpegCodecTest, RefusesInvalidImages) {
  std::unique_ptr<JpegEncoder> encoder = makeEncoder();
  ASSERT_NE(encoder, nullptr);
  const std::vector<uint8_t> image = test_support::gradientImage(/*width=*/16, /*height=*/16, /*phase=*/0);
  EXPECT_EQ(encoder->encode(image, /*width=*/0, /*height=*/16, /*quality=*/80, RowOrder::kTopDown).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(encoder->encode(image, /*width=*/16, /*height=*/-1, /*quality=*/80, RowOrder::kTopDown).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(encoder->encode(image, /*width=*/32, /*height=*/16, /*quality=*/80, RowOrder::kTopDown).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(encoder->encode(image, /*width=*/16, /*height=*/16, /*quality=*/0, RowOrder::kTopDown).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(encoder->encode(image, /*width=*/16, /*height=*/16, /*quality=*/101, RowOrder::kTopDown).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(JpegCodecTest, DecoderRefusesWhatIsNotAJpeg) {
  std::unique_ptr<JpegDecoder> decoder = makeDecoder();
  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(decoder->decode("").status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(decoder->decode("not a jpeg at all").status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace robot::web_viewer
