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

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "web_viewer/ViewerInput.h"
#include "web_viewer/ViewerInputQueue.h"

namespace robot::web_viewer {
namespace {

ViewerInput drag(ViewerMouseButton button, bool shift, double dx, double dy) {
  return ViewerInput{.type = ViewerInputType::kDrag,
                     .button = button,
                     .shift = shift,
                     .dx = dx,
                     .dy = dy};
}

ViewerInput key(char character) {
  return ViewerInput{.type = ViewerInputType::kKey, .key = character};
}

TEST(ViewerInputQueueTest, MergesConsecutiveDragsWithTheSameButtonAndShift) {
  ViewerInputQueue queue;
  queue.push(
      drag(ViewerMouseButton::kLeft, /*shift=*/false, /*dx=*/0.1, /*dy=*/0.2));
  queue.push(
      drag(ViewerMouseButton::kLeft, /*shift=*/false, /*dx=*/0.3, /*dy=*/-0.1));
  queue.push(
      drag(ViewerMouseButton::kLeft, /*shift=*/true, /*dx=*/0.5, /*dy=*/0.0));
  queue.push(
      drag(ViewerMouseButton::kRight, /*shift=*/true, /*dx=*/0.5, /*dy=*/0.0));
  queue.push(key('k'));
  queue.push(
      drag(ViewerMouseButton::kRight, /*shift=*/true, /*dx=*/0.25, /*dy=*/0.0));
  const std::vector<ViewerInput> drained = queue.drain();
  ASSERT_EQ(drained.size(), 5U);
  EXPECT_DOUBLE_EQ(drained[0].dx, 0.4);
  EXPECT_DOUBLE_EQ(drained[0].dy, 0.1);
  EXPECT_TRUE(drained[1].shift);
  EXPECT_EQ(drained[2].button, ViewerMouseButton::kRight);
  EXPECT_EQ(drained[3].type, ViewerInputType::kKey);
  // A key between two drags keeps them apart.
  EXPECT_DOUBLE_EQ(drained[4].dx, 0.25);
}

// The old merge clamped the sum to [-2, 2] and lost 5.5 of these 7.5 fractions:
// a fast swipe, which the page sends in pieces of at most 2 (page/index.html),
// turned the camera by less than the swipe.
TEST(ViewerInputQueueTest,
     AMergedDragKeepsTheRangeOfOneDragWithoutLosingMotion) {
  ViewerInputQueue queue;
  for (int index = 0; index < 5; ++index)
    queue.push(drag(ViewerMouseButton::kMiddle, /*shift=*/false, /*dx=*/-1.5,
                    /*dy=*/1.5));
  queue.push(drag(ViewerMouseButton::kMiddle, /*shift=*/false, /*dx=*/0.25,
                  /*dy=*/0.25));
  const std::vector<ViewerInput> drained = queue.drain();
  double dx = 0.0;
  double dy = 0.0;
  for (const ViewerInput& input : drained) {
    EXPECT_LE(std::abs(input.dx), ViewerInput::kMaxDragFraction);
    EXPECT_LE(std::abs(input.dy), ViewerInput::kMaxDragFraction);
    dx += input.dx;
    dy += input.dy;
  }
  EXPECT_DOUBLE_EQ(dx, -7.25);
  EXPECT_DOUBLE_EQ(dy, 7.75);
  // The small drag still merges into the last entry.
  EXPECT_EQ(drained.size(), 5U);
}

TEST(ViewerInputQueueTest, DropsTheOldestEventWhenFull) {
  ViewerInputQueue queue;
  constexpr int kPushed = ViewerInputQueue::kCapacity + 44;
  for (int index = 0; index < kPushed; ++index)
    queue.push(key(static_cast<char>('a' + index % 26)));
  EXPECT_EQ(queue.size(), ViewerInputQueue::kCapacity);
  EXPECT_EQ(queue.droppedCount(), 44);
  const std::vector<ViewerInput> drained = queue.drain();
  ASSERT_EQ(static_cast<int>(drained.size()), ViewerInputQueue::kCapacity);
  // The first 44 were dropped: the oldest kept is the 45th, 'a' + 44 % 26.
  EXPECT_EQ(drained.front().key, static_cast<char>('a' + 44 % 26));
  EXPECT_EQ(drained.back().key, static_cast<char>('a' + (kPushed - 1) % 26));
}

TEST(ViewerInputQueueTest, DrainReturnsInOrderAndEmpties) {
  ViewerInputQueue queue;
  EXPECT_TRUE(queue.drain().empty());
  queue.push(key('0'));
  queue.push(ViewerInput{.type = ViewerInputType::kResetCamera});
  queue.push(ViewerInput{.type = ViewerInputType::kScroll, .notches = 2.0});
  const std::vector<ViewerInput> drained = queue.drain();
  ASSERT_EQ(drained.size(), 3U);
  EXPECT_EQ(drained[0].key, '0');
  EXPECT_EQ(drained[1].type, ViewerInputType::kResetCamera);
  EXPECT_EQ(drained[2].type, ViewerInputType::kScroll);
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.drain().empty());
}

}  // namespace
}  // namespace robot::web_viewer
