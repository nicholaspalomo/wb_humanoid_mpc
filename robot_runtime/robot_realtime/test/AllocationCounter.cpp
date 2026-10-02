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

#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

#include <atomic>
#include <cerrno>
#include <cstddef>

namespace {

// Constant-initialized, so that it is ready before the first allocation of static initialization.
constinit std::atomic<std::size_t> allocations_since_start{0};

// Per thread. Constant-initialized and initial-exec, so that counting never calls __tls_get_addr, which may allocate.
constinit thread_local std::size_t allocations_on_this_thread __attribute__((tls_model("initial-exec"))) = 0;

void countAllocation() {
  ++allocations_since_start;
  ++allocations_on_this_thread;
}

}  // namespace

// glibc's own entry points. Defining the allocation functions in the binary interposes them on every caller in the
// process; free needs no counting and stays glibc's.
extern "C" {
void* __libc_malloc(std::size_t size);
void* __libc_calloc(std::size_t count, std::size_t size);
void* __libc_realloc(void* pointer, std::size_t size);
void* __libc_memalign(std::size_t alignment, std::size_t size);

void* malloc(std::size_t size) noexcept {
  countAllocation();
  return __libc_malloc(size);
}

void* calloc(std::size_t count, std::size_t size) noexcept {
  countAllocation();
  return __libc_calloc(count, size);
}

void* realloc(void* pointer, std::size_t size) noexcept {
  countAllocation();
  return __libc_realloc(pointer, size);
}

// The aligned entry points, which operator new uses for over-aligned types such as SpscQueue.
void* aligned_alloc(std::size_t alignment, std::size_t size) noexcept {
  countAllocation();
  return __libc_memalign(alignment, size);
}

void* memalign(std::size_t alignment, std::size_t size) noexcept {
  countAllocation();
  return __libc_memalign(alignment, size);
}

int posix_memalign(void** pointer, std::size_t alignment, std::size_t size) noexcept {
  countAllocation();
  if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
    return EINVAL;
  }
  void* const allocated = __libc_memalign(alignment, size);
  if (allocated == nullptr) {
    return ENOMEM;
  }
  *pointer = allocated;
  return 0;
}
}

namespace robot::realtime {

std::size_t heapAllocationCount() {
  return allocations_since_start.load();
}

std::size_t heapAllocationCountOnThisThread() {
  return allocations_on_this_thread;
}

}  // namespace robot::realtime
