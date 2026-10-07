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

// glibc's declarations of the functions this file defines: memalign() is in <malloc.h>, the others in <cstdlib>.
#include <malloc.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"

namespace {

// Constant-initialized, so that it is ready before the first allocation of static initialization.
constinit std::atomic<size_t> allocations_since_start{0};

// Per thread. Constant-initialized and initial-exec, so that counting never calls __tls_get_addr, which may allocate.
constinit thread_local size_t allocations_on_this_thread ABSL_ATTRIBUTE_INITIAL_EXEC = 0;

void countAllocation() {
  ++allocations_since_start;
  ++allocations_on_this_thread;
}

}  // namespace

// glibc's own entry points. Defining the allocation functions in the binary interposes them on every caller in the
// process; free needs no counting and stays glibc's. The parameters carry the C library's names (nmemb, ptr, memptr):
// glibc declares them as __nmemb, __ptr and __memptr, and a definition must not name them differently
// (readability-inconsistent-declaration-parameter-name).
extern "C" {
void* absl_nullable __libc_malloc(size_t size);
void* absl_nullable __libc_calloc(size_t nmemb, size_t size);
void* absl_nullable __libc_realloc(void* absl_nullable ptr, size_t size);
void* absl_nullable __libc_memalign(size_t alignment, size_t size);

void* absl_nullable malloc(size_t size) noexcept {
  countAllocation();
  return __libc_malloc(size);
}

void* absl_nullable calloc(size_t nmemb, size_t size) noexcept {
  countAllocation();
  return __libc_calloc(nmemb, size);
}

void* absl_nullable realloc(void* absl_nullable ptr, size_t size) noexcept {
  countAllocation();
  return __libc_realloc(ptr, size);
}

// The aligned entry points, which operator new uses for over-aligned types such as SpscQueue.
void* absl_nullable aligned_alloc(size_t alignment, size_t size) noexcept {
  countAllocation();
  return __libc_memalign(alignment, size);
}

void* absl_nullable memalign(size_t alignment, size_t size) noexcept {
  countAllocation();
  return __libc_memalign(alignment, size);
}

int posix_memalign(void* absl_nullable* absl_nonnull memptr, size_t alignment, size_t size) noexcept {
  countAllocation();
  if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
    return EINVAL;
  }
  void* absl_nullable const allocated = __libc_memalign(alignment, size);
  if (allocated == nullptr) {
    return ENOMEM;
  }
  *memptr = allocated;
  return 0;
}
}

namespace robot::realtime {

size_t heapAllocationCount() {
  return allocations_since_start.load();
}

size_t heapAllocationCountOnThisThread() {
  return allocations_on_this_thread;
}

}  // namespace robot::realtime
