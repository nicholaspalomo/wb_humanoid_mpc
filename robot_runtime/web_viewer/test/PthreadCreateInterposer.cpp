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

// Interposes glibc's pthread_create for the test binary that links it: a call
// for which takePthreadCreateFailure() returns true fails with EAGAIN, and
// every other call is passed on to glibc's.
//
// A translation unit of its own that includes no header declaring
// pthread_create (<pthread.h>, <atomic>, <thread>, <memory>): next to glibc's
// declaration, whose parameters carry reserved names, this definition would be
// a second declaration with other parameter names.

#include "robot_runtime/web_viewer/test/PthreadCreateInterposer.h"

#include <dlfcn.h>
#include <sys/types.h>

#include <cerrno>

#include "absl/base/nullability.h"

// The declaration glibc's <pthread.h> would give, with this file's parameter
// names.
extern "C" int pthread_create(
    pthread_t* absl_nonnull thread,
    const pthread_attr_t* absl_nullable attributes,
    void* absl_nullable (*absl_nonnull start)(void* absl_nullable),
    void* absl_nullable argument) noexcept;

extern "C" int pthread_create(
    pthread_t* absl_nonnull thread,
    const pthread_attr_t* absl_nullable attributes,
    void* absl_nullable (*absl_nonnull start)(void* absl_nullable),
    void* absl_nullable argument) noexcept {
  using PthreadCreate = int (*absl_nullable)(
      pthread_t* absl_nonnull, const pthread_attr_t* absl_nullable,
      void* absl_nullable (*absl_nonnull)(void* absl_nullable),
      void* absl_nullable);
  // glibc's own pthread_create, the next definition after this one; null only
  // in a process without libc's threads.
  static const PthreadCreate kGlibcPthreadCreate =
      reinterpret_cast<PthreadCreate>(dlsym(RTLD_NEXT, "pthread_create"));
  if (kGlibcPthreadCreate == nullptr) return ENOSYS;
  if (robot::web_viewer::takePthreadCreateFailure()) return EAGAIN;
  return kGlibcPthreadCreate(thread, attributes, start, argument);
}
