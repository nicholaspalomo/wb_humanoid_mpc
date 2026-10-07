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

#include "humanoid_common_mpc_app/robot/AtomicFileWrite.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {
namespace {

/** Bytes read per read() call. */
constexpr size_t kReadChunk = 64 * 1024;

/** Closes the descriptor `fd`; the errno's status naming `path` when that fails (a deferred write error). */
absl::Status closeFile(int fd, absl::string_view path) {
  if (::close(fd) != 0) return absl::ErrnoToStatus(errno, absl::StrCat("close(", path, ")"));
  return absl::OkStatus();
}

/** Writes all of `contents` to `fd`, retrying short writes and interrupted calls. */
absl::Status writeAll(int fd, absl::string_view contents, absl::string_view path) {
  // Null for an empty view, which writes nothing.
  const char* absl_nullable cursor = contents.data();
  size_t remaining = contents.size();
  while (remaining > 0) {
    const ssize_t written = ::write(fd, cursor, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      return absl::ErrnoToStatus(errno, absl::StrCat("write(", path, ")"));
    }
    cursor += written;
    remaining -= static_cast<size_t>(written);
  }
  return absl::OkStatus();
}

/** fsyncs the directory that holds `path`, so that a rename or an unlink in it survives a power loss. */
absl::Status syncDirectoryOf(absl::string_view path) {
  std::string directory = std::filesystem::path(std::string(path)).parent_path().string();
  if (directory.empty()) directory = ".";
  const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return absl::ErrnoToStatus(errno, absl::StrCat("open(", directory, ")"));
  absl::Status status = absl::OkStatus();
  if (::fsync(fd) != 0) status = absl::ErrnoToStatus(errno, absl::StrCat("fsync(", directory, ")"));
  const absl::Status closed = closeFile(fd, directory);
  return status.ok() ? closed : status;
}

/**
 * syncDirectoryOf(`path`) after `what` has taken effect: a failure is DataLoss, saying that `what` was done but may not
 * survive a power loss.
 */
absl::Status syncAfter(absl::string_view what, absl::string_view path) {
  absl::Status synced = syncDirectoryOf(path);
  if (synced.ok()) return synced;
  return absl::DataLossError(
      absl::StrCat(what, ", but its directory could not be synced, so it may not survive a power loss: ", synced.message()));
}

/** Creates `temporary` exclusively; one left by a dead process of this pid is removed and the creation retried once. */
absl::StatusOr<int> createTemporary(const std::string& temporary) {
  constexpr int kFlags = O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC;
  constexpr mode_t kMode = 0644;
  int fd = ::open(temporary.c_str(), kFlags, kMode);
  if (fd < 0 && errno == EEXIST && ::unlink(temporary.c_str()) == 0) {
    fd = ::open(temporary.c_str(), kFlags, kMode);
  }
  if (fd < 0) return absl::ErrnoToStatus(errno, absl::StrCat("open(", temporary, ", O_CREAT | O_EXCL)"));
  return fd;
}

/** Writes `contents` to the open temporary `fd`, fsyncs and closes it. */
absl::Status fillTemporary(int fd, const std::string& temporary, absl::string_view contents) {
  absl::Status status = writeAll(fd, contents, temporary);
  if (status.ok() && ::fsync(fd) != 0) status = absl::ErrnoToStatus(errno, absl::StrCat("fsync(", temporary, ")"));
  const absl::Status closed = closeFile(fd, temporary);
  return status.ok() ? closed : status;
}

}  // namespace

absl::Status writeFileAtomically(absl::string_view path, absl::string_view contents) {
  const std::string target(path);
  const std::string temporary = absl::StrCat(target, ".tmp.", ::getpid());
  const absl::StatusOr<int> fd = createTemporary(temporary);
  if (!fd.ok()) return fd.status();
  absl::Status status = fillTemporary(*fd, temporary, contents);
  if (status.ok() && std::rename(temporary.c_str(), target.c_str()) != 0) {
    status = absl::ErrnoToStatus(errno, absl::StrCat("rename(", temporary, ", ", target, ")"));
  }
  if (!status.ok()) {
    ::unlink(temporary.c_str());
    return status;
  }
  return syncAfter(absl::StrCat(target, " was replaced"), target);
}

absl::StatusOr<std::string> readFileBytes(absl::string_view path) {
  const std::string file(path);
  const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return absl::ErrnoToStatus(errno, absl::StrCat("open(", file, ")"));
  std::string contents;
  std::string chunk(kReadChunk, '\0');
  absl::Status status = absl::OkStatus();
  while (true) {
    const ssize_t got = ::read(fd, chunk.data(), chunk.size());
    if (got < 0) {
      if (errno == EINTR) continue;
      status = absl::ErrnoToStatus(errno, absl::StrCat("read(", file, ")"));
      break;
    }
    if (got == 0) break;
    contents.append(chunk.data(), static_cast<size_t>(got));
  }
  absl::Status closed = closeFile(fd, file);
  if (!status.ok()) return status;
  if (!closed.ok()) return closed;
  return contents;
}

absl::Status renameFile(absl::string_view from, absl::string_view to) {
  const std::string source(from);
  const std::string target(to);
  if (std::rename(source.c_str(), target.c_str()) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("rename(", source, ", ", target, ")"));
  }
  return syncAfter(absl::StrCat(source, " was renamed ", target), target);
}

absl::Status removeFile(absl::string_view path) {
  const std::string file(path);
  if (::unlink(file.c_str()) != 0) {
    if (errno == ENOENT) return absl::OkStatus();
    return absl::ErrnoToStatus(errno, absl::StrCat("unlink(", file, ")"));
  }
  return syncAfter(absl::StrCat(file, " was removed"), file);
}

}  // namespace ocs2::humanoid
