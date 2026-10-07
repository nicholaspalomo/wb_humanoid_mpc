# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Include spelling: quotes and full paths for everything but the C, C++ and POSIX system headers.

`include-style` (Google C++ style, Names and order of includes): only C and C++ standard library headers and
POSIX/Linux system headers use angle brackets; everything else, third-party libraries included, uses quotes and its
full path (`#include "pinocchio/fwd.hpp"`, `#include "humanoid_common_mpc/common/Types.h"`). A first-party header is
never included by its bare file name. The fix rewrites the brackets; clang-format then regroups the includes.

For Abseil the brackets are a build error waiting to happen, not only a style matter, and the finding says so. Bazel
makes the @abseil-cpp headers visible through `-iquote` paths, which only a quoted include searches. An angle include,
`#include <absl/log/log.h>`, skips them and falls through to the system include directories, which hold a different
Abseil release or none. The dev image once built Abseil 20240722 into /usr/local (removed from docker/Dockerfile
since), so such a file compiled there against headers of another version than the one it links against, and failed in
CI's clean ros:jazzy container with `absl/log/log.h: No such file or directory`.
"""

import re

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "include-style"

# The C++ standard library headers (cpplint's _CPP_HEADERS, without its pre-standard names).
CPP_STANDARD_HEADERS = frozenset(
    """
    algorithm any array atomic barrier bit bitset cassert ccomplex cctype cerrno cfenv cfloat charconv chrono cinttypes
    ciso646 climits clocale cmath codecvt compare complex concepts condition_variable coroutine csetjmp csignal cstdalign
    cstdarg cstdbool cstddef cstdint cstdio cstdlib cstring ctgmath ctime cuchar cwchar cwctype deque exception execution
    expected filesystem flat_map flat_set format forward_list fstream functional future generator initializer_list
    iomanip ios iosfwd iostream istream iterator latch limits list locale map mdspan memory memory_resource mutex new
    numbers numeric optional ostream print queue random ranges ratio regex scoped_allocator semaphore set shared_mutex
    source_location span spanstream sstream stack stacktrace stdexcept stdfloat stop_token streambuf string string_view
    strstream syncstream system_error thread tuple type_traits typeindex typeinfo unordered_map unordered_set utility
    valarray variant vector version
    """.split()
)
# The C, POSIX and Linux system headers (cpplint's _C_HEADERS), and the compiler's own.
C_SYSTEM_HEADERS = frozenset(
    """
    assert.h complex.h ctype.h errno.h fenv.h float.h inttypes.h iso646.h limits.h locale.h math.h setjmp.h signal.h
    stdalign.h stdarg.h stdatomic.h stdbool.h stddef.h stdint.h stdio.h stdlib.h stdnoreturn.h string.h tgmath.h
    threads.h time.h uchar.h wchar.h wctype.h stdbit.h stdckdint.h aio.h cpio.h dirent.h dlfcn.h fcntl.h fmtmsg.h
    fnmatch.h ftw.h glob.h grp.h iconv.h langinfo.h libgen.h monetary.h mqueue.h ndbm.h netdb.h nl_types.h poll.h
    pthread.h pwd.h regex.h sched.h search.h semaphore.h spawn.h strings.h stropts.h syslog.h tar.h termios.h trace.h
    ulimit.h unistd.h utime.h utmpx.h wordexp.h a.out.h aliases.h alloca.h ar.h argp.h argz.h byteswap.h crypt.h
    endian.h envz.h err.h error.h execinfo.h fpu_control.h fstab.h fts.h getopt.h gshadow.h ieee754.h ifaddrs.h
    libintl.h mcheck.h mntent.h obstack.h paths.h printf.h pty.h resolv.h shadow.h sysexits.h ttyent.h elf.h
    features.h gconv.h lastlog.h libio.h link.h malloc.h memory.h nss.h re_comp.h regexp.h sgtty.h stab.h stdc-predef.h
    stdio_ext.h syscall.h termio.h thread_db.h ucontext.h ustat.h utmp.h values.h wait.h xlocale.h
    cxxabi.h omp.h cpuid.h immintrin.h x86intrin.h xmmintrin.h emmintrin.h arm_neon.h
    """.split()
)
C_SYSTEM_DIRECTORIES = frozenset(
    "sys arpa asm asm-generic bits gnu net netinet protocols rpc rpcsvc scsi drm linux misc mtd rdma sound video xen".split()
)
# Third-party headers that have no directory of their own: cppzmq and libzmq, and HPIPM and BLASFEO, which install their
# headers flat (bazel/system_libs.bzl). .clang-format's include categories sort them with the other libraries.
# LINT.IfChange(directoryless_third_party)
DIRECTORYLESS_THIRD_PARTY = frozenset({"zmq.hpp", "zmq_addon.hpp", "zmq.h"})
_DIRECTORYLESS_THIRD_PARTY_PATTERN = re.compile(r"^(hpipm|blasfeo)_[a-z0-9_]+\.h$")
# LINT.ThenChange(//.clang-format:include_categories)


def is_directoryless_third_party(header: str) -> bool:
    """True for a third-party header that its library installs without a directory, such as `zmq.hpp`."""
    return header in DIRECTORYLESS_THIRD_PARTY or bool(
        _DIRECTORYLESS_THIRD_PARTY_PATTERN.match(header)
    )


_INCLUDE = re.compile(
    r'^(?P<lead>[ \t]*#[ \t]*include[ \t]*)(?P<open>[<"])(?P<header>[^>"\n]+)(?P<close>[>"])',
    re.MULTILINE,
)


def is_system_header(header: str) -> bool:
    """True for a C or C++ standard library header or a POSIX/Linux system header: the ones angle brackets are for."""
    if header in CPP_STANDARD_HEADERS or header in C_SYSTEM_HEADERS:
        return True
    return "/" in header and header.split("/", 1)[0] in C_SYSTEM_DIRECTORIES


def _includes(source: str) -> list[tuple[int, int, re.Match]]:
    """(line, column, match) of every #include of `source`, read with its comments blanked out."""
    without_comments = _without_comments(source)
    found = []
    for match in _INCLUDE.finditer(without_comments):
        line = without_comments.count("\n", 0, match.start()) + 1
        column = (
            match.start("open")
            - (without_comments.rfind("\n", 0, match.start()) + 1)
            + 1
        )
        found.append((line, column, match))
    return found


def _without_comments(source: str) -> str:
    return cpp_source.mask(source)[0]


_BRACKETS_REASON = "Angle brackets are for the C, C++ and POSIX system headers only (Google C++ style, Names and order of includes)."
# Why an angle Abseil include is worse than a style slip (the module docstring).
_ABSEIL_REASON = (
    "Bazel exposes @abseil-cpp to quoted includes only, and an angle include finds a system Abseil or none "
    "(tools/hooks/include_style.py)."
)


def _style_findings(source: str, path: str) -> list[check_types.Finding]:
    """The include-style findings of `source`: an angle include of a non-system header, or a directory-less one."""
    findings = []
    for line, column, match in _includes(source):
        header = match.group("header")
        if match.group("open") == "<" and not is_system_header(header):
            reason = _BRACKETS_REASON
            if header.startswith("absl/"):
                reason = _ABSEIL_REASON
            findings.append(
                check_types.Finding(
                    path,
                    line,
                    column,
                    NAME,
                    f'`#include <{header}>`: write `#include "{header}"`. {reason}',
                )
            )
        elif (
            match.group("open") == '"'
            and "/" not in header
            and not is_directoryless_third_party(header)
        ):
            findings.append(
                check_types.Finding(
                    path,
                    line,
                    column,
                    NAME,
                    f'`#include "{header}"` names no directory: include a project header by its full path below its '
                    "include root (Google C++ style, Names and order of includes).",
                )
            )
    return findings


def fix_include_style(source: str, path: str) -> str:
    """Rewrites the angle brackets around every header that is not a system header to quotes."""
    del path  # Unused.
    edits = []
    for _, _, match in _includes(source):
        if match.group("open") == "<" and not is_system_header(match.group("header")):
            edits.append(
                (match.start("open"), match.end("close"), f'"{match.group("header")}"')
            )
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_style_findings,
        fix_source=fix_include_style,
        fixed_by_format=True,
        description="angle brackets only for C, C++ and POSIX system headers; every other include is quoted with its "
        "full path (G: Names and order of includes).",
        hint="`make format` (or lint_code --fix --only include-style) rewrites the brackets; a bare file name needs its "
        "full path by hand.",
    ),
]
