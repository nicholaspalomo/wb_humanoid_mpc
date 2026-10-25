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


"""Loopback ports for the tests' servers: ones nothing is bound to, outside the kernel's ephemeral range.

A server the test starts (Rerun's, in a process of its own or not) binds its port by number, so it cannot be handed a
bound socket. A port the kernel chose for a probe and got back could be chosen again, for another test or an outgoing
connection, before the server binds it; one outside the ephemeral range is only ever bound by name, and one picked at
random collides with another test's pick about never.
"""

import random
import socket

# The lowest port the tests pick: above the bus (5600-5629), Jupyter (8888) and Rerun's defaults (9090, 9876).
_LOWEST_PORT = 20000


def _ephemeral_range() -> tuple[int, int]:
    """The ports the kernel picks from for bind(0) and outgoing connections (ip_local_port_range)."""
    try:
        with open(
            "/proc/sys/net/ipv4/ip_local_port_range", encoding="utf-8"
        ) as ports:
            low, high = (int(word) for word in ports.read().split())
            return low, high
    except (OSError, ValueError):
        return 32768, 60999


def unused_ports(count: int) -> tuple[int, ...]:
    """`count` distinct loopback ports nothing is bound to, outside the kernel's ephemeral range.

    Args:
        count: how many ports.

    Returns:
        The ports, in no particular order.

    Raises:
        RuntimeError: fewer than `count` ports below the ephemeral range are free.
    """
    low, _ = _ephemeral_range()
    candidates = list(range(_LOWEST_PORT, max(_LOWEST_PORT + count, low)))
    random.SystemRandom().shuffle(candidates)
    chosen: set[int] = set()
    for port in candidates:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        chosen.add(port)
        if len(chosen) == count:
            return tuple(chosen)
    raise RuntimeError(f"no {count} unused ports in [{_LOWEST_PORT}, {low})")
