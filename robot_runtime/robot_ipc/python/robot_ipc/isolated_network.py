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


"""A network of its own for every test process: the twin of robot_runtime/robot_ipc/test/TestNetwork.h."""

import os
import secrets

from robot_ipc import network_config

# The test domains: away from domain 0 of the shipped network file, and within the ports of the network file.
_FIRST_TEST_DOMAIN = 100
_TEST_DOMAINS = 100
# Drawn once per process (module import), so that every bus of a test process shares one domain.
_DRAW = secrets.randbits(64)
_DOMAIN_ID = _FIRST_TEST_DOMAIN + _DRAW % _TEST_DOMAINS
_DOMAIN_TAG = f"robot_ipc_test_{os.getpid()}_{_DRAW:x}"


def isolated_test_network(
    node_names: "list[str] | tuple[str, ...]",
) -> network_config.NetworkConfig:
    """A network of `node_names` on 127.0.0.1 that no other process joins.

    Its domain id and domain tag are drawn once per process, so the tests that Bazel runs at once never discover each
    other, while every bus of this process shares the domain. Hand it to a child process with
    format_network_config() to share it across processes.

    Args:
        node_names: The nodes of the network.

    Returns:
        The network.
    """
    return network_config.NetworkConfig(
        nodes=tuple(
            network_config.NodeEndpoint(name, network_config.LOOPBACK_HOST)
            for name in node_names
        ),
        domain_id=_DOMAIN_ID,
        domain_tag=_DOMAIN_TAG,
    )
