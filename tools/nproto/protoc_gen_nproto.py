"""protoc-gen-nproto: the protoc plugin that generates nproto's C++ structs (tools/nproto/README.md).

protoc runs it with a CodeGeneratorRequest on stdin and reads the CodeGeneratorResponse from stdout:

    protoc --plugin=protoc-gen-nproto=<this binary> --nproto_out=<dir> <files>

nproto_cc_library (tools/nproto/nproto.bzl) runs it on every proto_library it is given; nothing else needs to.
"""

import sys

from google.protobuf.compiler import plugin_pb2

import nproto_generator


def main() -> int:
    # nproto_generator has imported nproto's options_pb2, so the request's custom options parse as its extensions.
    request = plugin_pb2.CodeGeneratorRequest.FromString(sys.stdin.buffer.read())
    response = nproto_generator.generate(request)
    sys.stdout.buffer.write(response.SerializeToString())
    return 0


if __name__ == "__main__":
    sys.exit(main())
