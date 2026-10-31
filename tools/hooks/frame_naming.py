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

"""Names every frame-dependent quantity by its frames: `<quantity>_<frame1>_rt_<frame2>_ewrt_<frame3>`.

The convention is humanoid_nmpc/docs/frame_naming/README.md; this module holds its vocabulary (the quantities, the
frames, the qualifiers and the suffixes), which the README's tables list, and the check that measures the rename:

  frame-naming  in C++, Python and .proto (first-party code and lib/ocs2 without lib/ocs2/thirdparty):
                - an identifier with an `rt` or `ewrt` word follows the grammar: a known quantity, the frames its form
                  takes, known frame names, canonical qualifiers in order and a known suffix;
                - its words are spelled in one case style, the one of the identifier it replaces, and each splits back
                  into the same words: a C++ name camelCase (`getPosBaseRtWorldEwrtWorld`), kName or, where the code
                  was snake_case, snake_case; a proto field and a Python name snake_case, a Python module constant
                  UPPER_SNAKE; never the mixed `getPos_base_rt_world_ewrt_world` or `nomPos_base_rt_com_ewrt_base`;
                - an identifier without one that holds a legacy quantity word (position, velocity, orientation,
                  quaternion, rotation, acceleration, force, torque, wrench, twist, momentum, pose, ... and their
                  abbreviations) is a legacy name, unless it is exempt: a type, an enumerator, a macro, a module, a
                  joint-space, dimension or index name, a weight or gain, a third-party API name, or a test name.
                Comments and string literals are not read.

It is opt-in while the rename runs (`opt_in=True`): `make lint` skips it and
`python3 -m tools.hooks.lint_code --only frame-naming --paths <dir>` runs it. The rename's Verify phase deletes
`opt_in=True` below once the last finding is fixed.
"""

import io
import keyword
import re
import tokenize

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "frame-naming"
README = "humanoid_nmpc/docs/frame_naming/README.md"

# LINT.IfChange(dirs)
# The code that computes with frames: the robotics code, and the tools that read its messages and configurations. The
# lint machinery and the proto generator (tools/hooks, tools/nproto, ...) are not read: a `position` there is an offset
# into a text.
DIRS = (
    "lib/ocs2/",
    "humanoid_nmpc/",
    "robot_runtime/",
    "humanoid_state_estimation/",
    "robot_models/",
    "humanoid_learning/",
    "tools/config_dump/",
    "tools/config_registries/",
    "tools/locomotion_heuristics/",
    "tools/ipc/",
)
# LINT.ThenChange(//humanoid_nmpc/docs/frame_naming/README.md:units)

# The suffixes of the files nproto generates (tools/nproto). Their names are copied from the .proto fields, which the
# check reads, so the generated code is held to the convention through its .proto (README section 1).
_NPROTO_GENERATED_SUFFIXES = (".nproto.h", ".nproto.pb.h", ".nproto.pb.cc")

RT = "rt"
EWRT = "ewrt"

# The forms a quantity takes (README section 2.2).
RELATIVE_EXPRESSED = "relative_expressed"  # <q>_<f1>_rt_<f2>_ewrt_<f3>
RELATIVE = "relative"  # <q>_<f1>_rt_<f2>
APPLIED = "applied"  # <q>_<f1>_ewrt_<f3>
FREE = "free"  # <q>[_<subject>]_ewrt_<f3>

# LINT.IfChange(quantities)
# The quantity prefixes and their forms. The README's quantity table defines each one.
QUANTITIES: dict[str, str] = {
    "pos": RELATIVE_EXPRESSED,
    "linvel": RELATIVE_EXPRESSED,
    "angvel": RELATIVE_EXPRESSED,
    "linacc": RELATIVE_EXPRESSED,
    "angacc": RELATIVE_EXPRESSED,
    "twist": RELATIVE_EXPRESSED,
    "spatialacc": RELATIVE_EXPRESSED,
    "specforce": RELATIVE_EXPRESSED,
    "linmom": RELATIVE_EXPRESSED,
    "angmom": RELATIVE_EXPRESSED,
    "cmom": RELATIVE_EXPRESSED,
    "jac": RELATIVE_EXPRESSED,
    "linjac": RELATIVE_EXPRESSED,
    "angjac": RELATIVE_EXPRESSED,
    "quat": RELATIVE,
    "rotmat": RELATIVE,
    "rotvec": RELATIVE,
    "eulerzyx": RELATIVE,
    "pose": RELATIVE,
    "poseeulerzyx": RELATIVE,
    "yaw": RELATIVE,
    "dist": RELATIVE,
    "speed": RELATIVE,
    "force": APPLIED,
    "torque": APPLIED,
    "wrench": APPLIED,
    "normal": FREE,
    "axis": FREE,
    "dir": FREE,
    "gravity": FREE,
}
# LINT.ThenChange(//humanoid_nmpc/docs/frame_naming/README.md:quantities)

# LINT.IfChange(frames)
# The frame names. The README's frame table defines each one.
FRAMES = frozenset(
    {
        "world",
        "base",
        "com",
        "acom",
        "heading",
        "torso",
        "imu",
        "contact",
        "mjcontact",
        "foot",
        "lfoot",
        "rfoot",
        "hand",
        "lhand",
        "rhand",
        "knee",
        "lknee",
        "rknee",
        "hip",
        "lhip",
        "rhip",
        "ankle",
        "lankle",
        "rankle",
        "pad",
        "ee",
        "link",
        "surface",
        "icp",
        "dcm",
        "zmp",
        "cop",
        "cam",
        "ball",
        "param",
        "local",
    }
)
# The plurals of the generic roles, for a container indexed by that role; frame1 only.
PLURAL_FRAMES: dict[str, str] = {
    "contacts": "contact",
    "feet": "foot",
    "hands": "hand",
    "knees": "knee",
    "hips": "hip",
    "ankles": "ankle",
    "pads": "pad",
    "ees": "ee",
    "links": "link",
}
# What a free vector may belong to besides a frame.
FREE_SUBJECTS = frozenset({"terrain", "joint"})
# LINT.ThenChange(//humanoid_nmpc/docs/frame_naming/README.md:frames)

# LINT.IfChange(qualifiers)
# The qualifiers, in the order they are written when a name has several.
QUALIFIERS: tuple[str, ...] = (
    "ad",
    "init",
    "prev",
    "final",
    "nom",
    "ref",
    "des",
    "cmd",
    "meas",
    "est",
    "true",
    "filt",
    "min",
    "max",
    "err",
    "mnorm",
    "rate",
)
# Spellings of a qualifier that the convention replaces.
QUALIFIER_SYNONYMS: dict[str, str] = {
    "desired": "des",
    "reference": "ref",
    "measured": "meas",
    "measurement": "meas",
    "estimated": "est",
    "estimate": "est",
    "commanded": "cmd",
    "nominal": "nom",
    "initial": "init",
    "previous": "prev",
    "last": "prev",
    "default": "nom",
    "lower": "min",
    "upper": "max",
    "delta": "err",
    "actual": "true",
    "truth": "true",
    "filtered": "filt",
    "error": "err",
    "minimum": "min",
    "maximum": "max",
    "normalized": "mnorm",
    "dot": "rate",
    "derivative": "rate",
    "target": "des (a planner's output) or ref (the MPC's TargetTrajectories)",
}
# LINT.ThenChange(//humanoid_nmpc/docs/frame_naming/README.md:qualifiers)

# LINT.IfChange(suffixes)
# The suffixes after the frames: the components of a vector in frame3's axes, the coefficient order of a quaternion,
# one Euler angle, a time-indexed container, and the `_size` of a generated repeated-field accessor.
VECTOR_COMPONENTS = frozenset({"x", "y", "z", "xy", "xz", "yz"})
QUATERNION_LAYOUTS = frozenset({"xyzw", "wxyz"})
EULER_ANGLES = frozenset({"yaw", "pitch", "roll"})
CONTAINER_SUFFIXES = frozenset({"traj", "size"})
# LINT.ThenChange(//humanoid_nmpc/docs/frame_naming/README.md:suffixes)

# LINT.IfChange(legacy_words)
# Words of a legacy name: a frame-dependent quantity named without its frames.
LEGACY_WORDS = frozenset(
    {
        "position",
        "positions",
        "pos",
        "velocity",
        "velocities",
        "vel",
        "vels",
        "orientation",
        "orientations",
        "quaternion",
        "quaternions",
        "quat",
        "quats",
        "rotation",
        "rotations",
        "rot",
        "acceleration",
        "accelerations",
        "accel",
        "force",
        "forces",
        "torque",
        "torques",
        "wrench",
        "wrenches",
        "twist",
        "twists",
        "momentum",
        "momenta",
        "moment",
        "moments",
        "pose",
        "poses",
    }
)
# A name with one of these words is not a frame quantity: joint space, layout, tuning knobs, and objects named after a
# concept (README section 4).
EXEMPT_WORDS = frozenset(
    {
        "joint",
        "joints",
        "qpos",
        "qvel",
        "qacc",
        "generalized",
        "configuration",
        "dim",
        "dims",
        "dimension",
        "index",
        "indices",
        "idx",
        "count",
        "num",
        "layout",
        "slot",
        "slots",
        "weight",
        "weights",
        "gain",
        "gains",
        "cost",
        "costs",
        "constraint",
        "constraints",
        "penalty",
        "noise",
        "command",
        "commands",
        "message",
        "msg",
        "msgs",
        "pb2",
        "proto",
        "topic",
        "registry",
        "gui",
        "viewer",
        "visualization",
        "marker",
        "markers",
        "parameters",
        "params",
        "config",
        "configs",
        "settings",
        "options",
        "info",
        "model",
        "models",
        "cone",
        "cones",
        "generator",
        "generators",
        "start",
        "coefficients",
        "internal",
        "math",
        "name",
        "names",
        "color",
        "colors",
        "folder",
        "tracking",
        "interface",
        "callback",
        "rule",
        "gate",
        "enable",
        "disable",
        "mode",
        "brute",
    }
)
# The words of a generic operation on rotations or vectors whose frames are its arguments (the quaternion algebra of
# ocs2_core/manifold/UnitQuaternionMath.h, the conversions of orientation/EulerBoundary.h and RotationTransforms.h): the
# name says what it does, not which frames it relates.
OPERATION_WORDS = frozenset(
    {
        "exp",
        "log",
        "product",
        "conjugate",
        "slerp",
        "tangent",
        "safe",
        "safely",
        "between",
        "distance",
        "random",
        "unit",
    }
)
# A conversion between representations, `<a>From<b>` or `<a>To<b>`, is a generic operation when <b> is one of these.
REPRESENTATION_WORDS = frozenset(
    {
        "quaternion",
        "quat",
        "euler",
        "eulerzyx",
        "zyx",
        "rpy",
        "matrix",
        "xyzw",
        "wxyz",
        "unit",
        "rotation",
        "axis",
        "angles",
        "vectors",
    }
)
# LINT.ThenChange(//humanoid_nmpc/docs/frame_naming/README.md:exemptions)

# Third-party API names that hold a legacy word: Eigen, Pinocchio, MuJoCo and the Python libraries. A name that one of
# these namespaces or modules qualifies (`pinocchio::`, `np.`) is not read at all.
THIRD_PARTY_NAMES = frozenset(
    {
        # Eigen.
        "rotation",
        "toRotationMatrix",
        "rotate",
        "prerotate",
        # Pinocchio.
        "getFrameVelocity",
        "getFrameAcceleration",
        "getFrameClassicalAcceleration",
        "getVelocity",
        "getAcceleration",
        "getClassicalAcceleration",
        "computeForwardKinematicsDerivatives",
        "getCenterOfMassVelocityDerivatives",
        "computeCentroidalMomentum",
        "computeCentroidalMomentumTimeVariation",
        "framesForwardKinematics",
        # MuJoCo.
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "geom_pos",
        "geom_quat",
        "site_pos",
        "site_quat",
        "cam_pos",
        "cam_quat",
        "cam_pos0",
        "light_pos",
        "jnt_pos",
        "mocap_pos",
        "mocap_quat",
        "actuator_force",
        "actuator_velocity",
        "qfrc_actuator",
        "qfrc_applied",
        "xfrc_applied",
        "mj_contactForce",
        "mj_objectVelocity",
        "mj_fwdPosition",
        "mj_subtreeVel",
        "mjSENS_FORCE",
        "mju_rotVecQuat",
        "mju_quat2Mat",
        "mju_mat2Quat",
        "mju_mulQuat",
        "mju_negQuat",
        "mju_axisAngle2Quat",
        "mju_quatIntegrate",
        "frame_pos",
        "frame_quat",
        # GLFW.
        "glfwGetCursorPos",
        # scipy.spatial.transform.
        "Rotation",
        "from_quat",
        "as_quat",
        "from_rotvec",
        "as_rotvec",
    }
)
_THIRD_PARTY_NAMESPACES = frozenset(
    {"std", "absl", "Eigen", "pinocchio", "CppAD", "google", "mujoco", "rerun"}
)
_THIRD_PARTY_MODULES = frozenset(
    {
        "np",
        "numpy",
        "mujoco",
        "rr",
        "rerun",
        "scipy",
        "pin",
        "pinocchio",
        "math",
        "tk",
        "torch",
        "jax",
        "jnp",
        "gym",
        "gymnasium",
        "spatial",
        "transform",
    }
)

_CAMEL_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+[0-9]*")
_CONSTANT = re.compile(r"^k[A-Z0-9][A-Za-z0-9]*$")
_UPPER_SNAKE = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
_TYPE_NAME = re.compile(r"^_*[A-Z]")
_GRAMMAR_HINT = (
    "Name it <quantity>_<frame1>_rt_<frame2>_ewrt_<frame3> from the vocabulary of"
    f" {README}, spelled in the case style of its kind (a C++ local, member or"
    " function `posBaseRtWorldEwrtWorld`, a constant `kPosBaseRtWorldEwrtWorld`, a"
    " proto field or Python name `pos_base_rt_world_ewrt_world`; an orientation"
    " `quat_<f1>_rt_<f2>`, a force `force_<f1>_ewrt_<f3>`), or mark a genuine"
    " exception `NOLINT(frame-naming): <reason>`."
)

# The case styles an identifier is spelled in (README section 2.4).
SNAKE = "snake_case"  # pos_base_rt_world_ewrt_world: proto fields, Python, snake_case C++ struct fields
UPPER_SNAKE = (
    "UPPER_SNAKE"  # POS_BASE_RT_WORLD_EWRT_WORLD: Python module constants
)
CAMEL = "camelCase"  # posBaseRtWorldEwrtWorld: C++ locals, parameters, members and functions
CONSTANT = "kName"  # kPosBaseRtWorldEwrtWorld: C++ constants
MIXED = "mixed"  # getPos_base_rt_world_ewrt_world: humps and underscores, never correct


def style_of(identifier: str) -> str:
    """The case style of `identifier`, without the leading and trailing underscores of a member or private name.

    Args:
      identifier: A C++, Python or proto identifier.

    Returns:
      SNAKE, UPPER_SNAKE, CAMEL, CONSTANT or MIXED. A single lowercase word is SNAKE (and also a valid CAMEL).
    """
    core = identifier.strip("_")
    if _CONSTANT.match(core):
        return CONSTANT
    if re.fullmatch(r"[a-z][a-z0-9_]*", core):
        return SNAKE
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", core):
        return UPPER_SNAKE
    if re.fullmatch(r"[a-z][A-Za-z0-9]*", core):
        return CAMEL
    return MIXED


def split_words(identifier: str) -> list[str]:
    """Splits `identifier` into its words: at underscores, and at the humps of camelCase and kName pieces.

    Every word of the vocabulary is one lowercase run, so each spelling splits back into the same words:
    `posBaseRtWorld`, `kPosBaseRtWorld`, `pos_base_rt_world` and `POS_BASE_RT_WORLD` are all
    pos | base | rt | world.

    Args:
      identifier: A C++, Python or proto identifier.

    Returns:
      The words in order, lowercased; the leading and trailing underscores of a member or private name are dropped.
    """
    core = identifier.strip("_")
    if _CONSTANT.match(core):
        return [hump.lower() for hump in _CAMEL_WORD.findall(core[1:])]
    words = []
    for piece in core.split("_"):
        if not piece:
            continue
        if piece.islower() or piece.isupper() or piece.isdigit():
            words.append(piece.lower())
            continue
        words.extend(hump.lower() for hump in _CAMEL_WORD.findall(piece))
    return words


def spell(identifier: str, style: str) -> str:
    """Spells the words of `identifier` in `style`, keeping its leading and trailing underscores.

    Args:
      identifier: A C++, Python or proto identifier.
      style: SNAKE, UPPER_SNAKE, CAMEL or CONSTANT.

    Returns:
      The same words in that style: `getPos_base_rt_world` in CAMEL is `getPosBaseRtWorld`.
    """
    words = split_words(identifier)
    if not words:
        return identifier
    leading = identifier[: len(identifier) - len(identifier.lstrip("_"))]
    trailing = identifier[len(identifier.rstrip("_")) :]
    if style == SNAKE:
        body = "_".join(words)
    elif style == UPPER_SNAKE:
        body = "_".join(words).upper()
    elif style == CONSTANT:
        body = "k" + "".join(word.capitalize() for word in words)
    else:
        body = words[0] + "".join(word.capitalize() for word in words[1:])
    return leading + body + trailing


def _frame_problem(frame: str, role: str, plural_ok: bool) -> str | None:
    """Why `frame` cannot be the frame `role`, or None when it can."""
    if frame in FRAMES:
        return None
    if plural_ok and frame in PLURAL_FRAMES:
        return None
    if frame in PLURAL_FRAMES:
        return (
            f"{role} `{frame}` is a plural: only frame1 names a container"
            f" (use `{PLURAL_FRAMES[frame]}`)"
        )
    return f"{role} `{frame}` is not a frame of the vocabulary"


def _context_problems(context: list[str]) -> list[str]:
    """The problems of the words before the quantity: qualifier synonyms, their order, a second quantity."""
    problems = []
    last_rank = -1
    for word in context:
        if word in QUALIFIER_SYNONYMS:
            problems.append(
                f"write the qualifier `{word}` as"
                f" `{QUALIFIER_SYNONYMS[word]}`"
            )
        elif word in LEGACY_WORDS or word in QUANTITIES:
            problems.append(
                f"`{word}` before the quantity: a name holds one quantity"
            )
        elif word in QUALIFIERS:
            rank = QUALIFIERS.index(word)
            if rank <= last_rank:
                problems.append(
                    f"qualifier `{word}` out of order: write them in the"
                    f" order {', '.join(QUALIFIERS)}"
                )
            last_rank = rank
    return problems


def _suffix_problems(quantity: str, form: str, suffix: list[str]) -> list[str]:
    """The problems of the words after the frames: a component, a layout or an angle, then a container."""
    texts = list(suffix)
    if texts and texts[-1] in CONTAINER_SUFFIXES:
        texts = texts[:-1]
    if not texts:
        return []
    if len(texts) == 1:
        component = texts[0]
        if component in VECTOR_COMPONENTS and form in (
            RELATIVE_EXPRESSED,
            APPLIED,
            FREE,
        ):
            return []
        if component in QUATERNION_LAYOUTS and quantity == "quat":
            return []
        if component in EULER_ANGLES and quantity in (
            "eulerzyx",
            "poseeulerzyx",
        ):
            return []
    return [
        f"`{'_'.join(texts)}` after the frames is not a suffix of `{quantity}`"
        " (a component, a quaternion layout, an Euler angle, `traj` or `size`)"
    ]


def _parse_relative(words: list[str], at: int) -> tuple[list[str], int, str]:
    """Checks the quantity, frame1 and frame2 around the `rt` at `at`; returns the problems, the end and the form."""
    problems: list[str] = []
    if at < 2:
        return ["`rt` needs a quantity and frame1 before it"], len(words), ""
    quantity, subject = words[at - 2], words[at - 1]
    form = QUANTITIES.get(quantity, "")
    if not form:
        problems.append(f"`{quantity}` is not a quantity of the vocabulary")
    elif form in (APPLIED, FREE):
        problems.append(
            f"`{quantity}` names no relative frame: write"
            f" `{quantity}_<frame1>_ewrt_<frame3>`"
        )
    problem = _frame_problem(subject, "frame1", plural_ok=True)
    if problem:
        problems.append(problem)
    if at + 1 >= len(words):
        problems.append("`rt` needs frame2 after it")
        return problems, len(words), form
    frame2 = words[at + 1]
    problem = _frame_problem(frame2, "frame2", plural_ok=False)
    if problem:
        problems.append(problem)
    if PLURAL_FRAMES.get(subject, subject) == frame2:
        problems.append(f"frame1 and frame2 are both `{frame2}`")
    end = at + 2
    has_ewrt = end < len(words) and words[end] == EWRT
    if has_ewrt:
        if end + 1 >= len(words):
            problems.append("`ewrt` needs frame3 after it")
            return problems, len(words), form
        problem = _frame_problem(words[end + 1], "frame3", plural_ok=False)
        if problem:
            problems.append(problem)
        end += 2
    if form == RELATIVE_EXPRESSED and not has_ewrt:
        problems.append(
            f"`{quantity}` needs the frame it is expressed in:"
            " `_ewrt_<frame3>`"
        )
    if form == RELATIVE and has_ewrt:
        problems.append(
            f"`{quantity}` relates two frames and is expressed in none:"
            " drop `_ewrt_<frame3>`"
        )
    return problems, end, form


def _parse_expressed(words: list[str], at: int) -> tuple[list[str], int, str]:
    """Checks a name whose first frame word is the `ewrt` at `at`: an applied or a free quantity."""
    problems: list[str] = []
    if at < 1:
        return ["`ewrt` needs a quantity before it"], len(words), ""
    if words[at - 1] in QUANTITIES:
        quantity, subject = words[at - 1], None
    elif at >= 2:
        quantity, subject = words[at - 2], words[at - 1]
    else:
        return (
            [f"`{words[at - 1]}` is not a quantity of the vocabulary"],
            len(words),
            "",
        )
    form = QUANTITIES.get(quantity, "")
    if not form:
        problems.append(f"`{quantity}` is not a quantity of the vocabulary")
    elif form in (RELATIVE_EXPRESSED, RELATIVE):
        problems.append(
            f"`{quantity}` needs the frame it is relative to:"
            f" `{quantity}_<frame1>_rt_<frame2>"
            + ("_ewrt_<frame3>`" if form == RELATIVE_EXPRESSED else "`")
        )
    elif form == APPLIED and subject is None:
        problems.append(
            f"`{quantity}` needs the frame it acts at:"
            f" `{quantity}_<frame1>_ewrt_<frame3>`"
        )
    if subject is not None:
        if not (form == FREE and subject in FREE_SUBJECTS):
            problem = _frame_problem(subject, "frame1", plural_ok=True)
            if problem:
                problems.append(problem)
    if at + 1 >= len(words):
        problems.append("`ewrt` needs frame3 after it")
        return problems, len(words), form
    problem = _frame_problem(words[at + 1], "frame3", plural_ok=False)
    if problem:
        problems.append(problem)
    return problems, at + 2, form


def is_frame_name(identifier: str) -> bool:
    """True when `identifier` is written in the grammar: it has an `rt` or `ewrt` word and a quantity word.

    `rt` alone is another word (a gamepad's right trigger) when no quantity word comes with it.
    """
    words = split_words(identifier)
    return any(word in (RT, EWRT) for word in words) and any(
        word in QUANTITIES or word in LEGACY_WORDS for word in words
    )


def style_problem(
    identifier: str, allowed: tuple[str, ...], kind: str
) -> str | None:
    """Why the spelling of the frame name `identifier` does not fit its kind of identifier, or None (README 2.4).

    Args:
      identifier: The identifier.
      allowed: The case styles its kind is spelled in, the convention's own first.
      kind: What the identifier is, for the message: "a proto field".

    Returns:
      The problem, with the name respelled; None when it is no frame name or its style is allowed.
    """
    if not is_frame_name(identifier):
        return None
    style = style_of(identifier)
    if style in allowed:
        return None
    target = allowed[0]
    if style == CONSTANT and UPPER_SNAKE in allowed:
        target = UPPER_SNAKE
    if style == UPPER_SNAKE and CONSTANT in allowed:
        target = CONSTANT
    spelled = (
        "mixes camelCase humps and snake_case words"
        if style == MIXED
        else f"is {style}"
    )
    return f"it {spelled}, and {kind} is {target}: write `{spell(identifier, target)}`"


def grammar_problems(identifier: str) -> list[str]:
    """The ways `identifier`, a name with an `rt` or `ewrt` word, departs from the grammar (README section 2).

    Args:
      identifier: The identifier.

    Returns:
      One message per problem; empty when the name follows the grammar.
    """
    if not is_frame_name(identifier):
        return []
    words = split_words(identifier)
    texts = words
    at = next(i for i, text in enumerate(texts) if text in (RT, EWRT))
    if texts[at] == RT:
        problems, end, form = _parse_relative(words, at)
        start = at - 2
    else:
        problems, end, form = _parse_expressed(words, at)
        start = at - 1 if at >= 1 and texts[at - 1] in QUANTITIES else at - 2
    if any(text in (RT, EWRT) for text in texts[end:]):
        problems.append("one quantity per name: a second `rt` or `ewrt`")
        return problems
    quantity = texts[max(start, 0)]
    problems += _context_problems(words[: max(start, 0)])
    problems += _suffix_problems(quantity, form, words[end:])
    return problems


def is_legacy(identifier: str) -> bool:
    """True when `identifier` names a frame quantity without its frames and is not exempt (README section 4)."""
    ordered = split_words(identifier)
    words = set(ordered)
    if (RT in words or EWRT in words) and words & (
        QUANTITIES.keys() | LEGACY_WORDS
    ):
        return False
    if not words & LEGACY_WORDS or words & (EXEMPT_WORDS | OPERATION_WORDS):
        return False
    for before, after in zip(ordered, ordered[1:]):
        if before in ("from", "to") and after in REPRESENTATION_WORDS:
            return False
    # The quaternion algebra's maps of a quaternion argument: quaternionRotationMatrix, quaternionRateMatrix.
    if ordered[0] == "quaternion" and ordered[-1] in ("matrix", "map"):
        return False
    # The centroidal momentum matrix and its blocks, maps from velocities to momentum (README section 3.1):
    # getCentroidalMomentumMatrix, computeFloatingBaseCentroidalMomentumMatrixInverse.
    if words & LEGACY_WORDS == {"momentum"} and "matrix" in words:
        return False
    return identifier not in THIRD_PARTY_NAMES


def _is_type_like(identifier: str) -> bool:
    """A type, a macro or an enumerator spelling: PascalCase, UPPER_SNAKE or a `_t` / `_T` alias."""
    core = identifier.strip("_")
    if _CONSTANT.match(core):
        return False
    if identifier.endswith(("_t", "_T")):
        return True
    return bool(_TYPE_NAME.match(core)) and not core.isupper()


def _finding(
    path: str,
    line: int,
    column: int,
    identifier: str,
    allowed: tuple[str, ...],
    kind: str,
) -> check_types.Finding | None:
    """The finding for `identifier` at the position: a grammar or spelling error, a legacy name, or None.

    Args:
      path: The file.
      line: The identifier's line.
      column: The identifier's column.
      identifier: The identifier.
      allowed: The case styles its kind of identifier is spelled in (style_problem()).
      kind: What the identifier is, for the message.

    Returns:
      The finding, or None when the name is fine.
    """
    problems = grammar_problems(identifier)
    spelling = style_problem(identifier, allowed, kind)
    if spelling is not None:
        problems.append(spelling)
    if problems:
        return check_types.Finding(
            path,
            line,
            column,
            NAME,
            f"`{identifier}` does not follow the frame-naming grammar: "
            + "; ".join(problems)
            + ".",
        )
    if is_legacy(identifier):
        return check_types.Finding(
            path,
            line,
            column,
            NAME,
            f"`{identifier}` names a frame-dependent quantity without its"
            " frames.",
        )
    return None


# The case styles of each kind of identifier (README section 2.4). A new name keeps the style of the identifier it
# replaces. In C++ that is camelCase for most locals, parameters, members and functions and kName for constants, but
# snake_case for a snake_case local or a member that mirrors a snake_case field (a CentroidalModelInfo field, a
# protobuf accessor); the check cannot tell which style the replaced name had, so C++ accepts all three and rejects
# only the mixed spelling.
_CPP_STYLES: tuple[str, ...] = (CAMEL, CONSTANT, SNAKE)
_CPP_KIND = "a C++ name"
_PROTO_STYLES: tuple[str, ...] = (SNAKE,)
_PROTO_KIND = "a proto field"
_PYTHON_STYLES: tuple[str, ...] = (SNAKE, UPPER_SNAKE)
_PYTHON_KIND = "a Python name (a module constant: UPPER_SNAKE)"


def _cpp_findings(
    source: str, path: str, language: check_types.Language
) -> list[check_types.Finding]:
    """The findings in C++ or proto `source`, read with the C++ tokenizer (comments and strings skipped)."""
    tokens = cpp_source.code_tokens(source)
    in_scopes = (
        cpp_source.scopes(source)
        if language == check_types.Language.CPP
        else None
    )
    findings: list[check_types.Finding] = []
    seen: set[tuple[int, str]] = set()
    for k, token in enumerate(tokens):
        if token.kind != cpp_source.IDENTIFIER:
            continue
        name = token.text
        if language == check_types.Language.CPP and _UPPER_SNAKE.match(name):
            continue
        if language == check_types.Language.PROTO and _UPPER_SNAKE.match(name):
            continue
        if _is_type_like(name):
            continue
        if k >= 2 and tokens[k - 1].text == "::":
            qualifier = tokens[k - 2].text
            if qualifier in _THIRD_PARTY_NAMESPACES:
                continue
            # `Type::kName`: an enumerator or a class constant, read where it is declared.
            if _CONSTANT.match(name) and _TYPE_NAME.match(qualifier):
                continue
        if in_scopes is not None and in_scopes[k][-1:] == (
            cpp_source.ENUM_SCOPE,
        ):
            continue
        if (token.line, name) in seen:
            continue
        if language == check_types.Language.PROTO:
            allowed, kind = _PROTO_STYLES, _PROTO_KIND
        else:
            allowed, kind = _CPP_STYLES, _CPP_KIND
        finding = _finding(path, token.line, token.col, name, allowed, kind)
        if finding is not None:
            seen.add((token.line, name))
            findings.append(finding)
    return findings


def _python_findings(source: str, path: str) -> list[check_types.Finding]:
    """The findings in Python `source`, read with the tokenize module (imports and test names skipped)."""
    findings: list[check_types.Finding] = []
    seen: set[tuple[int, str]] = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return []
    in_import = False
    previous: list[tokenize.TokenInfo] = []
    for token in tokens:
        if token.type in (tokenize.NEWLINE, tokenize.NL):
            in_import = False
            previous = []
            continue
        if token.type in (tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            continue
        if token.type == tokenize.NAME and token.string in ("import", "from"):
            if not previous:
                in_import = True
        context = previous[-2:]
        previous.append(token)
        if token.type != tokenize.NAME or in_import:
            continue
        name = token.string
        if keyword.iskeyword(name) or _is_type_like(name):
            continue
        if name.startswith(("test_", "assert")):
            continue
        if (
            len(context) == 2
            and context[1].string == "."
            and context[0].string in _THIRD_PARTY_MODULES
        ):
            continue
        if (token.start[0], name) in seen:
            continue
        finding = _finding(
            path,
            token.start[0],
            token.start[1] + 1,
            name,
            _PYTHON_STYLES,
            _PYTHON_KIND,
        )
        if finding is not None:
            seen.add((token.start[0], name))
            findings.append(finding)
    return findings


def _findings(source: str, path: str) -> list[check_types.Finding]:
    normalized = lint_files.normalize(path)
    if not normalized.startswith(DIRS) or normalized.endswith(
        _NPROTO_GENERATED_SUFFIXES
    ):
        return []
    language = check_types.language_of(path)
    if language == check_types.Language.PYTHON:
        return _python_findings(source, path)
    if language in (check_types.Language.CPP, check_types.Language.PROTO):
        return _cpp_findings(source, path, language)
    return []


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset(
            {
                check_types.Language.CPP,
                check_types.Language.PYTHON,
                check_types.Language.PROTO,
            }
        ),
        scope=lint_files.Scope.FIRST_PARTY_AND_OCS2,
        check_source=_findings,
        description=(
            "a frame-dependent quantity is named"
            " <quantity>_<frame1>_rt_<frame2>_ewrt_<frame3> from the vocabulary"
            f" of tools/hooks/frame_naming.py ({README})."
        ),
        hint=_GRAMMAR_HINT,
        opt_in=True,
    ),
]
