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


"""The tunable parameters of a configuration file, read from its schema: what the tuning GUI renders, without Tk.

    from remote_control import config_schema

    for spec in config_schema.parameters(task, document):  # a parsed TaskFile and its TextprotoDocument
        if spec.renders:
            ...  # spec.path, spec.kind, spec.value, spec.slider_range(), config_schema.display_label(spec, task)

Nothing here names a hyperparameter. The walk follows the descriptor of the file's message, so a field added to a
schema is a parameter of the GUI with no code written here or in a tab (humanoid_mpc_config/README.md, "GUI
metadata"): every numeric, bool and enum field,
and every string field that names a registry entry, is one. A repeated string is a name list (the term lists, shown
read-only); any other string, and bytes, is text the GUI does not edit.

How a field renders is the schema's: its (humanoid_mpc_config.tuning) option (tuning_options.proto) gives the slider
range, log scale, unit, exclusion reason, reload class, registry, the conditions under which it applies and who reads
it. A block's options apply to every field below it unless that field sets its own (the registry excepted, which is
each field's own; the conditions of every block above add up), so a field of a shared message (Xyz, YawPitchRoll,
JointValue) is tuned where it is used. A top-level block's message option (humanoid_mpc_config.group) places it on a
tab.

A parameter is labeled by its field names alone, as the user decided: the dotted path of the fields from below its
block down to it, with the key of each element it is in. No option renames a field, and no comment of the file is part
of a label; display_label() adds only the unit and what annotations() says.

A condition names a singular scalar by its path from the nearest message above the field that has it: from the file
for a file's own blocks (`contact_input_parameterization` in a block of the task file), and from the task file inside
the MPC parameter update, which carries it as `task`. The walks hand every condition on as its path from the message
walked (`task.contact_input_parameterization`), which is what condition_value() and is_active() read.

The walk yields what the file holds:

- a scalar the file sets, with its value and its line;
- a scalar it leaves out, with the schema default, shown "(default)"; or, where the schema has no default because
  absence has its own meaning (std::optional in C++), as unset;
- the fields of a block the file leaves out, with their defaults, so that a slider can insert the block; but not those
  of an absent (nproto.optional_message) block, whose absence switches its feature off;
- every element of a repeated field, by the value of its key where it has one: the first string field whose values are
  set and unique across the elements (`joint_positions[joint=l_leg_aky].value`), otherwise by index (`points[0].x`).

Paths are textproto_document paths, so textproto_save.SetValue(spec.path, value) edits the file.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import functools
import math
from typing import Any, TypeAlias

from google.protobuf import descriptor as descriptor_module
from google.protobuf import descriptor_pb2
from humanoid_mpc_config import config_options_pb2
from humanoid_mpc_config import task_file_pb2
from humanoid_mpc_config import tuning_options_pb2
from nproto import options_pb2

from config_textproto import textproto_document

FieldDescriptor: TypeAlias = descriptor_module.FieldDescriptor
Descriptor: TypeAlias = descriptor_module.Descriptor

# The reload classes of TuningOptions.reload.
RELOAD_UNSPECIFIED = tuning_options_pb2.TuningOptions.RELOAD_UNSPECIFIED
RELOAD_HOT = tuning_options_pb2.TuningOptions.RELOAD_HOT
RELOAD_START_UP = tuning_options_pb2.TuningOptions.RELOAD_START_UP

_FLOATING_TYPES = (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT)
_INTEGER_TYPES = (
    FieldDescriptor.TYPE_INT32,
    FieldDescriptor.TYPE_INT64,
    FieldDescriptor.TYPE_UINT32,
    FieldDescriptor.TYPE_UINT64,
    FieldDescriptor.TYPE_SINT32,
    FieldDescriptor.TYPE_SINT64,
    FieldDescriptor.TYPE_FIXED32,
    FieldDescriptor.TYPE_FIXED64,
    FieldDescriptor.TYPE_SFIXED32,
    FieldDescriptor.TYPE_SFIXED64,
)
# A slider over decades without a range of its own spans this factor either side of its value.
_LOG_SCALE_DECADES = 100.0
# The path segment of "every element" in schema_fields().
EVERY_ELEMENT = "[*]"


class SchemaError(ValueError):
    """A tuning option that cannot hold: a condition on a field the file does not have, or not a value."""


class Kind(enum.Enum):
    """How a parameter is edited."""

    NUMBER = "number"  # a double or a float: a slider
    INTEGER = "integer"  # an integer: a slider in steps of one
    BOOL = "bool"  # a checkbox
    CHOICE = "choice"  # an enum, or a string naming a registry entry: a drop-down
    NAME_LIST = "name_list"  # a repeated string, such as a term list: shown, read-only
    TEXT = "text"  # a string without a registry, or bytes: not edited


class ValueSource(enum.Enum):
    """Where a parameter's value comes from."""

    FILE = "file"  # the file sets it
    DEFAULT = "default"  # the file leaves it out, and the schema default applies
    UNSET = "unset"  # the file leaves it out, and the schema has no default: absence has its own meaning


@dataclasses.dataclass(frozen=True)
class Condition:
    """The field applies only while the field `field_path` of the file has one of `values`.

    Attributes:
      field_path: A field of the file message, by its dotted path.
      values: The values, as text: a string, an enum value's name, true or false, a number.
    """

    field_path: str
    values: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Tuning:
    """The tuning options in effect for a field: its own, and what it inherits from the blocks above it.

    Attributes:
      slider_min: The slider's lower end; None: from the value (slider_range()).
      slider_max: The slider's upper end; None: from the value.
      log_scale: Whether the slider spans decades.
      unit: The unit, e.g. "m"; empty: none.
      exclude_reason: Why the field gets no widget; empty: it gets one.
      reload: RELOAD_HOT, RELOAD_START_UP or RELOAD_UNSPECIFIED.
      registry: The registry a string field names an entry of; empty: none.
      active_when: The conditions under which the field applies; all of them must hold.
      consumer: Who reads the field: "gui", "robot", "mpc"; empty: unspecified.
      formulations: The MPC formulations that read the field (FORMULATIONS); empty: every formulation.
    """

    slider_min: float | None = None
    slider_max: float | None = None
    log_scale: bool = False
    unit: str = ""
    exclude_reason: str = ""
    reload: int = RELOAD_UNSPECIFIED
    registry: str = ""
    active_when: tuple[Condition, ...] = ()
    consumer: str = ""
    formulations: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class SchemaField:
    """A scalar field of a file schema, wherever it is reachable.

    Attributes:
      path: Its path from the file message, every element of a repeated message as `[*]`:
        "state_weights.joint_positions[*].value".
      field: Its descriptor.
      kind: How it is edited.
      tuning: Its tuning options in effect there.
    """

    path: str
    field: FieldDescriptor
    kind: Kind
    tuning: Tuning

    @property
    def renders(self) -> bool:
        """Whether the GUI gives the field a widget: an editable kind, not excluded."""
        return _renders(self.kind, self.tuning)


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
    """One parameter of a file, as the GUI renders it.

    Attributes:
      path: Its textproto_document path, e.g. "state_weights.joint_positions[joint=l_leg_aky].value".
      field: The field's descriptor.
      kind: How it is edited.
      tuning: Its tuning options in effect there.
      value: Its value: a float, an int, a bool, a str (an enum value's name for an enum), a tuple of str for a name
        list; None when unset.
      source: Where the value comes from.
      block: The top-level field it is under; empty for a top-level scalar.
      tab: The tab its top-level block is placed on ((humanoid_mpc_config.group)); empty: the default, the MPC tab.
      order: Its block's place on the tab.
      label: Its name within its block: the field names from below the block down to it, each element of a repeated
        field by its key, the key alone for an element's only value ("joint_positions[l_leg_aky]").
      row: The key (or `field[index]`) of the repeated element it is in; None outside one.
      choices: The names a choice offers (an enum's values, or the registry's names given to parameters()); None when
        not known.
      line: Its line in the file; None when the file does not have it.
    """

    path: str
    field: FieldDescriptor
    kind: Kind
    tuning: Tuning
    value: Any
    source: ValueSource
    block: str
    tab: str
    order: int
    label: str
    row: str | None
    choices: tuple[str, ...] | None
    line: int | None

    @property
    def renders(self) -> bool:
        """Whether the GUI gives the parameter a widget: an editable kind, not excluded."""
        return _renders(self.kind, self.tuning)

    @property
    def read_only(self) -> bool:
        """Whether its widget only shows it: name lists (term toggling is a follow-up) and text."""
        return self.kind in (Kind.NAME_LIST, Kind.TEXT)

    def slider_range(self) -> tuple[float, float] | None:
        """The slider's ends for a number or an integer (slider_range()); None for the other kinds."""
        if self.kind not in (Kind.NUMBER, Kind.INTEGER):
            return None
        value = self.value if isinstance(self.value, (int, float)) else 0.0
        return slider_range(float(value), self.tuning)


# LINT.IfChange(rendered_fields)
def _renders(kind: Kind, tuning: Tuning) -> bool:
    return kind != Kind.TEXT and not tuning.exclude_reason


# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/ConfigReload.cpp:tunable_fields)


def slider_range(value: float, tuning: Tuning) -> tuple[float, float]:
    """The ends of a slider: the schema's slider_min / slider_max where it sets them, otherwise from the value.

    Linear: (0, max(4 |value|, 1)), symmetric about 0 for a negative value, as the GUI has always ranged a slider. Log
    scale: a factor of 100 either side of the value (of 1 for a value that is not positive).

    Args:
      value: The parameter's value.
      tuning: Its tuning options.

    Returns:
      (low, high).
    """
    if tuning.log_scale:
        center = value if math.isfinite(value) and value > 0.0 else 1.0
        low, high = center / _LOG_SCALE_DECADES, center * _LOG_SCALE_DECADES
    else:
        # The rule TuningOptions.slider_min documents for a field without a range.
        # LINT.IfChange(default_slider_range)
        magnitude = max(abs(value) * 4.0, 1.0) if math.isfinite(value) else 1.0
        low, high = (-magnitude, magnitude) if value < 0.0 else (0.0, magnitude)
        # LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/tuning_options.proto:default_slider_range)
    return (
        low if tuning.slider_min is None else tuning.slider_min,
        high if tuning.slider_max is None else tuning.slider_max,
    )


# ---- Options ----


@functools.lru_cache(maxsize=None)
def own_tuning(field: FieldDescriptor) -> Any:
    """The (humanoid_mpc_config.tuning) option of `field`, a TuningOptions; None when it has none."""
    options = descriptor_pb2.FieldOptions.FromString(
        field.GetOptions().SerializeToString()
    )
    if not options.HasExtension(config_options_pb2.tuning):
        return None
    return options.Extensions[config_options_pb2.tuning]


@functools.lru_cache(maxsize=None)
def group(descriptor: Descriptor) -> tuple[str, int]:
    """(tab, order) of the (humanoid_mpc_config.group) option of a block's message; ("", 0) without one."""
    options = descriptor_pb2.MessageOptions.FromString(
        descriptor.GetOptions().SerializeToString()
    )
    if not options.HasExtension(config_options_pb2.group):
        return "", 0
    placement = options.Extensions[config_options_pb2.group]
    return str(placement.tab), int(placement.order)


@functools.lru_cache(maxsize=None)
def is_optional_message(field: FieldDescriptor) -> bool:
    """Whether `field` is an (nproto.optional_message) block, whose absence has its own meaning."""
    options = descriptor_pb2.FieldOptions.FromString(
        field.GetOptions().SerializeToString()
    )
    return bool(options.Extensions[options_pb2.optional_message])


# LINT.IfChange(tuning_inheritance)
def inherit(parent: Tuning, field: FieldDescriptor) -> Tuning:
    """The tuning of `field` below a block whose tuning is `parent` (the module docstring).

    Args:
      parent: The tuning in effect for the block that holds `field`; Tuning() at the top.
      field: The field.

    Returns:
      Its own options where it sets them (formulations: where it names any), the block's otherwise; its own registry;
      every condition.
    """
    own = own_tuning(field)
    if own is None:
        return dataclasses.replace(parent, registry="")
    conditions = tuple(
        Condition(str(c.field_path), tuple(str(v) for v in c.values))
        for c in own.active_when
    )
    formulations = tuple(str(name) for name in own.formulations)
    return Tuning(
        slider_min=(
            float(own.slider_min) if own.HasField("slider_min") else parent.slider_min
        ),
        slider_max=(
            float(own.slider_max) if own.HasField("slider_max") else parent.slider_max
        ),
        log_scale=(
            bool(own.log_scale) if own.HasField("log_scale") else parent.log_scale
        ),
        unit=str(own.unit) if own.HasField("unit") else parent.unit,
        exclude_reason=(
            str(own.exclude_reason)
            if own.HasField("exclude_reason")
            else parent.exclude_reason
        ),
        reload=int(own.reload) if own.HasField("reload") else parent.reload,
        registry=str(own.registry),
        active_when=parent.active_when + conditions,
        consumer=str(own.consumer) if own.HasField("consumer") else parent.consumer,
        formulations=formulations or parent.formulations,
    )


# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/ConfigReload.cpp:tuning_inheritance)


# LINT.IfChange(tunable_fields)
def kind_of(field: FieldDescriptor, tuning: Tuning) -> Kind:
    """How a scalar field is edited (the Kind members)."""
    if field.type in _FLOATING_TYPES:
        return Kind.NUMBER
    if field.type in _INTEGER_TYPES:
        return Kind.INTEGER
    if field.type == FieldDescriptor.TYPE_BOOL:
        return Kind.BOOL
    if field.type == FieldDescriptor.TYPE_ENUM:
        return Kind.CHOICE
    if field.type == FieldDescriptor.TYPE_STRING:
        if field.is_repeated:
            return Kind.NAME_LIST
        return Kind.CHOICE if tuning.registry else Kind.TEXT
    return Kind.TEXT


# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/ConfigReload.cpp:tunable_fields)


def _is_map(field: FieldDescriptor) -> bool:
    return field.message_type is not None and bool(
        field.message_type.GetOptions().map_entry
    )


def _fields(descriptor: Descriptor) -> list[FieldDescriptor]:
    """The fields of a message the walks visit: not deprecated (a file may not set one), not maps (not edited)."""
    return [
        field
        for field in descriptor.fields
        if not field.GetOptions().deprecated and not _is_map(field)
    ]


# ---- Conditions ----


def _condition_field(root: Descriptor, field_path: str) -> FieldDescriptor:
    """The singular scalar field of `root` that a condition names; SchemaError when there is none."""
    descriptor: Descriptor | None = root
    field: FieldDescriptor | None = None
    for name in field_path.split("."):
        field = None if descriptor is None else descriptor.fields_by_name.get(name)
        if field is None or field.is_repeated:
            raise SchemaError(
                f"a condition names {field_path}, which is not a singular field of {root.full_name}"
            )
        descriptor = field.message_type
    if field is None or field.message_type is not None:
        raise SchemaError(
            f"a condition names {field_path}, which is a block of {root.full_name}, not a value"
        )
    return field


def condition_value(message: Any, field_path: str) -> str:
    """The value of the field `field_path` of the file `message`, as a condition compares it (Condition.values).

    Args:
      message: The file's message.
      field_path: A singular scalar field, by its dotted path.

    Returns:
      A string as it is, an enum value by name, a bool as true or false, a number as repr() writes it.

    Raises:
      SchemaError: `field_path` is not a singular scalar of the message.
    """
    field = _condition_field(message.DESCRIPTOR, field_path)
    current = message
    for name in field_path.split(".")[:-1]:
        current = getattr(current, name)
    return _text(field, getattr(current, field.name))


def _text(field: FieldDescriptor, value: Any) -> str:
    if field.type == FieldDescriptor.TYPE_ENUM:
        enum_value = field.enum_type.values_by_number.get(value)
        return str(value) if enum_value is None else str(enum_value.name)
    if field.type == FieldDescriptor.TYPE_BOOL:
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _condition_holds(message: Any, condition: Condition) -> bool:
    field = _condition_field(message.DESCRIPTOR, condition.field_path)
    current = condition_value(message, condition.field_path)
    if field.type not in _FLOATING_TYPES + _INTEGER_TYPES:
        return current in condition.values
    for text in condition.values:
        try:
            if textproto_document.parse_number(text) == float(current):
                return True
        except ValueError:
            continue
    return False


def unmet_conditions(tuning: Tuning, message: Any) -> list[Condition]:
    """The conditions of `tuning` that the file `message` does not meet; SchemaError for one naming no field."""
    return [
        condition
        for condition in tuning.active_when
        if not _condition_holds(message, condition)
    ]


def is_active(tuning: Tuning, message: Any) -> bool:
    """Whether a field with `tuning` applies to the file `message`: every one of its conditions holds."""
    return not unmet_conditions(tuning, message)


# ---- The schema ----


def schema_fields(descriptor: Descriptor) -> list[SchemaField]:
    """Every scalar field reachable from the file message `descriptor`, with its tuning there, in schema order.

    A message reachable from itself is walked once on each path. Every condition is checked against the file message.

    Args:
      descriptor: The file message's descriptor.

    Returns:
      The fields.

    Raises:
      SchemaError: A condition names no singular scalar of the file message.
    """
    found: list[SchemaField] = []
    _walk_schema(descriptor, descriptor, "", Tuning(), (), (("", descriptor),), found)
    return found


# The messages a condition may name a field of: (the path of each from the message walked, its descriptor), the
# message walked first. A message inside an element of a repeated field is none, as a path from the top cannot reach
# its fields without a selector.
Scopes: TypeAlias = tuple[tuple[str, Descriptor], ...]


def _resolved(parent: Tuning, tuning: Tuning, scopes: Scopes) -> Tuning:
    """`tuning` with the conditions its field declares itself made paths from the message walked.

    Args:
      parent: The tuning of the block that holds the field.
      tuning: What inherit() made of the field's own options below `parent`.
      scopes: The messages above the field that a condition may name a field of.

    Returns:
      `tuning`, its own conditions resolved (_resolved_condition()); the inherited ones are resolved already.

    Raises:
      SchemaError: A condition names no singular scalar of any message of `scopes`.
    """
    own = tuning.active_when[len(parent.active_when) :]
    if not own:
        return tuning
    return dataclasses.replace(
        tuning,
        active_when=parent.active_when
        + tuple(_resolved_condition(condition, scopes) for condition in own),
    )


def _resolved_condition(condition: Condition, scopes: Scopes) -> Condition:
    """`condition` with its path from the nearest message of `scopes` that has it to the top (the module docstring)."""
    for prefix, descriptor in reversed(scopes):
        try:
            _condition_field(descriptor, condition.field_path)
        except SchemaError:
            continue
        path = f"{prefix}.{condition.field_path}" if prefix else condition.field_path
        return Condition(path, condition.values)
    # The error of the message walked, which names the condition.
    _condition_field(scopes[0][1], condition.field_path)
    raise SchemaError(
        f"a condition names {condition.field_path}, which nothing above it has"
    )


def _inner_scopes(scopes: Scopes, path: str, descriptor: Descriptor) -> Scopes:
    """`scopes` and the singular block at `path` (`descriptor`), unless the block is inside an element of a repeated field."""
    if "[" in path:
        return scopes
    return (*scopes, (path, descriptor))


def _walk_schema(
    root: Descriptor,
    descriptor: Descriptor,
    prefix: str,
    tuning: Tuning,
    stack: tuple[str, ...],
    scopes: Scopes,
    found: list[SchemaField],
) -> None:
    """Appends to `found` the scalar fields below `descriptor`.

    Args:
      root: The file message, which every condition is checked against.
      descriptor: The message walked.
      prefix: Its path from the file message.
      tuning: The tuning in effect for it.
      stack: The messages above it, which are not walked again below it.
      scopes: The messages a condition of a field below may name a field of.
      found: Where the fields go.
    """
    for field in _fields(descriptor):
        field_tuning = _resolved(tuning, inherit(tuning, field), scopes)
        for condition in field_tuning.active_when:
            _condition_field(root, condition.field_path)
        path = f"{prefix}.{field.name}" if prefix else field.name
        if field.message_type is None:
            found.append(
                SchemaField(path, field, kind_of(field, field_tuning), field_tuning)
            )
            continue
        if field.message_type.full_name in stack:
            continue
        if field.is_repeated:
            path += EVERY_ELEMENT
        _walk_schema(
            root,
            field.message_type,
            path,
            field_tuning,
            (*stack, descriptor.full_name),
            _inner_scopes(scopes, path, field.message_type),
            found,
        )


# ---- The parameters of a file ----


@dataclasses.dataclass(frozen=True)
class _Place:
    """Where the walk is: the path so far, the tuning in effect, and what labels and groups what is below."""

    segments: tuple[textproto_document.PathSegment, ...]
    tuning: Tuning
    block: str
    tab: str
    order: int
    label: tuple[str, ...]  # the label's parts so far
    row: str | None
    present: bool  # whether the file has the message the walk is in
    value_field: (
        str | None
    )  # the one value field of the keyed element the walk is in, labeled by the key alone
    scopes: Scopes  # the messages a condition below may name a field of


def parameters(
    message: Any,
    document: textproto_document.TextprotoDocument | None = None,
    registries: Mapping[str, Sequence[str]] | None = None,
) -> list[ParameterSpec]:
    """Every parameter of a file (the module docstring), in schema order, elements in file order.

    Args:
      message: The file, parsed (nproto_textproto).
      document: Its text, for the lines of the values and the presence of the fields of proto3 messages; None:
        without them.
      registries: The names of each registry, for the choices of registry strings; None: unknown.

    Returns:
      The parameters.

    Raises:
      SchemaError: A condition names no singular scalar of the file message.
    """
    tab, order = group(message.DESCRIPTOR)
    walker = _Walker(message, document, registries or {})
    walker.message(
        message,
        _Place(
            (),
            Tuning(),
            "",
            tab,
            order,
            (),
            None,
            True,
            None,
            (("", message.DESCRIPTOR),),
        ),
    )
    return walker.specs


class _Walker:
    """Collects the ParameterSpec of every field below a message of a file."""

    def __init__(
        self,
        root: Any,
        document: textproto_document.TextprotoDocument | None,
        registries: Mapping[str, Sequence[str]],
    ) -> None:
        self._root = root
        self._document = document
        self._registries = registries
        self.specs: list[ParameterSpec] = []

    def message(self, message: Any, place: _Place) -> None:
        """Walks the fields of `message`, which is at `place`."""
        for field in _fields(message.DESCRIPTOR):
            tuning = _resolved(place.tuning, inherit(place.tuning, field), place.scopes)
            for condition in tuning.active_when:
                _condition_field(self._root.DESCRIPTOR, condition.field_path)
            if field.message_type is None:
                self._scalar(message, field, dataclasses.replace(place, tuning=tuning))
            elif field.is_repeated:
                self._elements(
                    message, field, dataclasses.replace(place, tuning=tuning)
                )
            else:
                self._block(message, field, dataclasses.replace(place, tuning=tuning))

    def _block(self, message: Any, field: FieldDescriptor, place: _Place) -> None:
        present = place.present and message.HasField(field.name)
        if not present and is_optional_message(field):
            return
        segment = textproto_document.PathSegment(field.name)
        child = _descend(place, field, segment, str(field.name))
        scopes = _inner_scopes(
            place.scopes,
            textproto_document.format_path(child.segments),
            field.message_type,
        )
        self.message(
            getattr(message, field.name),
            dataclasses.replace(child, present=present, scopes=scopes),
        )

    def _elements(self, message: Any, field: FieldDescriptor, place: _Place) -> None:
        """Walks every element of the repeated message field `field`, by key where its elements have one."""
        container = getattr(message, field.name)
        key_field = _key_field(field.message_type, container)
        value_field = _value_field(field.message_type, key_field)
        for index, element in enumerate(container):
            if key_field is not None:
                key = str(getattr(element, key_field.name))
                segment = textproto_document.PathSegment(
                    field.name, key=(key_field.name, key)
                )
                row = key
            else:
                segment = textproto_document.PathSegment(field.name, index=index)
                row = f"{field.name}[{index}]"
            name = row if not place.segments else f"{field.name}[{row}]"
            child = _descend(place, field, segment, name)
            self.message(
                element,
                dataclasses.replace(
                    child, row=row, present=True, value_field=value_field
                ),
            )

    def _scalar(self, message: Any, field: FieldDescriptor, place: _Place) -> None:
        """Adds the spec of a scalar field: one per element of a repeated one, one for a name list."""
        kind = kind_of(field, place.tuning)
        segment = textproto_document.PathSegment(field.name)
        if field.is_repeated and kind != Kind.NAME_LIST:
            for index, value in enumerate(getattr(message, field.name)):
                element = textproto_document.PathSegment(field.name, index=index)
                self._add(
                    field,
                    kind,
                    place,
                    element,
                    f"{field.name}[{index}]",
                    _python_value(field, value),
                    ValueSource.FILE,
                )
            return
        # An element's one value is labeled by the element's key alone: "joint_positions[l_leg_aky]".
        label = None if field.name == place.value_field else str(field.name)
        if field.is_repeated:
            values = tuple(str(value) for value in getattr(message, field.name))
            source = ValueSource.FILE if values else ValueSource.DEFAULT
            self._add(field, kind, place, segment, label, values, source)
            return
        value, source = self._value(message, field, place, segment)
        self._add(field, kind, place, segment, label, value, source)

    def _value(
        self,
        message: Any,
        field: FieldDescriptor,
        place: _Place,
        segment: textproto_document.PathSegment,
    ) -> tuple[Any, ValueSource]:
        """The value of a singular scalar and where it comes from."""
        value = _python_value(field, getattr(message, field.name))
        if field.has_presence:
            if place.present and message.HasField(field.name):
                return value, ValueSource.FILE
            if field.has_default_value:
                return value, ValueSource.DEFAULT
            return None, ValueSource.UNSET
        # Implicit presence (proto3): the file sets the field when its text has it.
        if place.present and self._document is not None:
            path = textproto_document.format_path((*place.segments, segment))
            if self._document.find(path) is not None:
                return value, ValueSource.FILE
            return value, ValueSource.DEFAULT
        return value, (
            ValueSource.FILE
            if place.present and value != field.default_value
            else ValueSource.DEFAULT
        )

    def _add(
        self,
        field: FieldDescriptor,
        kind: Kind,
        place: _Place,
        segment: textproto_document.PathSegment,
        label: str | None,
        value: Any,
        source: ValueSource,
    ) -> None:
        """Adds the spec of the value `segment` names below `place`; `label` is its part of the label (None: no part)."""
        path = textproto_document.format_path((*place.segments, segment))
        line = self._line(path, kind, source)
        parts = place.label if label is None else (*place.label, label)
        self.specs.append(
            ParameterSpec(
                path=path,
                field=field,
                kind=kind,
                tuning=place.tuning,
                value=value,
                source=source,
                block=place.block,
                tab=place.tab,
                order=place.order,
                label=".".join(parts),
                row=place.row,
                choices=self._choices(field, kind, place.tuning),
                line=line,
            )
        )

    def _line(self, path: str, kind: Kind, source: ValueSource) -> int | None:
        """The line of a value the file sets (of a name list's first name); None without the document."""
        if self._document is None or source != ValueSource.FILE:
            return None
        if kind == Kind.NAME_LIST:
            path = f"{path}[0]"
        try:
            if not self._document.has(path):
                return None
            return self._document.position(path)[0]
        except textproto_document.DocumentError:
            return None

    def _choices(
        self, field: FieldDescriptor, kind: Kind, tuning: Tuning
    ) -> tuple[str, ...] | None:
        if field.type == FieldDescriptor.TYPE_ENUM:
            return tuple(str(value.name) for value in field.enum_type.values)
        if (
            kind in (Kind.CHOICE, Kind.NAME_LIST)
            and tuning.registry in self._registries
        ):
            return tuple(self._registries[tuning.registry])
        return None


def _descend(
    place: _Place,
    field: FieldDescriptor,
    segment: textproto_document.PathSegment,
    name: str,
) -> _Place:
    """The place of the block `field` (an element of it: `segment` selects it) below `place`, labeled `name`."""
    top = not place.segments
    tab, order = group(field.message_type) if top else (place.tab, place.order)
    return dataclasses.replace(
        place,
        segments=(*place.segments, segment),
        block=field.name if top else place.block,
        tab=tab,
        order=order,
        label=() if top and not field.is_repeated else (*place.label, name),
        value_field=None,
    )


def _key_field(
    descriptor: Descriptor, elements: Sequence[Any]
) -> FieldDescriptor | None:
    """The key of a repeated message's elements: its first string field whose values are set and unique; else None."""
    if not elements:
        return None
    for field in descriptor.fields:
        if field.type != FieldDescriptor.TYPE_STRING or field.is_repeated:
            continue
        if field.has_presence and not all(
            element.HasField(field.name) for element in elements
        ):
            continue
        values = [getattr(element, field.name) for element in elements]
        if all(values) and len(set(values)) == len(values):
            return field
    return None


def _value_field(
    descriptor: Descriptor, key_field: FieldDescriptor | None
) -> str | None:
    """The one singular scalar of a keyed element besides its key (JointValue's value), which the key labels alone."""
    if key_field is None:
        return None
    others = [
        field.name
        for field in _fields(descriptor)
        if field.name != key_field.name
        and field.message_type is None
        and not field.is_repeated
    ]
    return others[0] if len(others) == 1 else None


def _python_value(field: FieldDescriptor, value: Any) -> Any:
    """A field's value as a parameter holds it: an enum value by name, the rest as reflection gives it."""
    if field.type == FieldDescriptor.TYPE_ENUM:
        return _text(field, value)
    return value


# ---- Labels ----


# The MPC formulations TuningOptions.formulations names, and how a label names each one's MPC.
# LINT.IfChange(formulation_names)
CENTROIDAL = "centroidal"
WHOLE_BODY = "whole_body"
_FORMULATION_MPCS = {
    CENTROIDAL: "the centroidal MPC",
    WHOLE_BODY: "the whole-body MPC",
}
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/config/ConfigReload.h:formulation_names, //humanoid_nmpc/humanoid_mpc_config/tuning_options.proto:formulations)
# Every formulation name, in the order of the schema's documentation.
FORMULATIONS: tuple[str, ...] = tuple(_FORMULATION_MPCS)


def formulation_of(task: Any) -> str:
    """The MPC formulation a task file configures: CENTROIDAL when it names a centroidal_model, WHOLE_BODY otherwise.

    The centroidal MPC requires a centroidal_model and the whole-body MPC reads none, so the field tells the two
    formulations' task files apart.

    Args:
      task: The task file, a humanoid_mpc_config.TaskFile.

    Returns:
      A name of FORMULATIONS.
    """
    # LINT.IfChange(formulation_marker)
    return CENTROIDAL if task.HasField("centroidal_model") else WHOLE_BODY
    # LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:centroidal_model)


def _formulation_note(spec: ParameterSpec, message: Any) -> str | None:
    """The "not applicable" note of a task file's field that the file's formulation does not read; None otherwise."""
    if (
        not spec.tuning.formulations
        or message.DESCRIPTOR.full_name != task_file_pb2.TaskFile.DESCRIPTOR.full_name
    ):
        return None
    formulation = formulation_of(message)
    if formulation in spec.tuning.formulations:
        return None
    return f"not applicable: {_FORMULATION_MPCS[formulation]} does not read it"


def annotations(spec: ParameterSpec, message: Any) -> list[str]:
    """What a parameter's label says besides its name: "restart", "not applicable: ...", "default", "unset".

    Every formulation's running MPC applies the RELOAD_HOT fields it reads, so only a RELOAD_START_UP field says
    "restart". A field of a task file that its formulation does not read (TuningOptions.formulations), and a field
    whose active_when conditions the file does not meet, say "not applicable" and why.

    Args:
      spec: The parameter.
      message: The file it is of, for its conditions and, for a task file, its formulation (formulation_of()).

    Returns:
      The annotations, in that order; empty for a value the file sets that the running stack applies.
    """
    notes: list[str] = []
    if spec.tuning.reload == RELOAD_START_UP:
        notes.append("restart")
    not_read = _formulation_note(spec, message)
    if not_read is not None:
        notes.append(not_read)
    unmet = unmet_conditions(spec.tuning, message)
    if unmet:
        shown = ", ".join(
            f"{c.field_path} is {condition_value(message, c.field_path)}" for c in unmet
        )
        notes.append(f"not applicable: {shown}")
    if spec.source == ValueSource.DEFAULT:
        notes.append("default")
    elif spec.source == ValueSource.UNSET:
        notes.append("unset")
    return notes


def display_label(spec: ParameterSpec, message: Any) -> str:
    """The label a widget shows: `label (unit) (annotations)`, each part only where it applies.

    The label is the parameter's field names (ParameterSpec.label); a comment of the file is never part of it.

    Args:
      spec: The parameter.
      message: The file it is of.

    Returns:
      e.g. "normalized_linear_momentum.x", "terrain_height (m) (default)".
    """
    parts = [spec.label]
    if spec.tuning.unit:
        parts.append(f"({spec.tuning.unit})")
    notes = annotations(spec, message)
    if notes:
        parts.append(f"({'; '.join(notes)})")
    return " ".join(parts)
