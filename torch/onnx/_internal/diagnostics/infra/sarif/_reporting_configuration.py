# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Optional

from typing_extensions import Literal

from torch.onnx._internal.diagnostics.infra.sarif import _property_bag


@dataclasses.dataclass
class ReportingConfiguration(object):
    """Information about a rule or notification that can be configured at runtime."""

    enabled: bool = dataclasses.field(
        default=True, metadata={"schema_property_name": "enabled"}
    )
    level: Literal["none", "note", "warning", "error"] = dataclasses.field(
        default="warning", metadata={"schema_property_name": "level"}
    )
    parameters: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "parameters"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    rank: float = dataclasses.field(
        default=-1.0, metadata={"schema_property_name": "rank"}
    )


# flake8: noqa
