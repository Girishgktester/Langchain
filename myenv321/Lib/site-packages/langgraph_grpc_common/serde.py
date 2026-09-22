import re
import uuid
from base64 import b64encode
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta, timezone
from decimal import Decimal
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)
from pathlib import Path
from re import Pattern
from typing import Any, NamedTuple, cast
from zoneinfo import ZoneInfo

import orjson
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Process-wide default serializer.
SERIALIZER: SerializerProtocol = JsonPlusSerializer()

# Task-local override and falls back to ``SERIALIZER`` when unset.
# (e.g. ``GrpcCheckpointer``) scope a custom serializer to its own RPC
# bodies without leaking into unrelated gRPC ops
_SCOPED_SERIALIZER: ContextVar[SerializerProtocol | None] = ContextVar(
    "_grpc_scoped_serializer", default=None
)


class Fragment(NamedTuple):
    buf: bytes


def fragment_loads(data: bytes) -> Fragment:
    return Fragment(data)


def decimal_encoder(dec_value: Decimal) -> int | float:
    """
    Encodes a Decimal as int of there's no exponent, otherwise float

    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
    where a integer (but not int typed) is used. Encoding this as a float
    results in failed round-tripping between encode and parse.
    Our Id type is a prime example of this.

    >>> decimal_encoder(Decimal("1.0"))
    1.0

    >>> decimal_encoder(Decimal("1"))
    1
    """
    if (
        # maps to float('nan') / float('inf') / float('-inf')
        not dec_value.is_finite()
        # or regular float
        or cast("int", dec_value.as_tuple().exponent) < 0
    ):
        return float(dec_value)
    return int(dec_value)


def default(obj):
    # Only need to handle types that orjson doesn't serialize by default
    # https://github.com/ijl/orjson#serialize
    if isinstance(obj, Fragment):
        return orjson.Fragment(obj.buf)
    if (
        hasattr(obj, "model_dump")
        and callable(obj.model_dump)
        and not isinstance(obj, type)
    ):
        return obj.model_dump()
    elif hasattr(obj, "dict") and callable(obj.dict) and not isinstance(obj, type):
        return obj.dict()
    elif (
        hasattr(obj, "_asdict") and callable(obj._asdict) and not isinstance(obj, type)
    ):
        return obj._asdict()
    elif isinstance(obj, BaseException):
        return {"error": type(obj).__name__, "message": str(obj)}
    elif isinstance(obj, (set, frozenset, deque)):
        return list(obj)
    elif isinstance(obj, (timezone, ZoneInfo)):
        return obj.tzname(None)
    elif isinstance(obj, timedelta):
        return obj.total_seconds()
    elif isinstance(obj, Decimal):
        return decimal_encoder(obj)
    elif isinstance(
        obj,
        (
            uuid.UUID,
            IPv4Address,
            IPv4Interface,
            IPv4Network,
            IPv6Address,
            IPv6Interface,
            IPv6Network,
            Path,
        ),
    ):
        return str(obj)
    elif isinstance(obj, Pattern):
        return obj.pattern
    elif isinstance(obj, bytes | bytearray):
        return b64encode(obj).decode()
    # Fallback for typed objects (e.g. Usage, custom classes) that orjson
    # doesn't natively serialize — use instance dict so they round-trip as dicts.
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return None


_option = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _replace_surr(s: str) -> str:
    return s if _SURROGATE_RE.search(s) is None else _SURROGATE_RE.sub("?", s)


def _sanitise(o: Any) -> Any:
    if isinstance(o, str):
        return _replace_surr(o)
    if isinstance(o, Mapping):
        return {_sanitise(k): _sanitise(v) for k, v in o.items()}
    if isinstance(o, list | tuple | set):
        if (
            isinstance(o, tuple)
            and hasattr(o, "_asdict")
            and callable(o._asdict)
            and hasattr(o, "_fields")
            and isinstance(o._fields, tuple)
        ):  # named tuple
            return {f: _sanitise(ov) for f, ov in zip(o._fields, o, strict=True)}
        ctor = list if isinstance(o, list) else type(o)
        return ctor(_sanitise(x) for x in o)
    return o


def json_dumpb(obj) -> bytes:
    try:
        dumped = orjson.dumps(obj, default=default, option=_option)
    except TypeError as e:
        if "surrogates not allowed" not in str(e):
            raise
        dumped = orjson.dumps(_sanitise(obj), default=default, option=_option)
    if rb"\u0000" not in dumped:
        return dumped  # fast path — no null bytes
    # Replace with U+FFFD (replacement character) instead of stripping,
    # so that "key\x00" becomes "key\uFFFD" rather than "key" — preventing
    # a poisoned key from projecting onto a legitimate one.
    return dumped.replace(rb"\\u0000", rb"\\ufffd").replace(rb"\u0000", rb"\ufffd")


def set_serializer(serializer: SerializerProtocol) -> None:
    """Replace the process-wide default serializer.

    This only affects every consumer of ``langgraph_grpc_common.conversion.*``.
    """
    global SERIALIZER
    SERIALIZER = serializer


def get_serializer() -> SerializerProtocol:
    """Return the serializer to use for gRPC payload (de)serialization.

    Resolution order:

    1. A task-local override bound via :func:`use_serializer`, if any.
    2. The process-wide default set by :func:`set_serializer` (or the
       ``JsonPlusSerializer()`` baseline if never replaced).
    """
    return _SCOPED_SERIALIZER.get() or SERIALIZER


@contextmanager
def use_serializer(serializer: SerializerProtocol) -> Iterator[None]:
    """Bind ``serializer`` for the current async task / sync context."""
    token = _SCOPED_SERIALIZER.set(serializer)
    try:
        yield
    finally:
        _SCOPED_SERIALIZER.reset(token)
