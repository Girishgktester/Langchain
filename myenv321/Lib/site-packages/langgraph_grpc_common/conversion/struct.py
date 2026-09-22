import re
from collections.abc import Mapping
from typing import Any

import orjson
from google.protobuf import struct_pb2


def struct_from_dict(d: Mapping[str, Any]) -> struct_pb2.Struct:
    s = struct_pb2.Struct()
    s.update(d)
    return s


def _default_serializer(obj: Any) -> Any:
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Type is not JSON serializable: {type(obj).__name__}")


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _replace_surrogates(o: Any) -> Any:
    """Recursively replace lone-surrogate codepoints (orjson rejects them).

    Mirrors ``_sanitise`` in ``langgraph_grpc_common.serde`` so metadata
    encode falls back gracefully instead of raising. Without this,
    fixtures that intentionally embed surrogates (e.g.
    ``test_thread_copy``'s ``"surrogate"`` field on
    ``langgraph==0.4.10`` which mirrors input into metadata) crash the
    whole put.
    """
    if isinstance(o, str):
        return _SURROGATE_RE.sub("?", o) if _SURROGATE_RE.search(o) else o
    if isinstance(o, Mapping):
        return {_replace_surrogates(k): _replace_surrogates(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return type(o)(_replace_surrogates(x) for x in o)
    return o


def raw_map_from_dict(d: Mapping[str, Any]) -> Mapping[str, bytes]:
    out: dict[str, bytes] = {}
    for k, v in d.items():
        try:
            out[k] = orjson.dumps(v, default=_default_serializer)
        except (TypeError, ValueError):
            # orjson rejects e.g. lone surrogates and a few other oddities
            # that the in-process Python ``Checkpointer.aput`` happily
            # stores via its own serde. Sanitize and retry rather than
            # failing the entire put.
            out[k] = orjson.dumps(_replace_surrogates(v), default=_default_serializer)
    return out


def dict_from_raw_map(m: Mapping[str, bytes]) -> dict[str, Any]:
    return {k: orjson.loads(v) for k, v in m.items()}
