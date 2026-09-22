"""Marshal langgraph.store.base GetOp/PutOp/Item/Result to and from store.proto."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import orjson
from google.protobuf import empty_pb2
from langgraph.store.base import (
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

from langgraph_grpc_common.proto import store_pb2

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.embeddings import Embeddings


def _ensure_index_config(
    index_config: IndexConfig,
) -> tuple[Embeddings | None, IndexConfig]:
    # Vendored from langgraph.store.postgres.base so grpc_common_py stays
    # backend-agnostic (no langgraph-checkpoint-postgres / psycopg dependency).
    index_config = index_config.copy()
    tokenized: list[tuple[str, str | list[str]]] = []
    tot = 0
    fields = index_config.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {fields}")
    for p in fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized  # ty: ignore[invalid-key]
    index_config["__estimated_num_vectors"] = tot  # ty: ignore[invalid-key]
    embeddings = ensure_embeddings(index_config.get("embed"))
    return embeddings, index_config


def _ttl_to_proto_minutes(ttl: float | None) -> int:
    if ttl is None:
        return 0
    if ttl != int(ttl):
        raise ValueError(
            f"gRPC store only supports integer-minute TTL; got fractional ttl={ttl}"
        )
    return int(ttl)


# Ordered comparisons compare the *text* form of the field (value->>key); $eq/$ne
# are handled separately as jsonb comparisons.
_FILTER_OP_BY_NAME = {
    "$gt": store_pb2.FILTER_OP_GT,
    "$gte": store_pb2.FILTER_OP_GTE,
    "$lt": store_pb2.FILTER_OP_LT,
    "$lte": store_pb2.FILTER_OP_LTE,
}

# pgvector distance / column types come from the index config as strings; map
# them to the wire enums (see store.proto DistanceType / VectorType).
_DISTANCE_TYPE_BY_NAME = {
    "cosine": store_pb2.DISTANCE_TYPE_COSINE,
    "l2": store_pb2.DISTANCE_TYPE_L2,
    "inner_product": store_pb2.DISTANCE_TYPE_INNER_PRODUCT,
}
_VECTOR_TYPE_BY_NAME = {
    "vector": store_pb2.VECTOR_TYPE_VECTOR,
    "halfvec": store_pb2.VECTOR_TYPE_HALFVEC,
}

# ListNamespaces match type: langgraph uses the strings "prefix"/"suffix".
_MATCH_TYPE_BY_NAME = {
    "prefix": store_pb2.MATCH_TYPE_PREFIX,
    "suffix": store_pb2.MATCH_TYPE_SUFFIX,
}
_MATCH_TYPE_TO_NAME = {op: name for name, op in _MATCH_TYPE_BY_NAME.items()}


def _match_type_to_proto(match_type: str) -> int:
    proto = _MATCH_TYPE_BY_NAME.get(match_type)
    if proto is None:
        raise ValueError(f"Unsupported list_namespaces match_type: {match_type!r}")
    return proto


def _filter_to_conditions(
    filter_dict: dict[str, Any],
) -> list[store_pb2.FilterCondition]:
    """Flatten a langgraph filter dict into typed FilterConditions.

    A plain value is an equality match; a dict is a set of operator comparisons.
    $eq/$ne compare as jsonb; $gt/$gte/$lt/$lte compare on the text form of the
    value (str(value), produced here so the store never re-derives it). Mirrors
    OSS PostgresStore._get_filter_condition, done in Python where the DSL is
    native so the store only maps op -> SQL.
    """
    conditions: list[store_pb2.FilterCondition] = []
    for key, value in filter_dict.items():
        if isinstance(value, dict):
            for op_name, op_val in value.items():
                if op_name in ("$eq", "$ne"):
                    op = (
                        store_pb2.FILTER_OP_EQ
                        if op_name == "$eq"
                        else store_pb2.FILTER_OP_NE
                    )
                    conditions.append(
                        store_pb2.FilterCondition(
                            key=key, op=op, operand=orjson.dumps(op_val).decode()
                        )
                    )
                elif op_name in _FILTER_OP_BY_NAME:
                    conditions.append(
                        store_pb2.FilterCondition(
                            key=key, op=_FILTER_OP_BY_NAME[op_name], operand=str(op_val)
                        )
                    )
                else:
                    raise ValueError(f"Unsupported filter operator: {op_name}")
        else:
            conditions.append(
                store_pb2.FilterCondition(
                    key=key,
                    op=store_pb2.FILTER_OP_EQ,
                    operand=orjson.dumps(value).decode(),
                )
            )
    return conditions


def op_to_proto(
    op: Op,
    *,
    vectors: Sequence[store_pb2.VectorValue] | None = None,
    search_vector: store_pb2.VectorSearch | None = None,
) -> store_pb2.Op:
    if isinstance(op, GetOp):
        return store_pb2.Op(
            get=store_pb2.GetOp(
                namespace=list(op.namespace),
                key=str(op.key),
                refreshTTL=op.refresh_ttl,
            )
        )
    if isinstance(op, PutOp):
        # A PutOp with value=None is a delete in OSS; on the gRPC path it maps to
        # an explicit DeleteOp so an empty value can't be mistaken for a delete.
        if op.value is None:
            return store_pb2.Op(
                delete=store_pb2.DeleteOp(
                    namespace=list(op.namespace),
                    key=str(op.key),
                )
            )
        put_kwargs: dict[str, Any] = {
            "namespace": list(op.namespace),
            "key": str(op.key),
            "value_json": orjson.dumps(op.value),
            "ttlMinutes": _ttl_to_proto_minutes(op.ttl),
        }
        if vectors is not None:
            put_kwargs["vectors"] = list(vectors)
        return store_pb2.Op(put=store_pb2.PutOp(**put_kwargs))
    if isinstance(op, SearchOp):
        # The natural-language query itself is not sent: Python embeds it into
        # `search_vector`, which is the only semantic-search signal the store needs.
        search_kwargs: dict[str, Any] = {
            "namespace_prefix": list(op.namespace_prefix),
            "limit": op.limit,
            "offset": op.offset,
            "refreshTTL": op.refresh_ttl,
        }
        if op.filter:
            search_kwargs["filter"] = _filter_to_conditions(op.filter)
        if search_vector is not None:
            search_kwargs["vector"] = search_vector
        return store_pb2.Op(search=store_pb2.SearchOp(**search_kwargs))
    if isinstance(op, ListNamespacesOp):
        list_kwargs: dict[str, Any] = {
            "limit": op.limit,
            "offset": op.offset,
        }
        if op.match_conditions:
            list_kwargs["match_conditions"] = [
                store_pb2.MatchCondition(
                    match_type=cast("Any", _match_type_to_proto(mc.match_type)),
                    path=list(mc.path),
                )
                for mc in op.match_conditions
            ]
        if op.max_depth is not None:
            list_kwargs["max_depth"] = op.max_depth
        return store_pb2.Op(list_namespaces=store_pb2.ListNamespacesOp(**list_kwargs))
    raise TypeError(f"unsupported op: {type(op).__name__}")


def op_from_proto(msg: store_pb2.Op) -> Op:
    kind = msg.WhichOneof("op")
    if kind == "get":
        g = msg.get
        return GetOp(
            namespace=tuple(g.namespace),
            key=g.key,
            refresh_ttl=g.refreshTTL,
        )
    if kind == "put":
        p = msg.put
        ttl: float | None = None
        if p.ttlMinutes:
            ttl = float(p.ttlMinutes)
        return PutOp(
            namespace=tuple(p.namespace),
            key=p.key,
            value=orjson.loads(p.value_json),
            ttl=ttl,
        )
    if kind == "delete":
        d = msg.delete
        return PutOp(namespace=tuple(d.namespace), key=d.key, value=None)
    if kind == "search":
        s = msg.search
        # `query` (embedded into `vector`) and `filter` (flattened to typed
        # conditions) are one-directional encodings, not reconstructed here; the
        # store consumes them and Python never decodes a SearchOp in production.
        return SearchOp(
            namespace_prefix=tuple(s.namespace_prefix),
            limit=s.limit,
            offset=s.offset,
            refresh_ttl=s.refreshTTL,
        )
    if kind == "list_namespaces":
        ln = msg.list_namespaces
        return ListNamespacesOp(
            match_conditions=tuple(
                MatchCondition(
                    match_type=cast("Any", _MATCH_TYPE_TO_NAME[mc.match_type]),
                    path=tuple(mc.path),
                )
                for mc in ln.match_conditions
            )
            or None,
            max_depth=ln.max_depth if ln.HasField("max_depth") else None,
            limit=ln.limit,
            offset=ln.offset,
        )
    raise ValueError("empty Op message (no oneof set)")


def result_to_proto(result: Result, op: Op) -> store_pb2.OpResult:
    if isinstance(op, GetOp):
        if result is None:
            return store_pb2.OpResult(none=empty_pb2.Empty())
        if not isinstance(result, Item):
            raise TypeError(
                f"expected Item for GetOp result, got {type(result).__name__}"
            )
        return store_pb2.OpResult(item=_item_to_proto(result))
    if isinstance(op, PutOp):
        return store_pb2.OpResult(none=empty_pb2.Empty())
    if isinstance(op, SearchOp):
        items = cast("list[SearchItem]", result or [])
        return store_pb2.OpResult(
            search=store_pb2.SearchResult(
                items=[_search_item_to_proto(item) for item in items]
            )
        )
    if isinstance(op, ListNamespacesOp):
        namespaces = cast("list[tuple[str, ...]]", result or [])
        return store_pb2.OpResult(
            list_namespaces=store_pb2.ListNamespacesResult(
                namespaces=[store_pb2.Namespace(parts=list(ns)) for ns in namespaces]
            )
        )
    raise TypeError(f"unsupported op for result encoding: {type(op).__name__}")


def result_from_proto(msg: store_pb2.OpResult) -> Result:
    kind = msg.WhichOneof("result")
    if kind == "none":
        return None
    if kind == "item":
        return _item_from_proto(msg.item)
    if kind == "search":
        return [_search_item_from_proto(item) for item in msg.search.items]
    if kind == "list_namespaces":
        return [tuple(ns.parts) for ns in msg.list_namespaces.namespaces]
    raise ValueError("empty OpResult message (no oneof set)")


def _item_to_proto(item: Item) -> store_pb2.Item:
    return store_pb2.Item(
        namespace=list(item.namespace),
        key=item.key,
        value_json=orjson.dumps(item.value),
        created_at=_format_dt(item.created_at),
        updated_at=_format_dt(item.updated_at),
    )


def _item_from_proto(msg: store_pb2.Item) -> Item:
    return Item(
        namespace=tuple(msg.namespace),
        key=msg.key,
        value=orjson.loads(msg.value_json),
        created_at=datetime.fromisoformat(msg.created_at),
        updated_at=datetime.fromisoformat(msg.updated_at),
    )


def _search_item_to_proto(item: SearchItem) -> store_pb2.SearchItem:
    kwargs: dict[str, Any] = {
        "namespace": list(item.namespace),
        "key": item.key,
        "value_json": orjson.dumps(item.value),
        "created_at": _format_dt(item.created_at),
        "updated_at": _format_dt(item.updated_at),
    }
    if item.score is not None:
        kwargs["score"] = item.score
    return store_pb2.SearchItem(**kwargs)


def _search_item_from_proto(msg: store_pb2.SearchItem) -> SearchItem:
    return SearchItem(
        namespace=tuple(msg.namespace),
        key=msg.key,
        value=orjson.loads(msg.value_json),
        created_at=datetime.fromisoformat(msg.created_at),
        updated_at=datetime.fromisoformat(msg.updated_at),
        score=msg.score if msg.HasField("score") else None,
    )


def _format_dt(dt: datetime) -> str:
    return dt.isoformat()


async def vectors_for_put_op(
    op: PutOp,
    index_config: IndexConfig,
    embeddings: Embeddings,
) -> list[store_pb2.VectorValue]:
    if op.index is False or op.value is None:
        return []

    value = op.value
    if op.index is None:
        paths = cast(
            "list[tuple[str, str | list[str]]]",
            index_config["__tokenized_fields"],  # ty: ignore[invalid-key]
        )
    else:
        paths = [(ix, tokenize_path(ix)) for ix in op.index]

    texts: list[str] = []
    pathnames: list[str] = []
    for path, tokenized_path in paths:
        extracted = get_text_at_path(value, tokenized_path)
        for i, text in enumerate(extracted):
            pathname = f"{path}.{i}" if len(extracted) > 1 else path
            pathnames.append(pathname)
            texts.append(text)

    if not texts:
        return []

    vectors = await embeddings.aembed_documents(texts)
    return [
        store_pb2.VectorValue(valuePath=pathname, embedding=vector)
        for pathname, vector in zip(pathnames, vectors, strict=False)
    ]


async def query_embedding_for_search_op(
    op: SearchOp,
    index_config: IndexConfig,
    embeddings: Embeddings,
) -> store_pb2.VectorSearch | None:
    """Embed a SearchOp's natural-language query for semantic search.

    Returns None when the op has no query (a plain filter/pagination search).
    Distance params are read from the index config and sent alongside the
    embedding so the Go store stays stateless.
    """
    if not op.query:
        return None

    vectors = await embeddings.aembed_documents([op.query])
    if not vectors:
        return None

    cfg = cast("dict[str, Any]", index_config)
    distance_type = cfg.get("distance_type", "cosine")
    vectors_per_doc = cfg.get("__estimated_num_vectors", 1)
    vector_type = cfg.get("ann_index_config", {}).get("vector_type", "vector")

    if distance_type not in _DISTANCE_TYPE_BY_NAME:
        raise ValueError(
            f"gRPC store semantic search does not support distance_type={distance_type!r} "
            "(only 'cosine', 'l2', 'inner_product')"
        )
    if vector_type not in _VECTOR_TYPE_BY_NAME:
        raise NotImplementedError(
            f"gRPC store semantic search does not support vector_type={vector_type!r} "
            "(only 'vector' and 'halfvec')"
        )

    return store_pb2.VectorSearch(
        query_embedding=vectors[0],
        distance_type=_DISTANCE_TYPE_BY_NAME[distance_type],
        vectors_per_doc=vectors_per_doc,
        vector_type=_VECTOR_TYPE_BY_NAME[vector_type],
    )


def normalize_index_config(
    index: IndexConfig | None,
) -> tuple[Embeddings | None, IndexConfig | None]:
    if not index:
        return None, None
    return _ensure_index_config(index)
