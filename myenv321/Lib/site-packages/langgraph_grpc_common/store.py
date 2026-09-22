"""GrpcStore — BaseStore subclass that proxies to core-server Store."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, TypeVar, cast

import grpc.aio
import orjson
from langgraph.store.base import BaseStore, IndexConfig, Op, PutOp, Result, SearchOp
from langgraph.store.base.batch import AsyncBatchedBaseStore

from langgraph_grpc_common.conversion.store import (
    normalize_index_config,
    op_to_proto,
    query_embedding_for_search_op,
    result_from_proto,
    vectors_for_put_op,
)
from langgraph_grpc_common.proto import store_pb2
from langgraph_grpc_common.proto.store_pb2_grpc import StoreStub

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable, Mapping

T = TypeVar("T")


class GrpcStore(AsyncBatchedBaseStore):
    """BaseStore client that dispatches ``abatch`` over gRPC.

    Supports get/put/delete, search (filter + semantic), list_namespaces, TTL,
    and vector indexing. Embeddings (put vectors and search query embeddings) are
    computed in Python; the Go core-server handles persistence and ranking.
    """

    supports_ttl = True

    # Writes must retain their caller's request-scoped encryption context;
    # AsyncBatchedBaseStore.aput dispatches from a long-lived background task.
    aput = BaseStore.aput

    def __init__(
        self,
        address: str,
        *,
        index: IndexConfig | None = None,
        ttl: dict[str, Any] | None = None,
        encryption_context_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.ttl_config = ttl
        self._embeddings, self._index_config = normalize_index_config(index)
        self._address = address
        self._encryption_context_provider = encryption_context_provider
        self._channel_lock = threading.Lock()
        self._channel: grpc.aio.Channel | None = None
        self._stub: StoreStub | None = None

    def _get_stub(self) -> StoreStub:
        if self._stub is None:
            with self._channel_lock:
                if self._stub is None:
                    # Internal loopback to core-server; matches langgraph_api.grpc.client.
                    self._channel = grpc.aio.insecure_channel(self._address)
                    self._stub = StoreStub(self._channel)
        return self._stub

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        op_list = list(ops)
        if not op_list:
            return []

        proto_ops: list[store_pb2.Op] = []
        index_config = self._index_config
        embeddings = self._embeddings
        has_index = index_config is not None and embeddings is not None
        for op in op_list:
            if (
                has_index
                and isinstance(op, PutOp)
                and op.value is not None
                and op.index is not False
            ):
                vectors = await vectors_for_put_op(
                    op,
                    cast("Any", index_config),
                    cast("Any", embeddings),
                )
                proto_ops.append(op_to_proto(op, vectors=vectors))
            elif has_index and isinstance(op, SearchOp) and op.query:
                search_vector = await query_embedding_for_search_op(
                    op,
                    cast("Any", index_config),
                    cast("Any", embeddings),
                )
                proto_ops.append(op_to_proto(op, search_vector=search_vector))
            else:
                proto_ops.append(op_to_proto(op))

        encryption_context = (
            self._encryption_context_provider()
            if self._encryption_context_provider is not None
            else {}
        )
        request = store_pb2.BatchRequest(
            ops=proto_ops,
            encryption_context_json={
                key: orjson.dumps(value) for key, value in encryption_context.items()
            },
        )
        stub = self._get_stub()
        response = await stub.Batch(request)
        if len(response.results) != len(op_list):
            raise RuntimeError(
                f"store-server returned {len(response.results)} results for "
                f"{len(op_list)} ops"
            )
        return [result_from_proto(r) for r in response.results]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._run_sync(self.abatch(ops))

    def _run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def aclose(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
