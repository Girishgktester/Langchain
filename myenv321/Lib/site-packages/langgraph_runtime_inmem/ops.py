"""Implementation of the LangGraph API using in-memory checkpointer & store."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import typing
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import croniter as croniter_mod
import orjson
import structlog
from langgraph.checkpoint.serde.jsonplus import _msgpack_ext_hook_to_json
from langgraph.pregel.debug import CheckpointPayload
from langgraph.types import Interrupt, StateSnapshot
from langgraph.version import __version__
from langgraph_sdk import Auth
from starlette.exceptions import HTTPException

from langgraph_runtime_inmem.checkpoint import Checkpointer
from langgraph_runtime_inmem.database import InMemConnectionProto, connect
from langgraph_runtime_inmem.inmem_stream import (
    THREADLESS_KEY,
    ContextQueue,
    Message,
    get_stream_manager,
)

if typing.TYPE_CHECKING:
    from langgraph_api.asyncio import ValueEvent
    from langgraph_api.config import ThreadTTLConfig
    from langgraph_api.schema import (
        Assistant,
        AssistantSelectField,
        Checkpoint,
        Config,
        Context,
        Cron,
        CronSelectField,
        DeprecatedInterrupt,
        IfNotExists,
        MetadataInput,
        MetadataValue,
        MultitaskStrategy,
        OnConflictBehavior,
        PoolStats,
        QueueStats,
        Run,
        RunSelectField,
        RunStatus,
        StreamMode,
        Thread,
        ThreadSelectField,
        ThreadStatus,
        ThreadStreamMode,
        ThreadUpdateResponse,
    )
    from langgraph_api.schema import Interrupt as InterruptSchema
    from langgraph_api.utils import AsyncConnectionProto

StreamHandler = ContextQueue

logger = structlog.stdlib.get_logger(__name__)

# Only gate features on the major.minor version; Lets you ignore the rc/alpha/etc. releases anyway
LANGGRAPH_PY_MINOR = tuple(map(int, __version__.split(".")[:2]))
USE_NEW_INTERRUPTS = LANGGRAPH_PY_MINOR >= (0, 6)


async def _decrypt_response(data: dict, model: str, fields: list[str]) -> dict:
    from langgraph_api.encryption.middleware import decrypt_response  # noqa: PLC0415

    return await decrypt_response(data, model, fields)  # type: ignore[arg-type]


async def _encrypt_request(data: dict, model: str, fields: list[str]) -> dict:
    from langgraph_api.encryption.middleware import encrypt_request  # noqa: PLC0415

    return await encrypt_request(data, model, fields)  # type: ignore[arg-type]


def _using_custom_encryption() -> bool:
    from langgraph_api.encryption.shared import (  # noqa: PLC0415
        using_custom_encryption,
    )

    return using_custom_encryption()


def _run_stream_mode_matches(event_mode: str, stream_mode: list[str] | None) -> bool:
    """Return True if a published run-stream event matches the join filter."""
    if not stream_mode:
        return True
    if event_mode in stream_mode:
        return True
    if (
        "messages" in stream_mode or "messages-tuple" in stream_mode
    ) and event_mode.startswith("messages"):
        return True
    if "|" in event_mode:
        base_mode, _, _ = event_mode.partition("|")
        if base_mode in stream_mode:
            return True
    return False


def _ensure_uuid(id_: str | uuid.UUID | None) -> uuid.UUID:
    if isinstance(id_, str):
        return uuid.UUID(id_)
    if id_ is None:
        return uuid4()
    return id_


def _snapshot_defaults():
    # Support older versions of langgraph
    if not hasattr(StateSnapshot, "interrupts"):
        return {}
    return {"interrupts": tuple()}


class WrappedHTTPException(Exception):
    def __init__(self, http_exception: HTTPException):
        self.http_exception = http_exception


# Right now the whole API types as UUID but frequently passes a str
# We ensure UUIDs for eveerything EXCEPT the checkpoint storage/writes,
# which we leave as strings. This is because I'm too lazy to subclass fully
# and we use non-UUID examples in the OSS version


class Authenticated:
    resource: Literal["threads", "crons", "assistants"]

    @classmethod
    def _context(
        cls,
        ctx: Auth.types.BaseAuthContext | None,
        action: Literal["create", "read", "update", "delete", "create_run"],
    ) -> Auth.types.AuthContext | None:
        if not ctx:
            return
        return Auth.types.AuthContext(
            user=ctx.user,
            permissions=ctx.permissions,
            resource=cls.resource,
            action=action,
        )

    @classmethod
    async def handle_event(
        cls,
        ctx: Auth.types.BaseAuthContext | None,
        action: Literal["create", "read", "update", "delete", "search", "create_run"],
        value: Any,
    ) -> Auth.types.FilterType | None:
        from langgraph_api.auth.custom import handle_event  # noqa: PLC0415
        from langgraph_api.utils import get_auth_ctx  # noqa: PLC0415

        ctx = ctx or get_auth_ctx()
        if not ctx:
            return
        return await handle_event(cls._context(ctx, action), value)


class Assistants(Authenticated):
    resource = "assistants"

    @staticmethod
    async def search(
        conn: InMemConnectionProto,
        *,
        graph_id: str | None,
        name: str | None,
        metadata: MetadataInput,
        limit: int,
        offset: int,
        sort_by: str | None = None,
        sort_order: str | None = None,
        select: list[AssistantSelectField] | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> tuple[AsyncIterator[Assistant], int]:
        from langgraph_api.graph import assert_graph_exists  # noqa: PLC0415

        metadata = metadata if metadata is not None else {}
        filters = await Assistants.handle_event(
            ctx,
            "search",
            Auth.types.AssistantsSearch(
                graph_id=graph_id, metadata=metadata, limit=limit, offset=offset
            ),
        )

        if graph_id is not None:
            assert_graph_exists(graph_id)

        # Get all assistants and filter them
        assistants = conn.store["assistants"]
        filtered_assistants = [
            assistant
            for assistant in assistants
            if (not graph_id or assistant["graph_id"] == graph_id)
            and (not name or name.lower() in assistant["name"].lower())
            and (not metadata or is_jsonb_contained(assistant["metadata"], metadata))
            and (not filters or _check_filter_match(assistant["metadata"], filters))
        ]

        # Sort based on sort_by and sort_order
        sort_by = sort_by.lower() if sort_by else None
        if sort_by and sort_by in (
            "assistant_id",
            "graph_id",
            "name",
            "created_at",
            "updated_at",
        ):
            reverse = False if sort_order and sort_order.upper() == "ASC" else True
            # Use case-insensitive sorting for string fields
            if sort_by in ["name", "graph_id"]:
                filtered_assistants.sort(
                    key=lambda x: (
                        str(x.get(sort_by, "")).lower() if x.get(sort_by) else ""
                    ),
                    reverse=reverse,
                )
            else:
                filtered_assistants.sort(key=lambda x: x.get(sort_by), reverse=reverse)
        else:
            sort_by = "created_at"
            # Default sorting by created_at in descending order
            filtered_assistants.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        paginated_assistants = filtered_assistants[offset : offset + limit]
        cur = offset + limit if len(filtered_assistants) > offset + limit else None

        async def assistant_iterator() -> AsyncIterator[Assistant]:
            for assistant in paginated_assistants:
                if select:
                    # Filter to only selected fields
                    filtered_assistant = {
                        k: v for k, v in assistant.items() if k in select
                    }
                    yield filtered_assistant
                else:
                    yield assistant

        return assistant_iterator(), cur

    @staticmethod
    async def get(
        conn: InMemConnectionProto,
        assistant_id: UUID | str,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        """Get an assistant by ID."""
        assistant_id = _ensure_uuid(assistant_id)
        filters = await Assistants.handle_event(
            ctx,
            "read",
            Auth.types.AssistantsRead(assistant_id=assistant_id),
        )

        async def _yield_result():
            for assistant in conn.store["assistants"]:
                if assistant["assistant_id"] == assistant_id and (
                    not filters or _check_filter_match(assistant["metadata"], filters)
                ):
                    yield copy.deepcopy(assistant)

        return _yield_result()

    @staticmethod
    async def put(
        conn: InMemConnectionProto,
        assistant_id: UUID | str,
        *,
        graph_id: str,
        config: Config,
        context: Context,
        metadata: MetadataInput,
        if_exists: OnConflictBehavior,
        name: str,
        ctx: Auth.types.BaseAuthContext | None = None,
        description: str | None = None,
        system: bool = False,
    ) -> AsyncIterator[Assistant]:
        """Insert an assistant."""
        from langgraph_api.graph import assert_graph_exists  # noqa: PLC0415

        assistant_id = _ensure_uuid(assistant_id)
        metadata = metadata if metadata is not None else {}
        custom_encryption = _using_custom_encryption()
        if custom_encryption:
            decrypted = await _decrypt_response(
                {"config": config, "context": context, "metadata": metadata},
                "assistant",
                ["config", "context", "metadata"],
            )
            config, context, metadata = (
                decrypted["config"],
                decrypted["context"],
                decrypted["metadata"],
            )
        filters = await Assistants.handle_event(
            ctx,
            "create",
            Auth.types.AssistantsCreate(
                assistant_id=assistant_id,
                graph_id=graph_id,
                config=config,
                context=context,
                metadata=metadata,
                name=name,
            ),
        )

        if config.get("configurable") and context:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both configurable and context. Prefer setting context alone. Context was introduced in LangGraph 0.6.0 and is the long term planned replacement for configurable.",
            )

        assert_graph_exists(graph_id)

        # Keep config and context up to date with one another
        if config.get("configurable"):
            context = config["configurable"]
        elif context:
            config["configurable"] = context

        existing_assistant = next(
            (a for a in conn.store["assistants"] if a["assistant_id"] == assistant_id),
            None,
        )
        if existing_assistant:
            existing_metadata = existing_assistant["metadata"]
            if custom_encryption:
                existing_metadata = (
                    await _decrypt_response(
                        {"metadata": existing_metadata}, "assistant", ["metadata"]
                    )
                )["metadata"]
            if filters and not _check_filter_match(existing_metadata, filters):
                raise HTTPException(
                    status_code=409, detail=f"Assistant {assistant_id} already exists"
                )
            if if_exists == "raise":
                raise HTTPException(
                    status_code=409, detail=f"Assistant {assistant_id} already exists"
                )
            elif if_exists == "do_nothing":

                async def _yield_existing():
                    yield existing_assistant

                return _yield_existing()

        now = datetime.now(UTC)
        new_assistant: Assistant = {
            "assistant_id": assistant_id,
            "graph_id": graph_id,
            "config": config or {},
            "context": context or {},
            "metadata": metadata or {},
            "name": name,
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "description": description,
        }
        new_version = {
            "assistant_id": assistant_id,
            "version": 1,
            "graph_id": graph_id,
            "config": config or {},
            "context": context or {},
            "metadata": metadata or {},
            "created_at": now,
            "name": name,
            "description": description,
        }
        if custom_encryption and not system:
            new_assistant = await _encrypt_request(
                new_assistant, "assistant", ["config", "context", "metadata"]
            )
            new_version = await _encrypt_request(
                new_version, "assistant", ["config", "context", "metadata"]
            )
        conn.store["assistants"].append(new_assistant)
        conn.store["assistant_versions"].append(new_version)

        async def _yield_new():
            yield new_assistant

        return _yield_new()

    @staticmethod
    async def patch(
        conn: InMemConnectionProto,
        assistant_id: UUID,
        *,
        config: Config | None = None,
        context: Context | None = None,
        graph_id: str | None = None,
        metadata: MetadataInput | None = None,
        name: str | None = None,
        description: str | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        """Update an assistant.

        Args:
            conn: The connection to the in-memory store.
            assistant_id: The assistant ID.
            graph_id: The graph ID.
            config: The assistant config.
            context: The assistant's static context.
            metadata: The assistant metadata.
            name: The assistant name.
            description: The assistant description.
            ctx: The auth context.

        Returns:
            return the updated assistant model.
        """
        from langgraph_api.graph import assert_graph_exists  # noqa: PLC0415

        assistant_id = _ensure_uuid(assistant_id)
        metadata = metadata if metadata is not None else {}
        config = config if config is not None else {}
        custom_encryption = _using_custom_encryption()
        if custom_encryption:
            decrypted = await _decrypt_response(
                {"config": config, "context": context, "metadata": metadata},
                "assistant",
                ["config", "context", "metadata"],
            )
            config, context, metadata = (
                decrypted["config"],
                decrypted["context"],
                decrypted["metadata"],
            )
        filters = await Assistants.handle_event(
            ctx,
            "update",
            Auth.types.AssistantsUpdate(
                assistant_id=assistant_id,
                graph_id=graph_id,
                config=config,
                context=context,
                metadata=metadata,
                name=name,
            ),
        )

        if config.get("configurable") and context:
            raise HTTPException(
                status_code=400,
                detail="Cannot specify both configurable and context. Prefer setting context alone. Context was introduced in LangGraph 0.6.0 and is the long term planned replacement for configurable.",
            )

        if graph_id is not None:
            assert_graph_exists(graph_id)

        # Keep config and context up to date with one another
        if config.get("configurable"):
            context = config["configurable"]
        elif context:
            config["configurable"] = context

        assistant = next(
            (a for a in conn.store["assistants"] if a["assistant_id"] == assistant_id),
            None,
        )
        if not assistant:
            raise HTTPException(
                status_code=404, detail=f"Assistant {assistant_id} not found"
            )
        stored_assistant = assistant
        if custom_encryption:
            assistant = await _decrypt_response(
                assistant, "assistant", ["config", "context", "metadata"]
            )
        if filters and not _check_filter_match(assistant["metadata"], filters):
            raise HTTPException(
                status_code=404, detail=f"Assistant {assistant_id} not found"
            )

        now = datetime.now(UTC)
        new_version = (
            max(
                v["version"]
                for v in conn.store["assistant_versions"]
                if v["assistant_id"] == assistant_id
            )
            + 1
            if conn.store["assistant_versions"]
            else 1
        )

        new_version_entry = {
            "assistant_id": assistant_id,
            "version": new_version,
            "graph_id": graph_id if graph_id is not None else assistant["graph_id"],
            "config": config if config else assistant["config"],
            "context": context if context is not None else assistant.get("context", {}),
            "metadata": (
                {**assistant["metadata"], **metadata}
                if metadata is not None
                else assistant["metadata"]
            ),
            "created_at": now,
            "name": name if name is not None else assistant["name"],
            "description": (
                description if description is not None else assistant.get("description")
            ),
        }
        if custom_encryption:
            new_version_entry = await _encrypt_request(
                new_version_entry,
                "assistant",
                ["config", "context", "metadata"],
            )
        conn.store["assistant_versions"].append(new_version_entry)

        # Update assistants table
        stored_assistant.update(
            {
                "graph_id": new_version_entry["graph_id"],
                "config": new_version_entry["config"],
                "context": new_version_entry["context"],
                "metadata": new_version_entry["metadata"],
                "name": name if name is not None else assistant["name"],
                "description": (
                    description
                    if description is not None
                    else assistant.get("description")
                ),
                "updated_at": now,
                "version": new_version,
            }
        )

        async def _yield_updated():
            yield stored_assistant

        return _yield_updated()

    @staticmethod
    async def delete(
        conn: InMemConnectionProto | None,
        assistant_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
        *,
        delete_threads: bool = False,
    ) -> AsyncIterator[UUID]:
        """Delete an assistant by ID."""
        async with AsyncExitStack() as stack:
            if conn is None:
                conn = await stack.enter_async_context(connect())

            assistant_id = _ensure_uuid(assistant_id)
            filters = await Assistants.handle_event(
                ctx,
                "delete",
                Auth.types.AssistantsDelete(
                    assistant_id=assistant_id,
                ),
            )
            assistant = next(
                (
                    a
                    for a in conn.store["assistants"]
                    if a["assistant_id"] == assistant_id
                ),
                None,
            )

            if not assistant:
                raise HTTPException(
                    status_code=404,
                    detail=f"Assistant with ID {assistant_id} not found",
                )
            elif filters and not _check_filter_match(assistant["metadata"], filters):
                raise HTTPException(
                    status_code=404,
                    detail=f"Assistant with ID {assistant_id} not found",
                )

            if delete_threads:
                threads_to_delete = [
                    t["thread_id"]
                    for t in conn.store["threads"]
                    if t.get("metadata", {}).get("assistant_id") == str(assistant_id)
                ]
                for thread_id in threads_to_delete:
                    try:
                        async for _ in await Threads.delete(conn, thread_id, ctx=ctx):
                            pass
                    except HTTPException:
                        await logger.awarning(
                            "Skipping thread deletion during cascade delete (user lacks permission)",
                            thread_id=thread_id,
                            assistant_id=assistant_id,
                        )

            # 3. Cancel in-flight runs AFTER auth validation
            await Runs.cancel(
                conn,
                assistant_id=assistant_id,
                action="interrupt",
                ctx=ctx,
            )

            # 4. Delete assistant
            conn.store["assistants"] = [
                a for a in conn.store["assistants"] if a["assistant_id"] != assistant_id
            ]
            # Cascade delete assistant versions
            conn.store["assistant_versions"] = [
                v
                for v in conn.store["assistant_versions"]
                if v["assistant_id"] != assistant_id
            ]
            # Cascade delete crons
            conn.store["crons"] = [
                c
                for c in conn.store["crons"]
                if str(c["assistant_id"]) != str(assistant_id)
            ]

            async def _yield_deleted():
                yield assistant_id

            return _yield_deleted()

    @staticmethod
    async def set_latest(
        conn: InMemConnectionProto,
        assistant_id: UUID,
        version: int,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        """Change the version of an assistant."""
        assistant_id = _ensure_uuid(assistant_id)
        filters = await Assistants.handle_event(
            ctx,
            "update",
            Auth.types.AssistantsUpdate(
                assistant_id=assistant_id,
                version=version,
            ),
        )
        assistant = next(
            (a for a in conn.store["assistants"] if a["assistant_id"] == assistant_id),
            None,
        )
        if not assistant:
            raise HTTPException(
                status_code=404, detail=f"Assistant {assistant_id} not found"
            )
        elif filters and not _check_filter_match(assistant["metadata"], filters):
            raise HTTPException(
                status_code=404, detail=f"Assistant {assistant_id} not found"
            )

        version_data = next(
            (
                v
                for v in conn.store["assistant_versions"]
                if v["assistant_id"] == assistant_id and v["version"] == version
            ),
            None,
        )
        if not version_data:
            raise HTTPException(
                status_code=404,
                detail=f"Version {version} not found for assistant {assistant_id}",
            )

        assistant.update(
            {
                "config": version_data["config"],
                "metadata": version_data["metadata"],
                "version": version_data["version"],
                "updated_at": datetime.now(UTC),
                "name": version_data["name"],
                "description": version_data["description"],
            }
        )

        async def _yield_updated():
            yield assistant

        return _yield_updated()

    @staticmethod
    async def get_versions(
        conn: InMemConnectionProto,
        assistant_id: UUID,
        metadata: MetadataInput,
        limit: int,
        offset: int,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        """Get all versions of an assistant."""
        assistant_id = _ensure_uuid(assistant_id)
        filters = await Assistants.handle_event(
            ctx,
            "read",
            Auth.types.AssistantsRead(assistant_id=assistant_id),
        )
        assistant = next(
            (a for a in conn.store["assistants"] if a["assistant_id"] == assistant_id),
            None,
        )
        if not assistant:
            raise HTTPException(
                status_code=404, detail=f"Assistant {assistant_id} not found"
            )
        versions = [
            v
            for v in conn.store["assistant_versions"]
            if v["assistant_id"] == assistant_id
            and (not metadata or is_jsonb_contained(v["metadata"], metadata))
            and (not filters or _check_filter_match(v["metadata"], filters))
        ]

        # Previously, the name was not included in the assistant_versions table. So we should add them here.
        description = assistant.get("description")
        for v in versions:
            if "name" not in v:
                v["name"] = assistant["name"]
            if "description" not in v:
                v["description"] = description
            else:
                description = v["description"]

        versions.sort(key=lambda x: x["version"], reverse=True)

        async def _yield_versions():
            for version in versions[offset : offset + limit]:
                yield version

        return _yield_versions()

    @staticmethod
    async def count(
        conn: InMemConnectionProto,
        *,
        graph_id: str | None = None,
        name: str | None = None,
        metadata: MetadataInput = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> int:
        """Get count of assistants."""
        from langgraph_api.graph import assert_graph_exists  # noqa: PLC0415

        metadata = metadata if metadata is not None else {}
        filters = await Assistants.handle_event(
            ctx,
            "search",
            Auth.types.AssistantsSearch(
                graph_id=graph_id, metadata=metadata, limit=0, offset=0
            ),
        )

        if graph_id is not None:
            assert_graph_exists(graph_id)

        count = 0
        for assistant in conn.store["assistants"]:
            if (
                (not graph_id or assistant["graph_id"] == graph_id)
                and (not name or name.lower() in assistant["name"].lower())
                and (
                    not metadata or is_jsonb_contained(assistant["metadata"], metadata)
                )
                and (not filters or _check_filter_match(assistant["metadata"], filters))
            ):
                count += 1

        return count


def is_jsonb_contained(superset: dict[str, Any], subset: dict[str, Any]) -> bool:
    """
    Implements Postgres' @> (containment) operator for dictionaries.
    Returns True if superset contains all key/value pairs from subset.
    """
    for key, value in subset.items():
        if key not in superset:
            return False
        if isinstance(value, dict) and isinstance(superset[key], dict):
            if not is_jsonb_contained(superset[key], value):
                return False
        elif superset[key] != value:
            return False
    return True


def bytes_decoder(obj):
    """Custom JSON decoder that converts base64 back to bytes."""
    if "__type__" in obj and obj["__type__"] == "bytes":
        return base64.b64decode(obj["value"].encode("utf-8"))
    return obj


def _replace_thread_id(data, new_thread_id, thread_id):
    class BytesEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles bytes by converting them to base64."""

        def default(self, obj):
            if isinstance(obj, bytes | bytearray):
                return {
                    "__type__": "bytes",
                    "value": base64.b64encode(
                        obj.replace(
                            str(thread_id).encode(), str(new_thread_id).encode()
                        )
                    ).decode("utf-8"),
                }

            return super().default(obj)

    try:
        json_str = json.dumps(data, cls=BytesEncoder, indent=2)
    except Exception as e:
        raise ValueError(data) from e
    json_str = json_str.replace(str(thread_id), str(new_thread_id))

    # Decoding back from JSON
    d = json.loads(json_str, object_hook=bytes_decoder)
    return d


def _patch_interrupt(
    interrupt: Interrupt | dict,
) -> InterruptSchema | DeprecatedInterrupt:
    """Convert a langgraph interrupt (v0 or v1) to standard interrupt schema.

    In v0.4 and v0.5, interrupt_id is a property on the langgraph.types.Interrupt object,
    so we reconstruct the type in order to access the id, with compatibility for the new
    v0.6 interrupt format as well.
    """
    if USE_NEW_INTERRUPTS:
        interrupt = Interrupt(**interrupt) if isinstance(interrupt, dict) else interrupt

        return {
            "id": interrupt.id,
            "value": interrupt.value,
        }
    else:
        if isinstance(interrupt, dict):
            # interrupt_id is a deprecated property on Interrupt and should not be used for initialization
            # id is the new field we use for identification, also not supported on init for old versions
            interrupt.pop("interrupt_id", None)
            interrupt.pop("id", None)
            interrupt = Interrupt(**interrupt)

        return {
            "id": (
                interrupt.interrupt_id if hasattr(interrupt, "interrupt_id") else None
            ),
            "value": interrupt.value,
            "resumable": interrupt.resumable,
            "ns": interrupt.ns,
            "when": interrupt.when,  # type: ignore[unresolved-attribute]
        }


class Threads(Authenticated):
    resource = "threads"

    @staticmethod
    async def search(
        conn: InMemConnectionProto,
        *,
        ids: list[str] | list[UUID] | None = None,
        metadata: MetadataInput,
        values: MetadataInput,
        status: ThreadStatus | None,
        limit: int,
        offset: int,
        sort_by: str | None = None,
        sort_order: str | None = None,
        select: list[ThreadSelectField] | None = None,
        extract: dict[str, str] | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> tuple[AsyncIterator[Thread], int]:
        threads = conn.store["threads"]
        filtered_threads: list[Thread] = []
        metadata = metadata if metadata is not None else {}
        values = values if values is not None else {}
        filters = await Threads.handle_event(
            ctx,
            "search",
            Auth.types.ThreadsSearch(
                metadata=metadata,
                values=values,
                status=status,
                limit=limit,
                offset=offset,
            ),
        )

        # Apply filters
        id_set: set[UUID] | None = None
        if ids:
            id_set = set()
            for i in ids:
                try:
                    id_set.add(_ensure_uuid(i))
                except Exception:
                    raise HTTPException(
                        status_code=400, detail="Invalid thread ID " + str(i)
                    ) from None
        for thread in threads:
            if id_set is not None and thread.get("thread_id") not in id_set:
                continue
            if filters and not _check_filter_match(thread["metadata"], filters):
                continue

            if metadata and not is_jsonb_contained(thread["metadata"], metadata):
                continue

            if (
                values
                and "values" in thread
                and not is_jsonb_contained(thread["values"], values)
            ):
                continue

            if status and thread.get("status") != status:
                continue

            thread.setdefault("state_updated_at", thread.get("updated_at"))
            filtered_threads.append(thread)

        if sort_by and sort_by in [
            "thread_id",
            "created_at",
            "updated_at",
            "state_updated_at",
            "status",
        ]:
            reverse = False if sort_order and sort_order.upper() == "ASC" else True
            sorted_threads = sorted(
                filtered_threads, key=lambda x: x.get(sort_by), reverse=reverse
            )
        else:
            sort_by = "created_at"
            # Default sorting by created_at in descending order
            sorted_threads = sorted(
                filtered_threads, key=lambda x: x["updated_at"], reverse=True
            )

        # Apply limit and offset
        paginated_threads = sorted_threads[offset : offset + limit]
        cursor = offset + limit if len(sorted_threads) > offset + limit else None

        async def thread_iterator() -> AsyncIterator[Thread]:
            if extract:
                from langgraph_api.utils.extract import (  # noqa: PLC0415
                    extract_path_value,
                )

            for thread in paginated_threads:
                if select:
                    # Filter to only selected fields
                    filtered_thread = {k: v for k, v in thread.items() if k in select}
                else:
                    filtered_thread = dict(thread)

                if extract:
                    filtered_thread["extracted"] = {
                        alias: extract_path_value(thread, path)
                        for alias, path in extract.items()
                    }

                yield filtered_thread

        return thread_iterator(), cursor

    @staticmethod
    async def _get_with_filters(
        conn: InMemConnectionProto,
        thread_id: UUID,
        filters: Auth.types.FilterType | None,
    ) -> Thread | None:
        thread_id = _ensure_uuid(thread_id)
        matching_thread = next(
            (
                thread
                for thread in conn.store["threads"]
                if thread["thread_id"] == thread_id
            ),
            None,
        )
        if not matching_thread or (
            filters and not _check_filter_match(matching_thread["metadata"], filters)
        ):
            return

        matching_thread.setdefault(
            "state_updated_at", matching_thread.get("updated_at")
        )
        return matching_thread

    @staticmethod
    async def _get(
        conn: InMemConnectionProto,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> Thread | None:
        """Get a thread by ID."""
        thread_id = _ensure_uuid(thread_id)
        filters = await Threads.handle_event(
            ctx,
            "read",
            Auth.types.ThreadsRead(thread_id=thread_id),
        )
        return await Threads._get_with_filters(conn, thread_id, filters)

    @staticmethod
    async def get(
        conn: InMemConnectionProto,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
        include_ttl: bool = False,
        read_mask_paths: list[str] | None = None,
    ) -> AsyncIterator[Thread]:
        """Get a thread by ID.

        Args:
            conn: In-memory connection
            thread_id: Thread ID
            ctx: Auth context
            include_ttl: Not supported in inmem - parameter ignored.
            read_mask_paths: Column restriction hint for the postgres runtime;
                ignored for inmem since there's no values-column I/O to skip.
        """
        matching_thread = await Threads._get(conn, thread_id, ctx)

        if not matching_thread:
            raise HTTPException(
                status_code=404, detail=f"Thread with ID {thread_id} not found"
            )

        async def _yield_result():
            if matching_thread:
                yield matching_thread

        return _yield_result()

    @staticmethod
    async def put(
        conn: InMemConnectionProto | Any,
        thread_id: UUID | str,
        *,
        metadata: MetadataInput,
        if_exists: OnConflictBehavior,
        ttl: ThreadTTLConfig | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Thread]:
        """Insert or update a thread."""
        thread_id = _ensure_uuid(thread_id)
        if metadata is None:
            metadata = {}

        # Check if thread already exists
        existing_thread = next(
            (t for t in conn.store["threads"] if t["thread_id"] == thread_id), None
        )
        filters = await Threads.handle_event(
            ctx,
            "create",
            Auth.types.ThreadsCreate(
                thread_id=thread_id, metadata=metadata, if_exists=if_exists
            ),
        )
        # Re-fetch in case an auth handler replaced the thread object in the store
        # (e.g. via a loopback patch call, which deep-copies and replaces the element).
        existing_thread = next(
            (t for t in conn.store["threads"] if t["thread_id"] == thread_id), None
        )

        if existing_thread:
            if filters and not _check_filter_match(
                existing_thread["metadata"], filters
            ):
                # Should we use a different status code here?
                raise HTTPException(
                    status_code=409, detail=f"Thread with ID {thread_id} already exists"
                )
            if if_exists == "raise":
                raise HTTPException(
                    status_code=409, detail=f"Thread with ID {thread_id} already exists"
                )
            elif if_exists == "do_nothing":

                async def _yield_existing():
                    yield existing_thread

                return _yield_existing()
        # Create new thread
        now = datetime.now(UTC)
        new_thread: Thread = {
            "thread_id": thread_id,
            "created_at": now,
            "updated_at": now,
            "state_updated_at": now,
            "metadata": copy.deepcopy(metadata),
            "status": "idle",
            "config": {},
            "values": None,
        }

        # Add to store
        conn.store["threads"].append(new_thread)

        async def _yield_new():
            yield new_thread

        return _yield_new()

    @staticmethod
    async def patch(
        conn: InMemConnectionProto,
        thread_id: UUID,
        *,
        metadata: MetadataValue,
        ttl: ThreadTTLConfig | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
        read_mask_paths: list[str] | None = None,
    ) -> AsyncIterator[Thread]:
        """Update a thread."""
        thread_list = conn.store["threads"]
        thread_idx = None
        thread_id = _ensure_uuid(thread_id)

        for idx, thread in enumerate(thread_list):
            if thread["thread_id"] == thread_id:
                thread_idx = idx
                break

        if thread_idx is not None:
            filters = await Threads.handle_event(
                ctx,
                "update",
                Auth.types.ThreadsUpdate(thread_id=thread_id, metadata=metadata),
            )
            if not filters or _check_filter_match(
                thread_list[thread_idx]["metadata"], filters
            ):
                thread = copy.deepcopy(thread_list[thread_idx])
                thread.setdefault("state_updated_at", thread.get("updated_at"))
                thread["metadata"] = {**thread["metadata"], **metadata}
                thread["updated_at"] = datetime.now(UTC)
                thread_list[thread_idx] = thread

                async def thread_iterator() -> AsyncIterator[Thread]:
                    yield thread

                return thread_iterator()

        async def empty_iterator() -> AsyncIterator[Thread]:
            if False:  # This ensures the iterator is empty
                yield

        return empty_iterator()

    @staticmethod
    async def set_status(
        conn: InMemConnectionProto,
        thread_id: UUID,
        checkpoint: CheckpointPayload | None,
        exception: BaseException | None,
        # This does not accept the auth context since it's only used internally
    ) -> None:
        """Set the status of a thread."""
        from langgraph_api.serde import json_dumpb, json_loads  # noqa: PLC0415

        thread_id = _ensure_uuid(thread_id)

        async def has_pending_runs(conn_: InMemConnectionProto, tid: UUID) -> bool:
            """Check if thread has any pending runs."""
            return any(
                run["status"] in ("pending", "running") and run["thread_id"] == tid
                for run in conn_.store["runs"]
            )

        # Find the thread
        thread = next(
            (
                thread
                for thread in conn.store["threads"]
                if thread["thread_id"] == thread_id
            ),
            None,
        )

        if not thread:
            raise HTTPException(
                status_code=404, detail=f"Thread {thread_id} not found."
            )

        # Determine has_next from checkpoint
        has_next = False if checkpoint is None else bool(checkpoint["next"])

        # Determine base status
        if exception:
            status = "error"
        elif has_next:
            status = "interrupted"
        else:
            status = "idle"

        # Check for pending runs and update to busy if found
        if await has_pending_runs(conn, thread_id):
            status = "busy"

        # Update thread
        now = datetime.now(UTC)
        update: dict = {
            "updated_at": now,
            "state_updated_at": now,
            "status": status,
            "interrupts": (
                {
                    t["id"]: [_patch_interrupt(i) for i in t["interrupts"]]
                    for t in checkpoint["tasks"]
                    if t.get("interrupts")
                }
                if checkpoint
                else {}
            ),
            "error": json_loads(json_dumpb(exception)) if exception else None,
        }
        if checkpoint is not None:
            update["values"] = checkpoint["values"]
        thread.update(update)

    @staticmethod
    async def set_joint_status(
        conn: InMemConnectionProto,
        thread_id: UUID,
        run_id: UUID,
        run_status: RunStatus | Literal["rollback"],
        graph_id: str,
        checkpoint: CheckpointPayload | None = None,
        exception: BaseException | None = None,
    ) -> None:
        """Set the status of both thread and run atomically in a single query.

        This is an optimized version that combines the logic from Threads.set_status
        and Runs.set_status to minimize database round trips and ensure atomicity.

        Args:
            conn: Database connection
            thread_id: Thread ID to update
            run_id: Run ID to update
            run_status: New status for the run (or "rollback" to delete the run)
            checkpoint: Checkpoint payload for thread status calculation
            exception: Exception that occurred (affects thread status)
        """
        # No auth since it's internal
        from langgraph_api.errors import UserInterrupt, UserRollback  # noqa: PLC0415
        from langgraph_api.serde import json_dumpb, json_loads  # noqa: PLC0415

        thread_id = _ensure_uuid(thread_id)
        run_id = _ensure_uuid(run_id)

        def _thread_has_active_runs() -> bool:
            return any(
                r["thread_id"] == thread_id and r["status"] in ("pending", "running")
                for r in conn.store["runs"]
            )

        thread = next(
            (t for t in conn.store["threads"] if t["thread_id"] == thread_id), None
        )
        if thread is None:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

        run = next(
            (
                r
                for r in conn.store["runs"]
                if r["run_id"] == run_id and r["thread_id"] == thread_id
            ),
            None,
        )
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        has_next = bool(checkpoint and checkpoint["next"])
        if exception and not isinstance(exception, UserInterrupt | UserRollback):
            base_thread_status: ThreadStatus = "error"
        elif has_next:
            base_thread_status = "interrupted"
        else:
            base_thread_status = "idle"

        interrupts = (
            {
                t["id"]: [_patch_interrupt(i) for i in t["interrupts"]]
                for t in checkpoint["tasks"]
                if t.get("interrupts")
            }
            if checkpoint
            else {}
        )

        now = datetime.now(UTC)

        if run_status == "rollback":
            await Runs.delete(conn, run_id, thread_id=run["thread_id"])
            final_thread_status: ThreadStatus = (
                "busy" if _thread_has_active_runs() else base_thread_status
            )

        else:
            run.update({"status": run_status, "updated_at": now})

            if run_status in ("pending", "running") or _thread_has_active_runs():
                final_thread_status = "busy"
            else:
                final_thread_status = base_thread_status
        thread["metadata"]["graph_id"] = graph_id
        update: dict = {
            "updated_at": now,
            "state_updated_at": now,
            "interrupts": interrupts,
            "status": final_thread_status,
            "error": json_loads(json_dumpb(exception)) if exception else None,
        }
        if checkpoint is not None:
            update["values"] = checkpoint["values"]
        thread.update(update)

    @staticmethod
    async def delete(
        conn: InMemConnectionProto,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[UUID]:
        """Delete a thread by ID and cascade delete all associated runs."""
        thread_list = conn.store["threads"]
        thread_idx = None
        thread_id = _ensure_uuid(thread_id)

        # Find the thread to delete
        for idx, thread in enumerate(thread_list):
            if thread["thread_id"] == thread_id:
                thread_idx = idx
                break
        filters = await Threads.handle_event(
            ctx,
            "delete",
            Auth.types.ThreadsDelete(thread_id=thread_id),
        )
        if (filters and not _check_filter_match(thread["metadata"], filters)) or (
            thread_idx is None
        ):
            raise HTTPException(
                status_code=404, detail=f"Thread with ID {thread_id} not found"
            )
        # Cascade delete all runs associated with this thread
        conn.store["runs"] = [
            run for run in conn.store["runs"] if run["thread_id"] != thread_id
        ]
        # Cascade delete crons associated with this thread
        conn.store["crons"] = [
            c for c in conn.store["crons"] if str(c.get("thread_id")) != str(thread_id)
        ]
        await _delete_checkpoints_for_thread(thread_id, conn)

        if thread_idx is not None:
            # Remove the thread from the store
            deleted_thread = thread_list.pop(thread_idx)

            # Return an async iterator with the deleted thread_id
            async def id_iterator() -> AsyncIterator[UUID]:
                yield deleted_thread["thread_id"]

            return id_iterator()

        # If thread not found, return empty iterator
        async def empty_iterator() -> AsyncIterator[UUID]:
            if False:  # This ensures the iterator is empty
                yield

        return empty_iterator()

    @staticmethod
    async def prune(
        thread_ids: Sequence[str] | Sequence[UUID],
        strategy: Literal["delete", "keep_latest"] = "delete",
        batch_size: int = 100,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> int:
        """Prune threads by ID (inmem implementation).

        Args:
            thread_ids: List of thread IDs to prune
            strategy: Prune strategy ("delete" supported, "keep_latest" not supported)
            batch_size: Not used in inmem implementation
            ctx: Auth context for permission checks

        Returns:
            Number of threads successfully pruned
        """
        if not thread_ids:
            return 0

        auth_filters = await Threads.handle_event(
            ctx,
            "delete",
            {"thread_ids": thread_ids},
        )
        if auth_filters:
            # Validate access to all threads
            async with connect() as conn:
                matching_threads = await asyncio.gather(
                    *[
                        Threads._get_with_filters(conn, tid, auth_filters)
                        for tid in thread_ids
                    ]
                )
                if any(not thread for thread in matching_threads):
                    raise HTTPException(
                        status_code=404,
                        detail="At least one thread not found or not authorized",
                    )

        if strategy == "keep_latest":
            raise HTTPException(
                status_code=422,
                detail="keep_latest strategy is not supported in in-memory runtime",
            )

        pruned = 0
        async with connect() as conn:
            for tid in thread_ids:
                try:
                    tid_uuid = _ensure_uuid(tid)
                    iter_result = await Threads.delete(conn, tid_uuid, ctx)
                    # Consume the iterator to ensure deletion
                    async for _ in iter_result:
                        pruned += 1
                except HTTPException:
                    # Thread not found or no permission - skip silently
                    pass

        return pruned

    @staticmethod
    async def _delete_with_run(
        conn: InMemConnectionProto,
        thread_id: UUID,
        run_id: UUID,
    ) -> UUID:
        """Delete a thread by ID."""
        # We don't really care about "optimal" here.
        return await Threads.delete(conn, thread_id)

    @staticmethod
    async def copy(
        conn: InMemConnectionProto,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Thread]:
        """Create a copy of an existing thread."""
        thread_id = _ensure_uuid(thread_id)
        new_thread_id = uuid4()
        read_filters = await Threads.handle_event(
            ctx,
            "read",
            Auth.types.ThreadsRead(
                thread_id=thread_id,
            ),
        )
        # Assert that the user has permissions to create a new thread.
        # (We don't actually need the filters.)
        await Threads.handle_event(
            ctx,
            "create",
            Auth.types.ThreadsCreate(
                thread_id=new_thread_id,
            ),
        )

        async with conn.pipeline():
            # Find the original thread in our store
            original_thread = next(
                (t for t in conn.store["threads"] if t["thread_id"] == thread_id), None
            )

            if not original_thread:
                return _empty_generator()
            if read_filters and not _check_filter_match(
                original_thread["metadata"], read_filters
            ):
                return _empty_generator()

            # Create new thread with copied metadata
            now = datetime.now(tz=UTC)
            new_thread: Thread = {
                "thread_id": new_thread_id,
                "created_at": now,
                "updated_at": now,
                "state_updated_at": now,
                "metadata": copy.deepcopy(original_thread["metadata"]),
                "status": "idle",
                "config": {},
            }

            # Add new thread to store
            conn.store["threads"].append(new_thread)

            from langgraph_api import config as api_config  # noqa: PLC0415

            if api_config.USE_CUSTOM_CHECKPOINTER:
                from langgraph_api import (  # noqa: PLC0415
                    _checkpointer as api_checkpointer,
                )

                checkpointer = await api_checkpointer.get_checkpointer()
                await checkpointer.acopy_thread(str(thread_id), str(new_thread_id))
            else:
                checkpointer = Checkpointer()
                copied_storage = _replace_thread_id(
                    checkpointer.storage[str(thread_id)],
                    new_thread_id,
                    thread_id,
                )
                checkpointer.storage[str(new_thread_id)] = copied_storage
                # Copy the writes over (if any)
                outer_keys = []
                for k in checkpointer.writes:
                    if k[0] == str(thread_id):
                        outer_keys.append(k)
                for tid, checkpoint_ns, checkpoint_id in outer_keys:
                    mapped = {
                        k: _replace_thread_id(v, new_thread_id, thread_id)
                        for k, v in checkpointer.writes[
                            (str(tid), checkpoint_ns, checkpoint_id)
                        ].items()
                    }

                    checkpointer.writes[
                        (str(new_thread_id), checkpoint_ns, checkpoint_id)
                    ] = mapped
                # Copy the blobs
                for k in list(checkpointer.blobs):
                    if str(k[0]) == str(thread_id):
                        new_key = (str(new_thread_id), *k[1:])
                        checkpointer.blobs[new_key] = checkpointer.blobs[k]

            async def row_generator() -> AsyncIterator[Thread]:
                yield new_thread

            return row_generator()

    @staticmethod
    async def sweep_ttl(
        conn: InMemConnectionProto,
        *,
        limit: int | None = None,
        batch_size: int = 100,
    ) -> tuple[int, int]:
        # Not implemented for inmem server
        return (0, 0)

    class State(Authenticated):
        # We will treat this like a runs resource for now.
        resource = "threads"

        @staticmethod
        async def get(
            conn: InMemConnectionProto,
            config: Config,
            subgraphs: bool = False,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> StateSnapshot:
            """Get state for a thread."""
            from langgraph_api.graph import get_graph  # noqa: PLC0415
            from langgraph_api.store import get_store  # noqa: PLC0415

            checkpointer = await _get_checkpointer(
                conn, unpack_hook=_msgpack_ext_hook_to_json
            )
            thread_id = _ensure_uuid(config["configurable"]["thread_id"])
            # Auth will be applied here so no need to use filters downstream
            thread_iter = await Threads.get(conn, thread_id, ctx=ctx)
            thread = await anext(thread_iter)
            if not thread:
                return StateSnapshot(
                    values={},
                    next=[],
                    config=None,
                    metadata=None,
                    created_at=None,
                    parent_config=None,
                    tasks=tuple(),
                    **_snapshot_defaults(),
                )

            metadata = thread.get("metadata", {})
            thread_config = cast(dict[str, Any], thread.get("config", {}))
            thread_config = {
                **thread_config,
                "configurable": {
                    **thread_config.get("configurable", {}),
                    **config.get("configurable", {}),
                },
            }

            # Fallback to graph_id from run if not in thread metadata
            graph_id = metadata.get("graph_id")
            if not graph_id:
                for run in conn.store["runs"]:
                    if run["thread_id"] == thread_id:
                        graph_id = run["kwargs"]["config"]["configurable"]["graph_id"]
                        break

            if graph_id:
                # Prefetch checkpoint to avoid redundant aget_tuple in aget_state
                checkpointer.latest_iter = await checkpointer.aget(config)
                async with get_graph(
                    graph_id,
                    thread_config,
                    checkpointer=checkpointer,
                    store=(await get_store()),
                    access_context="threads.read",
                ) as graph:
                    result = await graph.aget_state(config, subgraphs=subgraphs)
                    if (
                        result.metadata is not None
                        and "checkpoint_ns" in result.metadata
                        and result.metadata["checkpoint_ns"] == ""
                    ):
                        result.metadata.pop("checkpoint_ns")
                    return result
            else:
                return StateSnapshot(
                    values={},
                    next=[],
                    config=None,
                    metadata=None,
                    created_at=None,
                    parent_config=None,
                    tasks=tuple(),
                    **_snapshot_defaults(),
                )

        @staticmethod
        async def post(
            conn: InMemConnectionProto,
            config: Config,
            values: Sequence[dict] | dict[str, Any] | None,
            as_node: str | None = None,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> ThreadUpdateResponse:
            """Add state to a thread."""
            from langgraph_api.graph import get_graph  # noqa: PLC0415
            from langgraph_api.schema import ThreadUpdateResponse  # noqa: PLC0415
            from langgraph_api.state import (  # noqa: PLC0415
                state_snapshot_to_thread_state,
            )
            from langgraph_api.store import get_store  # noqa: PLC0415
            from langgraph_api.utils import fetchone  # noqa: PLC0415

            thread_id = _ensure_uuid(config["configurable"]["thread_id"])
            filters = await Threads.handle_event(
                ctx,
                "update",
                Auth.types.ThreadsUpdate(thread_id=thread_id),
            )

            checkpointer = await _get_checkpointer()

            thread_iter = await Threads.get(conn, thread_id, ctx=ctx)
            thread = await fetchone(
                thread_iter, not_found_detail=f"Thread {thread_id} not found."
            )
            if not thread:
                raise HTTPException(status_code=404, detail="Thread not found")
            if not _check_filter_match(thread["metadata"], filters):
                raise HTTPException(status_code=403, detail="Forbidden")

            metadata = thread["metadata"]
            thread_config = thread["config"]
            # Check that there are no in-flight runs
            pending_runs = [
                run
                for run in conn.store["runs"]
                if run["thread_id"] == thread_id
                and run["status"] in ("pending", "running")
            ]
            if pending_runs:
                raise HTTPException(
                    status_code=409,
                    detail=f"Thread {thread_id} has in-flight runs: {pending_runs}",
                )
            thread_config = {
                **thread_config,
                "configurable": {
                    **thread_config.get("configurable", {}),
                    **config.get("configurable", {}),
                },
            }

            # Fallback to graph_id from run if not in thread metadata
            graph_id = metadata.get("graph_id")
            if not graph_id:
                for run in conn.store["runs"]:
                    if run["thread_id"] == thread_id:
                        graph_id = run["kwargs"]["config"]["configurable"]["graph_id"]
                        break

            if graph_id:
                config["configurable"].setdefault("graph_id", graph_id)

                # Prefetch checkpoint to avoid redundant aget_tuple in aupdate_state
                checkpointer.latest_iter = await checkpointer.aget(config)
                async with get_graph(
                    graph_id,
                    thread_config,
                    checkpointer=checkpointer,
                    store=(await get_store()),
                    access_context="threads.update",
                ) as graph:
                    update_config = config.copy()
                    update_config["configurable"] = {
                        **config["configurable"],
                        "checkpoint_ns": config["configurable"].get(
                            "checkpoint_ns", ""
                        ),
                    }
                    next_config = await graph.aupdate_state(
                        update_config, values, as_node=as_node
                    )

                    # Get current state
                    state = await Threads.State.get(
                        conn, config, subgraphs=False, ctx=ctx
                    )
                    # Update thread status, values, and interrupts
                    await Threads.set_status(
                        conn,
                        thread_id,
                        {
                            "next": list(state.next),
                            "values": state.values,
                            "tasks": [
                                {
                                    "id": t.id,
                                    "interrupts": list(t.interrupts),
                                }
                                for t in state.tasks
                            ],
                        },
                        None,
                    )

                    # Publish state update event
                    from langgraph_api.serde import json_dumpb  # noqa: PLC0415

                    event_data = {
                        "state": state_snapshot_to_thread_state(state),
                        "thread_id": str(thread_id),
                    }
                    await Threads.Stream.publish(
                        thread_id,
                        "state_update",
                        json_dumpb(event_data),
                    )

                    return ThreadUpdateResponse(
                        checkpoint=next_config["configurable"],
                        # Including deprecated fields
                        configurable=next_config["configurable"],
                        checkpoint_id=next_config["configurable"]["checkpoint_id"],
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Thread '{thread['thread_id']}' has no assigned graph ID. This usually occurs when no runs have been made on this particular thread."
                    " This operation requires a graph ID. Please ensure a run has been made for the thread or manually update the thread metadata (by setting the 'graph_id' field) before running this operation.",
                )

        @staticmethod
        async def bulk(
            conn: InMemConnectionProto,
            *,
            config: Config,
            supersteps: Sequence[dict],
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> ThreadUpdateResponse:
            """Update a thread with a batch of state updates."""

            from langgraph.types import StateUpdate  # noqa: PLC0415
            from langgraph_api.command import map_cmd  # noqa: PLC0415
            from langgraph_api.graph import get_graph  # noqa: PLC0415
            from langgraph_api.schema import ThreadUpdateResponse  # noqa: PLC0415
            from langgraph_api.store import get_store  # noqa: PLC0415
            from langgraph_api.utils import fetchone  # noqa: PLC0415

            thread_id = _ensure_uuid(config["configurable"]["thread_id"])
            filters = await Threads.handle_event(
                ctx,
                "update",
                Auth.types.ThreadsUpdate(thread_id=thread_id),
            )

            thread_iter = await Threads.get(conn, thread_id, ctx=ctx)
            thread = await fetchone(
                thread_iter, not_found_detail=f"Thread {thread_id} not found."
            )

            thread_config = cast(dict[str, Any], thread["config"])
            thread_config = {
                **thread_config,
                "configurable": {
                    **thread_config.get("configurable", {}),
                    **config.get("configurable", {}),
                },
            }
            metadata = thread["metadata"]

            if not thread:
                raise HTTPException(status_code=404, detail="Thread not found")

            if not _check_filter_match(metadata, filters):
                raise HTTPException(status_code=403, detail="Forbidden")

            if graph_id := metadata.get("graph_id"):
                config["configurable"].setdefault("graph_id", graph_id)
                config["configurable"].setdefault("checkpoint_ns", "")

                checkpointer = await _get_checkpointer()
                async with get_graph(
                    graph_id,
                    thread_config,
                    checkpointer=checkpointer,
                    store=(await get_store()),
                    access_context="threads.update",
                ) as graph:
                    next_config = await graph.abulk_update_state(
                        config,
                        [
                            [
                                StateUpdate(
                                    (
                                        map_cmd(update.get("command"))
                                        if update.get("command")
                                        else update.get("values")
                                    ),
                                    update.get("as_node"),
                                )
                                for update in superstep.get("updates", [])
                            ]
                            for superstep in supersteps
                        ],
                    )

                    state = await Threads.State.get(
                        conn, config, subgraphs=False, ctx=ctx
                    )

                    # Update thread status, values, and interrupts
                    await Threads.set_status(
                        conn,
                        thread_id,
                        {
                            "next": list(state.next),
                            "values": state.values,
                            "tasks": [
                                {
                                    "id": t.id,
                                    "interrupts": list(t.interrupts),
                                }
                                for t in state.tasks
                            ],
                        },
                        None,
                    )

                    # Publish state update event
                    from langgraph_api.serde import json_dumpb  # noqa: PLC0415
                    from langgraph_api.state import (  # noqa: PLC0415
                        state_snapshot_to_thread_state,
                    )

                    event_data = {
                        "state": state_snapshot_to_thread_state(state),
                        "thread_id": str(thread_id),
                    }
                    await Threads.Stream.publish(
                        thread_id,
                        "state_update",
                        json_dumpb(event_data),
                    )

                    return ThreadUpdateResponse(
                        checkpoint=next_config["configurable"],
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Thread '{thread['thread_id']}' has no assigned graph ID. This usually occurs when no runs have been made on this particular thread."
                    " This operation requires a graph ID. Please ensure a run has been made for the thread or manually update the thread metadata (by setting the 'graph_id' field) before running this operation.",
                )

        @staticmethod
        async def list(
            conn: InMemConnectionProto,
            *,
            config: Config,
            limit: int = 1,
            before: str | Checkpoint | None = None,
            metadata: MetadataInput = None,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> list[StateSnapshot]:
            """Get the history of a thread."""
            from langgraph_api.graph import get_graph  # noqa: PLC0415
            from langgraph_api.store import get_store  # noqa: PLC0415
            from langgraph_api.utils import fetchone  # noqa: PLC0415

            thread_id = _ensure_uuid(config["configurable"]["thread_id"])
            thread = None
            filters = await Threads.handle_event(
                ctx,
                "read",
                Auth.types.ThreadsRead(thread_id=thread_id),
            )
            thread = await fetchone(
                await Threads.get(conn, config["configurable"]["thread_id"], ctx=ctx)
            )

            # Parse thread metadata and config
            thread_metadata = thread["metadata"]
            if not _check_filter_match(thread_metadata, filters):
                return []

            thread_config = cast(dict[str, Any], thread["config"])
            thread_config = {
                **thread_config,
                "configurable": {
                    **thread_config.get("configurable", {}),
                    **config.get("configurable", {}),
                },
            }
            # If graph_id exists, get state history
            if graph_id := thread_metadata.get("graph_id"):
                checkpointer = await _get_checkpointer(
                    conn, unpack_hook=_msgpack_ext_hook_to_json
                )
                async with get_graph(
                    graph_id,
                    thread_config,
                    checkpointer=checkpointer,
                    store=(await get_store()),
                    access_context="threads.read",
                ) as graph:
                    # Convert before parameter if it's a string
                    before_param = (
                        {"configurable": {"checkpoint_id": before}}
                        if isinstance(before, str)
                        else before
                    )

                    states = [
                        state
                        async for state in graph.aget_state_history(
                            config, limit=limit, filter=metadata, before=before_param
                        )
                    ]

                    return states

            return []

    class Stream(Authenticated):
        resource = "threads"

        @staticmethod
        async def subscribe(
            conn: InMemConnectionProto | AsyncConnectionProto,
            thread_id: UUID,
            seen_runs: set[UUID],
        ) -> list[tuple[UUID, asyncio.Queue]]:
            """Subscribe to the thread stream, creating queues for unseen runs."""
            stream_manager = get_stream_manager()
            queues = []

            # Create new queues only for runs not yet seen
            thread_id = _ensure_uuid(thread_id)

            # Add thread stream queue
            if thread_id not in seen_runs:
                queue = await stream_manager.add_thread_stream(thread_id)
                queues.append((thread_id, queue))
                seen_runs.add(thread_id)

            for run in conn.store["runs"]:
                if run["thread_id"] == thread_id:
                    run_id = run["run_id"]
                    if run_id not in seen_runs:
                        queue = await stream_manager.add_queue(run_id, thread_id)
                        queues.append((run_id, queue))
                        seen_runs.add(run_id)

            return queues

        @staticmethod
        async def join(
            thread_id: UUID,
            *,
            last_event_id: str | None = None,
            stream_modes: list[ThreadStreamMode],
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> AsyncIterator[tuple[bytes, bytes, bytes | None]]:
            async for (
                event,
                payload,
                stream_id,
                _run_id,
            ) in Threads.Stream.join_event_streaming(
                thread_id,
                last_event_id=last_event_id,
                stream_modes=stream_modes,
                ctx=ctx,
            ):
                yield event, payload, stream_id

        @staticmethod
        async def join_event_streaming(
            thread_id: UUID,
            *,
            last_event_id: str | None = None,
            stream_modes: list[ThreadStreamMode],
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> AsyncIterator[tuple[bytes, bytes, bytes | None, str | None]]:
            """Stream the thread output."""
            await Threads.Stream.check_thread_stream_auth(thread_id, ctx)

            from langgraph_api.utils.stream_codec import (  # noqa: PLC0415
                decode_stream_message,
            )

            def should_filter_event(event_name: str, message_bytes: bytes) -> bool:
                """Check if an event should be filtered out based on stream_modes."""
                if "run_modes" in stream_modes and event_name != "state_update":
                    return False
                if "state_update" in stream_modes and event_name == "state_update":
                    return False
                if "lifecycle" in stream_modes and event_name == "metadata":
                    try:
                        message_data = orjson.loads(message_bytes)
                        if message_data.get("status") == "run_done":
                            return False
                        if "attempt" in message_data and "run_id" in message_data:
                            return False
                    except (orjson.JSONDecodeError, TypeError):
                        pass
                return True

            stream_manager = get_stream_manager()
            seen_runs: set[UUID] = set()
            created_queues: list[tuple[UUID, asyncio.Queue]] = []

            try:
                async with connect() as conn:
                    await logger.ainfo(
                        "Joined thread stream",
                        thread_id=str(thread_id),
                    )

                    # Restore messages if resuming from a specific event
                    if last_event_id is not None:
                        # ``message_stores`` is keyed by ``UUID`` (see
                        # :meth:`StreamManager.put`). Callers can hand us
                        # ``thread_id`` as either ``str`` or ``UUID``, so
                        # normalize before the lookup — otherwise replay
                        # always misses and yields nothing.
                        store_key = _ensure_uuid(thread_id)
                        all_events = []
                        for run_id in stream_manager.message_stores.get(store_key, []):
                            for message in stream_manager.restore_messages(
                                run_id, thread_id, last_event_id
                            ):
                                all_events.append((message, run_id))

                        # Sort by message ID (which is ms-seq format)
                        all_events.sort(key=lambda x: x[0].id.decode())

                        # Yield sorted events
                        for message, run_id in all_events:
                            decoded = decode_stream_message(
                                message.data, channel=message.topic
                            )
                            event_bytes = decoded.event_bytes
                            message_bytes = decoded.message_bytes

                            if event_bytes == b"control":
                                if message_bytes == b"done":
                                    event_bytes = b"metadata"
                                    message_bytes = orjson.dumps(
                                        {"status": "run_done", "run_id": run_id}
                                    )
                            if not should_filter_event(
                                event_bytes.decode("utf-8"), message_bytes
                            ):
                                yield (
                                    event_bytes,
                                    message_bytes,
                                    message.id,
                                    str(run_id),
                                )

                    # Listen for live messages from all queues.
                    #
                    # Hot loop is non-blocking: a burst of N events drains in
                    # one outer-loop iteration via ``get_nowait``, instead of
                    # the previous "one event per queue per 200ms timeout"
                    # pattern that throttled fast-publishing runs (the empty
                    # thread-stream queue alone forced every iteration to
                    # wait the full timeout). When everything is idle we fall
                    # back to a short ``asyncio.sleep`` so new runs joining
                    # the thread get picked up by the next ``subscribe``
                    # without burning CPU.
                    while True:
                        # Refresh queues to pick up any new runs that joined this thread
                        new_queue_tuples = await Threads.Stream.subscribe(
                            conn, thread_id, seen_runs
                        )
                        # Track new queues for cleanup
                        for run_id, queue in new_queue_tuples:
                            created_queues.append((run_id, queue))

                        drained_any = False
                        for run_id, queue in created_queues:
                            while True:
                                try:
                                    message = queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                                try:
                                    decoded = decode_stream_message(
                                        message.data, channel=message.topic
                                    )
                                except (ValueError, KeyError):
                                    continue
                                event = decoded.event_bytes
                                event_name = event.decode("utf-8")
                                payload = decoded.message_bytes

                                if event == b"control" and payload == b"done":
                                    # Don't shadow the queue-iteration
                                    # ``run_id`` with the topic-extracted
                                    # string — non-control events later in
                                    # this drain pass would yield the
                                    # rebound value. Wire output is
                                    # identical (``str(UUID)`` matches the
                                    # topic suffix), but the rebinding is
                                    # fragile if the topic format moves.
                                    topic = message.topic.decode()
                                    done_run_id = topic.split("run:")[1].split(":")[0]
                                    meta_event = b"metadata"
                                    meta_payload = orjson.dumps(
                                        {"status": "run_done", "run_id": done_run_id}
                                    )
                                    if not should_filter_event(
                                        "metadata", meta_payload
                                    ):
                                        yield (
                                            meta_event,
                                            meta_payload,
                                            message.id,
                                            done_run_id,
                                        )
                                        drained_any = True
                                else:
                                    if not should_filter_event(event_name, payload):
                                        yield (
                                            event,
                                            payload,
                                            message.id,
                                            str(run_id),
                                        )
                                        drained_any = True

                        if drained_any:
                            # Yield once so other tasks (worker, send) can
                            # advance, then loop immediately to drain any
                            # follow-up burst.
                            await asyncio.sleep(0)
                        else:
                            # All queues empty — short poll interval keeps
                            # ``subscribe`` rechecking for newly-spawned
                            # runs and lets the worker emit without a
                            # >5-events-per-second delivery cap.
                            await asyncio.sleep(0.02)

            except WrappedHTTPException as e:
                raise e.http_exception from None
            except asyncio.CancelledError:
                await logger.awarning(
                    "Thread stream client disconnected",
                    thread_id=str(thread_id),
                )
                raise
            except:
                raise
            finally:
                # Clean up all created queues
                for run_id, queue in created_queues:
                    try:
                        await stream_manager.remove_queue(run_id, thread_id, queue)
                    except Exception:
                        # Ignore cleanup errors
                        pass

        @staticmethod
        async def publish(
            thread_id: UUID | str,
            event: str,
            message: bytes,
        ) -> None:
            """Publish a thread-level event to the thread stream."""
            from langgraph_api.utils.stream_codec import STREAM_CODEC  # noqa: PLC0415

            topic = f"thread:{thread_id}:stream".encode()

            stream_manager = get_stream_manager()
            payload = STREAM_CODEC.encode(event, message)
            await stream_manager.put_thread(
                str(thread_id), Message(topic=topic, data=payload)
            )

        @staticmethod
        async def check_thread_stream_auth(
            thread_id: UUID,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> None:
            async with connect() as conn:
                filters = await Threads.Stream.handle_event(
                    ctx,
                    "read",
                    Auth.types.ThreadsRead(thread_id=thread_id),
                )
                if filters:
                    thread = await Threads._get_with_filters(
                        cast(InMemConnectionProto, conn), thread_id, filters
                    )
                    if not thread:
                        raise HTTPException(status_code=404, detail="Thread not found")

    @staticmethod
    async def count(
        conn: InMemConnectionProto,
        *,
        metadata: MetadataInput = None,
        values: MetadataInput = None,
        status: ThreadStatus | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> int:
        """Get count of threads."""
        threads = conn.store["threads"]
        metadata = metadata if metadata is not None else {}
        values = values if values is not None else {}
        filters = await Threads.handle_event(
            ctx,
            "search",
            Auth.types.ThreadsSearch(
                metadata=metadata,
                values=values,
                status=status,
                limit=0,
                offset=0,
            ),
        )

        count = 0
        for thread in threads:
            if filters and not _check_filter_match(thread["metadata"], filters):
                continue

            if metadata and not is_jsonb_contained(thread["metadata"], metadata):
                continue

            if (
                values
                and "values" in thread
                and not is_jsonb_contained(thread["values"], values)
            ):
                continue

            if status and thread.get("status") != status:
                continue

            count += 1

        return count


RUN_LOCK = asyncio.Lock()


class Runs(Authenticated):
    resource = "threads"

    @staticmethod
    async def stats(conn: InMemConnectionProto) -> QueueStats:
        """Get stats about the queue."""
        pending_runs = [run for run in conn.store["runs"] if run["status"] == "pending"]
        running_runs = [run for run in conn.store["runs"] if run["status"] == "running"]

        if not pending_runs and not running_runs:
            return {
                "n_pending": 0,
                "pending_runs_wait_time_max_secs": None,
                "pending_runs_wait_time_med_secs": None,
                "pending_unblocked_runs_wait_time_max_secs": None,
                "n_running": 0,
            }

        now = datetime.now(UTC)
        pending_waits: list[float] = []
        for run in pending_runs:
            created_at = run.get("created_at")
            if not isinstance(created_at, datetime):
                continue
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            pending_waits.append((now - created_at).total_seconds())

        max_pending_wait = max(pending_waits) if pending_waits else None
        if pending_waits:
            sorted_waits = sorted(pending_waits)
            half = len(sorted_waits) // 2
            if len(sorted_waits) % 2 == 1:
                med_pending_wait = sorted_waits[half]
            else:
                med_pending_wait = (sorted_waits[half - 1] + sorted_waits[half]) / 2
        else:
            med_pending_wait = None

        # Calculate max wait time for unblocked runs (runs not blocked by another run on the same thread)
        pending_unblocked_waits: list[float] = []
        for run in pending_runs:
            thread_id = run.get("thread_id")
            # Check if there's a running run on the same thread
            has_running_on_thread = any(
                r.get("thread_id") == thread_id and r.get("status") == "running"
                for r in conn.store["runs"]
            )
            if not has_running_on_thread:
                created_at = run.get("created_at")
                if not isinstance(created_at, datetime):
                    continue
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                if created_at < now:
                    pending_unblocked_waits.append((now - created_at).total_seconds())

        max_unblocked_wait = (
            max(pending_unblocked_waits) if pending_unblocked_waits else None
        )

        return {
            "n_pending": len(pending_runs),
            "n_running": len(running_runs),
            "pending_runs_wait_time_max_secs": max_pending_wait,
            "pending_runs_wait_time_med_secs": med_pending_wait,
            "pending_unblocked_runs_wait_time_max_secs": max_unblocked_wait,
        }

    @staticmethod
    async def pool_stats() -> PoolStats:
        """This method is for fetching the grpc pool stats, which don't exist for inmem, so we return empty dict"""
        return {}

    @staticmethod
    async def next(wait: bool, limit: int = 1) -> AsyncIterator[tuple[Run, int]]:
        """Get the next run from the queue, and the attempt number.
        1 is the first attempt, 2 is the first retry, etc."""
        now = datetime.now(UTC)

        if wait:
            await asyncio.sleep(0.5)
        else:
            await asyncio.sleep(0)

        async with connect() as conn, RUN_LOCK:
            pending_runs = sorted(
                [
                    run
                    for run in conn.store["runs"]
                    if run["status"] == "pending" and run.get("created_at", now) < now
                ],
                key=lambda x: x.get("created_at", datetime.min),
            )

            if not pending_runs:
                return

            # Try to lock and get the first available run
            for _, run in zip(range(limit), pending_runs, strict=False):
                if run["status"] != "pending":
                    continue

                run_id = run["run_id"]
                thread_id = run["thread_id"]
                thread = next(
                    (t for t in conn.store["threads"] if t["thread_id"] == thread_id),
                    None,
                )

                if thread is None:
                    await logger.awarning(
                        "Unexpected missing thread in Runs.next",
                        thread_id=run["thread_id"],
                    )
                    continue

                if run["status"] != "pending":
                    continue

                if any(
                    run["status"] == "running"
                    for run in conn.store["runs"]
                    if run["thread_id"] == thread_id
                ):
                    continue
                # Increment attempt counter
                attempt = await conn.retry_counter.increment(run_id)
                # Set run as "running"
                run["status"] = "running"
                yield run, attempt

    @asynccontextmanager
    @staticmethod
    async def enter(
        run_id: UUID,
        thread_id: UUID | None,
        loop: asyncio.AbstractEventLoop,
        resumable: bool,
    ) -> AsyncIterator[ValueEvent]:
        """Enter a run, listen for cancellation while running, signal when done."
        This method should be called as a context manager by a worker executing a run.
        """
        from langgraph_api.asyncio import SimpleTaskGroup, ValueEvent  # noqa: PLC0415
        from langgraph_api.utils.stream_codec import STREAM_CODEC  # noqa: PLC0415

        stream_manager = get_stream_manager()
        # Get control queue for this run (normal queue is created during run creation)
        control_queue = await stream_manager.add_control_queue(run_id, thread_id)

        async with SimpleTaskGroup(cancel=True, taskgroup_name="Runs.enter") as tg:
            done = ValueEvent()
            tg.create_task(
                listen_for_cancellation(control_queue, run_id, thread_id, done)
            )

            # Give done event to caller
            yield done
            # Store the control message for late subscribers
            control_message = Message(
                topic=f"run:{run_id}:control".encode(), data=b"done"
            )
            await stream_manager.put(run_id, thread_id, control_message)

            # Signal done to all subscribers using stream codec
            stream_message = Message(
                topic=f"run:{run_id}:stream".encode(),
                data=STREAM_CODEC.encode("control", b"done"),
            )
            await stream_manager.put(
                run_id, thread_id, stream_message, resumable=resumable
            )

            # Remove the control_queue (normal queue is cleaned up during run deletion)
            await stream_manager.remove_control_queue(run_id, thread_id, control_queue)

    @staticmethod
    async def sweep() -> None:
        """Sweep runs that are no longer running"""
        pass

    @staticmethod
    def _merge_jsonb(*objects: dict) -> dict:
        """Mimics PostgreSQL's JSONB merge behavior"""
        result = {}
        for obj in objects:
            if obj is not None:
                result.update(copy.deepcopy(obj))
        return result

    @staticmethod
    def _get_configurable(config: dict) -> dict:
        """Extract configurable from config, mimicking PostgreSQL's coalesce"""
        return config.get("configurable", {})

    @staticmethod
    async def put(
        conn: InMemConnectionProto | AsyncConnectionProto,
        assistant_id: UUID,
        kwargs: dict,
        *,
        thread_id: UUID | None = None,
        user_id: str | None = None,
        run_id: UUID | None = None,
        status: RunStatus | None = "pending",
        metadata: MetadataInput,
        prevent_insert_if_inflight: bool,
        multitask_strategy: MultitaskStrategy = "reject",
        if_not_exists: IfNotExists = "reject",
        after_seconds: int = 0,
        ctx: Auth.types.BaseAuthContext | None = None,
        langsmith_session_name: str | None = None,
    ) -> AsyncIterator[Run]:
        """Create a run."""
        from langgraph_api.schema import Run, Thread  # noqa: PLC0415

        assistant_id = _ensure_uuid(assistant_id)
        assistant = next(
            (a for a in conn.store["assistants"] if a["assistant_id"] == assistant_id),
            None,
        )

        if not assistant:
            return _empty_generator()

        custom_encryption = _using_custom_encryption()
        if custom_encryption:
            assistant = await _decrypt_response(
                assistant, "assistant", ["metadata", "config", "context"]
            )
            kwargs = await _decrypt_response(
                kwargs, "run", ["input", "command", "config", "context"]
            )
            metadata = (
                await _decrypt_response({"metadata": metadata}, "run", ["metadata"])
            )["metadata"]

        thread_id = _ensure_uuid(thread_id) if thread_id else None
        run_id = _ensure_uuid(run_id) if run_id else None
        metadata = metadata if metadata is not None else {}
        config = kwargs.get("config", {})
        temporary = kwargs.get("temporary", False)

        # Handle thread creation/update
        existing_thread = next(
            (t for t in conn.store["threads"] if t["thread_id"] == thread_id), None
        )
        create_run_value = Auth.types.RunsCreate(
            thread_id=None if temporary else thread_id,
            assistant_id=assistant_id,
            run_id=run_id,
            status=status,
            metadata=metadata,
            prevent_insert_if_inflight=prevent_insert_if_inflight,
            multitask_strategy=multitask_strategy,
            if_not_exists=if_not_exists,
            after_seconds=after_seconds,
            kwargs=kwargs,
        )
        filters = await Runs.handle_event(
            ctx,
            "create_run",
            create_run_value,
        )
        # Re-fetch in case an auth handler replaced the thread object in the store
        # (e.g. via a loopback patch call, which deep-copies and replaces the element).
        if thread_id is not None:
            existing_thread = next(
                (t for t in conn.store["threads"] if t["thread_id"] == thread_id), None
            )
        if custom_encryption and existing_thread:
            existing_thread = await _decrypt_response(
                existing_thread,
                "thread",
                ["metadata", "config"],
            )
        # Automatically enforce assistant ownership for non-system assistants
        # by calling the user's assistant read auth handler.
        if assistant.get("metadata", {}).get("created_by") != "system":
            assistant_request_data = Auth.types.AssistantsRead(
                assistant_id=assistant_id
            )
            assistant_filters = await Assistants.handle_event(
                ctx, "read", assistant_request_data
            )
            if assistant_filters and not _check_filter_match(
                assistant.get("metadata", {}), assistant_filters
            ):
                return _empty_generator()

        if existing_thread and filters:
            # Reject if the user doesn't own the thread
            if not _check_filter_match(existing_thread["metadata"], filters):
                return _empty_generator()

        if not existing_thread and (thread_id is None or if_not_exists == "create"):
            # Create new thread
            if thread_id is None:
                thread_id = uuid4()

            thread = Thread(
                thread_id=thread_id,
                status="busy",
                metadata={
                    "graph_id": assistant["graph_id"],
                    "assistant_id": str(assistant_id),
                    **(config.get("metadata") or {}),
                    **metadata,
                },
                config=Runs._merge_jsonb(
                    assistant["config"],
                    config,
                    {
                        "configurable": Runs._merge_jsonb(
                            Runs._get_configurable(assistant["config"]),
                        )
                    },
                ),
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                values=None,
            )

            await logger.ainfo("Creating thread", thread_id=thread_id)
            conn.store["threads"].append(thread)
        elif existing_thread:
            # Update existing thread
            if existing_thread["status"] != "busy":
                existing_thread["status"] = "busy"
                existing_thread["metadata"] = Runs._merge_jsonb(
                    existing_thread["metadata"],
                    {
                        "graph_id": assistant["graph_id"],
                        "assistant_id": str(assistant_id),
                    },
                )
                existing_thread["config"] = Runs._merge_jsonb(
                    assistant["config"],
                    existing_thread["config"],
                    config,
                    {
                        "configurable": Runs._merge_jsonb(
                            Runs._get_configurable(assistant["config"]),
                            Runs._get_configurable(existing_thread["config"]),
                        )
                    },
                )
                existing_thread["updated_at"] = datetime.now(UTC)
        else:
            return _empty_generator()

        # Check for inflight runs if needed
        inflight_runs = [
            r
            for r in conn.store["runs"]
            if r["thread_id"] == thread_id and r["status"] in ("pending", "running")
        ]
        if prevent_insert_if_inflight:
            if inflight_runs:

                async def _return_inflight():
                    for run in inflight_runs:
                        yield run

                return _return_inflight()

        # Create new run
        configurable = Runs._merge_jsonb(
            Runs._get_configurable(assistant["config"]),
            (
                Runs._get_configurable(existing_thread["config"])
                if existing_thread
                else {}
            ),
            Runs._get_configurable(config),
            {
                "run_id": str(run_id),
                "thread_id": str(thread_id),
                "graph_id": assistant["graph_id"],
                "assistant_id": str(assistant_id),
                "user_id": (
                    config.get("configurable", {}).get("user_id")
                    or (
                        existing_thread["config"].get("configurable", {}).get("user_id")
                        if existing_thread
                        else None
                    )
                    or assistant["config"].get("configurable", {}).get("user_id")
                    or user_id
                ),
            },
        )
        merged_metadata = Runs._merge_jsonb(
            assistant["metadata"],
            existing_thread["metadata"] if existing_thread else {},
            config.get("metadata") or {},
            metadata,
        )
        # Always overwrite assistant_id to prevent user spoofing
        merged_metadata["assistant_id"] = str(assistant_id)
        new_run = Run(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            metadata=merged_metadata,
            status=status,
            kwargs=Runs._merge_jsonb(
                kwargs,
                {
                    "config": Runs._merge_jsonb(
                        assistant["config"],
                        config,
                        {"configurable": configurable},
                        {
                            "metadata": merged_metadata,
                        },
                    ),
                    "context": Runs._merge_jsonb(
                        assistant.get("context", {}), kwargs.get("context", {})
                    ),
                },
            ),
            multitask_strategy=multitask_strategy,
            created_at=datetime.now(UTC) + timedelta(seconds=after_seconds),
            updated_at=datetime.now(UTC),
            langsmith_session_name=langsmith_session_name,
        )
        if custom_encryption:
            # Unlike Go core, in-memory workers read nested run kwargs and thread
            # config directly from the store, so those containers must stay readable.
            encrypted_thread = await _encrypt_request(
                thread if not existing_thread else existing_thread,
                "thread",
                ["metadata"],
            )
            stored_thread = next(
                t
                for t in conn.store["threads"]
                if t["thread_id"] == encrypted_thread["thread_id"]
            )
            stored_thread.update(encrypted_thread)
            new_run["kwargs"] = await _encrypt_request(
                new_run["kwargs"],
                "run",
                ["input", "config", "context", "command"],
            )
            new_run = Run(**await _encrypt_request(new_run, "run", ["metadata"]))
        conn.store["runs"].append(new_run)

        async def _yield_new():
            yield new_run
            for r in inflight_runs:
                yield r

        return _yield_new()

    @staticmethod
    async def get(
        conn: InMemConnectionProto,
        run_id: UUID,
        *,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Run]:
        """Get a run by ID."""

        run_id, thread_id = _ensure_uuid(run_id), _ensure_uuid(thread_id)
        filters = await Runs.handle_event(
            ctx,
            "read",
            Auth.types.ThreadsRead(thread_id=thread_id),
        )

        async def _yield_result():
            matching_run = None
            for run in conn.store["runs"]:
                if run["run_id"] == run_id and run["thread_id"] == thread_id:
                    matching_run = run
                    break
            if matching_run:
                if filters:
                    thread = await Threads._get_with_filters(
                        conn, matching_run["thread_id"], filters
                    )
                    if not thread:
                        return
                yield matching_run

        return _yield_result()

    @staticmethod
    async def delete(
        conn: InMemConnectionProto,
        run_id: UUID,
        *,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[UUID]:
        """Delete a run by ID."""
        run_id, thread_id = _ensure_uuid(run_id), _ensure_uuid(thread_id)
        filters = await Runs.handle_event(
            ctx,
            "delete",
            Auth.types.ThreadsDelete(run_id=run_id, thread_id=thread_id),
        )

        if filters:
            thread = await Threads._get_with_filters(conn, thread_id, filters)
            if not thread:
                return _empty_generator()
        await _delete_checkpoints_for_thread(thread_id, conn, run_id=run_id)

        found = False
        for i, run in enumerate(conn.store["runs"]):
            if run["run_id"] == run_id and run["thread_id"] == thread_id:
                del conn.store["runs"][i]
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Run not found")

        async def _yield_deleted():
            await logger.ainfo("Run deleted", run_id=run_id)
            yield run_id

        return _yield_deleted()

    @staticmethod
    async def cancel(
        conn: InMemConnectionProto | AsyncConnectionProto,
        run_ids: Sequence[UUID | str] | None = None,
        *,
        action: Literal["interrupt", "rollback"] = "interrupt",
        thread_id: UUID | None = None,
        status: Literal["pending", "running", "all"] | None = None,
        assistant_id: UUID | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> None:
        """
        Cancel runs in memory. Must provide either:
        1) thread_id + run_ids, or
        2) status in {"pending", "running", "all"}, or
        3) assistant_id (cancels all in-flight runs for that assistant).

        Steps:
        - Validate arguments (one usage pattern or the other).
        - Auth check: 'update' event via handle_event().
        - Gather runs matching either the (thread_id, run_ids) set or the given status.
        - For each run found:
            * Send a cancellation message through the stream manager.
            * If 'pending', set to 'interrupted' or delete (if action='rollback' and not actively queued).
            * If 'running', the worker will pick up the message.
            * Otherwise, log a warning for non-cancelable states.
        - 404 if no runs are found or authorized (unless assistant_id is provided).
        """
        # 1. Validate arguments
        if assistant_id is not None:
            # If assistant_id is set, user must NOT specify other filters
            if thread_id is not None or run_ids is not None or status is not None:
                raise HTTPException(
                    status_code=422,
                    detail="Cannot specify 'thread_id', 'run_ids', or 'status' when using 'assistant_id'",
                )
            assistant_id = _ensure_uuid(assistant_id)
        elif status is not None:
            # If status is set, user must NOT specify thread_id or run_ids
            if thread_id is not None or run_ids is not None:
                raise HTTPException(
                    status_code=422,
                    detail="Cannot specify 'thread_id' or 'run_ids' when using 'status'",
                )
        else:
            # If status is not set, user must specify both thread_id and run_ids
            if thread_id is None or run_ids is None:
                raise HTTPException(
                    status_code=422,
                    detail="Must provide either a status, an assistant_id, or both 'thread_id' and 'run_ids'",
                )

        # Convert and normalize inputs
        if run_ids is not None:
            run_ids = [_ensure_uuid(rid) for rid in run_ids]
        if thread_id is not None:
            thread_id = _ensure_uuid(thread_id)

        filters = await Runs.handle_event(
            ctx,
            "update",
            Auth.types.ThreadsUpdate(
                thread_id=thread_id,  # type: ignore
                action=action,
                metadata={
                    "run_ids": run_ids,
                    "status": status,
                },
            ),
        )

        status_list: tuple[str, ...] = ()
        if status is not None:
            if status == "all":
                status_list = ("pending", "running")
            elif status in ("pending", "running"):
                status_list = (status,)
            else:
                raise ValueError(f"Unsupported status: {status}")

        def is_run_match(r: dict) -> bool:
            """
            Check whether a run in `conn.store["runs"]` meets the selection criteria.
            """
            if assistant_id is not None:
                return r["assistant_id"] == assistant_id and r["status"] in (
                    "pending",
                    "running",
                )
            elif status_list:
                return r["status"] in status_list
            else:
                return r["thread_id"] == thread_id and r["run_id"] in run_ids  # type: ignore

        candidate_runs = [r for r in conn.store["runs"] if is_run_match(r)]

        if filters:
            if thread_id:
                thread = await Threads._get_with_filters(conn, thread_id, filters)
                if not thread:
                    candidate_runs = []
            else:
                candidate_runs = [
                    r
                    for r in candidate_runs
                    if await Threads._get_with_filters(conn, r["thread_id"], filters)
                ]

        if not candidate_runs:
            # When cancelling by assistant_id, it's valid to have no runs
            if assistant_id is not None:
                return
            raise HTTPException(status_code=404, detail="No runs found to cancel.")

        stream_manager = get_stream_manager()
        coros = []
        cancelable_runs = []

        for run in candidate_runs:
            run_id = run["run_id"]
            control_message = Message(
                topic=f"run:{run_id}:control".encode(),
                data=action.encode(),
            )
            coros.append(stream_manager.put(run_id, thread_id, control_message))

            queues = stream_manager.get_queues(run_id, thread_id)

            if run["status"] in ("pending", "running"):
                cancelable_runs.append(run)
                if queues or action != "rollback":
                    if run["status"] == "pending":
                        thread = next(
                            (
                                t
                                for t in conn.store["threads"]
                                if t["thread_id"] == run["thread_id"]
                            ),
                            None,
                        )
                        if thread:
                            thread["status"] = "idle"
                            thread["updated_at"] = datetime.now(tz=UTC)
                    run["status"] = "interrupted"
                    run["updated_at"] = datetime.now(tz=UTC)
                else:
                    await logger.ainfo(
                        "Eagerly deleting pending run with rollback action",
                        run_id=str(run_id),
                        status=run["status"],
                    )
                    coros.append(Runs.delete(conn, run_id, thread_id=run["thread_id"]))
            else:
                await logger.awarning(
                    "Attempted to cancel non-pending run.",
                    run_id=str(run_id),
                    status=run["status"],
                )

        if not cancelable_runs:
            # When cancelling by assistant_id, it's valid to have no cancelable runs
            if assistant_id is not None:
                return
            raise HTTPException(
                status_code=404,
                detail="No matching runs to cancel. Please verify the thread ID and run IDs are correct, and the runs haven't been deleted or completed.",
            )

        if coros:
            await asyncio.gather(*coros)

        await logger.ainfo(
            "Cancelled runs",
            run_ids=[str(r["run_id"]) for r in cancelable_runs],
            thread_id=str(thread_id) if thread_id else None,
            status=status,
            action=action,
        )

    @staticmethod
    async def search(
        conn: InMemConnectionProto,
        thread_id: UUID,
        *,
        limit: int = 10,
        offset: int = 0,
        status: RunStatus | None = None,
        select: list[RunSelectField] | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Run]:
        """List all runs by thread."""
        runs = conn.store["runs"]
        metadata = {}
        thread_id = _ensure_uuid(thread_id)
        filters = await Runs.handle_event(
            ctx,
            "search",
            Auth.types.ThreadsSearch(thread_id=thread_id, metadata=metadata),
        )
        filtered_runs = [
            run
            for run in runs
            if run["thread_id"] == thread_id
            and is_jsonb_contained(run["metadata"], metadata)
            and (
                not filters
                or (await Threads._get_with_filters(conn, thread_id, filters))
            )
            and (status is None or run["status"] == status)
        ]
        sorted_runs = sorted(filtered_runs, key=lambda x: x["created_at"], reverse=True)
        sliced_runs = sorted_runs[offset : offset + limit]

        async def _return():
            for run in sliced_runs:
                if select:
                    # Filter to only selected fields
                    filtered_run = {k: v for k, v in run.items() if k in select}
                    yield filtered_run
                else:
                    yield run

        return _return()

    @staticmethod
    async def set_status(
        conn: InMemConnectionProto, run_id: UUID, status: RunStatus
    ) -> None:
        """Set the status of a run."""
        # Find the run in the store
        run_id = _ensure_uuid(run_id)
        run = next((run for run in conn.store["runs"] if run["run_id"] == run_id), None)

        if run:
            # Update the status and updated_at timestamp
            run["status"] = status
            run["updated_at"] = datetime.now(tz=UTC)
            return run
        return None

    class Stream:
        @staticmethod
        async def subscribe(
            run_id: UUID,
            thread_id: UUID | None = None,
        ) -> ContextQueue:
            """Subscribe to the run stream, returning a queue."""
            stream_manager = get_stream_manager()
            queue = await stream_manager.add_queue(_ensure_uuid(run_id), thread_id)

            # If there's a control message already stored, send it to the new subscriber
            if thread_id is None:
                thread_id = THREADLESS_KEY
            if control_queues := stream_manager.control_queues.get(thread_id, {}).get(
                run_id
            ):
                for control_queue in control_queues:
                    # NOTE: must use ``get_nowait``. ``asyncio.Queue.get`` is a
                    # coroutine — calling it without ``await`` returns a
                    # coroutine object and never raises ``QueueEmpty``, which
                    # turns this drain into an infinite loop that blocks the
                    # event loop (the coroutine objects get pushed straight
                    # into ``queue`` via ``put_nowait`` with no yield point).
                    while True:
                        try:
                            control_msg = control_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        await queue.put(control_msg)
            return queue

        @staticmethod
        async def join(
            run_id: UUID,
            *,
            stream_channel: asyncio.Queue,
            thread_id: UUID,
            ignore_404: bool = False,
            cancel_on_disconnect: bool = False,
            stream_mode: list[StreamMode] | StreamMode | None = None,
            last_event_id: str | None = None,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> AsyncIterator[tuple[bytes, bytes, bytes | None]]:
            """Stream the run output."""
            from langgraph_api.asyncio import create_task  # noqa: PLC0415
            from langgraph_api.serde import json_dumpb  # noqa: PLC0415
            from langgraph_api.utils.stream_codec import (  # noqa: PLC0415
                decode_stream_message,
            )

            queue = stream_channel
            try:
                async with connect() as conn:
                    try:
                        await Runs.Stream.check_run_stream_auth(run_id, thread_id, ctx)
                    except HTTPException as e:
                        raise WrappedHTTPException(e) from None
                    run_iter = await Runs.get(
                        conn, run_id, thread_id=thread_id, ctx=ctx
                    )
                    run = await anext(run_iter, None)

                    for message in get_stream_manager().restore_messages(
                        run_id, thread_id, last_event_id
                    ):
                        data, id = message.data, message.id
                        decoded = decode_stream_message(data, channel=message.topic)
                        mode = decoded.event_bytes.decode("utf-8")
                        payload = decoded.message_bytes

                        if mode == "control":
                            if payload == b"done":
                                return
                        elif _run_stream_mode_matches(mode, stream_mode):
                            yield mode.encode(), payload, id
                            logger.debug(
                                "Replayed run event",
                                run_id=str(run_id),
                                message_id=id,
                                stream_mode=mode,
                                data=data,
                            )

                    while True:
                        try:
                            # Wait for messages with a timeout
                            message = await asyncio.wait_for(queue.get(), timeout=0.5)
                            data, id = message.data, message.id
                            decoded = decode_stream_message(data, channel=message.topic)
                            mode = decoded.event_bytes.decode("utf-8")
                            payload = decoded.message_bytes

                            if mode == "control":
                                if payload == b"done":
                                    break
                            elif _run_stream_mode_matches(mode, stream_mode):
                                # We only return a stream ID if the run is resumable
                                stream_id = (
                                    id
                                    if run.get("kwargs", {}).get("resumable")
                                    else None
                                )
                                yield mode.encode(), payload, stream_id
                                logger.debug(
                                    "Streamed run event",
                                    run_id=str(run_id),
                                    stream_mode=mode,
                                    message_id=id,
                                    data=payload,
                                )
                        except TimeoutError:
                            # Check if the run is still pending
                            run_iter = await Runs.get(
                                conn, run_id, thread_id=thread_id, ctx=ctx
                            )
                            run = await anext(run_iter, None)

                            if ignore_404 and run is None:
                                break
                            elif run is None:
                                yield (
                                    b"error",
                                    json_dumpb(
                                        HTTPException(
                                            status_code=404, detail="Run not found"
                                        )
                                    ),
                                    None,
                                )
                                break
                            elif run["status"] not in ("pending", "running"):
                                break
            except WrappedHTTPException as e:
                raise e.http_exception from None
            except:
                if cancel_on_disconnect:
                    create_task(cancel_run(thread_id, run_id))
                raise
            finally:
                stream_manager = get_stream_manager()
                await stream_manager.remove_queue(run_id, thread_id, queue)

        @staticmethod
        async def check_run_stream_auth(
            run_id: UUID,
            thread_id: UUID,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> None:
            async with connect() as conn:
                filters = await Runs.handle_event(
                    ctx,
                    "read",
                    Auth.types.ThreadsRead(thread_id=thread_id),
                )
                if filters:
                    thread = await Threads._get_with_filters(
                        cast(InMemConnectionProto, conn), thread_id, filters
                    )
                    if not thread:
                        raise HTTPException(status_code=404, detail="Thread not found")

        @staticmethod
        async def publish(
            run_id: UUID | str,
            event: str,
            message: bytes,
            *,
            thread_id: UUID | str | None = None,
            resumable: bool = False,
        ) -> None:
            """Publish a message to all subscribers of the run stream."""
            from langgraph_api.utils.stream_codec import STREAM_CODEC  # noqa: PLC0415

            topic = f"run:{run_id}:stream".encode()

            stream_manager = get_stream_manager()
            # Send to all queues subscribed to this run_id using protocol frame
            payload = STREAM_CODEC.encode(event, message)
            await stream_manager.put(
                run_id, thread_id, Message(topic=topic, data=payload), resumable
            )


async def listen_for_cancellation(
    queue: asyncio.Queue, run_id: UUID, thread_id: UUID | None, done: ValueEvent
):
    """Listen for cancellation messages and set the done event accordingly."""
    from langgraph_api.errors import UserInterrupt, UserRollback  # noqa: PLC0415

    stream_manager = get_stream_manager()

    if control_key := stream_manager.get_control_key(run_id, thread_id):
        payload = control_key.data
        if payload == b"rollback":
            done.set(UserRollback())
        elif payload == b"interrupt":
            done.set(UserInterrupt())

    while not done.is_set():
        try:
            # This task gets cancelled when Runs.enter exits anyway,
            # so we can have a pretty lengthy timeout here
            message = await asyncio.wait_for(queue.get(), timeout=240)
            payload = message.data
            if payload == b"rollback":
                done.set(UserRollback())
            elif payload == b"interrupt":
                done.set(UserInterrupt())
            elif payload == b"done":
                done.set()
                break
        except TimeoutError:
            break


class Crons(Authenticated):
    resource = "crons"

    @staticmethod
    def _validate_cron_schedule_or_throw(schedule: str) -> None:
        """Validate cron schedule format and raise HTTPException if invalid."""
        if not croniter_mod.croniter.is_valid(schedule):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid cron schedule: '{schedule}'. Reason: Invalid cron schedule. "
                    "Ensure the schedule uses the standard cron format (minute hour day_of_month month day_of_week). "
                    "Example: '*/5 * * * *' for every 5 minutes."
                ),
            )

    @staticmethod
    async def put(
        conn: InMemConnectionProto,
        *,
        payload: dict,
        schedule: str,
        cron_id: UUID | None = None,
        thread_id: UUID | None = None,
        on_run_completed: Literal["delete", "keep"] | None = None,
        end_time: datetime | None = None,
        metadata: dict | None = None,
        enabled: bool,
        timezone: str | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Cron]:
        from langgraph_api.graph import get_assistant_id  # noqa: PLC0415
        from langgraph_api.utils import (  # noqa: PLC0415
            get_auth_ctx,
            next_cron_date,
            uuid7,
        )

        ctx = ctx or get_auth_ctx()
        user_id = ctx.user.identity if ctx is not None else None
        cron_id = cron_id or uuid7()

        try:
            thread_id = UUID(str(thread_id)) if thread_id else None
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid thread ID {thread_id}. Expected a UUID.",
            ) from None

        if thread_id is not None:
            effective_on_run_completed = None
        else:
            effective_on_run_completed = on_run_completed or "delete"

        metadata = metadata if metadata is not None else {}
        payload = payload if payload is not None else {}
        config = payload.get("config")
        if config is None:
            config = {}
            payload["config"] = config
        configurable = config.get("configurable")
        if configurable is None:
            configurable = {}
            config["configurable"] = configurable
        configurable["cron_id"] = str(cron_id)

        cron_request_data = Auth.types.CronsCreate(
            payload=payload,
            schedule=schedule,
            cron_id=cron_id,
            thread_id=thread_id,
            user_id=user_id,
            end_time=end_time,
        )
        cron_request_data["metadata"] = metadata  # type: ignore
        filters = await Crons.handle_event(ctx, "create", cron_request_data)

        Crons._validate_cron_schedule_or_throw(schedule)

        assistant_id = get_assistant_id(payload["assistant_id"])
        payload["assistant_id"] = assistant_id

        from langgraph_api.graph import SYSTEM_ASSISTANT_IDS  # noqa: PLC0415

        assistant_filters: list[Any] = []
        if assistant_id not in SYSTEM_ASSISTANT_IDS:
            # Do not populate `metadata` on the AssistantsRead request value:
            # client-supplied cron metadata must not influence the auth
            # decision (defense vs. forged payloads), and the gRPC/postgres
            # path also omits it — keeping handler behavior consistent across
            # runtimes and mirroring Runs.put.
            assistant_request_data = Auth.types.AssistantsRead(
                assistant_id=payload["assistant_id"]
            )
            assistant_filters = await Assistants.handle_event(
                ctx, "read", assistant_request_data
            )

        # Validate assistant exists
        assistant = next(
            (
                a
                for a in conn.store["assistants"]
                if str(a["assistant_id"]) == str(assistant_id)
            ),
            None,
        )
        if not assistant:
            raise HTTPException(
                status_code=404,
                detail=f"Assistant '{assistant_id}' not found",
            )
        if assistant_filters and not _check_filter_match(
            assistant.get("metadata", {}), assistant_filters
        ):
            raise HTTPException(
                status_code=404,
                detail=f"Assistant '{assistant_id}' not found",
            )

        # Validate thread exists if provided
        if thread_id is not None:
            # Get thread-specific auth filters
            thread_request_data = Auth.types.ThreadsRead(thread_id=thread_id)
            thread_request_data["metadata"] = metadata  # type: ignore
            thread_filters = await Threads.handle_event(
                ctx, "read", thread_request_data
            )

            thread = next(
                (
                    t
                    for t in conn.store["threads"]
                    if str(t["thread_id"]) == str(thread_id)
                ),
                None,
            )
            if not thread:
                raise HTTPException(
                    status_code=404,
                    detail=f"Thread with ID '{thread_id}' not found. Please verify the ID is correct and the thread hasn't been deleted or expired.",
                )
            if thread_filters and not _check_filter_match(
                thread.get("metadata", {}), thread_filters
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Thread with ID '{thread_id}' not found. Please verify the ID is correct and the thread hasn't been deleted or expired.",
                )

        # Check if cron already exists (ON CONFLICT DO NOTHING equivalent)
        existing_cron = next(
            (c for c in conn.store["crons"] if str(c["cron_id"]) == str(cron_id)),
            None,
        )
        if existing_cron:
            if filters and not _check_filter_match(
                existing_cron.get("metadata", {}), filters
            ):

                async def _empty():
                    return
                    yield  # type: ignore[misc]

                return _empty()

            async def _yield_existing():
                yield existing_cron

            return _yield_existing()

        now = datetime.now(UTC)
        new_cron: Cron = {
            "cron_id": cron_id,
            "assistant_id": UUID(str(assistant_id)),
            "thread_id": thread_id,
            "user_id": user_id,
            "end_time": end_time,
            "schedule": schedule,
            "timezone": timezone,
            "payload": payload,
            "next_run_date": next_cron_date(schedule, now, timezone=timezone),
            "metadata": metadata,
            "on_run_completed": effective_on_run_completed,
            "enabled": enabled,
            "created_at": now,
            "updated_at": now,
        }
        conn.store["crons"].append(new_cron)

        async def _yield_new():
            yield new_cron

        return _yield_new()

    @staticmethod
    async def update(
        conn: InMemConnectionProto,
        *,
        cron_id: UUID,
        schedule: str | None = None,
        end_time: datetime | None = None,
        clear_end_time: bool = False,
        enabled: bool | None = None,
        on_run_completed: Literal["delete", "keep"] | None = None,
        payload: dict | None = None,
        metadata: dict | None = None,
        timezone: str | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Cron]:
        from langgraph_api.utils import get_auth_ctx, next_cron_date  # noqa: PLC0415

        ctx = ctx or get_auth_ctx()
        request_data = Auth.types.CronsUpdate(
            cron_id=cron_id,
            schedule=schedule,
            end_time=end_time,
            enabled=enabled,
            on_run_completed=on_run_completed,
            payload=payload,
        )
        if metadata is not None:
            request_data["metadata"] = metadata
        filters = await Crons.handle_event(ctx, "update", request_data)

        # Check if anything to update
        has_updates = clear_end_time or any(
            v is not None
            for v in [
                schedule,
                end_time,
                enabled,
                on_run_completed,
                payload,
                metadata,
                timezone,
            ]
        )
        if not has_updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        cron = next(
            (c for c in conn.store["crons"] if str(c["cron_id"]) == str(cron_id)),
            None,
        )
        if not cron:
            raise HTTPException(
                status_code=404,
                detail=f"Cron '{cron_id}' not found",
            )
        if filters and not _check_filter_match(cron.get("metadata", {}), filters):
            raise HTTPException(
                status_code=404,
                detail=f"Cron '{cron_id}' not found",
            )

        if timezone is not None:
            cron["timezone"] = timezone

        if schedule is not None:
            Crons._validate_cron_schedule_or_throw(schedule)
            cron["schedule"] = schedule
            cron["next_run_date"] = next_cron_date(
                schedule, datetime.now(UTC), timezone=cron.get("timezone")
            )
        elif timezone is not None:
            # Timezone changed but schedule didn't — recompute next_run_date
            cron["next_run_date"] = next_cron_date(
                cron["schedule"], datetime.now(UTC), timezone=timezone
            )

        if clear_end_time:
            cron["end_time"] = None
        elif end_time is not None:
            cron["end_time"] = end_time

        if enabled is not None:
            cron["enabled"] = enabled

        if on_run_completed is not None:
            cron["on_run_completed"] = on_run_completed

        if metadata is not None:
            cron["metadata"] = {**cron.get("metadata", {}), **metadata}

        if payload is not None:
            # Shallow merge payload, preserve assistant_id and config.configurable.cron_id
            existing_payload = cron.get("payload") or {}
            merged = {**existing_payload, **payload}
            # Preserve assistant_id from existing
            merged["assistant_id"] = existing_payload.get(
                "assistant_id", merged.get("assistant_id")
            )
            # Ensure config.configurable.cron_id is preserved
            merged_config = merged.get("config") or {}
            merged_configurable = merged_config.get("configurable") or {}
            merged_configurable["cron_id"] = str(cron_id)
            merged_config["configurable"] = merged_configurable
            merged["config"] = merged_config
            cron["payload"] = merged

        cron["updated_at"] = datetime.now(UTC)

        async def _yield_updated():
            yield cron

        return _yield_updated()

    @staticmethod
    async def get(
        conn: InMemConnectionProto,
        cron_id: UUID | str,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Cron]:
        """Get a cron by ID."""
        cron_id = _ensure_uuid(cron_id)
        filters = await Crons.handle_event(
            ctx,
            "read",
            Auth.types.CronsRead(cron_id=cron_id),
        )

        async def _yield_result():
            for cron in conn.store["crons"]:
                if str(cron["cron_id"]) == str(cron_id) and (
                    not filters
                    or _check_filter_match(cron.get("metadata", {}), filters)
                ):
                    yield copy.deepcopy(cron)

        return _yield_result()

    @staticmethod
    async def delete(
        conn: InMemConnectionProto,
        cron_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[UUID]:
        filters = await Crons.handle_event(
            ctx,
            "delete",
            Auth.types.CronsDelete(cron_id=cron_id),
        )

        original_len = len(conn.store["crons"])
        if filters:
            conn.store["crons"] = [
                c
                for c in conn.store["crons"]
                if not (
                    str(c["cron_id"]) == str(cron_id)
                    and _check_filter_match(c.get("metadata", {}), filters)
                )
            ]
        else:
            conn.store["crons"] = [
                c for c in conn.store["crons"] if str(c["cron_id"]) != str(cron_id)
            ]

        deleted = original_len > len(conn.store["crons"])

        async def _yield_deleted():
            if deleted:
                yield cron_id

        return _yield_deleted()

    @staticmethod
    async def next(
        conn: InMemConnectionProto,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Cron]:
        now = datetime.now(UTC)
        for cron in conn.store["crons"]:
            if not cron.get("enabled", False):
                continue
            if cron.get("end_time") is not None and cron["end_time"] < now:
                continue
            if cron["next_run_date"] > now:
                continue
            yield {**cron, "now": now}

    @staticmethod
    async def set_next_run_date(
        conn: InMemConnectionProto,
        cron_id: UUID,
        next_run_date: datetime,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> None:
        for cron in conn.store["crons"]:
            if str(cron["cron_id"]) == str(cron_id):
                cron["next_run_date"] = next_run_date
                return

    @staticmethod
    async def search(
        conn: InMemConnectionProto,
        *,
        assistant_id: UUID | None,
        thread_id: UUID | None,
        enabled: bool | None,
        limit: int,
        offset: int,
        select: list[CronSelectField] | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
        sort_by: str | None = None,
        sort_order: Literal["asc", "desc"] | None = None,
        metadata: dict | None = None,
    ) -> tuple[AsyncIterator[Cron], int | None]:
        filters = await Crons.handle_event(
            ctx,
            "search",
            {
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "limit": limit,
                "offset": offset,
                "metadata": metadata or {},
            },
        )

        if thread_id:
            thread_filters = await Threads.handle_event(
                ctx,
                "read",
                Auth.types.ThreadsRead(thread_id=thread_id),
            )
        else:
            thread_filters = await Threads.handle_event(
                ctx,
                "search",
                Auth.types.ThreadsSearch(),
            )

        crons = conn.store["crons"]
        # First pass: filter on cron-level criteria
        filtered_crons = [
            c
            for c in crons
            if (assistant_id is None or str(c["assistant_id"]) == str(assistant_id))
            and (thread_id is None or str(c.get("thread_id")) == str(thread_id))
            and (enabled is None or c.get("enabled") == enabled)
            and (not metadata or is_jsonb_contained(c.get("metadata", {}), metadata))
            and (not filters or _check_filter_match(c.get("metadata", {}), filters))
        ]

        # Second pass: apply thread-level auth filters
        # Crons without a thread_id are exempt from thread filtering.
        if thread_filters:
            # Build lookup only for threads referenced by matching crons
            cron_thread_ids = {
                str(c["thread_id"])
                for c in filtered_crons
                if c.get("thread_id") is not None
            }
            threads_by_id = {
                str(t["thread_id"]): t
                for t in conn.store["threads"]
                if str(t["thread_id"]) in cron_thread_ids
            }
            filtered_crons = [
                c
                for c in filtered_crons
                if c.get("thread_id") is None
                or _check_filter_match(
                    threads_by_id.get(str(c["thread_id"]), {}).get("metadata", {}),
                    thread_filters,
                )
            ]

        # Sort
        sort_by = sort_by.lower() if sort_by else None
        if sort_by and sort_by in (
            "cron_id",
            "assistant_id",
            "thread_id",
            "next_run_date",
            "end_time",
            "created_at",
            "updated_at",
        ):
            reverse = False if sort_order and sort_order.upper() == "ASC" else True
            if sort_by in ["cron_id", "assistant_id", "thread_id"]:
                filtered_crons.sort(
                    key=lambda x: str(x.get(sort_by, "")).lower(),
                    reverse=reverse,
                )
            else:
                filtered_crons.sort(
                    key=lambda x: x.get(sort_by) or datetime.min.replace(tzinfo=UTC),
                    reverse=reverse,
                )
        elif sort_by is None:
            filtered_crons.sort(key=lambda x: x["created_at"], reverse=True)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid sort_by field: '{sort_by}'. Valid options are: cron_id, assistant_id, thread_id, next_run_date, end_time, created_at, updated_at",
            )

        # Paginate — fetch limit+1 to determine cursor
        paginated = filtered_crons[offset : offset + limit + 1]
        if len(paginated) > limit:
            cursor = offset + limit
            paginated = paginated[:limit]
        else:
            cursor = None

        async def cron_iterator() -> AsyncIterator[Cron]:
            for cron in paginated:
                if select:
                    yield {k: v for k, v in cron.items() if k in select}
                else:
                    yield cron

        return cron_iterator(), cursor

    @staticmethod
    async def count(
        conn: InMemConnectionProto,
        *,
        assistant_id: UUID | None = None,
        thread_id: UUID | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Get count of crons."""
        filters = await Crons.handle_event(
            ctx,
            "search",
            {
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "limit": 0,
                "offset": 0,
                "metadata": metadata or {},
            },
        )

        if thread_id:
            thread_filters = await Threads.handle_event(
                ctx,
                "read",
                Auth.types.ThreadsRead(thread_id=thread_id),
            )
        else:
            thread_filters = await Threads.handle_event(
                ctx,
                "search",
                Auth.types.ThreadsSearch(),
            )

        # First pass: cron-level filtering
        filtered_crons = []
        for c in conn.store["crons"]:
            if assistant_id is not None and str(c["assistant_id"]) != str(assistant_id):
                continue
            if thread_id is not None and str(c.get("thread_id")) != str(thread_id):
                continue
            if metadata and not is_jsonb_contained(c.get("metadata", {}), metadata):
                continue
            if filters and not _check_filter_match(c.get("metadata", {}), filters):
                continue
            filtered_crons.append(c)

        # Second pass: thread-level auth filtering
        # Crons without a thread_id are exempt from thread filtering.
        if thread_filters:
            cron_thread_ids = {
                str(c["thread_id"])
                for c in filtered_crons
                if c.get("thread_id") is not None
            }
            threads_by_id = {
                str(t["thread_id"]): t
                for t in conn.store["threads"]
                if str(t["thread_id"]) in cron_thread_ids
            }
            filtered_crons = [
                c
                for c in filtered_crons
                if c.get("thread_id") is None
                or _check_filter_match(
                    threads_by_id.get(str(c["thread_id"]), {}).get("metadata", {}),
                    thread_filters,
                )
            ]

        return len(filtered_crons)


async def cancel_run(
    thread_id: UUID, run_id: UUID, ctx: Auth.types.BaseAuthContext | None = None
) -> None:
    async with connect() as conn:
        await Runs.cancel(conn, [run_id], thread_id=thread_id, ctx=ctx)


async def _get_checkpointer(
    conn: InMemConnectionProto | None = None,
    *,
    unpack_hook=None,
):
    """Get the appropriate checkpointer (custom or built-in)."""
    from langgraph_api import config as api_config  # noqa: PLC0415

    if api_config.USE_CUSTOM_CHECKPOINTER:
        from langgraph_api import _checkpointer as api_checkpointer  # noqa: PLC0415

        return await api_checkpointer.get_checkpointer()
    if conn is not None:
        return await asyncio.to_thread(Checkpointer, conn, unpack_hook=unpack_hook)
    return Checkpointer()


async def _delete_checkpoints_for_thread(
    thread_id: str | UUID,
    conn: InMemConnectionProto,
    run_id: str | UUID | None = None,
):
    from langgraph_api import config as api_config  # noqa: PLC0415

    if api_config.USE_CUSTOM_CHECKPOINTER:
        from langgraph_api import _checkpointer as api_checkpointer  # noqa: PLC0415

        checkpointer = await api_checkpointer.get_checkpointer()
        if run_id:
            await checkpointer.adelete_for_runs([str(run_id)])
        else:
            await checkpointer.adelete_thread(str(thread_id))
        return

    checkpointer = Checkpointer()
    thread_id = str(thread_id)
    if thread_id not in checkpointer.storage:
        return
    if run_id:
        # Look through metadata
        run_id = str(run_id)
        for checkpoint_ns, checkpoints in list(checkpointer.storage[thread_id].items()):
            for checkpoint_id, (_, metadata_b, _) in list(checkpoints.items()):
                metadata = checkpointer.serde.loads_typed(metadata_b)
                if metadata.get("run_id") == run_id:
                    del checkpointer.storage[thread_id][checkpoint_ns][checkpoint_id]
                    if not checkpointer.storage[thread_id][checkpoint_ns]:
                        del checkpointer.storage[thread_id][checkpoint_ns]
    else:
        del checkpointer.storage[thread_id]
        # Keys are (thread_id, checkpoint_ns, checkpoint_id)
        checkpointer.writes = defaultdict(
            dict, {k: v for k, v in checkpointer.writes.items() if k[0] != thread_id}
        )


def _validate_filter_structure(
    filters: Auth.types.FilterType | None,
    nesting_level: int = 0,
) -> None:
    """Validate the structure of filter conditions without checking matches.

    Args:
        filters: The filter conditions to validate
        nesting_level: Current depth of nested operators (max 2)

    Raises:
        HTTPException: If the filter structure is invalid
    """
    if nesting_level > 2:
        raise HTTPException(
            status_code=500,
            detail="Your auth handler returned a filter with too many nested operators. The maximum depth for nested operators is 2. Please simplify your filter.",
        )

    if not filters:
        return

    # Handle $or operator
    if "$or" in filters:
        or_groups = filters["$or"]
        if not isinstance(or_groups, list) or not len(or_groups) >= 2:
            raise HTTPException(
                status_code=500,
                detail="Your auth handler returned a filter with an invalid $or operator. The $or operator must be a list of at least 2 filter objects. Check the filter returned by your auth handler.",
            )

        # Recursively validate all groups
        for group in or_groups:
            _validate_filter_structure(group, nesting_level=nesting_level + 1)

        # Validate remaining filters (implicit AND with the $or)
        remaining_filters = {k: v for k, v in filters.items() if k != "$or"}
        if remaining_filters:
            _validate_filter_structure(
                remaining_filters, nesting_level=nesting_level + 1
            )

    # Handle $and operator
    if "$and" in filters:
        and_groups = filters["$and"]
        if not isinstance(and_groups, list) or not len(and_groups) >= 2:
            raise HTTPException(
                status_code=500,
                detail="Your auth handler returned a filter with an invalid $and operator. The $and operator must be a list of at least 2 filter objects. Check the filter returned by your auth handler.",
            )

        # Recursively validate all groups
        for group in and_groups:
            _validate_filter_structure(group, nesting_level=nesting_level + 1)

        # Validate remaining filters (implicit AND with the $and)
        remaining_filters = {k: v for k, v in filters.items() if k != "$and"}
        if remaining_filters:
            _validate_filter_structure(
                remaining_filters, nesting_level=nesting_level + 1
            )


def _check_filter_match(
    metadata: dict,
    filters: Auth.types.FilterType | None,
    nesting_level: int = 0,
) -> bool:
    """Check if metadata matches the filter conditions.

    Args:
        metadata: The metadata to check
        filters: The filter conditions to apply
        nesting_level: Current depth of nested operators (max 2)

    Returns:
        True if the metadata matches all filter conditions, False otherwise
    """
    if nesting_level > 2:
        raise HTTPException(
            status_code=500,
            detail="Your auth handler returned a filter with too many nested operators. The maximum depth for nested operators is 2. Please simplify your filter.",
        )

    if not filters:
        return True

    # Handle $or operator
    if "$or" in filters:
        or_groups = filters["$or"]
        if not isinstance(or_groups, list) or not len(or_groups) >= 2:
            raise HTTPException(
                status_code=500,
                detail="Your auth handler returned a filter with an invalid $or operator. The $or operator must be a list of at least 2 filter objects. Check the filter returned by your auth handler.",
            )

        # Validate all groups first to ensure nesting limits are respected
        # (even if we short-circuit during matching)
        for group in or_groups:
            _validate_filter_structure(group, nesting_level=nesting_level + 1)

        # At least one group must match
        or_match = False
        for group in or_groups:
            if _check_filter_match(metadata, group, nesting_level=nesting_level + 1):
                or_match = True
                break

        if not or_match:
            return False

        # Check remaining filters (implicit AND with the $or)
        remaining_filters = {k: v for k, v in filters.items() if k != "$or"}
        if remaining_filters:
            return _check_filter_match(
                metadata, remaining_filters, nesting_level=nesting_level + 1
            )
        return True

    # Handle $and operator
    if "$and" in filters:
        and_groups = filters["$and"]
        if not isinstance(and_groups, list) or not len(and_groups) >= 2:
            raise HTTPException(
                status_code=500,
                detail="Your auth handler returned a filter with an invalid $and operator. The $and operator must be a list of at least 2 filter objects. Check the filter returned by your auth handler.",
            )

        # Validate all groups first to ensure nesting limits are respected
        for group in and_groups:
            _validate_filter_structure(group, nesting_level=nesting_level + 1)

        # All groups must match
        for group in and_groups:
            if not _check_filter_match(
                metadata, group, nesting_level=nesting_level + 1
            ):
                return False

        # Check remaining filters (implicit AND with the $and)
        remaining_filters = {k: v for k, v in filters.items() if k != "$and"}
        if remaining_filters:
            return _check_filter_match(
                metadata, remaining_filters, nesting_level=nesting_level + 1
            )
        return True

    # Regular filter logic (implicit AND)
    for key, value in filters.items():
        if isinstance(value, dict):
            op = next(iter(value))
            filter_value = value[op]

            if op == "$eq":
                if key not in metadata or metadata[key] != filter_value:
                    return False
            elif op == "$contains":
                if key not in metadata or not isinstance(metadata[key], list):
                    return False

                if isinstance(filter_value, list):
                    # Mimick Postgres containment operator behavior.
                    # It would be more efficient to use set operations here,
                    # but we can't assume that elements are hashable.
                    # The Postgres algorithm is also O(n^2).
                    for filter_element in filter_value:
                        if filter_element not in metadata[key]:
                            return False
                elif filter_value not in metadata[key]:
                    return False
        else:
            # Direct equality
            if key not in metadata or metadata[key] != value:
                return False

    return True


async def _empty_generator():
    if False:
        yield


__all__ = [
    "StreamHandler",
    "Assistants",
    "Crons",
    "Runs",
    "Threads",
]
