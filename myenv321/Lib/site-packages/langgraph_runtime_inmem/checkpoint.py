from __future__ import annotations

import logging
import os
import typing
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import Any

from langgraph.checkpoint.memory import (
    InMemorySaver as InMemorySaverBase,
)
from langgraph.checkpoint.memory import (
    PersistentDict,
)

from langgraph_runtime_inmem._persistence import (
    register_persistent_dict,
    stop_flush_loop,
)

if typing.TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import (
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        SerializerProtocol,
    )

logger = logging.getLogger(__name__)

_EXCLUDED_KEYS = {"checkpoint_ns", "checkpoint_id", "run_id", "thread_id"}

# Configurable keys that are transient (per-request) and should not be persisted in checkpoints
_TRANSIENT_CONFIGURABLE_KEYS = frozenset(
    {
        "langgraph_request_id",
        "langgraph_auth_user",
        "langgraph_auth_user_id",
        "langgraph_auth_permissions",
    }
)

# Not in public docs: internal, disables pickle file persistence for inmem runtime
DISABLE_FILE_PERSISTENCE = (
    os.getenv("LANGGRAPH_DISABLE_FILE_PERSISTENCE", "false").lower() == "true"
)


class InMemorySaver(InMemorySaverBase):
    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
        __persistence_hook__: Callable[[PersistentDict], None] | None = None,
    ) -> None:
        self.filename = os.path.join(".langgraph_api", ".langgraph_checkpoint.")
        self.latest_iter: AsyncIterator[CheckpointTuple] | None = None
        i = 0

        def factory(*args):
            nonlocal i
            i += 1

            if not os.path.exists(".langgraph_api"):
                os.mkdir(".langgraph_api")
            thisfname = self.filename + str(i) + ".pckl"
            d = PersistentDict(*args, filename=thisfname)
            if __persistence_hook__:
                __persistence_hook__(d)

            try:
                d.load()
            except FileNotFoundError:
                pass
            except ModuleNotFoundError:
                logger.error(
                    "Unable to load cached data - your code has changed in a way that's incompatible with the cache."
                    "\nThis usually happens when you've:"
                    "\n  - Renamed or moved classes"
                    "\n  - Changed class structures"
                    "\n  - Pulled updates that modified class definitions in a way that's incompatible with the cache"
                    "\n\nRemoving invalid cache data stored at path: .langgraph_api"
                )
                try:
                    os.remove(self.filename)
                except Exception:
                    pass
            except Exception as e:
                logger.error("Failed to load cached data: %s", str(e))
                try:
                    os.remove(self.filename)
                except Exception:
                    pass
            return d

        from langgraph_api.serde import Serializer  # noqa: PLC0415

        super().__init__(
            serde=serde if serde is not None else Serializer(),
            factory=factory if not DISABLE_FILE_PERSISTENCE else defaultdict,
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, str | int | float],
    ) -> RunnableConfig:
        # TODO: Should this be done in OSS as well?
        # Filter out transient fields that are request-scoped, not checkpoint-scoped
        config_metadata = config.get("metadata", {})
        metadata = {
            **{
                k: v
                for k, v in config["configurable"].items()
                if not k.startswith("__")
                and k not in _EXCLUDED_KEYS
                and k not in _TRANSIENT_CONFIGURABLE_KEYS
            },
            **{
                k: v
                for k, v in config_metadata.items()
                if k not in _TRANSIENT_CONFIGURABLE_KEYS
            },
            **{
                k: v
                for k, v in metadata.items()
                if k not in _TRANSIENT_CONFIGURABLE_KEYS
            },
        }
        if not isinstance(checkpoint["id"], uuid.UUID):
            # Avoid type inconsistencies
            checkpoint = checkpoint.copy()
            checkpoint["id"] = str(checkpoint["id"])
        return super().put(config, checkpoint, metadata, new_versions)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        if isinstance(config["configurable"].get("checkpoint_id"), uuid.UUID):
            # Avoid type inconsistencies....
            config = config.copy()

            config["configurable"] = {
                **config["configurable"],
                "checkpoint_id": str(config["configurable"]["checkpoint_id"]),
            }
        return super().get_tuple(config)

    def clear(self):
        self.storage.clear()
        self.writes.clear()
        for suffix in ["1", "2"]:
            file_path = f"{self.filename}{suffix}.pckl"
            if os.path.exists(file_path):
                os.remove(file_path)

    async def _decrypt_json(
        self, data: dict[str, Any], model: str, field: str
    ) -> dict[str, Any]:
        """Decrypt a dict if custom encryption is configured."""
        from langgraph_api import config as api_config  # noqa: PLC0415

        if not api_config.LANGGRAPH_ENCRYPTION:
            return data
        from langgraph_api.encryption import get_encryption  # noqa: PLC0415
        from langgraph_api.encryption.middleware import (  # noqa: PLC0415
            decrypt_json_if_needed,
        )

        result = await decrypt_json_if_needed(data, get_encryption(), model, field)
        if result is None:
            raise ValueError("decrypt_json_if_needed returned None for non-None input")
        return result

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get checkpoint tuple with decrypted metadata."""
        tuple_ = self.get_tuple(config)
        if tuple_ is None:
            return None

        # Decrypt metadata if encryption is enabled
        decrypted_metadata = await self._decrypt_json(
            tuple_.metadata, "run", "metadata"
        )

        from langgraph.checkpoint.base import (  # noqa: PLC0415
            CheckpointTuple as CPTuple,
        )

        return CPTuple(
            config=tuple_.config,
            checkpoint=tuple_.checkpoint,
            metadata=decrypted_metadata,
            parent_config=tuple_.parent_config,
            pending_writes=tuple_.pending_writes,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints with decrypted metadata."""
        from langgraph.checkpoint.base import (  # noqa: PLC0415
            CheckpointTuple as CPTuple,
        )

        for tuple_ in self.list(config, filter=filter, before=before, limit=limit):
            # Decrypt metadata if encryption is enabled
            decrypted_metadata = await self._decrypt_json(
                tuple_.metadata, "run", "metadata"
            )

            yield CPTuple(
                config=tuple_.config,
                checkpoint=tuple_.checkpoint,
                metadata=decrypted_metadata,
                parent_config=tuple_.parent_config,
                pending_writes=tuple_.pending_writes,
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        stop_flush_loop()
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def aget_iter(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        tup = await self.aget_tuple(config)
        if tup is not None:
            yield tup


MEMORY = None


def Checkpointer(*args, unpack_hook=None, **kwargs):
    global MEMORY
    if MEMORY is None:
        MEMORY = InMemorySaver(
            __persistence_hook__=register_persistent_dict,
        )
    if unpack_hook is not None:
        from langgraph_api.serde import Serializer  # noqa: PLC0415

        # Prefer the API-level feature flag when available; older
        # langgraph-api versions may not define it yet.
        try:
            from langgraph_api.feature_flags import (  # noqa: PLC0415
                DELTA_CHANNEL_SUPPORT,
            )
        except ImportError:
            DELTA_CHANNEL_SUPPORT = False

        # DeltaChannel snapshots only exist on langgraph >= 1.2; on older
        # installs the ``EXT_DELTA_SNAPSHOT`` codepoint can never appear in
        # serialized payloads, so the bare ``unpack_hook`` is sufficient.
        if DELTA_CHANNEL_SUPPORT:
            from langgraph.checkpoint.serde.jsonplus import (  # noqa: PLC0415
                EXT_DELTA_SNAPSHOT,  # ty: ignore[unresolved-import]
            )
            from langgraph.checkpoint.serde.types import (  # noqa: PLC0415
                _DeltaSnapshot,  # ty: ignore[unresolved-import]
            )

            _inner_hook = unpack_hook

            def _delta_aware_hook(code: int, data: bytes) -> Any:
                if code == EXT_DELTA_SNAPSHOT:
                    import ormsgpack  # noqa: PLC0415

                    return _DeltaSnapshot(
                        ormsgpack.unpackb(
                            data,
                            ext_hook=_delta_aware_hook,
                            option=ormsgpack.OPT_NON_STR_KEYS,
                        )
                    )
                return _inner_hook(code, data)

            ext_hook = _delta_aware_hook
        else:
            ext_hook = unpack_hook

        saver = InMemorySaver(
            serde=Serializer(__unpack_ext_hook__=ext_hook),
            __persistence_hook__=register_persistent_dict,
            **kwargs,
        )
        saver.writes = MEMORY.writes
        saver.blobs = MEMORY.blobs
        saver.storage = MEMORY.storage
        return saver
    return MEMORY


__all__ = ["Checkpointer"]
