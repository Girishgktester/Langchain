"""Thread-scoped run manager for protocol v2 transports.

This module replaces the previous session-based registry with a thread-centric
design (protocol v0.5.0). There is no server-side session state:

* SSE event streams are connection-scoped — the route handler creates a
  ``ThreadRunManager`` directly and closes it when the connection ends.
* WebSocket connections own a ``ThreadRunManager`` per socket; subscription
  IDs persist only for the lifetime of the socket.

``ThreadRunManager`` owns:

* A single bound ``EventStreamingSession`` (per thread) that normalizes raw
  stream events into protocol events.
* Run creation/resume for the thread (``run.start``, ``input.respond``).
* Command routing to the underlying session.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast, get_args
from uuid import UUID, uuid4

import orjson
import structlog

from langgraph_api.event_streaming.capabilities import (
    ProtocolV2Capabilities,
    ProtocolV2UnsupportedError,
    probe_event_streaming_v2_capabilities,
)
from langgraph_api.event_streaming.constants import (
    DEFAULT_RUN_STREAM_MODES,
    EVENT_STREAMING_V2_CONFIG_KEY,
)
from langgraph_api.event_streaming.session import (
    EventStreamingSession,
    _is_supported_channel,
)
from langgraph_api.event_streaming.types import Subscription
from langgraph_api.feature_flags import FF_V2_EVENT_STREAMING, PREFER_GRPC_CHECKPOINTER
from langgraph_api.schema import MultitaskStrategy

logger = structlog.stdlib.get_logger(__name__)


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _checkpoint_id_from_run_start_config(params: dict[str, Any]) -> str | None:
    """Extract ``config.configurable.checkpoint_id`` from ``run.start`` params.

    Protocol v2 clients pass fork/time-travel targets via RunnableConfig::

        {
            "input": null,
            "config": { "configurable": { "checkpoint_id": "<uuid>" } },
        }

    Returns the checkpoint id (validated as a non-empty string) or ``None``
    when no fork target is present.
    """
    config = params.get("config")
    if not _is_record(config):
        return None
    configurable = config.get("configurable")
    if not _is_record(configurable):
        return None
    checkpoint_id = configurable.get("checkpoint_id")
    if checkpoint_id is None:
        return None
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError(
            "config.configurable.checkpoint_id must be a non-empty string."
        )
    return checkpoint_id


def validate_checkpoint_id_from_runnable_config(params: dict[str, Any]) -> str | None:
    """Return a client-facing validation message, or ``None`` if valid/absent."""
    try:
        checkpoint_id = _checkpoint_id_from_run_start_config(params)
    except ValueError as exc:
        return str(exc)
    if checkpoint_id is None:
        return None
    try:
        UUID(checkpoint_id)
    except ValueError:
        return f"Invalid config.configurable.checkpoint_id: {checkpoint_id!r}."
    return None


# Concurrency strategies accepted on ``run.start`` — the four values the
# SDK's ``multitaskStrategy`` option can take, derived from the canonical
# ``MultitaskStrategy`` literal so the two never drift.
VALID_MULTITASK_STRATEGIES: tuple[str, ...] = get_args(MultitaskStrategy)
# Legacy stream-endpoint default: queue new runs behind active ones instead
# of interrupting. Used when the caller omits ``multitaskStrategy``.
DEFAULT_MULTITASK_STRATEGY = "enqueue"


def _multitask_strategy_from_run_start(params: dict[str, Any]) -> str:
    """Resolve the per-run ``multitaskStrategy`` from ``run.start`` params.

    The SDK forwards the caller's ``multitaskStrategy`` on every
    ``run.start`` (one of ``reject`` | ``rollback`` | ``interrupt`` |
    ``enqueue``). Honor it when it is one of the recognized strategies,
    otherwise fall back to ``DEFAULT_MULTITASK_STRATEGY`` (the legacy
    stream-endpoint default). This mirrors the JS reference server's
    lenient normalization so both behave identically.
    """
    strategy = params.get("multitaskStrategy")
    if isinstance(strategy, str) and strategy in VALID_MULTITASK_STRATEGIES:
        return strategy
    return DEFAULT_MULTITASK_STRATEGY


EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]


# ``set_joint_status`` commits interrupts after the stream event; gRPC can lag.
_IN_FLIGHT_THREAD_STATUS = "busy"
_INTERRUPT_SETTLE_TIMEOUT_SECONDS = 5.0
_INTERRUPT_SETTLE_POLL_INTERVAL_SECONDS = 0.05


# Protocol v2 commands this server implements. Anything outside this
# set returns ``unknown_command`` up front from ``handle_command`` so
# pre-session dispatch does not mislabel typos (or removed methods
# like ``state.get`` / ``input.inject``) as ``no_such_run``. The
# ``input.respond`` and ``agent.getTree`` entries flow through to the
# session dispatcher which handles their per-run semantics.
_KNOWN_COMMANDS: frozenset[str] = frozenset(
    {
        "run.start",
        "input.respond",
        "agent.getTree",
        "subscription.subscribe",
        "subscription.unsubscribe",
        "subscription.reconnect",
    }
)


class ThreadRunManager:
    """Owns a ``EventStreamingSession`` bound to a specific thread.

    One ``ThreadRunManager`` is created per connection (HTTP SSE stream or
    WebSocket). It is not shared across connections — each observer owns
    its own view of the thread's run stream.
    """

    def __init__(
        self,
        *,
        thread_id: str,
        runs: Any,
        threads: Any,
        send_event: EventSink | None = None,
    ) -> None:
        self._thread_id = thread_id
        self._runs = runs
        self._threads = threads
        self._send_event = send_event
        self._session: EventStreamingSession | None = None
        self._current_run_id: str | None = None
        self._queued_events: list[dict[str, Any]] = []
        self._seq: int = 0
        self._thread_source_task: asyncio.Task[None] | None = None
        # Set by ``_consume_thread_stream`` when the background task
        # exits — success, cancellation, or unexpected failure all
        # transition this. Route handlers (SSE body_iter, WS watchdog)
        # await this so a transient ``join_event_streaming`` failure can no
        # longer leave the connection wedged with no events flowing.
        self._thread_stream_done: asyncio.Event = asyncio.Event()
        # WebSocket clients typically call ``subscription.subscribe``
        # before ``run.start`` — natural ordering since they want to
        # observe before firing. Buffer those subscriptions here and
        # install them on the ``EventStreamingSession`` once it's bound
        # (in ``_ensure_run_session``). SSE doesn't hit this path
        # because each connection carries its filter in the request
        # body rather than as an in-band command.
        self._pending_subscriptions: dict[str, Subscription] = {}

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @property
    def current_run_id(self) -> str | None:
        return self._current_run_id

    @property
    def session(self) -> EventStreamingSession | None:
        return self._session

    @property
    def seq(self) -> int:
        return self._seq

    async def attach_event_sink(self, send_event: EventSink) -> None:
        """Attach a live transport consumer and flush buffered events."""
        self._send_event = send_event
        if inspect.iscoroutinefunction(send_event):
            async_sink = cast("Callable[[dict[str, Any]], Awaitable[None]]", send_event)
            for event in self._queued_events[:]:
                await async_sink(event)
        else:
            sync_sink = cast("Callable[[dict[str, Any]], None]", send_event)
            for event in self._queued_events[:]:
                sync_sink(event)
        self._queued_events.clear()

    async def close(self) -> None:
        if self._thread_source_task is not None:
            self._thread_source_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._thread_source_task
            self._thread_source_task = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    def install_subscription(self, subscription: Subscription) -> None:
        """Install a subscription on the current and future run sessions."""
        subscription.active = True
        self._pending_subscriptions[subscription.id] = subscription
        if self._session is not None:
            self._session.subscriptions[subscription.id] = Subscription(
                id=subscription.id,
                channels=set(subscription.channels),
                namespaces=subscription.namespaces,
                depth=subscription.depth,
                active=True,
            )

    def start_thread_stream(self) -> None:
        """Start consuming the thread-level stream for this connection.

        Always start from the beginning of the thread's event stream
        (``last_event_id="-"``). The SDK's shared-stream rotation
        (``ThreadStream.#computeUnionFilter``) drops any caller-supplied
        ``since`` when it widens to the union filter, so the contract
        the SDK was built against is "open a fresh SSE → replay all
        history matching the filter." Without this, namespace-scoped
        projections (``useMessages`` etc.) opened on a finished or
        already-streaming subagent see only events that arrive after
        registration and the UI sits at "Waiting for the subagent to
        produce output…". The SSE/WS sink applies the caller's ``since``
        as a seq filter on outbound events, and the SDK dedups by
        ``event_id``, so repeat observers don't see duplicates.
        """
        if self._thread_source_task is not None:
            return
        self._thread_source_task = asyncio.create_task(
            self._consume_thread_stream(last_event_id="-")
        )

    async def _consume_thread_stream(self, *, last_event_id: str | None) -> None:
        try:
            async for (
                event,
                message,
                stream_id,
                run_id,
            ) in self._threads.Stream.join_event_streaming(
                self._thread_id,
                stream_modes=["run_modes", "lifecycle"],
                last_event_id=last_event_id,
            ):
                if not run_id or run_id == "*":
                    continue
                session = await self._ensure_run_session_for_id(run_id)
                if session is None:
                    continue
                if event == b"metadata":
                    if self._is_run_done_metadata(message, run_id):
                        await session.complete()
                    continue
                await session.ingest_source_event((event, message, stream_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await logger.aexception(
                "protocol thread stream failed",
                thread_id=str(self._thread_id),
                error=str(exc),
                error_type=type(exc).__name__,
            )
        finally:
            # Wake up SSE body_iter / WS watchdog so the connection can
            # be torn down. Without this, a transient ``join_event_streaming``
            # failure leaves the response generator blocked on its
            # ``flush_pending`` event with no events ever arriving, and
            # the client has to time out to learn the stream is dead.
            self._thread_stream_done.set()

    async def wait_for_thread_stream_end(self) -> None:
        """Block until the thread-stream consumer exits (any reason).

        Used by the SSE body iterator and WebSocket route to detect
        consumer death and tear the connection down rather than wedge
        forever. Returns immediately if the consumer was never started
        or has already finished.
        """
        if self._thread_source_task is None:
            return
        await self._thread_stream_done.wait()

    async def _ensure_run_session_for_id(
        self, run_id: str
    ) -> EventStreamingSession | None:
        if self._session is not None and self._current_run_id == run_id:
            return self._session

        run = await self._fetch_run_by_id(run_id)
        if run is None:
            return None
        await self._ensure_run_session(run)
        self._current_run_id = run_id
        return self._session

    @staticmethod
    def _is_run_done_metadata(message: bytes, run_id: str) -> bool:
        try:
            data = orjson.loads(message)
        except Exception:
            return False
        return (
            _is_record(data)
            and data.get("status") == "run_done"
            and data.get("run_id") == run_id
        )

    async def handle_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a protocol command for this thread."""
        method = command.get("method", "")

        # Reject unknown commands up front so pre-session dispatch does
        # not mislabel them as ``no_such_run``. Commands that are only
        # meaningful after a session exists (``input.respond``,
        # ``agent.getTree``, the subscription family) still flow through
        # the session dispatch below.
        if method not in _KNOWN_COMMANDS:
            return self._error(
                command.get("id"),
                "unknown_command",
                f"Unknown protocol command: {method}",
            )

        if method == "run.start":
            return await self._handle_run_start(command)
        if method == "input.respond":
            return await self._handle_input_respond(command)

        # Subscription commands can arrive before a run exists (typical
        # on WebSocket: subscribe → run.start → events). Buffer them on
        # the manager and install them when a session gets bound.
        if self._session is None:
            if method == "subscription.subscribe":
                return self._buffer_subscribe(command)
            if method == "subscription.unsubscribe":
                return self._buffer_unsubscribe(command)

        # Forward subscription, agent, flow commands to the bound session.
        return await self._forward_to_run_session(command)

    def _buffer_subscribe(self, command: dict[str, Any]) -> dict[str, Any]:
        """Queue a ``subscription.subscribe`` command before a session exists.

        Validates the params the same way the session does and returns a
        success response with a generated ``subscription_id`` and
        ``replayed_events: 0`` — the session hasn't emitted anything to
        replay. The buffered subscription transfers onto the session in
        ``_ensure_run_session``.
        """
        params = command.get("params", {}) if _is_record(command.get("params")) else {}
        raw_channels = params.get("channels")
        if not isinstance(raw_channels, list) or not raw_channels:
            return self._error(
                command.get("id"),
                "invalid_argument",
                "subscription.subscribe requires a non-empty channels array.",
            )
        channels = [
            c for c in raw_channels if isinstance(c, str) and _is_supported_channel(c)
        ]
        if len(channels) != len(raw_channels):
            return self._error(
                command.get("id"),
                "invalid_argument",
                "subscription.subscribe received an unsupported channel.",
            )

        namespaces: list[list[str]] | None = None
        raw_ns = params.get("namespaces")
        if isinstance(raw_ns, list) and all(
            isinstance(ns, list) and all(isinstance(s, str) for s in ns)
            for ns in raw_ns
        ):
            namespaces = raw_ns

        depth: int | None = None
        raw_depth = params.get("depth")
        # Reject bool explicitly — ``isinstance(False, int)`` is ``True``
        # and would silently coerce to ``depth=0`` (root-only).
        if (
            isinstance(raw_depth, int)
            and not isinstance(raw_depth, bool)
            and raw_depth >= 0
        ):
            depth = raw_depth

        sub = Subscription(
            id=str(uuid4()),
            channels=set(channels),
            namespaces=namespaces,
            depth=depth,
            active=True,
        )
        self._pending_subscriptions[sub.id] = sub
        cmd_id = command.get("id")
        resp: dict[str, Any] = {
            "type": "success",
            "id": cmd_id if isinstance(cmd_id, int) else 0,
            "result": {"subscription_id": sub.id, "replayed_events": 0},
            "meta": self._meta(),
        }
        return resp

    def _buffer_unsubscribe(self, command: dict[str, Any]) -> dict[str, Any]:
        """Remove a subscription that was buffered before the session bound."""
        params = command.get("params", {}) if _is_record(command.get("params")) else {}
        sub_id = params.get("subscription_id")
        if not isinstance(sub_id, str):
            return self._error(
                command.get("id"),
                "invalid_argument",
                "subscription.unsubscribe requires a subscription_id.",
            )
        if sub_id not in self._pending_subscriptions:
            return self._error(
                command.get("id"),
                "no_such_subscription",
                f"Unknown subscription: {sub_id}",
            )
        del self._pending_subscriptions[sub_id]
        cmd_id = command.get("id")
        return {
            "type": "success",
            "id": cmd_id if isinstance(cmd_id, int) else 0,
            "result": {},
            "meta": self._meta(),
        }

    def _meta(self) -> dict[str, Any]:
        return {"applied_through_seq": self._seq}

    # ------------------------------------------------------------------
    # run.start
    # ------------------------------------------------------------------

    async def _handle_run_start(self, command: dict[str, Any]) -> dict[str, Any]:
        params = command.get("params", {}) if _is_record(command.get("params")) else {}

        assistant_id = params.get("assistant_id")
        if not isinstance(assistant_id, str) or not assistant_id:
            return self._error(
                command.get("id"),
                "invalid_argument",
                "run.start requires an assistant_id.",
            )

        checkpoint_error = validate_checkpoint_id_from_runnable_config(params)
        if checkpoint_error is not None:
            return self._error(command.get("id"), "invalid_argument", checkpoint_error)

        try:
            run = await self._create_or_resume_run(assistant_id, params)
        except ProtocolV2UnsupportedError as exc:
            return self._error(command.get("id"), "unsupported", str(exc))
        except Exception as exc:
            return self._error(command.get("id"), "unknown_error", str(exc))

        # ``create_valid_run`` returns a ``UUID`` for ``run_id``; the CDDL
        # wire shape requires ``text``, and Starlette's ``JSONResponse``
        # uses stdlib ``json`` which can't serialize ``UUID``. Coerce here.
        run_id = run["run_id"] if _is_record(run) else getattr(run, "run_id", "")
        return {
            "type": "success",
            "id": command.get("id", 0),
            "result": {"run_id": str(run_id) if run_id else ""},
            "meta": self._meta(),
        }

    @staticmethod
    def _coerce_respond_namespace(raw: Any) -> list[str] | None:
        """Validate an ``input.respond`` namespace.

        CDDL ``InputRespondParams`` marks ``namespace`` as required, but an
        omitted value is treated as the root namespace (``[]``). Returns the
        coerced list, or ``None`` when the value is not a list of strings.
        """
        if not isinstance(raw, list) or not all(isinstance(seg, str) for seg in raw):
            return None
        return list(raw)

    def _normalize_respond_entries(
        self, cmd_id: int | None, params: dict[str, Any]
    ) -> tuple[list[tuple[str, list[str], Any]] | None, dict[str, Any] | None]:
        """Normalize ``input.respond`` params into resume entries.

        Clients send either a single ``interrupt_id`` / ``response`` (with an
        optional ``namespace``) or a ``responses`` batch — a list of
        ``{interrupt_id, response, namespace?}`` objects. The batch form
        resumes several interrupts pending at the same checkpoint (e.g.
        parallel tool-authorization prompts) in one command; sequential
        single resumes cannot, since the first resume starts a run, leaving
        the rest with no interrupted run to respond to. Both forms are part
        of the streaming protocol (``InputRespondParams`` =
        ``InputRespondOne / InputRespondMany``); the batch entries are read
        leniently to tolerate clients pinned to older bindings.

        Returns ``(entries, None)`` on success or ``(None, error)`` with a
        protocol error envelope. Each entry is
        ``(interrupt_id, namespace, response)``.
        """
        raw_responses = params.get("responses")
        if isinstance(raw_responses, list):
            if not raw_responses:
                return None, self._error(
                    cmd_id,
                    "invalid_argument",
                    "input.respond requires at least one response.",
                )
            entries: list[tuple[str, list[str], Any]] = []
            for entry in raw_responses:
                if not _is_record(entry):
                    return None, self._error(
                        cmd_id,
                        "invalid_argument",
                        "input.respond responses entries must be objects.",
                    )
                entry_id = entry.get("interrupt_id")
                if not isinstance(entry_id, str):
                    return None, self._error(
                        cmd_id,
                        "invalid_argument",
                        "input.respond responses entries require an interrupt_id.",
                    )
                namespace = self._coerce_respond_namespace(entry.get("namespace", []))
                if namespace is None:
                    return None, self._error(
                        cmd_id,
                        "invalid_argument",
                        "input.respond requires namespace to be a list of strings.",
                    )
                entries.append((entry_id, namespace, entry.get("response")))
            return entries, None

        interrupt_id = params.get("interrupt_id")
        if not isinstance(interrupt_id, str):
            return None, self._error(
                cmd_id,
                "invalid_argument",
                "input.respond requires an interrupt_id.",
            )
        namespace = self._coerce_respond_namespace(params.get("namespace", []))
        if namespace is None:
            return None, self._error(
                cmd_id,
                "invalid_argument",
                "input.respond requires namespace to be a list of strings.",
            )
        return [(interrupt_id, namespace, params.get("response"))], None

    async def _handle_input_respond(self, command: dict[str, Any]) -> dict[str, Any]:
        params = command.get("params", {}) if _is_record(command.get("params")) else {}

        entries, normalize_error = self._normalize_respond_entries(
            command.get("id"), params
        )
        if normalize_error is not None:
            return normalize_error
        if entries is None:  # narrowed by the error branch above
            return self._error(
                command.get("id"), "unknown_error", "No entries to respond to."
            )

        # Cross-check every targeted interrupt against the session's pending
        # interrupts so we can return ``no_such_interrupt`` for
        # unknown/mismatched pairs. When the session was attached fresh by
        # this HTTP request (the stateless ``POST /commands`` transport) its
        # source task hasn't had a chance to emit ``input.requested`` yet —
        # ``_pending_interrupts`` is still empty. In that case, fall back to
        # the thread-state check (``_collect_settled_interrupt_ids``) so we
        # don't reject legitimate resumes with a fresh handle.
        #
        # The thread-state fallback is fetched at most once per batch and
        # cached in ``thread_state_ids``.
        thread_state_ids: set[str] | None = None
        thread_state_fetched = False
        for interrupt_id, claimed_namespace, _ in entries:
            pending_namespace = (
                self._session.lookup_pending_interrupt(interrupt_id)
                if self._session is not None
                else None
            )
            if pending_namespace is not None:
                # Authoritative: the session observed the ``input.requested``
                # event and recorded the exact subgraph namespace. Enforce
                # strict comparison.
                if pending_namespace != claimed_namespace:
                    return self._error(
                        command.get("id"),
                        "no_such_interrupt",
                        "Interrupt namespace does not match the pending interrupt.",
                    )
                # WS can surface ``input.requested`` before postgres/gRPC
                # persists the thread-row interrupt. Reuse the HTTP settle
                # barrier before enqueueing the resume run.
                if not thread_state_fetched:
                    thread_state_ids = await self._collect_settled_interrupt_ids()
                    thread_state_fetched = True
                continue
            # HTTP fallback: the persisted lookup surfaces interrupts by id
            # only, so trust the client-claimed namespace and validate by
            # existence (the id is a namespace hash, so that's sufficient).
            if not thread_state_fetched:
                thread_state_ids = await self._collect_settled_interrupt_ids()
                thread_state_fetched = True
            # Fail open when the lookup is unavailable (``None``): only reject
            # when a definitive id set was read and lacks the target. The
            # checkpointer resume below is authoritative, so a transient lookup
            # failure must not masquerade as a missing interrupt.
            if thread_state_ids is not None and interrupt_id not in thread_state_ids:
                return self._error(
                    command.get("id"),
                    "no_such_interrupt",
                    f"Unknown or already-consumed interrupt: {interrupt_id}",
                )

        # At this point we've confirmed every interrupt exists on the
        # thread state (or was surfaced via the session). No need for a
        # second ``_has_pending_interrupts`` round-trip.

        # Resolve the assistant from thread metadata. Run creation keeps
        # the thread's assistant_id current, avoiding a latest-run search
        # on stateless command requests.
        assistant_id = await self._resolve_resume_assistant_id()
        if assistant_id is None:
            return self._error(
                command.get("id"),
                "no_such_run",
                "No interrupted run is bound to this thread.",
            )

        # Merge every entry into a single ``{interrupt_id: response}`` resume
        # map. ``_create_or_resume_run`` forwards it verbatim as
        # ``Command(resume=...)``, which resumes all targeted interrupts in
        # one run.
        #
        # ``update`` / ``goto`` are the optional state mutation and directed
        # jump applied in the same superstep as the resume (HITL "push card
        # into state + resume", or resume-and-redirect flows). They are folded
        # into ``Command(resume, update, goto)`` by ``_create_or_resume_run``,
        # so the resumed run produces a single checkpoint reflecting all of
        # them. Run-level (not per-entry): one resumed run services the batch.
        resume_input = {interrupt_id: response for interrupt_id, _, response in entries}
        try:
            await self._create_or_resume_run(
                assistant_id,
                {
                    "assistant_id": assistant_id,
                    "input": resume_input,
                    "force_resume": True,
                    "update": params.get("update"),
                    "goto": params.get("goto"),
                    "config": params.get("config"),
                    "metadata": params.get("metadata"),
                },
            )
        except ProtocolV2UnsupportedError as exc:
            return self._error(command.get("id"), "unsupported", str(exc))
        except Exception as exc:
            return self._error(command.get("id"), "unknown_error", str(exc))

        if self._session is not None:
            for interrupt_id, _, _ in entries:
                self._session.clear_pending_interrupt(interrupt_id)

        return {
            "type": "success",
            "id": command.get("id", 0),
            "result": {},
            "meta": self._meta(),
        }

    async def _create_or_resume_run(
        self, assistant_id: str, params: dict[str, Any]
    ) -> Any:
        from langgraph_api.models.run import create_valid_run  # noqa: PLC0415
        from langgraph_api.utils import uuid7 as make_uuid7  # noqa: PLC0415
        from langgraph_runtime.database import connect  # noqa: PLC0415

        # Defense in depth — the v2 routes are gated at registration
        # time on ``FF_V2_EVENT_STREAMING`` (see ``api/__init__.py``), so
        # this branch is unreachable in normal operation. Direct callers
        # (tests, future internal code) still hit this and get a clear
        # ``unsupported`` envelope rather than a half-completed run.
        if not FF_V2_EVENT_STREAMING:
            raise ProtocolV2UnsupportedError(ProtocolV2Capabilities.disabled_by_flag())
        # Verify the installed ``langgraph`` / ``langchain-core`` can serve
        # a Protocol v2 run before we persist the new run row. Otherwise the
        # mismatch only surfaces later as an ``error`` SSE event from inside
        # ``astream_state`` (invalid ``stream_mode``, missing ``StreamMux``,
        # or — worst case — a silently-collapsed message stream). Cached,
        # so the second ``run.start`` on the same process skips the work.
        capabilities = probe_event_streaming_v2_capabilities()
        if not capabilities.ok:
            raise ProtocolV2UnsupportedError(capabilities)

        config = params.get("config") if _is_record(params.get("config")) else {}

        current_run = await self._fetch_current_run()

        current_status = None
        if current_run is not None:
            current_status = (
                current_run["status"]
                if _is_record(current_run)
                else getattr(current_run, "status", None)
            )

        has_interrupts = False
        if params.get("input") is not None:
            has_interrupts = await self._has_pending_interrupts()

        is_resume = params.get("input") is not None and (
            params.get("force_resume")
            or (current_run is not None and current_status == "interrupted")
            or has_interrupts
        )

        run_config: dict[str, Any] = {
            **(config or {}),
            "configurable": {
                **(config.get("configurable", {}) if _is_record(config) else {}),
                EVENT_STREAMING_V2_CONFIG_KEY: True,
                "thread_id": self._thread_id,
            },
        }

        # The fork target arrives only via ``config.configurable.checkpoint_id``
        # (the SDK folds its ergonomic ``forkFrom`` option into this field
        # client-side, so there is a single way to provide it). Extract it
        # here and map it onto the top-level ``checkpoint_id`` / ``checkpoint``
        # fields of ``run_payload`` below: ``create_valid_run`` — the shared
        # run-creation path also used by the legacy REST run-create endpoints —
        # reads the fork target from those fields (UUID-validating it and
        # injecting it back into ``config.configurable``), so populating them is
        # what makes the run replay from the requested checkpoint rather than
        # the thread's latest state.
        checkpoint_id = _checkpoint_id_from_run_start_config(params)

        run_payload: dict[str, Any] = {
            "assistant_id": assistant_id,
            "input": None if is_resume else params.get("input"),
            # On resume, fold the optional state update / directed jump into the
            # same ``Command`` as the resume value so the resumed run produces a
            # single checkpoint (see ``langgraph_api.command.map_cmd``). Only
            # set on the ``input.respond`` path — ``run.start`` params carry no
            # ``update`` / ``goto``, so ``.get`` returns ``None`` and the keys
            # are omitted.
            "command": (
                {
                    "resume": params["input"],
                    **(
                        {"update": params["update"]}
                        if params.get("update") is not None
                        else {}
                    ),
                    **(
                        {"goto": params["goto"]}
                        if params.get("goto") is not None
                        else {}
                    ),
                }
                if is_resume
                else None
            ),
            "config": run_config,
            "metadata": params.get("metadata"),
            "checkpoint_id": checkpoint_id,
            "checkpoint": {"checkpoint_id": checkpoint_id} if checkpoint_id else None,
            "context": None,
            "webhook": None,
            "stream_mode": list(DEFAULT_RUN_STREAM_MODES),
            "stream_subgraphs": True,
            "stream_resumable": True,
            "interrupt_before": None,
            "interrupt_after": None,
            "feedback_keys": None,
            "on_completion": "keep",
            # ``after_seconds`` must be a concrete int — ``create_valid_run``
            # uses it verbatim in a ``timedelta(seconds=...)`` call and
            # chokes on None. Zero means "start immediately".
            "after_seconds": 0,
            "if_not_exists": "create",
            # Honor the caller's ``multitaskStrategy`` (the SDK sends it on
            # every run.start), falling back to ``enqueue`` — the legacy
            # stream-endpoint default — when omitted.
            "multitask_strategy": _multitask_strategy_from_run_start(params),
            "langsmith_tracer": None,
            "durability": None,
        }

        run_id = make_uuid7()
        try:
            async with connect() as conn:
                run = await create_valid_run(
                    conn,
                    self._thread_id,
                    run_payload,
                    {},
                    run_id=run_id,
                )
        except Exception:
            raise

        return run

    async def _ensure_run_session(
        self,
        run: Any,
    ) -> None:
        # ``create_valid_run`` returns ``run_id`` as a ``UUID`` for the
        # inmem runtime; the postgres path returns a row with string
        # fields. ``self._current_run_id`` is always a ``str``, so the
        # short-circuit guard below would silently fail if we kept the
        # ``UUID`` here. Coerce both run_id / thread_id at the boundary.
        run_id = (
            str(run["run_id"]) if _is_record(run) else str(getattr(run, "run_id", ""))
        )
        thread_id = (
            str(run["thread_id"])
            if _is_record(run)
            else str(getattr(run, "thread_id", self._thread_id))
        )

        if self._session is not None and self._current_run_id == run_id:
            return

        old_subscriptions = (
            list(self._session.subscriptions.values())
            if self._session is not None
            else []
        )
        # Preserve the prior session's seq cursor so events on the new
        # run keep climbing monotonically. SSE deduplicates by
        # ``event_id`` (``str(seq)``) — if the follow-up session
        # restarted at seq=0, event ids would collide with the
        # terminated run's and ``on_event``'s ``delivered`` set would
        # drop them. See ``EventStreamingSession.set_initial_seq``.
        carry_seq = self._session.next_seq if self._session is not None else 0
        if self._session is not None:
            await self._session.close()

        session = EventStreamingSession(
            run_id=run_id,
            thread_id=thread_id,
            initial_run=(
                run
                if _is_record(run)
                else {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "assistant_id": getattr(run, "assistant_id", ""),
                    "status": getattr(run, "status", "pending"),
                    "kwargs": getattr(run, "kwargs", {}),
                }
            ),
            get_run=self._make_get_run(run_id, thread_id),
            get_thread_state=self._make_get_thread_state(thread_id),
            source=None,
            send=self._make_send(),
        )

        self._session = session
        session.set_initial_seq(carry_seq)

        # Carry over active subscriptions from the previous run session.
        for sub_def in old_subscriptions:
            session.subscriptions[sub_def.id] = Subscription(
                id=sub_def.id,
                channels=set(sub_def.channels),
                namespaces=sub_def.namespaces,
                depth=sub_def.depth,
                active=True,
            )

        # Install subscriptions that arrived on the manager before a
        # session existed (WebSocket path: ``subscription.subscribe``
        # before ``run.start``). They receive the initial lifecycle
        # event emitted by ``session.start()`` below.
        for sub_def in self._pending_subscriptions.values():
            session.subscriptions[sub_def.id] = Subscription(
                id=sub_def.id,
                channels=set(sub_def.channels),
                namespaces=sub_def.namespaces,
                depth=sub_def.depth,
                active=True,
            )
        self._pending_subscriptions.clear()

        await session.start()

    def _make_send(self) -> Callable[[str], Awaitable[None] | None]:
        async def send(payload: str) -> None:
            parsed = orjson.loads(payload)
            self._seq = max(self._seq, parsed.get("seq", self._seq))
            sink = self._send_event
            if sink is None:
                self._queued_events.append(parsed)
                return
            if inspect.iscoroutinefunction(sink):
                await cast("Callable[[dict[str, Any]], Awaitable[None]]", sink)(parsed)
            else:
                cast("Callable[[dict[str, Any]], None]", sink)(parsed)

        return send

    def _make_get_run(
        self, run_id: str, thread_id: str
    ) -> Callable[[], Awaitable[dict[str, Any] | None]]:
        """Build a closure that fetches the bound run on demand.

        Used by ``EventStreamingSession._emit_terminal_lifecycle`` to read
        the run's final status. ``Runs.get`` requires keyword-only
        ``thread_id`` / ``ctx`` args, so a positional lambda would blow
        up at call time.
        """

        async def get_run() -> dict[str, Any] | None:
            from langgraph_runtime.database import connect  # noqa: PLC0415

            try:
                async with connect() as conn:
                    results = await self._runs.get(conn, run_id, thread_id=thread_id)
                    async for run in results:
                        return run if _is_record(run) else None
            except Exception:
                return None
            return None

        return get_run

    def _make_get_thread_state(
        self, thread_id: str
    ) -> Callable[[], Awaitable[dict[str, Any] | None]]:
        async def get_thread_state() -> dict[str, Any] | None:
            from langgraph_runtime.database import connect  # noqa: PLC0415

            # When ``PREFER_GRPC_CHECKPOINTER`` is false the postgres backend
            # yields a real connection for the Python checkpointer; when true
            # it yields a no-op stand-in and checkpoint I/O goes through gRPC.
            try:
                async with connect(supports_core_api=PREFER_GRPC_CHECKPOINTER) as conn:
                    return await self._threads.State.get(
                        conn,
                        {"configurable": {"thread_id": thread_id}},
                        subgraphs=True,
                    )
            except Exception as exc:
                logger.warning(
                    "terminal_thread_state_fetch_failed",
                    thread_id=thread_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                return None

        return get_thread_state

    async def _get_thread_assistant_id(self) -> str | None:
        from langgraph_runtime.database import connect  # noqa: PLC0415

        try:
            async with connect() as conn:
                result = await self._threads.get(conn, self._thread_id)
                thread = await anext(result) if hasattr(result, "__anext__") else result
        except Exception:
            return None
        metadata = thread.get("metadata") if _is_record(thread) else None
        assistant_id = metadata.get("assistant_id") if _is_record(metadata) else None
        return assistant_id if isinstance(assistant_id, str) else None

    async def _resolve_resume_assistant_id(self) -> str | None:
        assistant_id = await self._get_thread_assistant_id()
        if assistant_id is not None:
            return assistant_id
        if self._session is None:
            return None
        initial_run = getattr(self._session, "_initial_run", None)
        if not _is_record(initial_run):
            return None
        assistant_id = initial_run.get("assistant_id")
        return assistant_id if isinstance(assistant_id, str) else None

    # ------------------------------------------------------------------
    # Run lookup
    # ------------------------------------------------------------------

    async def _fetch_current_run(self) -> dict[str, Any] | None:
        """Fetch the currently-bound run from storage.

        ``Runs.get`` uses keyword-only ``thread_id`` / ``ctx`` args so we
        cannot call it positionally. We also rely on ``ApiRoute``'s
        ``with_user`` to populate the auth context var instead of
        passing ``ctx=self._auth`` (which is a starlette
        ``AuthCredentials``, not a ``BaseAuthContext``).
        """
        if self._current_run_id is None:
            return None
        return await self._fetch_run_by_id(self._current_run_id)

    async def _fetch_run_by_id(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a run by id without searching for the latest run."""
        from langgraph_runtime.database import connect  # noqa: PLC0415

        try:
            async with connect() as conn:
                results = await self._runs.get(
                    conn,
                    run_id,
                    thread_id=self._thread_id,
                )
                async for run in results:
                    return run if _is_record(run) else None
        except Exception as exc:
            await logger.awarning(
                "protocol fetch_run failed",
                thread_id=str(self._thread_id),
                run_id=str(run_id),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None
        return None

    # ------------------------------------------------------------------
    # Interrupt detection
    # ------------------------------------------------------------------

    async def _lookup_interrupt_in_thread_state(
        self, interrupt_id: str
    ) -> list[str] | None:
        """Existence check for ``interrupt_id`` against persisted thread state.

        Fallback for ``input.respond`` on stateless HTTP transports, where the
        fresh session hasn't observed ``input.requested`` yet. Backed by the
        durable thread-row map (see :meth:`_collect_persisted_interrupt_ids`).

        Returns ``[]`` if the interrupt exists, else ``None``. The ``[]`` is a
        sentinel, not a namespace — the map carries no namespace, so callers
        must treat it as "found, namespace unknown" and trust the client claim.
        """
        found = await self._collect_persisted_interrupt_ids()
        if found is None:
            return None
        return [] if interrupt_id in found else None

    async def _collect_persisted_interrupt_ids(self) -> set[str] | None:
        """Collect interrupt ids from the durable thread-row ``interrupts`` map.

        ``Threads.set_status`` writes ``{task_id: [{id, value}, ...]}`` onto the
        thread row atomically with the status whenever a run pauses. Reading it
        is a plain row fetch (same ``connect()`` + ``Threads.get`` path as
        :meth:`_get_thread_assistant_id`) — unlike ``Threads.State.get`` it does
        not rebuild the graph, so it can't spuriously raise and reject a
        legitimate resume, and it survives reconnects/redeploys.

        Returns the set of interrupt ids (empty when none are pending), or
        ``None`` if the fetch failed — callers treat ``None`` as "unavailable"
        and fall open rather than fabricating ``no_such_interrupt``.
        """
        from langgraph_runtime.database import connect  # noqa: PLC0415

        try:
            async with connect() as conn:
                result = await self._threads.get(conn, self._thread_id)
                thread = await anext(result) if hasattr(result, "__anext__") else result
        except Exception as exc:
            logger.warning(
                "interrupt_lookup_failed",
                thread_id=self._thread_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None

        interrupts_map = thread.get("interrupts") if _is_record(thread) else None
        if not _is_record(interrupts_map):
            return set()
        found: set[str] = set()
        for entries in interrupts_map.values():
            if not isinstance(entries, (list, tuple)):
                continue
            for entry in entries:
                entry_id = (
                    entry.get("id") if _is_record(entry) else getattr(entry, "id", None)
                )
                if isinstance(entry_id, str):
                    found.add(entry_id)
        return found

    async def _fetch_thread_status(self) -> str | None:
        """Return durable thread status, or ``None`` if unreadable."""
        from langgraph_runtime.database import connect  # noqa: PLC0415

        try:
            async with connect() as conn:
                result = await self._threads.get(conn, self._thread_id)
                thread = await anext(result) if hasattr(result, "__anext__") else result
        except Exception:
            return None
        status = thread.get("status") if _is_record(thread) else None
        return status if isinstance(status, str) else None

    async def _collect_settled_interrupt_ids(self) -> set[str] | None:
        """Persisted interrupt ids, polling while the thread is still ``busy``."""
        ids = await self._collect_persisted_interrupt_ids()
        if ids is None:
            return None
        if ids:
            return ids
        deadline = time.monotonic() + _INTERRUPT_SETTLE_TIMEOUT_SECONDS
        while True:
            status = await self._fetch_thread_status()
            if status != _IN_FLIGHT_THREAD_STATUS:
                return await self._collect_persisted_interrupt_ids()
            if time.monotonic() >= deadline:
                return set()
            await asyncio.sleep(_INTERRUPT_SETTLE_POLL_INTERVAL_SECONDS)
            ids = await self._collect_persisted_interrupt_ids()
            if ids is None:
                return None
            if ids:
                return ids

    async def _has_pending_interrupts(self) -> bool:
        # Prefer the session's in-memory pending interrupts (populated the
        # instant ``input.requested`` surfaces) to skip a DB round-trip;
        # otherwise consult the durable thread-row map. A lookup failure
        # (``None``) reads as "none pending" — ``run.start`` then falls back to
        # the ``current_status == "interrupted"`` signal for resume-vs-start.
        if self._session is not None and self._session._pending_interrupts:
            return True
        found = await self._collect_settled_interrupt_ids()
        return bool(found)

    # ------------------------------------------------------------------
    # Command forwarding
    # ------------------------------------------------------------------

    async def _forward_to_run_session(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            return self._error(
                command.get("id"),
                "no_such_run",
                "No active run is bound to this thread.",
            )
        return await self._session.handle_event_streaming_command(command, self._meta())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error(
        cmd_id: int | None,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        return {"type": "error", "id": cmd_id, "error": code, "message": message}
