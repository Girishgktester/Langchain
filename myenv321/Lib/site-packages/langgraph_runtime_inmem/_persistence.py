"""Periodic flushing for all PersistentDict stores."""

from __future__ import annotations

import functools
import logging
import os
import threading
import weakref

from langgraph.checkpoint.memory import PersistentDict

logger = logging.getLogger(__name__)

_stores: dict[str, weakref.ref[PersistentDict]] = {}
_flush_thread: tuple[threading.Event, threading.Thread] | None = None
_flush_interval: int = 10
DISABLE_FILE_PERSISTENCE = (
    os.getenv("LANGGRAPH_DISABLE_FILE_PERSISTENCE", "false").lower() == "true"
)


def register_persistent_dict(d: PersistentDict) -> None:
    """Register a PersistentDict for periodic flushing."""
    if DISABLE_FILE_PERSISTENCE:
        return
    global _flush_thread
    _stores[d.filename] = weakref.ref(d)
    if _flush_thread is None:
        logger.info("Starting dev persistence flush loop")
        stop_event = threading.Event()
        _flush_thread = (
            stop_event,
            threading.Thread(
                target=functools.partial(_flush_loop, stop_event), daemon=True
            ),
        )
        _flush_thread[1].start()


def stop_flush_loop() -> None:
    """Stop the background flush thread."""
    global _flush_thread
    if _flush_thread is not None:
        logger.info("Stopping dev persistence flush loop")
        _flush_thread[0].set()
        _flush_thread[1].join()
        _flush_thread = None


def _flush_loop(stop_event: threading.Event) -> None:
    drop = set()
    while not stop_event.wait(timeout=_flush_interval):
        keys = list(_stores.keys())
        for store_key in keys:
            if store := _stores[store_key]():
                store.sync()
            else:
                drop.add(store_key)
        if drop:
            for store_key in drop:
                del _stores[store_key]
            drop.clear()
    logger.info("dev persistence flush loop exiting")
