"""LangSmith tracing session helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langsmith.utils import get_tracer_project


def resolve_tracing_session_name(payload: Mapping[str, Any]) -> str:
    """Return the LangSmith session (project) name for a run.

    Per-run ``langsmith_tracer.project_name`` overrides the deployment default
    from ``LANGSMITH_PROJECT`` / ``LANGCHAIN_PROJECT``.
    """
    ls_tracing = payload.get("langsmith_tracer")
    if isinstance(ls_tracing, Mapping):
        project_name = ls_tracing.get("project_name")
        if isinstance(project_name, str) and project_name:
            return project_name
    return get_tracer_project()
