"""Build hook to conditionally vendor grpc_common into wheels.

For wheel/sdist builds from repo: vendors ../grpc_common_py/langgraph_grpc_common
For wheel from sdist: vendors the already-included langgraph_grpc_common/
For editable installs: does nothing (grpc_common installed as dev dependency)
"""

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import (  # type: ignore[import]
    BuildHookInterface,
)


class ConditionalGrpcCommonHook(BuildHookInterface):
    PLUGIN_NAME = "conditional_grpc_common"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        # Skip vendoring for editable installs - grpc_common installed as dev dependency
        if version == "editable":
            return

        # Path when building directly from the repo
        repo_path = Path(self.root) / ".." / "grpc_common_py" / "langgraph_grpc_common"
        # Path when building wheel from sdist (already vendored)
        vendored_path = Path(self.root) / "langgraph_grpc_common"

        if repo_path.resolve().exists():
            if "force_include" not in build_data:
                build_data["force_include"] = {}
            build_data["force_include"]["../grpc_common_py/langgraph_grpc_common"] = (
                "langgraph_grpc_common"
            )
        elif vendored_path.exists():
            # Building wheel from sdist - grpc_common already vendored
            if "force_include" not in build_data:
                build_data["force_include"] = {}
            build_data["force_include"]["langgraph_grpc_common"] = (
                "langgraph_grpc_common"
            )
        # else: editable install or grpc_common not available - skip silently
