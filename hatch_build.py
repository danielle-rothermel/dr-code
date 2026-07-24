"""Install the immutable preprocessing-viewer assets into wheels."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Consume the one checked-in frontend archive without a JS toolchain."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        package = root / "src" / "dr_code" / "viewer"
        assets = _load_assets_module(package)
        temporary = tempfile.TemporaryDirectory(
            prefix="dr-code-viewer-assets-"
        )
        try:
            static_dir = assets.extract_prebuilt_viewer_archive(
                package / assets.ARCHIVE_FILENAME,
                package / assets.DIGEST_FILENAME,
                Path(temporary.name),
            )
        except assets.PrebuiltViewerAssetsError as exc:
            temporary.cleanup()
            raise RuntimeError(
                f"{exc}; run scripts/build_viewer_assets.py"
            ) from exc
        except BaseException:
            temporary.cleanup()
            raise
        self._temporary = temporary
        force_include = build_data["force_include"]
        force_include[str(static_dir)] = "dr_code/viewer/static"

    def finalize(
        self,
        version: str,
        build_data: dict[str, Any],
        artifact_path: str,
    ) -> None:
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()


def _load_assets_module(package: Path) -> ModuleType:
    module_path = package / "assets.py"
    spec = importlib.util.spec_from_file_location(
        "_dr_code_viewer_build_assets", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load viewer asset implementation: {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
