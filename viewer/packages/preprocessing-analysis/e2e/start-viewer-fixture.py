"""Start the browser fixture on an OS-assigned, continuously owned port."""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path
from types import ModuleType
from typing import cast

import uvicorn
from uvicorn._types import ASGIApplication

HOST = "127.0.0.1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_PATH = REPOSITORY_ROOT / "tests/browser/serve_viewer_fixture.py"


class _ReadyServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, url: str) -> None:
        super().__init__(config)
        self._url = url

    async def startup(
        self, sockets: list[socket.socket] | None = None
    ) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            print(f"DR_CODE_VIEWER_API_URL={self._url}", flush=True)


def _load_fixture() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_dr_code_playwright_viewer_fixture", FIXTURE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load browser fixture from {FIXTURE_PATH}"
        )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    return fixture


def _run_fixture_server(
    app: ASGIApplication,
    *,
    host: str,
    port: int,
    workers: int,
) -> None:
    if host != HOST or port != 0 or workers != 1:
        raise RuntimeError(
            "Playwright fixture must use one worker on an OS-assigned "
            f"{HOST} port, got host={host!r}, port={port}, workers={workers}"
        )

    config = uvicorn.Config(app, host=host, port=port, workers=workers)
    with socket.create_server(
        (host, port), backlog=config.backlog
    ) as listener:
        assigned_port = cast("tuple[str, int]", listener.getsockname())[1]
        server = _ReadyServer(config, f"http://{host}:{assigned_port}")
        server.run(sockets=[listener])


def main() -> None:
    fixture = _load_fixture()
    fixture.PORT = 0
    fixture.uvicorn.run = _run_fixture_server
    fixture.main()


if __name__ == "__main__":
    main()
