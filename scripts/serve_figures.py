#!/usr/bin/env python3
"""Serve the local figure viewer and optionally open it in a browser."""

from __future__ import annotations

import argparse
import contextlib
import http.server
import socket
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def pick_port(host: str, port: int) -> int:
    """Return the requested port when free, otherwise ask the OS for one."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            sock.bind((host, 0))
        return int(sock.getsockname()[1])


def build_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/web/index.html"


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the interactive figure viewer locally.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Preferred port")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the browser",
    )
    args = parser.parse_args()

    port = pick_port(args.host, args.port)
    handler = lambda *handler_args, **handler_kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *handler_args, directory=str(ROOT), **handler_kwargs
    )
    server = http.server.ThreadingHTTPServer((args.host, port), handler)
    url = build_url(args.host, port)

    print(f"Serving {ROOT}")
    print(f"Figure viewer: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
