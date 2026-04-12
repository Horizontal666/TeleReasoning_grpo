#!/usr/bin/env python3
"""Validate that the current W&B credentials can authenticate with the server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


def _resolve_api_key() -> str | None:
    key = os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key")
    if key:
        return key.strip()

    key_file = os.environ.get("WANDB_API_KEY_FILE")
    if key_file:
        path = Path(key_file).expanduser()
        if not path.is_file():
            raise SystemExit(f"WANDB_API_KEY_FILE not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    return None


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip()
    if "://" not in base_url:
        base_url = f"https://{base_url}"
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise SystemExit(f"Invalid WANDB_BASE_URL: {base_url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _pick_viewer_field(viewer: object, *names: str) -> str | None:
    for name in names:
        value = getattr(viewer, name, None)
        if value:
            return str(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that the current W&B API key can authenticate with the W&B server."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Login verification timeout in seconds. Default: 30",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ModuleNotFoundError as exc:
        print(f"W&B check failed: wandb is not installed: {exc}", file=sys.stderr)
        return 2

    base_url = _normalize_base_url(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"))
    api_key = _resolve_api_key()
    if not api_key:
        print(
            "W&B check failed: no API key found. Set WANDB_API_KEY, wandb_api_key, or WANDB_API_KEY_FILE.",
            file=sys.stderr,
        )
        return 2

    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_BASE_URL"] = base_url

    try:
        logged_in = wandb.login(
            key=api_key,
            host=base_url,
            relogin=True,
            verify=True,
            timeout=args.timeout,
        )
        if not logged_in:
            print("W&B check failed: wandb.login() returned False.", file=sys.stderr)
            return 1

        api = wandb.Api(overrides={"base_url": base_url}, api_key=api_key)
        viewer = api.viewer
    except Exception as exc:  # noqa: BLE001
        print(f"W&B check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    username = _pick_viewer_field(viewer, "username", "name")
    entity = _pick_viewer_field(viewer, "entity", "default_entity")
    email = _pick_viewer_field(viewer, "email")
    server = _pick_viewer_field(viewer, "server_info")

    print("W&B authentication succeeded.")
    print(f"base_url={base_url}")
    if username:
        print(f"username={username}")
    if entity:
        print(f"entity={entity}")
    if email:
        print(f"email={email}")
    if server:
        print(f"server_info={server}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
