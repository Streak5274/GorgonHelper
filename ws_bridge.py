"""
ws_bridge.py — GorgonHelper WebSocket bridge

Watches the Project: Gorgon Reports folder and Player.log, then streams
changes to connected browser clients over WebSocket. Enables automatic
folder-watch and live player-log support in Firefox and any browser that
doesn't support the File System Access API.

Usage:
    python ws_bridge.py

The bridge listens on ws://localhost:8765. GorgonHelper.html connects
automatically on page load and falls back to the native File System
Access API when the bridge is not running.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    sys.exit(
        "ERROR: 'websockets' is not installed.\n"
        "Run:  pip install -r requirements.txt"
    )

# ── Configuration ──────────────────────────────────────────────────────────
PORT          = 8765
POLL_INTERVAL = 2.0        # seconds between file-system polls
LOG_TAIL_INIT = 200_000    # bytes of Player.log to send on first connect

# ── Path detection ─────────────────────────────────────────────────────────
def _locate_game() -> tuple[Path, Path]:
    """Return (reports_folder, player_log) auto-detected from %APPDATA%."""
    appdata = Path(os.environ.get("APPDATA", ""))
    base    = appdata.parent / "LocalLow" / "Elder Game" / "Project Gorgon"
    return base / "Reports", base / "Player.log"

REPORTS_FOLDER, PLAYER_LOG = _locate_game()

# ── Per-connection state ───────────────────────────────────────────────────
_clients: set = set()

# Global file-mtime cache shared across all poll cycles
_file_mtimes: dict[str, float] = {}

# Player-log position — starts at end-of-file so we don't replay old history
_log_pos: int = 0

# ── Helpers ────────────────────────────────────────────────────────────────
_CHAR_RE = re.compile(r"^Character_.+\.json$")

def _iter_watched() -> list[tuple[Path, str]]:
    """
    Yield (abs_path, bridge_path) for every file the bridge tracks.

    bridge_path is the relative path sent to the client:
      - Root character files   →  "Character_Foo.json"
      - character_exports/     →  "character_exports/Character_Foo.json"
      - Json/ game data        →  "Json/items.json"
    """
    dirs = [
        (REPORTS_FOLDER,                          "",                    _CHAR_RE),
        (REPORTS_FOLDER / "character_exports",    "character_exports/",  None),
        (REPORTS_FOLDER / "Json",                 "Json/",               None),
    ]
    result = []
    for folder, prefix, pattern in dirs:
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if not p.is_file() or p.suffix != ".json":
                continue
            if pattern and not pattern.match(p.name):
                continue
            result.append((p, prefix + p.name))
    return result


async def _broadcast(msg: dict) -> None:
    if not _clients:
        return
    data = json.dumps(msg, ensure_ascii=False)
    await asyncio.gather(
        *(c.send(data) for c in list(_clients)),
        return_exceptions=True,
    )


def _read_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _tail_log(from_pos: int, max_bytes: int | None = None) -> tuple[str, int]:
    """
    Read new (or last max_bytes) bytes from Player.log.
    Returns (text_chunk, new_position).
    """
    if not PLAYER_LOG.exists():
        return "", from_pos
    try:
        size = PLAYER_LOG.stat().st_size
        read_from = from_pos
        if max_bytes is not None:
            read_from = max(0, size - max_bytes)
        if size <= read_from:
            return "", from_pos
        with open(PLAYER_LOG, "rb") as f:
            f.seek(read_from)
            raw = f.read(size - read_from)
        text = raw.decode("utf-8", errors="replace")
        # Only process up to the last complete newline
        last_nl = text.rfind("\n")
        if last_nl < 0:
            return "", from_pos
        chunk = text[: last_nl + 1]
        new_pos = read_from + len(chunk.encode("utf-8"))
        return chunk, new_pos
    except Exception:
        return "", from_pos


# ── WebSocket handler ──────────────────────────────────────────────────────
async def _handler(ws) -> None:
    global _log_pos

    _clients.add(ws)
    addr = getattr(ws, "remote_address", "?")
    print(f"[bridge] + client {addr}  ({len(_clients)} connected)")

    try:
        # 1. Hello
        await ws.send(json.dumps({
            "type":      "hello",
            "version":   "1",
            "folder":    str(REPORTS_FOLDER),
            "playerLog": str(PLAYER_LOG),
        }))

        # 2. Initial file dump
        for path, bpath in _iter_watched():
            content = _read_file(path)
            if content is not None:
                await ws.send(json.dumps({"type": "file", "path": bpath, "content": content}))

        # 3. Initial player-log tail (last LOG_TAIL_INIT bytes so NPC data is fresh)
        chunk, _ = _tail_log(0, max_bytes=LOG_TAIL_INIT)
        if chunk:
            lines = [l for l in chunk.splitlines() if l]
            if lines:
                await ws.send(json.dumps({"type": "playerlog", "lines": lines}))

        # 4. Signal that the initial dump is complete
        await ws.send(json.dumps({"type": "ready"}))

        await ws.wait_closed()

    finally:
        _clients.discard(ws)
        print(f"[bridge] - client {addr}  ({len(_clients)} connected)")


# ── Poll loop ──────────────────────────────────────────────────────────────
async def _poll_loop() -> None:
    global _log_pos

    # Initialise log position to current end so we don't replay history
    if PLAYER_LOG.exists():
        _log_pos = PLAYER_LOG.stat().st_size

    while True:
        await asyncio.sleep(POLL_INTERVAL)

        # Watch for new or changed JSON files
        for path, bpath in _iter_watched():
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            key = str(path)
            if _file_mtimes.get(key) == mtime:
                continue
            _file_mtimes[key] = mtime
            content = _read_file(path)
            if content is not None:
                await _broadcast({"type": "file", "path": bpath, "content": content})
                print(f"[bridge] → {bpath}")

        # Tail Player.log for new lines
        chunk, new_pos = _tail_log(_log_pos)
        if chunk:
            _log_pos = new_pos
            lines = [l for l in chunk.splitlines() if l]
            if lines:
                await _broadcast({"type": "playerlog", "lines": lines})


# ── Entry point ────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"[bridge] Reports folder : {REPORTS_FOLDER}")
    print(f"[bridge] Player.log     : {PLAYER_LOG}")
    if not REPORTS_FOLDER.exists():
        print("[bridge] WARNING: Reports folder not found — will watch when it appears")
    if not PLAYER_LOG.exists():
        print("[bridge] WARNING: Player.log not found — will watch when it appears")
    print(f"[bridge] Listening on   ws://localhost:{PORT}")
    print(f"[bridge] Press Ctrl+C to stop.\n")

    async with websockets.serve(_handler, "localhost", PORT):
        await _poll_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[bridge] Stopped.")
