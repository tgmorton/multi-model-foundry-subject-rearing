"""Load the shared bad-node list (configs/bad_nodes.txt)."""
from __future__ import annotations
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "configs" / "bad_nodes.txt"


def load() -> list[str]:
    out = []
    for line in PATH.read_text().splitlines():
        host = line.split("#", 1)[0].strip()
        if host:
            out.append(host)
    return out
