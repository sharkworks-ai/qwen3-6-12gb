from __future__ import annotations

from dataclasses import dataclass


PERMISSIONS = {
    "viewer": {"read"},
    "operator": {"read", "run", "stop"},
    "publisher": {"read", "run", "stop", "publish"},
    "admin": {"read", "run", "stop", "publish", "secrets", "users"},
}


def allowed(role: str, action: str) -> bool:
    return action in PERMISSIONS.get(role, set())
