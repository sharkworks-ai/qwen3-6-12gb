from __future__ import annotations

import json
import urllib.request
from typing import Any


def send_webhook(url: str, payload: dict[str, Any], timeout: float = 10.0) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status >= 300:
            raise RuntimeError(f"webhook failed with HTTP {response.status}")


def notify(config: dict[str, Any], event: str, payload: dict[str, Any]) -> None:
    for target in config.get("webhooks", []):
        if event not in target.get("events", [event]):
            continue
        send_webhook(
            target["url"],
            {"event": event, "payload": payload},
        )
