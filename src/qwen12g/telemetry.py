from __future__ import annotations

import csv
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

QUERY = (
    "index,name,memory.total,memory.used,utilization.gpu,"
    "temperature.gpu,power.draw"
)


class NvidiaSmiSampler:
    def __init__(self, output: Path, interval_seconds: float = 1.0) -> None:
        self.output = output
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2))

    def _run(self) -> None:
        with self.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp",
                    "index",
                    "name",
                    "memory_total_mib",
                    "memory_used_mib",
                    "utilization_gpu_percent",
                    "temperature_c",
                    "power_w",
                ]
            )
            handle.flush()

            while not self._stop.is_set():
                timestamp = datetime.now(UTC).isoformat()
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--query-gpu={QUERY}",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        text=True,
                        capture_output=True,
                    )
                    for line in result.stdout.splitlines():
                        if line.strip():
                            writer.writerow(
                                [timestamp, *[part.strip() for part in line.split(",")]]
                            )
                    handle.flush()
                except (OSError, subprocess.CalledProcessError) as exc:
                    writer.writerow([timestamp, "error", str(exc)])
                    handle.flush()

                self._stop.wait(self.interval_seconds)


def peak_memory_by_gpu(path: Path) -> dict[str, float]:
    peaks: dict[str, float] = {}
    if not path.exists():
        return peaks

    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = row.get("index")
            used = row.get("memory_used_mib")
            if not index or index == "error" or not used:
                continue
            try:
                value = float(used)
            except ValueError:
                continue
            peaks[index] = max(peaks.get(index, 0.0), value)
    return peaks
