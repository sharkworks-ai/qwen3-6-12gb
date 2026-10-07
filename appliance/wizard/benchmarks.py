"""Automatically call an operator-configured isolated benchmark service."""

from __future__ import annotations

import argparse
import os

import httpx

from appliance.proof.benchmarks import compare
from appliance.runtime.packed import sha256
from appliance.stages.common import data_path, load_config, save_json
from appliance.wizard.config import harness_profiles


def run(cfg):
    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    report = load_config(data_path(cfg["report_file"], root))
    profile = harness_profiles()[cfg["benchmark_profile"]]
    endpoint = profile["url"]
    if not endpoint.startswith(("https://", "http://")) or not profile.get("revision"):
        raise ValueError("Harness requires an HTTP endpoint and pinned revision")
    candidates = {
        name: {
            "bundle": v["bundle"],
            "artifact_manifest_sha256": sha256(
                data_path(v["bundle"], root) / "mixed-manifest.json"
            ),
        }
        for name, v in report["variants"].items()
    }
    payload = {
        "schema_version": 1,
        "request_id": cfg["request_id"],
        "harness_revision": profile["revision"],
        "runtime": "qwen12g_packed_eager",
        "candidates": candidates,
    }
    headers = {}
    if profile.get("token_env"):
        token = os.environ.get(profile["token_env"])
        if not token:
            raise ValueError("Configured benchmark service credential is unavailable")
        headers["Authorization"] = f"Bearer {token}"
    # This service owns task sandboxes. Generated code never runs in this container.
    # The service must honor request_id for retries after interrupted runs.
    with httpx.Client(
        timeout=float(profile.get("timeout_seconds", 86400)),
        follow_redirects=False,
        trust_env=False,
    ) as client:
        response = client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        evidence = response.json()
    result = compare(report, evidence, root)
    if any(
        item["harness_revisions"].get("profile") != profile["revision"]
        for item in evidence.values()
    ):
        raise ValueError("Benchmark harness revision mismatch")
    output = data_path(cfg["output_dir"], root)
    save_json(output / "evidence.json", evidence)
    result["evidence_sha256"] = sha256(output / "evidence.json")
    save_json(output / "benchmark-report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
