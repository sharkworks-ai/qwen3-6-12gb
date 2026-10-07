"""Compare provenance-bound external scores without re-running compression."""

from __future__ import annotations

import argparse
import os

from appliance.eval.oxcoder_gate import BASELINE, score
from appliance.runtime.packed import sha256
from appliance.stages.common import data_path, load_config, save_json


def compare(report, evidence, root):
    if report.get("synthetic") or report.get("status") != "completed":
        raise ValueError("Require a completed non-synthetic validation report")
    result = {}
    for name, variant in report["variants"].items():
        candidate = evidence.get(name, {})
        bundle = data_path(variant["bundle"], root)
        manifest = bundle / "mixed-manifest.json"
        if load_config(manifest).get("synthetic_proof"):
            raise ValueError("Synthetic artifacts cannot satisfy release benchmark gates")
        if candidate.get("artifact_manifest_sha256") != sha256(manifest) or not candidate.get(
            "harness_revisions"
        ):
            raise ValueError(f"Benchmark provenance mismatch: {name}")
        scores = candidate.get("scores", {})
        if set(scores) != set(BASELINE) or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not 0 <= v <= 100
            for v in scores.values()
        ):
            raise ValueError("Require all 11 benchmark scores on a 0–100 scale")
        result[name] = {
            "gate": score(scores),
            "harness_revisions": candidate["harness_revisions"],
            "artifact_manifest_sha256": candidate["artifact_manifest_sha256"],
        }
    return {"candidates": result, "source": "external_results", "release_approved": False}


def run(cfg):
    root = os.environ.get("QWEN12G_DATA_ROOT", "/data")
    output = data_path(cfg["output_dir"], root)
    report_path = output / "report.json"
    evidence_path = data_path(cfg["benchmark_results"], root)
    result = compare(load_config(report_path), load_config(evidence_path), root)
    result["input_hashes"] = {"report": sha256(report_path), "evidence": sha256(evidence_path)}
    save_json(output / "benchmark-report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
