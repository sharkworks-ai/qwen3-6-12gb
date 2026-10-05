from __future__ import annotations

from fastapi import APIRouter, HTTPException

from appliance.search.pareto import frontier


router = APIRouter(prefix="/api/v1", tags=["api"])


@router.get("/health")
def health():
    return {"ok": True}


@router.post("/pareto")
def pareto(payload: dict):
    candidates = payload.get("candidates", [])
    return {"frontier": frontier(candidates)}


@router.get("/features")
def features():
    return {
        "sft": True,
        "profiling": True,
        "pruning": True,
        "recovery": True,
        "quantization": [
            "llama_cpp",
            "turboquant",
            "bittern_catq",
        ],
        "release_search": True,
        "dataset_builder": True,
        "contamination_check": True,
        "lineage": True,
        "runtime_matrix": True,
        "release_packaging": True,
        "notifications": True,
        "rbac": True,
        "secret_store": True,
    }
