from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from appliance.api.router import router as api_router
from appliance.core import (
    Auth,
    ApplianceDB,
    JOBS,
    JobManager,
    Settings,
    gpu_info,
    storage_info,
)
from appliance.datasets.contamination import flag_overlaps
from appliance.datasets.manifest import write_manifest
from appliance.db_ext import WorkbenchDB
from appliance.lineage.graph import graph as lineage_graph
from appliance.publish import push_github, push_huggingface
from appliance.runtime.matrix import runtime_matrix
from appliance.search.pareto import frontier
from appliance.search.release_search import ReleaseSearch, ReleaseSearchConfig
from appliance.quant.presets import preset
from appliance.quant.precision import validate as validate_precision


settings = Settings.from_env()
settings.ensure_dirs()
db = ApplianceDB(settings.data_root / "db" / "appliance.sqlite3")
workbench_db = WorkbenchDB(settings.data_root / "db" / "appliance.sqlite3")
jobs = JobManager(db, settings.data_root / "runs")
auth = Auth(settings.web_token)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _registered_launch(stage: str, config: dict) -> str:
    if stage not in JOBS:
        raise ValueError(
            f"Stage {stage!r} is not registered in the appliance job registry yet"
        )
    return jobs.start(stage, config)


release_search = ReleaseSearch(workbench_db, launch_stage=_registered_launch)


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs.reconcile()
    yield


app = FastAPI(title="Qwen3.6 12GB Lab", lifespan=lifespan)
app.include_router(api_router)


@app.middleware("http")
async def require_auth(request: Request, call_next):
    if request.url.path in {"/login", "/healthz", "/api/v1/health"}:
        return await call_next(request)

    authorization = request.headers.get("authorization", "")
    bearer_ok = authorization.startswith("Bearer ") and auth.verify_token(
        authorization.removeprefix("Bearer ").strip()
    )
    if not bearer_ok and not auth.verify_cookie(
        request.cookies.get(Auth.COOKIE)
    ):
        if request.url.path.startswith("/api/"):
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "authentication required"}, status_code=401)
        return RedirectResponse("/login", status_code=303)
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None},
    )


@app.post("/login")
def login(request: Request, token: str = Form(...)):
    if not auth.verify_token(token):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Invalid token"},
            status_code=401,
        )
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        Auth.COOKIE,
        auth.cookie_value(),
        httponly=True,
        samesite="strict",
        secure=settings.secure_cookie,
        max_age=86400 * 30,
    )
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(Auth.COOKIE)
    return response


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    jobs.reconcile()
    storage = storage_info(settings.data_root)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "gpus": gpu_info(),
            "jobs": db.list_jobs(10),
            "job_specs": JOBS.values(),
            "storage": {
                "total_gib": round(storage["total"] / 1024**3, 1),
                "free_gib": round(storage["free"] / 1024**3, 1),
            },
        },
    )


@app.get("/workbench", response_class=HTMLResponse)
def workbench(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="workbench.html",
        context={},
    )


@app.get("/workflows", response_class=HTMLResponse)
def workflows(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="workflows.html",
        context={},
    )


@app.post("/jobs/start")
def start_job(kind: str = Form(...), config_json: str = Form("{}")):
    try:
        config = json.loads(config_json or "{}")
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object")
        if kind in {"mixed_quant", "mixed_pipeline"}:
            validate_precision(config)
            if kind == "mixed_pipeline" and config["preset"] == "extreme":
                recovery = config.get("recovery", {})
                if not recovery.get("enabled", True) or not recovery.get("dataset"):
                    raise ValueError("Extreme requires QAT recovery and a recovery dataset")
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    run_id = jobs.start(kind, config)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.get("/compression", response_class=HTMLResponse)
def compression_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="compression.html",
        context={"presets": {name: preset(name) for name in ("aggressive", "extreme")}},
    )


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    jobs.reconcile()
    return templates.TemplateResponse(
        request=request,
        name="runs.html",
        context={"jobs": db.list_jobs(200)},
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: str):
    jobs.reconcile()
    job = db.get_job(run_id)
    run_dir = settings.data_root / "runs" / run_id

    def read(name: str) -> str:
        path = run_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return ""

    return templates.TemplateResponse(
        request=request,
        name="run.html",
        context={
            "job": job,
            "stdout": read("stdout.log"),
            "stderr": read("stderr.log"),
            "result": read("result.json"),
            "config": read("config.json"),
        },
    )


@app.post("/runs/{run_id}/stop")
def stop_job(run_id: str):
    jobs.stop(run_id)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.get("/candidates", response_class=HTMLResponse)
def candidates_page(request: Request):
    candidates = workbench_db.list_candidates(1000)
    return templates.TemplateResponse(
        request=request,
        name="candidates.html",
        context={
            "candidates": candidates,
            "frontier_ids": {
                item["candidate_id"] for item in frontier(candidates)
            },
        },
    )


@app.get("/lineage", response_class=HTMLResponse)
def lineage_page(request: Request, candidate_id: str | None = None):
    payload = None
    error = None
    if candidate_id:
        try:
            payload = lineage_graph(workbench_db, candidate_id)
        except KeyError:
            error = f"Unknown candidate: {candidate_id}"
    return templates.TemplateResponse(
        request=request,
        name="lineage.html",
        context={
            "candidate_id": candidate_id or "",
            "payload": json.dumps(payload, indent=2) if payload else "",
            "error": error,
        },
    )


@app.get("/release-search", response_class=HTMLResponse)
def release_search_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="release_search.html",
        context={
            "plan": None,
            "default_source": "tuned-reference",
        },
    )


@app.post("/release-search", response_class=HTMLResponse)
def release_search_plan(
    request: Request,
    source_candidate: str = Form(...),
    max_candidates: int = Form(64),
):
    config = ReleaseSearchConfig(
        source_candidate=source_candidate,
        work_dir=settings.data_root / "artifacts" / "release-search",
        max_candidates=max_candidates,
    )
    plan = release_search.plan(config)
    release_search.write_plan(config)
    return templates.TemplateResponse(
        request=request,
        name="release_search.html",
        context={
            "plan": json.dumps(plan, indent=2),
            "default_source": source_candidate,
        },
    )


@app.get("/runtime-matrix", response_class=HTMLResponse)
def runtime_matrix_page(request: Request):
    matrix = runtime_matrix({})
    return templates.TemplateResponse(
        request=request,
        name="runtime_matrix.html",
        context={
            "matrix": matrix,
            "matrix_json": json.dumps(matrix, indent=2),
        },
    )


@app.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="datasets.html",
        context={"message": None},
    )


@app.post("/datasets/manifest", response_class=HTMLResponse)
def dataset_manifest(
    request: Request,
    name: str = Form(...),
    files_json: str = Form("[]"),
    sources_json: str = Form("[]"),
    licenses_json: str = Form("[]"),
):
    try:
        file_paths = [Path(value) for value in json.loads(files_json)]
        manifest_path = (
            settings.data_root
            / "datasets"
            / "manifests"
            / f"{name}.json"
        )
        payload = write_manifest(
            manifest_path,
            name=name,
            sources=json.loads(sources_json),
            licenses=json.loads(licenses_json),
            stats={"files": len(file_paths)},
            files=file_paths,
        )
        message = json.dumps(payload, indent=2)
    except Exception as exc:
        message = f"ERROR: {exc}"

    return templates.TemplateResponse(
        request=request,
        name="datasets.html",
        context={"message": message},
    )


@app.get("/failures", response_class=HTMLResponse)
def failures_page(request: Request):
    traces_root = settings.data_root / "artifacts" / "traces"
    traces = sorted(traces_root.glob("*.json")) if traces_root.exists() else []
    return templates.TemplateResponse(
        request=request,
        name="failures.html",
        context={"traces": traces},
    )


@app.get("/system", response_class=HTMLResponse)
def system_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="system.html",
        context={
            "gpus_json": json.dumps(gpu_info(), indent=2),
            "storage_json": json.dumps(
                storage_info(settings.data_root), indent=2
            ),
            "data_root": settings.data_root,
            "github_ready": bool(settings.github_token),
            "hf_ready": bool(settings.hf_token),
        },
    )


@app.get("/publish", response_class=HTMLResponse)
def publish_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="publish.html",
        context={
            "message": None,
            "github_ready": bool(settings.github_token),
            "hf_ready": bool(settings.hf_token),
        },
    )


@app.post("/publish/github", response_class=HTMLResponse)
def publish_github(
    request: Request,
    source_path: str = Form(...),
    repo_url: str = Form(...),
    branch: str = Form("main"),
    message: str = Form("Publish model artifacts"),
):
    try:
        result = push_github(
            data_root=settings.data_root,
            source_dir=Path(source_path),
            repo_url=repo_url,
            branch=branch,
            commit_message=message,
            token=settings.github_token,
        )
    except Exception as exc:
        result = f"ERROR: {exc}"
    return templates.TemplateResponse(
        request=request,
        name="publish.html",
        context={
            "message": result,
            "github_ready": bool(settings.github_token),
            "hf_ready": bool(settings.hf_token),
        },
    )


@app.post("/publish/hf", response_class=HTMLResponse)
def publish_hf(
    request: Request,
    source_path: str = Form(...),
    repo_id: str = Form(...),
    repo_type: str = Form("model"),
    private: str | None = Form(None),
):
    try:
        result = push_huggingface(
            data_root=settings.data_root,
            source_dir=Path(source_path),
            repo_id=repo_id,
            repo_type=repo_type,
            token=settings.hf_token,
            private=private is not None,
        )
    except Exception as exc:
        result = f"ERROR: {exc}"
    return templates.TemplateResponse(
        request=request,
        name="publish.html",
        context={
            "message": result,
            "github_ready": bool(settings.github_token),
            "hf_ready": bool(settings.hf_token),
        },
    )


if __name__ == "__main__":
    uvicorn.run("appliance.app:app", host="0.0.0.0", port=8080, reload=False)
