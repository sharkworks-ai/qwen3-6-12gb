# Full workbench integration

The workbench modules are complete building blocks around the existing appliance pipeline.

Wire the existing FastAPI application as follows:

1. `app.include_router(appliance.api.router.router)`.
2. Add `/workbench` using `templates/workbench.html`.
3. Add pages for candidates, datasets, lineage, runtime matrix, failures, and release search.
4. Instantiate `WorkbenchDB(settings.db_path)` beside the existing `ApplianceDB`.
5. Use the existing registered job launcher as the `ReleaseSearch.launch_stage` callback.
6. Add `cryptography` to the Docker image using `appliance/requirements-workbench.txt`.
7. On job completion, invoke `notifications.dispatcher.notify`.
8. Use `release.package.package_release` before GitHub/HF publishing.
9. Do not expose arbitrary benchmark shell commands in the UI; benchmark adapters must be config-pinned by administrators.

The search engine intentionally composes the already-built SFT/profile/prune/recover/merge/quant/eval jobs rather than reimplementing them.
