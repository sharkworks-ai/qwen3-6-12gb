# Apply this bundle to the dev branch

Copy the bundle contents into the repository root while checked out on `dev`.

```bash
git add appliance docs/APPLIANCE.md .github/workflows/appliance.yml tests/test_appliance.py
git commit -m "Build self-contained GPU training appliance with web UI"
git push origin dev
```

These files are additive and are intended to coexist with the current project while the appliance architecture replaces the old remote-worker direction on `dev`.
