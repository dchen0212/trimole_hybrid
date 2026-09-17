# Release Artifacts

The generated release candidate is a lightweight, immutable code-and-audit
snapshot intended for both GitHub Releases and Zenodo. It is not generated from
the historical May 2026 server ZIP.

Build it from a clean frozen worktree with:

```bash
python tools/build_zenodo_archive_v1.py \
  --repo-root . \
  --output-dir release_artifacts/build
```

The builder writes a ZIP, `ARCHIVE_MANIFEST.json` and `SHA256SUMS`. Review all
three before uploading the unchanged ZIP to both services.

The public archive excludes official TDC datasets, full sample-level label
files, trained weights, cached embeddings, arrays and serialized estimators.
Private reviewer-only evidence must not be added to the public ZIP unless its
license and disclosure status have been reviewed.
