# GitHub and Zenodo Release Checklist

Complete these checks only after the revised manuscript, controlled baselines,
statistical analyses and supplementary tables are frozen.

1. Confirm the final repository URL: `https://github.com/dchen0212/trimole_hybrid`.
2. Confirm the revised title and author order in `CITATION.cff` and `.zenodo.json`.
3. Choose a license and replace `LICENSE_PENDING.md` with the final license file.
4. Run the full test suite and record the command, commit and result in the release notes.
5. Confirm that official TDC data, trained weights, cached embeddings, serialized models and unrestricted sample-level labels are not committed or archived.
6. Rebuild S19--S23 and verify their provenance JSON/checksum files against the frozen commit.
7. Run `tools/build_zenodo_archive_v1.py` from a clean checkout. Review its allowlist report and `SHA256SUMS`.
8. Create a GitHub Release from the same commit and attach the generated Zenodo-ready ZIP.
9. After explicit author confirmation, upload that exact ZIP to Zenodo and mint the DOI.
10. Add the DOI to `CITATION.cff`, the README, the manuscript Availability statement and the response letter. Rebuild the final manuscript PDFs without changing experimental files.

Release gates:

- The Git worktree is clean and the archived commit matches the GitHub Release tag.
- `ARCHIVE_MANIFEST.json` contains the commit, file sizes and SHA-256 hashes.
- No rejected binary suffix or file above the configured size limit is present.
- All 22 tasks are represented in the frozen data and prediction manifests.
- No DOI is published until the corresponding author confirms the final snapshot.
