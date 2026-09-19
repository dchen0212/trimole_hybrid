# GitHub and Zenodo Release Checklist

Complete these checks only after the revised manuscript, controlled baselines,
statistical analyses and supplementary tables are frozen.

1. Confirm the final repository URL: `https://github.com/dchen0212/trimole_hybrid`.
2. Confirm the revised title and author order in `CITATION.cff` and `.zenodo.json`.
3. Confirm the Apache-2.0 top-level license and retained third-party notices in `NOTICE`.
4. Run the full test suite and record the command, commit and result in the release notes.
5. Confirm that official TDC data, trained weights, cached embeddings, serialized models and unrestricted sample-level labels are not committed or archived.
6. Rebuild S19--S24l and verify their provenance JSON/checksum files against the frozen commit. Confirm the `target_only` primary scores are not mislabeled as family transfer, and that S20c--S20d remain labeled transductive sensitivity evidence.
7. Run `tools/build_zenodo_archive_v1.py` from a clean checkout. Review its allowlist report and `SHA256SUMS`.
8. Create a GitHub Release from the same commit and attach the generated Zenodo-ready ZIP plus the separate label-free prediction bundle. Check that all 2,860 prediction CSVs omit labels and SMILES; record the bundle SHA-256.
9. After explicit author confirmation, upload both unchanged assets to one Zenodo deposition and mint the DOI.
10. Add the DOI to `CITATION.cff`, the README, the manuscript Availability statement and the response letter. Rebuild the final manuscript PDFs without changing experimental files.

Release gates:

- The Git worktree is clean and the archived commit matches the GitHub Release tag.
- `ARCHIVE_MANIFEST.json` contains the commit, file sizes and SHA-256 hashes.
- No rejected binary suffix or file above the configured size limit is present.
- All 22 tasks are represented in the frozen data and prediction manifests.
- The release notes state that this is a retrospective candidate-pool-matched reanalysis, not an untouched prospective test or an end-to-end compute-matched comparison.
- No DOI is published until the corresponding author confirms the final snapshot.
