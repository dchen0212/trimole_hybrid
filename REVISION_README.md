# Trimole-Hybrid revision code and analysis archive

Version 2026.10.24. DOI: https://doi.org/10.5281/zenodo.23602309

This release collects the retained source, aggregate results and supplementary material supporting the October 2026 revision. The historical repository is preserved under `historical_repository/`. Use `revision/` for the added controls, corrected result ledgers, target-only training scripts, paired uncertainty and subgroup analyses. The companion table identifiers retain their legacy filenames; `revision/NUMBERING_MAP.md` maps them to Tables S1–S14 and Figures S1–S3.

## Start here

- `revision/Supplement_v92_20pages.pdf`: current 20-page supplement (v92).
- `revision/supplement_source/main.tex`: corresponding supplement source and figures.
- `revision/Supplementary_Data_v92.zip`: complete aggregate result tables and protocols; includes corrected FLAML CYP2D6 substrate v02 records.
- `revision/code/`: experiment and plotting source snapshots, including target-only KPGT/Uni-Mol fusion and CYP3A4 kernel refinement.
- `revision/flaml_evidence/`: the authoritative 110-run search ledger, configuration counts and source hashes.
- `revision/target_only_evidence/`: final retained scores, seed records, lineage and the history of attempted methods.
- `revision/statistics/`: saved interval results, protocols and analysis source.
- `RESULT_CODE_MAP.csv` and `EXTERNAL_DEPENDENCIES.md`: entry points and required inputs.
- `SOURCE_MANIFEST.json` and `MANIFEST_SHA256.json`: provenance and byte-level inventory.

The 22-task recovered-pool controls, 20-task development study and six-task reused official-test evaluation answer different questions. Their protocols and results are separate. Historical endpoint adoption used official test results; the tables are historical/descriptive records. Target-only retraining removes related-task supervision and is benchmark reuse, with public-pretraining exposure outside this archive's verification scope.

## Verify the download

Run `python3 verify_archive.py` from this folder. It checks SHA-256 and file sizes only. It does not fit, predict, score or bootstrap any model.

## Build the supplement

With TeX Live, run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice in `revision/supplement_source/`. This source and the included PDF use the same result values and numbering as the manuscript v92 and reviewer reply v92 at packaging. Their availability statements are updated with this DOI in separate delivery copies.

## Scope

The archive supports source inspection, file verification and rebuilding the supplement. Training and statistical recalculation additionally require the official datasets, pretrained checkpoints and saved row-level prediction inputs documented in `EXTERNAL_DEPENDENCIES.md`. They are not bundled. No new training, inference, evaluation or bootstrap was performed to prepare this release. We checked the newly archived Python files for syntax without executing their experiment entry points.

Original project code follows the existing Apache-2.0 release metadata. Bundled third-party components retain their own notices and licenses. OUP template files and third-party checkpoint binaries are not newly redistributed. See `LICENSE` and `NOTICE`.

## Updates in version 2026.10.24

- `revision/s4_pool_180s/`: completed 22-task, five-run 180-s searches, source bindings, protocols and run evidence.
- `revision/revision_budget_checks/`: aggregate budget comparisons for both candidate-pool regimes.
- `revision/Manuscript_v92_clean.pdf` and `revision/Supplement_v92_20pages.pdf`: synchronized compiled documents.
- Solubility main-table and recovered-pool coverage: 1,995 test rows; external baseline full-test coverage: 1,997.
- Root LICENSE contains the complete Apache-2.0 text. Third-party notices and licenses are retained.
- Files prefixed PREVIOUS_RELEASE document the preceding archive; MANIFEST_SHA256.json inventories this release.
