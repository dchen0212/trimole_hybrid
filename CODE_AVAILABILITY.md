# Code and Artifact Availability

This repository contains the source code, Supplementary Information files, supplementary tables and lightweight audit material supporting the Trimole-Hybrid ADMET manuscript.

## Public repository

The public development repository is:

- `https://github.com/dchen0212/trimole_hybrid`

An archival DOI snapshot is not yet available. It must be created for the revision after the leakage-controlled reruns, manuscript-facing audit artifacts and license have been finalized.

## Included artifacts

- Source code for molecular representation wrappers, fusion models, endpoint heads, prediction-level ensembles and audit scripts.
- Supplementary tables and summary-level formal benchmark, ablation, endpoint-selection and case-study audits.
- Supplementary Information source/PDF, supplementary figures and supplementary tables used by the manuscript.

## External dependencies

Official TDC benchmark data should be obtained from Therapeutics Data Commons. Large trained weights, cached features and serialized estimators are not included in the Git repository. Split-level prediction audits may contain labels and should be distributed only when compatible with the relevant data-use terms.

## Release artifact

For reviewer inspection, attach the full lightweight server-audit package to a GitHub Release:

- `trimole_hybrid_server_code_pull_20260524.zip`

Do not treat the historical archive as the revised formal result bundle. Create a new release asset only after the leakage-controlled reruns and provenance checks are complete, then archive that exact release with Zenodo or an equivalent DOI-granting service.
