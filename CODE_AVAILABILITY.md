# Code and Artifact Availability

This repository contains the source code, supplementary tables and lightweight audit material supporting the Trimole-Hybrid ADMET manuscript.

## Public repository

Planned public repository:

- `https://github.com/dchen0212/trimole_hybird`

The repository is currently private and can be made public after license and artifact-release decisions are finalized.

## Included artifacts

- Source code for molecular representation wrappers, fusion models, endpoint heads, prediction-level ensembles and audit scripts.
- Supplementary tables and summary-level formal benchmark, ablation, endpoint-selection and case-study audits.
- Supplementary tables used by the manuscript.

## External dependencies

Official TDC benchmark data should be obtained from Therapeutics Data Commons. Large trained weights, cached features and serialized estimators are not included in the Git repository. Split-level prediction audits may contain labels and should be distributed only when compatible with the relevant data-use terms.

## Release artifact

For reviewer inspection, attach the full lightweight server-audit package to a GitHub Release:

- `trimole_hybrid_server_code_pull_20260524.zip`

Before public release, upload the local release asset with this filename to a GitHub Release.
