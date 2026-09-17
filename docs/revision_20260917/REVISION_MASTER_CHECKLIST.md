# BIOINF-2026-2212 Revision Master Checklist

Status values: `DONE-EVIDENCE`, `RUNNING`, `PENDING-EXPERIMENT`, `PENDING-TEXT`, `BLOCKED`.

## Editor and submission requirements

| Item | Required action | Evidence/output | Status |
| --- | --- | --- | --- |
| Potential data leakage | Audit overlaps, implement leakage-safe protocol, rerun, independently validate, replace affected claims | `docs/revision_20260917/P0_FINDINGS.md` and validation JSON files | DONE-EVIDENCE |
| Archival code DOI | Freeze the exact revision code and data manifest on Zenodo/Figshare/Software Heritage | DOI plus GitHub URL in Availability statement | PENDING-TEXT |
| Marked manuscript | Produce a red-text marked LaTeX/PDF version | Marked PDF and complete LaTeX source | PENDING-TEXT |
| Clean supplement | Upload final supplement without revision colors | Clean supplementary PDF | PENDING-TEXT |
| Figure alt text | Add concise alt text under every figure legend | Main manuscript figure captions | PENDING-TEXT |

## Reviewer 1

| ID | Reviewer request | Required change | Status |
| --- | --- | --- | --- |
| R1.1 | Explain broader ADMET heterogeneity and distinction between ADME and toxicology | Expand Introduction/Discussion; avoid treating 22 TDC tasks as the full ADMET problem | PENDING-TEXT |
| R1.2 | Cite the peer-reviewed TDC paper | Add Huang et al., Nature Chemical Biology (2022), and use it as the primary TDC citation | PENDING-TEXT |
| R1.3 | Discuss dataset provenance, quality and applicability domain | Add task-level provenance/assay/size/endpoint table and limitations of scaffold-split benchmark transferability | PENDING-EXPERIMENT |
| R1.4 | Discuss OpenADMET and measurement noise | Add OpenADMET/avoid-ome and Landrum-Riniker assay-noise context; temper real-world claims | PENDING-TEXT |

## Reviewer 2

| ID | Reviewer request | Required change | Evidence/output | Status |
| --- | --- | --- | --- | --- |
| R2.1 | Clarify novelty relative to model selection, stacking and AutoML | Reframe as task-wise validation-based model configuration/ensemble system; remove unsupported representation-learning novelty; add controlled AutoML comparison | Controlled baseline protocol; AutoML still pending | RUNNING |
| R2.2 | Resolve ambiguous validation protocol | Specify feature extraction, base fitting, tuning, blend estimation, calibration, selection and final refit as separate stages; prohibit candidate ranking on test | Leakage-safe staged runners and tests | DONE-EVIDENCE |
| R2.3 | Add fair controlled baselines | Global single, per-task single without custom ensembles, uniform average, OOF stacking and equal-budget AutoML | 330/330 selection predictions complete; OOF final running | RUNNING |
| R2.4 | Add uncertainty and avoid overstated top-reference claims | Five-seed uncertainty; paired bootstrap CI/tests when matched baseline predictions exist; descriptive comparison only when public predictions are absent | Sample-level predictions available for 22/22 tasks | PENDING-EXPERIMENT |
| R2.5 | Replace incompatible cross-metric ablation aggregation | Use within-task normalized rank/relative effect; report selection instability separately; define no-task-adaptive baseline exactly | Reanalysis script and replacement Figure 3 required | PENDING-EXPERIMENT |
| R2.6 | Audit family-transfer leakage | Report source tasks, split restrictions, overlap counts and leakage-safe reruns for every affected endpoint | Two independent validation reports; old scores replaced | DONE-EVIDENCE |
| R2.7 | Repair Figure 2d subgroup analysis | Report subgroup N, class prevalence, valid-metric conditions, bootstrap CIs and matched-baseline delta; delete overall-only comparison | New subgroup table and figure required | PENDING-EXPERIMENT |

## Decisions from the meeting

1. Historical results may be reused only when sample-level predictions, official split alignment, metric recomputation and provenance are verified.
2. Public TDC top-2/top-3 scores are descriptive references unless their predictions or methods are rerun under the same protocol.
3. The weak control must have an exact reproducible definition. The current controls use one global modality, per-task single-modality selection, uniform averaging and OOF stacking.
4. Agent-based routing should not be added solely to manufacture novelty. It is optional future work unless a frozen, fair and reproducible baseline can be completed.
5. The response must not claim that no leakage existed. Cross-task molecule overlap did affect two submitted family-transfer results, and the revised scores are lower.
6. The response must not claim that every task was independently trained with no transfer, because the submitted system included family-transfer endpoints.
7. Cross-metric ablations will use within-task rank or normalized effects, not raw signed metric differences.
8. Figure 2d will be replaced rather than defended in its current form if a matched reproducible baseline cannot be established.

## Final document format

- Reviewer comments: black, reproduced verbatim.
- Author responses: blue, with key outcomes in bold.
- Every response: `Response`, `Changes in manuscript`, and exact `Page/Lines` fields.
- Revised manuscript: changed text in red, matching the GS-DTI marked-manuscript example.
- Final supplement: clean formatting without colored revision marks.
