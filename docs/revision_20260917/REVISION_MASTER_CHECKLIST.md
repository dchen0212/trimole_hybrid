# BIOINF-2026-2212 Revision Master Checklist

Status values: `DONE-EVIDENCE`, `DONE-TEXT`, `AUTHOR-ACTION`.

## Editor and submission requirements

| Item | Evidence/output | Status |
| --- | --- | --- |
| Potential data leakage | Target-wide exact/similarity filtering, corrected reruns, independent validation, Tables S20b--d | DONE-EVIDENCE |
| Archival code DOI | Zenodo-ready release package and GitHub release checklist prepared; DOI must be minted by an author | AUTHOR-ACTION |
| Marked manuscript | Red-text LaTeX source and PDF | DONE-TEXT |
| Clean supplement | Clean LaTeX source, PDF and S1--S24 workbook | DONE-TEXT |
| Figure alt text | Main and supplementary figures have independent alt-text descriptions | DONE-TEXT |

## Reviewer 1

| ID | Completed action | Status |
| --- | --- | --- |
| R1.1 | Distinguished ADME from toxicology and limited claims to the 22-task benchmark-defined subset | DONE-TEXT |
| R1.2 | Added the peer-reviewed TDC paper as the primary benchmark citation | DONE-TEXT |
| R1.3 | Added task-level provenance, assay, sample-size, split and applicability-domain information in Table S19 | DONE-EVIDENCE |
| R1.4 | Added OpenADMET/avoid-ome and assay-noise context and tempered real-world claims | DONE-TEXT |

## Reviewer 2

| ID | Completed action | Evidence/output | Status |
| --- | --- | --- | --- |
| R2.1 | Reframed contribution as validation-governed task-wise selection/ensemble learning, not a new encoder | revised title, abstract, Figure 1 and Discussion | DONE-TEXT |
| R2.2 | Separated feature fitting, validation decisions, frozen refit and test scoring; manifest is written before test labels are read | staged runners, tests and Figure S2 | DONE-EVIDENCE |
| R2.3 | Compared eight strategies on one frozen nine-family pool, identical splits and five seeds; task-wise selector and AutoML each search six validation configurations | Tables S24g--j | DONE-EVIDENCE |
| R2.4 | Added seed SD and seed-by-sample hierarchical bootstrap; restricted single-ensemble BBB historical inference | Table S24i and revised statistical language | DONE-EVIDENCE |
| R2.5 | Replaced raw cross-metric averages with within-task normalized rank loss and reported selection frequency, entropy and switching | Table S23 and revised Figure 3 | DONE-EVIDENCE |
| R2.6 | Replaced stage-specific filtering with one pre-split target identity universe plus Tanimoto/scaffold audits; reran both affected tasks | Tables S20b--d and validation JSON files | DONE-EVIDENCE |
| R2.7 | Added subgroup N, prevalence, validity checks, matched controls and paired CIs; removed the old overall-average interpretation | Table S22 and revised Figure 2d | DONE-EVIDENCE |

## Interpretation constraints

1. The common-pool task-wise selector is not uniformly superior: it has mean rank 3.727 and is out-ranked by validation top-2/top-3 averaging overall.
2. Against per-task single selection it improves 11/22 tasks, with five positive BH-FDR-significant results and no significant negative result under hierarchical bootstrap.
3. Against common-pool AutoML it improves 13/22 tasks, with three significant positive and three significant negative results.
4. Historical public-reference and three-view FLAML comparisons are sensitivity analyses, not matched-budget evidence.
5. Corrected family-transfer results replace all submitted values; no top-reference claim is retained for those endpoints.
6. Scaffold overlap is an applicability-domain audit, not automatically leakage. Exact and high-similarity source rows are excluded by a rule fixed before fitting.

## Final document format

- Reviewer comments: black and reproduced verbatim.
- Author responses: blue, with key outcomes in bold.
- Every response includes `Response`, `Action taken`, `Changes in manuscript`, and a source location.
- Revised manuscript: changed text in red.
- Final supplement: clean formatting without revision colours.
