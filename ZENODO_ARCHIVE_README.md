# Zenodo Archive Scope

This archive is the lightweight, reproducible snapshot associated with the
BIOINF-2026-2212 major revision. It contains source code, executable tests,
environment specifications, frozen task/configuration manifests, aggregate
results and leakage/statistical audit summaries.

Original project code and revision workflows are released under Apache-2.0.
Bundled third-party components retain their original licenses and notices as
documented in `NOTICE` and the license files located with those components.

It does not redistribute official TDC datasets, unrestricted sample-level
labels, trained model binaries, cached embeddings, NumPy arrays or serialized
estimators. Dataset acquisition and split provenance are documented by the
versioned manifests and Supplementary Tables S13, S18 and S19.

A separate companion ZIP contains the strict common-pool candidate and method
predictions with labels and SMILES removed. It improves score auditing but does
not replace the official TDC datasets or excluded encoder weights. Table S24k
shows that the common-pool comparison matches candidate and option counts, not
end-to-end compute; the frozen revision manifest does not establish prospective
independence from historical benchmark development.

The archive manifest records the exact Git commit, path, size and SHA-256 hash
of every included file. The ZIP must be uploaded unchanged to GitHub Releases
and Zenodo so that both resources identify the same experimental snapshot.
