# GitHub Release Checklist

Before making the repository public:

1. Confirm the final repository URL: `https://github.com/dchen0212/trimole_hybrid`.
2. Add the final author list, affiliation and publication citation after acceptance.
3. Choose a license and replace `LICENSE_PENDING.md` with the final license file.
4. Confirm that official TDC data, trained weights, cached embeddings and serialized models are not committed.
5. This public-upload copy already excludes `results_audit/` because some split-level prediction files contain `y_true` labels. Provide those audits only as a private reviewer archive if appropriate.
6. Upload `trimole_hybrid_server_code_pull_20260524.zip` as a GitHub Release asset rather than committing it to git.
7. Optionally archive the release on Zenodo and add the DOI to the manuscript and README.

Suggested repository layout:

- GitHub repository upload folder: `trimole-hybrid-admet-github-public-upload/`
- GitHub Release asset: `trimole_hybrid_server_code_pull_20260524.zip`
