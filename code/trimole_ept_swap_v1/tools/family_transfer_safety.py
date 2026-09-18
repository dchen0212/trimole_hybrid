"""Molecule-identity safeguards for cross-task family transfer experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping, Sequence, TypeVar


Candidate = TypeVar("Candidate", bound=Mapping[str, object])


@dataclass(frozen=True)
class MoleculeIdentity:
    canonical_smiles: str
    inchikey: str
    connectivity_key: str


def canonical_identity(smiles: object) -> MoleculeIdentity:
    """Return a conservative identity suitable for cross-task overlap checks."""
    try:
        from rdkit import Chem
        from rdkit.Chem import inchi
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise RuntimeError("RDKit is required for molecular identity auditing") from exc

    text = str(smiles).strip()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"invalid SMILES: {text!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    inchikey = inchi.MolToInchiKey(mol)
    if not inchikey:
        raise ValueError(f"could not generate InChIKey for SMILES: {text!r}")
    return MoleculeIdentity(canonical, inchikey, inchikey.split("-", 1)[0])


def canonical_connectivity_keys(smiles_values: Iterable[object]) -> list[str]:
    """Canonicalize SMILES and return connectivity-level InChIKey blocks."""
    return [canonical_identity(value).connectivity_key for value in smiles_values]


def forbidden_keys(
    target_valid_keys: Sequence[str],
    target_test_keys: Sequence[str],
    stage: str,
) -> set[str]:
    """Build the holdout set that must be absent from a pooled training stage."""
    if stage == "selection":
        return set(target_valid_keys) | set(target_test_keys)
    if stage == "final":
        return set(target_test_keys)
    raise ValueError(f"unknown stage: {stage!r}")


def presplit_target_universe_keys(
    target_train_keys: Sequence[str],
    target_valid_keys: Sequence[str],
    target_test_keys: Sequence[str],
) -> set[str]:
    """Build one split-membership-independent target identity set.

    Cross-task sources are filtered against this same set during selection and
    final refitting. The rule depends on target-dataset membership, not on which
    target rows later happen to be assigned to the test partition.
    """

    return set(target_train_keys) | set(target_valid_keys) | set(target_test_keys)


def keep_mask(row_keys: Sequence[str], blocked_keys: set[str]) -> list[bool]:
    return [key not in blocked_keys for key in row_keys]


def overlap_stats(source_keys: Sequence[str], holdout_keys: Sequence[str]) -> dict[str, int]:
    holdout = set(holdout_keys)
    overlapping_rows = [key for key in source_keys if key in holdout]
    return {
        "source_rows": len(source_keys),
        "source_unique_molecules": len(set(source_keys)),
        "holdout_rows": len(holdout_keys),
        "holdout_unique_molecules": len(holdout),
        "overlap_source_rows": len(overlapping_rows),
        "overlap_unique_molecules": len(set(overlapping_rows)),
    }


def molecular_similarity_audit(
    source_smiles: Sequence[object],
    target_smiles: Sequence[object],
    threshold: float = 0.90,
) -> dict[str, int | float]:
    """Audit exact scaffold and high Morgan-Tanimoto cross-dataset overlap."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("Tanimoto threshold must be in (0, 1]")
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError("RDKit is required for molecular similarity auditing") from exc

    def molecules(values: Sequence[object]):
        output = []
        for value in values:
            mol = Chem.MolFromSmiles(str(value).strip())
            if mol is None:
                raise ValueError(f"invalid SMILES: {value!r}")
            output.append(mol)
        return output

    source_molecules = molecules(source_smiles)
    target_molecules = molecules(target_smiles)
    target_scaffolds = {
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        for mol in target_molecules
    }
    target_scaffolds.discard("")
    source_scaffolds = [
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        for mol in source_molecules
    ]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target_fingerprints = [generator.GetFingerprint(mol) for mol in target_molecules]
    high_similarity_rows = 0
    maximum_similarity = 0.0
    for mol in source_molecules:
        fingerprint = generator.GetFingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, target_fingerprints)
        row_maximum = max(similarities, default=0.0)
        maximum_similarity = max(maximum_similarity, float(row_maximum))
        high_similarity_rows += int(row_maximum >= threshold)
    return {
        "source_rows": len(source_molecules),
        "target_rows": len(target_molecules),
        "source_rows_with_target_scaffold": sum(
            bool(scaffold) and scaffold in target_scaffolds for scaffold in source_scaffolds
        ),
        "source_unique_scaffolds": len({value for value in source_scaffolds if value}),
        "target_unique_scaffolds": len(target_scaffolds),
        "source_rows_tanimoto_ge_threshold": high_similarity_rows,
        "tanimoto_threshold": threshold,
        "maximum_tanimoto": maximum_similarity,
    }


def molecular_similarity_keep_mask(
    source_smiles: Sequence[object],
    target_smiles: Sequence[object],
    threshold: float = 0.90,
) -> list[bool]:
    """Keep source rows whose Morgan similarity to every target row is below threshold."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("Tanimoto threshold must be in (0, 1]")
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError("RDKit is required for molecular similarity filtering") from exc

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(value: object):
        mol = Chem.MolFromSmiles(str(value).strip())
        if mol is None:
            raise ValueError(f"invalid SMILES: {value!r}")
        return generator.GetFingerprint(mol)

    target_fingerprints = [fingerprint(value) for value in target_smiles]
    output = []
    for value in source_smiles:
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint(value), target_fingerprints
        )
        output.append(max(similarities, default=0.0) < threshold)
    return output


def anonymized_key(connectivity_key: str) -> str:
    """Hash an identity before writing row-level overlap details."""
    return sha256(connectivity_key.encode("utf-8")).hexdigest()


def select_validation_candidate(
    candidates: Sequence[Candidate], selection_stat: str
) -> Candidate:
    """Select deterministically while rejecting any test-derived selector."""
    if selection_stat not in {"valid_mean", "valid_adjusted"}:
        raise ValueError(f"selection must be validation-only, got {selection_stat!r}")
    if not candidates:
        raise ValueError("candidate list is empty")
    return sorted(
        candidates,
        key=lambda row: (
            -float(row[selection_stat]),
            str(row["feature_set"]),
            int(row["config_index"]),
        ),
    )[0]
