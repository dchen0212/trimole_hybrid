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
