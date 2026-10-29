"""Original fixed standard chemical blocks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdMolDescriptors
from . import core as base

def fp_to_array(fp) -> np.ndarray:
    if hasattr(fp, "GetLength"):
        n = int(fp.GetLength())
    else:
        n = int(fp.GetNumBits())
    arr = np.zeros((n,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def mols_from_smiles(smiles: pd.Series) -> list[Chem.Mol]:
    empty = Chem.MolFromSmiles("")
    mols: list[Chem.Mol] = []
    for smi in smiles.astype(str):
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol if mol is not None else empty)
    return mols


def stack_fps(mols: list[Chem.Mol], fn) -> np.ndarray:
    return np.stack([fp_to_array(fn(mol)) for mol in mols], axis=0)


def extra_chemical_blocks(smiles: pd.Series) -> dict[str, np.ndarray]:
    RDLogger.DisableLog("rdApp.*")
    mols = mols_from_smiles(smiles)
    core = base.get_fingerprints(smiles)
    maccs = stack_fps(mols, MACCSkeys.GenMACCSKeys)
    morgan3 = stack_fps(
        mols,
        lambda mol: rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=3, nBits=1024),
    )
    fcfp2 = stack_fps(
        mols,
        lambda mol: rdMolDescriptors.GetHashedMorganFingerprint(
            mol, radius=2, nBits=1024, useFeatures=True
        ),
    )
    atom_pair = stack_fps(
        mols,
        lambda mol: rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=1024),
    )
    torsion = stack_fps(
        mols,
        lambda mol: rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol, nBits=1024),
    )
    rdk_path = stack_fps(mols, lambda mol: RDKFingerprint(mol, fpSize=1024))

    blocks = {
        "core": core,
        "core_maccs_fcfp": np.concatenate([core, maccs, fcfp2], axis=1),
        "core_pair_torsion": np.concatenate([core, atom_pair, torsion], axis=1),
        "core_path": np.concatenate([core, rdk_path], axis=1),
        "wide_chem": np.concatenate([core, maccs, morgan3, fcfp2, atom_pair, torsion, rdk_path], axis=1),
    }
    return {name: base.sanitize_features(value.astype(np.float32)) for name, value in blocks.items()}
