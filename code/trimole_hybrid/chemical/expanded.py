"""Original expanded chemical blocks at the recorded 2,048-bit setting."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdMolDescriptors
from . import core as base
v3 = SimpleNamespace(XL_FP_BITS=2048)

def fp_to_array(fp) -> np.ndarray:
    n = int(fp.GetLength()) if hasattr(fp, "GetLength") else int(fp.GetNumBits())
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


def extra_chemical_blocks_xl(smiles: pd.Series) -> dict[str, np.ndarray]:
    RDLogger.DisableLog("rdApp.*")
    n_bits = int(getattr(v3, "XL_FP_BITS", 2048))
    mols = mols_from_smiles(smiles)

    core = base.get_fingerprints(smiles)
    maccs = stack_fps(mols, MACCSkeys.GenMACCSKeys)
    morgan1 = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=1, nBits=n_bits))
    morgan2 = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=2, nBits=n_bits))
    morgan3 = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=3, nBits=n_bits))
    fcfp2 = stack_fps(
        mols,
        lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=2, nBits=n_bits, useFeatures=True),
    )
    fcfp3 = stack_fps(
        mols,
        lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, radius=3, nBits=n_bits, useFeatures=True),
    )
    atom_pair = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedAtomPairFingerprint(m, nBits=n_bits))
    torsion = stack_fps(mols, lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(m, nBits=n_bits))
    rdk_path = stack_fps(mols, lambda m: RDKFingerprint(m, fpSize=n_bits))
    layered = stack_fps(mols, lambda m: Chem.LayeredFingerprint(m, fpSize=n_bits))
    pattern = stack_fps(mols, lambda m: Chem.PatternFingerprint(m, fpSize=n_bits))

    morgan_family = np.concatenate([core, maccs, morgan1, morgan2, morgan3, fcfp2, fcfp3], axis=1)
    topology_family = np.concatenate([core, maccs, atom_pair, torsion, rdk_path, layered, pattern], axis=1)
    full = np.concatenate(
        [
            core,
            maccs,
            morgan1,
            morgan2,
            morgan3,
            fcfp2,
            fcfp3,
            atom_pair,
            torsion,
            rdk_path,
            layered,
            pattern,
        ],
        axis=1,
    )
    return {
        "xl_morgan_family": base.sanitize_features(morgan_family.astype(np.float32)),
        "xl_topology_family": base.sanitize_features(topology_family.astype(np.float32)),
        "xl_full_chemical_prior": base.sanitize_features(full.astype(np.float32)),
    }
