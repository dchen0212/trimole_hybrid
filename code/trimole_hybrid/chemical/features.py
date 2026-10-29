"""Assemble the archived chemical feature bases in their original column order."""
import numpy as np
import pandas as pd
from rdkit import Chem
from . import core

STANDARD = ("core_maccs_fcfp", "core_pair_torsion", "core_path", "wide_chem")
EXPANDED = ("xl_morgan_family", "xl_topology_family", "xl_full_chemical_prior")
PROFILES = ("core", "descriptors", "morgan1024", "avalon1024", "erg") + STANDARD + EXPANDED


def _columns(prefix, width):
    return [prefix + "." + str(i) for i in range(width)]


def _series(smiles):
    series = pd.Series(list(smiles), dtype=str)
    if len(series) == 0:
        raise ValueError("Provide at least one SMILES record.")
    invalid = [i for i, smi in enumerate(series) if Chem.MolFromSmiles(smi) is None]
    if invalid:
        raise ValueError("Invalid SMILES at zero-based row(s): " + ", ".join(map(str, invalid)))
    return series


def build_features(smiles, profile="core", embedding=None, base_prediction=None):
    if profile not in PROFILES:
        raise ValueError("Unknown fixed feature profile: " + profile)
    series = _series(smiles)
    mols = series.apply(Chem.MolFromSmiles)
    erg_width = core.get_erg_fingerprints(mols.iloc[:1]).shape[1]
    descriptor_names = ["rdkit." + name for name in core.get_chosen_descriptors()]
    core_names = _columns("morgan_count_r2", 1024) + _columns("avalon_count", 1024)
    core_names += _columns("erg", erg_width) + descriptor_names
    if profile == "core":
        matrix, names = core.get_fingerprints(series), core_names
    elif profile == "descriptors":
        matrix = core.sanitize_features(core.get_rdkit_features(mols))
        names = descriptor_names
    elif profile == "morgan1024":
        matrix = core.sanitize_features(core.get_morgan_fingerprints(mols))
        names = _columns("morgan_count_r2", 1024)
    elif profile == "avalon1024":
        matrix = core.sanitize_features(core.get_avalon_fingerprints(mols))
        names = _columns("avalon_count", 1024)
    elif profile == "erg":
        matrix = core.sanitize_features(core.get_erg_fingerprints(mols))
        names = _columns("erg", erg_width)
    elif profile in STANDARD:
        from .standard import extra_chemical_blocks
        matrix = extra_chemical_blocks(series)[profile]
        extra = {
            "core_maccs_fcfp": [("maccs", 167), ("fcfp_count_r2", 1024)],
            "core_pair_torsion": [("atom_pair_count", 1024), ("torsion_count", 1024)],
            "core_path": [("rdkit_path", 1024)],
            "wide_chem": [("maccs", 167), ("morgan_count_r3", 1024),
                          ("fcfp_count_r2", 1024), ("atom_pair_count", 1024),
                          ("torsion_count", 1024), ("rdkit_path", 1024)],
        }[profile]
        names = core_names + [name for prefix, width in extra for name in _columns(prefix, width)]
    else:
        from .expanded import extra_chemical_blocks_xl
        matrix = extra_chemical_blocks_xl(series)[profile]
        morgan = [("morgan_count_r1", 2048), ("morgan_count_r2", 2048),
                  ("morgan_count_r3", 2048), ("fcfp_count_r2", 2048), ("fcfp_count_r3", 2048)]
        topology = [("atom_pair_count", 2048), ("torsion_count", 2048),
                    ("rdkit_path", 2048), ("layered", 2048), ("pattern", 2048)]
        extra = morgan if profile == "xl_morgan_family" else topology
        if profile == "xl_full_chemical_prior":
            extra = morgan + topology
        # Prefix the appended families to distinguish repeated Morgan/core columns.
        names = core_names + _columns("xl.maccs", 167)
        names += [name for prefix, width in extra for name in _columns("xl." + prefix, width)]
    if embedding is not None:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 2 or embedding.shape[0] != len(series):
            raise ValueError("Embedding rows must match the SMILES rows in identical order.")
        matrix = np.concatenate([embedding, matrix], axis=1)
        names = _columns("embedding", embedding.shape[1]) + names
    if base_prediction is not None:
        pred = np.asarray(base_prediction, dtype=np.float32)
        if pred.shape != (len(series),):
            raise ValueError("Base predictions must be one aligned vector.")
        matrix = np.concatenate([matrix, pred[:, None]], axis=1)
        names += ["base_prediction"]
    matrix = core.sanitize_features(matrix)
    if matrix.shape != (len(series), len(names)):
        raise ValueError("Feature columns differ from the archived feature schema.")
    return matrix, names


def task_profile(config):
    if config["task_id"] in ("ames", "bbb_martins"):
        return {"profile": "core", "candidate": None, "base_prediction": False}
    params = config.get("training_parameters", {})
    variant = params.get("feature_block")
    if not variant:
        raise ValueError("This task has component-specific inputs. Use its recorded chemical profile with --block.")
    use_embedding = variant.startswith("embed_chem_")
    profile = variant
    for prefix in ("embed_chem_", "chem_"):
        if profile.startswith(prefix):
            profile = profile[len(prefix):]
            break
    use_prediction = profile.endswith("_base_pred")
    if use_prediction:
        profile = profile[:-len("_base_pred")]
    candidate = params["feature_family"].removeprefix("xl_v4_") if use_embedding else None
    return {"profile": profile, "candidate": candidate, "base_prediction": use_prediction}


def embedding_features(blocks, candidate, n_rows):
    order = {
        "chemberta": ["C"], "kpgt": ["K"], "ept": ["E"],
        "chemberta_kpgt": ["C", "K"], "chemberta_ept": ["C", "E"],
        "kpgt_ept": ["K", "E"], "chemberta_kpgt_ept_gated": ["C", "K", "E"],
    }
    if candidate not in order:
        raise ValueError("Unknown recorded embedding candidate: " + str(candidate))
    arrays = []
    for name in order[candidate]:
        if name not in blocks:
            raise ValueError("Missing aligned embedding block: " + name)
        array = np.asarray(blocks[name], dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != n_rows:
            raise ValueError("Embedding block " + name + " has different rows.")
        arrays.append(array)
    return np.concatenate(arrays, axis=1)
