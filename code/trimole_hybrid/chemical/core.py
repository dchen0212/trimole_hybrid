"""Original fixed Morgan/Avalon/ErG/descriptor feature basis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem import rdReducedGraphs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

def count_to_array(fingerprint):
    n = int(fingerprint.GetLength()) if hasattr(fingerprint, "GetLength") else 0
    array = np.zeros((n,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def get_avalon_fingerprints(molecules, n_bits=1024):
    fps = molecules.apply(lambda x: GetAvalonCountFP(x, nBits=n_bits))
    fps = fps.apply(count_to_array)
    return np.stack(fps.values)


def get_morgan_fingerprints(molecules, n_bits=1024, radius=2):
    fps = molecules.apply(lambda x: GetHashedMorganFingerprint(x, nBits=n_bits, radius=radius))
    fps = fps.apply(count_to_array)
    return np.stack(fps.values)


def get_erg_fingerprints(molecules):
    fps = molecules.apply(rdReducedGraphs.GetErGFingerprint)
    return np.stack(fps.values)


def get_chosen_descriptors():
    return ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1',
        'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
        'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
        'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount',
        'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
        'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge',
        'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
        'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
        'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors',
        'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
        'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
        'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1',
        'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
        'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
        'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
        'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
        'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
        'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
        'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
        'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
        'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
        'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
        'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo',
        'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
        'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
        'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
        'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
        'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
        'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
        'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
        'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
        'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
        'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
        'fr_unbrch_alkane', 'fr_urea', 'qed']


def get_rdkit_features(molecules):
    calc = MolecularDescriptorCalculator(get_chosen_descriptors())
    X = molecules.apply(lambda x: np.array(calc.CalcDescriptors(x)))
    return np.vstack(X.values)


def sanitize_features(x: np.ndarray, clip_value: float = 1e6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32, copy=False)


def get_fingerprints(smiles_series: pd.Series) -> np.ndarray:
    RDLogger.DisableLog('rdApp.*')
    mols = smiles_series.apply(Chem.MolFromSmiles)
    return sanitize_features(np.concatenate(
        [
            get_morgan_fingerprints(mols),
            get_avalon_fingerprints(mols),
            get_erg_fingerprints(mols),
            get_rdkit_features(mols),
        ],
        axis=1,
    ))
