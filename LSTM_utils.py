import numpy as np
import re
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem, rdFingerprintGenerator
import sascorer

import umap
import matplotlib.pyplot as plt
import numpy as np

import selfies as sf

# Disable error messages when a non-valid smiles is encountered
rdBase.DisableLog("rdApp.error")


def conditions_tokens(df):

    def conditions_mapping(condition: str, dict_length: int):
        top_unique = list(df[condition].value_counts().head(dict_length).index)
        stoi = {key: value for value, key in enumerate(top_unique)}
        itos = {value: key for key, value in stoi.items()}

        return stoi, itos

    conditions = ["antibody_name", "payload_name", "indication"]
    stoi_dicts = []
    itos_dicts = []
    dict_length = 5
    for condition in conditions:
        stoi, itos = conditions_mapping(condition, dict_length)
        stoi_dicts.append(stoi)
        itos_dicts.append(itos)

    return stoi_dicts, itos_dicts, dict_length


def canonicalize_motifs(motif_dict):
    clean_dict = {}
    for key, smiles_list in motif_dict.items():
        clean_list = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol:
                clean_smi = Chem.MolToSmiles(mol, canonical=True)
                clean_list.append(clean_smi)

        clean_dict[key] = list(set(clean_list))
    return clean_dict


def tag_dataset_with_motifs(df, motif_dict):
    print("Tagging dataset with ADC motif replacements")

    replacements = []
    counter = 1
    tag_map = {}

    for category in ["cleavable", "heads", "tails"]:
        if category not in motif_dict:
            continue

        for i, smi in enumerate(motif_dict[category]):
            pattern = Chem.MolFromSmiles(smi)
            if pattern is None:
                continue

            # Replace with a dummy atom that would never be used
            tag_smiles = f"[{counter}Po]"
            replacement = Chem.MolFromSmiles(tag_smiles)
            token_name = f"<{category.upper()}_{i}>"

            replacements.append((pattern, replacement, token_name, tag_smiles))
            tag_map[tag_smiles] = token_name
            counter += 1

    # Sort replacements by size
    # Ensures we capture all the big stuff before the small stuff
    replacements.sort(key=lambda x: x[0].GetNumAtoms(), reverse=True)

    def process_row(smi):
        if not smi:
            return ""
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return ""

        for pat, rep, name, tag in replacements:
            if mol.HasSubstructMatch(pat):
                try:
                    test_mol = AllChem.ReplaceSubstructs(
                        mol, pat, rep, replaceAll=True
                    )[0]

                    # check for valence errors
                    Chem.SanitizeMol(test_mol)

                    # if we have a broken bond then default to regular smile not motif dummy
                    if "." not in Chem.MolToSmiles(test_mol):
                        mol = test_mol
                    else:
                        # if it failed then skip the conversion
                        pass
                except:
                    # sometimes rdkit is giving me errors...
                    pass

        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return ""

    # add the tagged smiles to their own column
    df["tagged_smiles"] = df["smiles"].apply(process_row)

    # Filter out empty strings
    df = df[df["tagged_smiles"] != ""]

    print(f"Tagging Complete.")
    return df, tag_map


def build_tag_to_smiles_map(motif_dict):
    tag_to_smiles = {}
    counter = 1

    for category in ["cleavable", "heads", "tails"]:
        for smi in motif_dict[category]:
            tag = f"[{counter}Po]"
            tag_to_smiles[tag] = smi
            counter += 1

    return tag_to_smiles


def selfie_sanitizer(real_smi, tag_to_smiles):
    """
    Applies structural fixes, regex cleanup, and tag replacement to a raw SMILES string.
    """
    # Structural Fixes Dictionary
    fixes = {
        "CCN=[N+]=[N-]": "[N-]=[N+]=NCC",
        "ON1C(=O)CCC1=O": "O=C1CCC(=O)N1O",
        "O=C1CCC(=O)N1O": "O=C1CCC(=O)N1O",
        "ON1C(=O)CC(S(=O)(=O)O)C1=O": "OS(=O)(=O)CC1C(=O)N(O)C1=O",
        "O=C1CC(S(=O)(=O)O)C(=O)N1O": "OS(=O)(=O)CC1C(=O)N(O)C1=O",
        "OS(=O)(=O)CC1C(=O)N(O)C1(=O)": "OS(=O)(=O)CC1C(=O)N(O)C1=O",
        "O=C1C=CC(=O)N1": "O=C1C=CC(=O)N1",
        "Oc1ccc(N=Nc2ccccc2)cc1": "Oc1ccc(cc1)N=Nc2ccccc2",
        "Nc1ccc(CO)cc1": "Nc1ccc(cc1)CO",
        "CC1c2ccccc2-c2ccccc21": "NC(=O)OCC1c2ccccc2-c2ccccc21",
        "O=CC(F)(F)F": "C(=O)C(F)(F)F",
        "Fc1c(F)c(F)c(F)c(F)c1OC(=O)": "C(=O)Oc1c(F)c(F)c(F)c(F)c1F",
        "c1c(F)c(F)c(F)c(F)c1OC=O": "c1c(F)c(F)c(F)c(F)c1OC(=O)",
    }

    # Tag Replacement
    for tag, structure in tag_to_smiles.items():
        if tag in real_smi:
            structure_to_use = fixes.get(structure, structure)
            real_smi = real_smi.replace(tag, structure_to_use)

    # Fluorine/Sulfonyl Specific Patches
    real_smi = re.sub(
        r"\[F\]c(\d)c\(F\)c\(F\)c\(F\)c\(F\)c\1",
        r"c\1c(F)c(F)c(F)c(F)c\1F",
        real_smi,
    )

    if "C(=O)ON1C(=O)CC(S(=O)(=O)[O-])C1(=O)" in real_smi:
        real_smi = real_smi.replace(
            "C(=O)ON1C(=O)CC(S(=O)(=O)[O-])C1(=O)",
            "O=C1CC(S(=O)(=O)[O-])C(=O)N1O",
        )

    if "N1OC=O" in real_smi:
        real_smi = real_smi.replace("N1OC=O", "N1O")

    # Standardizing Carbonyls and Bonds
    real_smi = real_smi.replace("O=C", "C(=O)")

    # Fix Hallucinated Triple Bonds to Oxygen
    real_smi = re.sub(r"O#", "O", real_smi)
    real_smi = re.sub(r"#O", "O", real_smi)

    # Fix Backbone Carbonyls (C=O -> C(=O))
    real_smi = re.sub(r"(C[0-9]*)=O", r"\1(=O)", real_smi)

    # Fix S=O and P=O
    for atom in ["S", "P"]:
        real_smi = real_smi.replace(f"{atom}=O", f"{atom}(=O)")

    # Amide fix
    real_smi = real_smi.replace("C(N)=O", "C(=O)N")

    # 5. Strip Explicit Tags
    real_smi = re.sub(r"\[CH\d?\]", "C", real_smi)
    real_smi = re.sub(r"\[OH\d?\]", "O", real_smi)
    real_smi = re.sub(r"\[NH\d?\]", "N", real_smi)
    real_smi = re.sub(r"\[C\]", "C", real_smi)  # Crucial [C] fix

    real_smi = real_smi.replace("[N+1]", "N").replace("[n+1]", "n")
    real_smi = real_smi.replace("[F]c", "c")
    real_smi = real_smi.replace("Fc", "c")

    # Tetrazine Tautomer Fixes
    real_smi = real_smi.replace("n1cnnc1C", "c1nnc(C)nn1")
    real_smi = real_smi.replace("C1=NN=C(C)N=N1", "c1nnc(C)nn1")

    # 6. Cleanup Trailing Garbage
    real_smi = re.sub(r"(=O)+$", "", real_smi)

    # 7. Aromaticity Fallback Check
    # (If RDKit fails, try converting aromatic lower case to aliphatic upper case)
    if Chem.MolFromSmiles(real_smi) is None:
        if any(char in real_smi for char in ["c", "n", "o", "s"]):
            fallback = (
                real_smi.replace("n", "N")
                .replace("c", "C")
                .replace("o", "O")
                .replace("s", "S")
            )
            if Chem.MolFromSmiles(fallback) is not None:
                real_smi = fallback

    return real_smi


def compute_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=np.float32)

    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = mfgen.GetFingerprintAsNumPy(mol)
    # need float 32 for pytorch compatibility
    return np.array(fp, dtype=np.float32)
