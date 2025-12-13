import pandas as pd
import torch
import os
from rdkit import Chem
import selfies as sf

os.environ["RL_WARNINGS"] = "False"

from LSTM_classes import Tokenizer, LSTMGenModel, CriticModel, adcDataset
from LSTM_helpers import (
    train_LSTM_gen,
    LSTM_model_RL,
    train_classifiers,
)

from LSTM_utils import (
    conditions_tokens,
    canonicalize_motifs,
    tag_dataset_with_motifs,
    build_tag_to_smiles_map,
)

if __name__ == "__main__":

    print()
    print("#" * 30)
    print("ADC LINKER GENERATOR TRAINING")
    print("#" * 30)
    print()

    print("Name a folder to save your models")
    print("'models' is reserved for pre-trained")
    while True:
        try:
            model_directory = input("Folder name: ")
            if isinstance(model_directory, str) and model_directory != "models":
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                break
        except ValueError:
            print("Invalid directory, try again")

    """LSTM Setup"""
    print("LSTM setup starting...")

    # List of different adc motifs for tokenization
    adc_motifs = {
        "cleavable": [
            "Nc1ccc(cc1)CO",  # PABC Core
            "SS",  # Disulfide Bond
            "CC(C)[C@@H]C=O",  # Valine Backbone
            "CCC(=O)NNC=O",  # Hydrazone
            "CC[C@@H]C=O",  # Alanine
            "OC(=O)[C@@H](CCC)C=O",  # Glutamic acid derivative
            "Oc1ccc(cc1)N=Nc2ccccc2",  # Azo-linker
            "O=[N+]([O-])c1ccccc1",  # Nitro-aromatic
            "NC(=O)[C@@H](CCCC)C=O",  # Citrulline/Glutamine sidechain
            "N[C@@H](C(C)C)C(=O)",  # Valine derivative
            "N[C@@H](CCC(=O)O)C(=O)",  # Glutamic acid
            "N[C@@H](CCCCN)C(=O)",  # Lysine/Ornithine
        ],
        "heads": [
            "O=C1C=CC(=O)N1",  # Maleimide (Cysteine conjugation - Standard)
            "O=C1CCC(=O)N1O",  # NHS Ester (Lysine conjugation - Standard)
            "BrCC(=O)",  # Bromoacetyl (Cysteine conjugation - Alternative)
            "ICC(=O)",  # Iodoacetyl (Cysteine conjugation - Alternative)
            # --- Click Chemistry (Spacer/Linker Handles) ---
            "CCN=[N+]=[N-]",  # Azide (Standard Click)
            "CC#C",  # Alkyne (Standard Click)
            "C#Cc1ccccc1",  # DBCO (Strain-promoted Click - Very Common)
            "C#CC1CC2CC1C2",  # BCN (Strain-promoted Click - High Utility)
            "C1=CCCC=CC1",  # TCO (Trans-cyclooctene - Fast Click)
            # --- Older/Specific but Simple ---
            "O=Cc1ccccc1",  # Benzaldehyde (Hydrazone formation)
        ],
        "tails": [
            "c1ccccc1",  # Phenyl
            "CC(=O)S",  # Thioacetate
            "NC(=O)OCC1c2ccccc2-c2ccccc21",  # Fmoc (Corrected Carbamate)
            "c1ccncc1",  # Pyridine
            "CCS(=O)(=O)O",  # Sulfonate
            "CC(=O)CCC=O",  # Levulinyl
            "C(=O)OC(C)(C)C",  # Boc
            "C(=O)OCc1ccccc1",  # Cbz
        ],
    }

    # load ADC data
    adc_directory = "data/adc_data_filtered.pkl"
    synthetic_directory = "data/synthetic_data.pkl"
    adc_df = pd.read_pickle(adc_directory)
    synth_df = pd.read_pickle(synthetic_directory)
    adc_df["data_type"] = "real"
    synth_df["data_type"] = "synthetic"

    combo_df = pd.concat([adc_df, synth_df])
    combo_df = combo_df.fillna(
        {
            "indication": "unknown",
            "antibody_name": "unknown",
            "payload_name": "unknown",
            "data_type": "real",
        }
    )

    # remove any empty entries first
    combo_df = combo_df[combo_df["smiles"] != ""].reset_index(drop=True)

    # Clean Motifs
    adc_motifs_clean = canonicalize_motifs(adc_motifs)

    # Apply Graph Tags
    # This replaces complex motifs with our dummy atoms
    combo_df, tag_map = tag_dataset_with_motifs(combo_df, adc_motifs_clean)

    # create our tag to smiles map
    tag_to_smiles = build_tag_to_smiles_map(adc_motifs_clean)

    # start conversion of smiles to selfies
    valid_indices = []
    selfies_list = []

    # iterate through each row in df and convert SMILES to SELFIES
    for idx, row in combo_df.iterrows():
        smi = row["tagged_smiles"]
        # attempt the selfies conversion
        try:
            sel = sf.encoder(smi)

            if sel:
                selfies_list.append(sel)
                valid_indices.append(idx)
        except Exception as e:
            continue

    # filter df based on rows where conversion was successful
    combo_df = combo_df.loc[valid_indices].reset_index(drop=True)
    combo_df["selfies"] = selfies_list

    tokenizer = Tokenizer(combo_df, tag_map=tag_map)
    stoi_dicts, itos_dicts, dict_lengths = conditions_tokens(combo_df)

    # aka the grease trap
    def is_boring_linker(smi):
        if "CCCCCC" in smi:
            return True
        if "COCCOCCO" in smi:
            return True
        if "OCCOCCOC" in smi:
            return True

        # if more than 70% is just carbon and oxygen then its boring
        c_count = smi.count("C") + smi.count("c")
        o_count = smi.count("O") + smi.count("o")
        if len(smi) > 20 and (c_count + o_count) / len(smi) > 0.8:
            return True

        return False

    # apply the grease filtering
    boring_mask = combo_df["smiles"].apply(is_boring_linker)
    boring_df = combo_df[boring_mask]
    interesting_df = combo_df[~boring_mask]

    boring_sample = boring_df.sample(frac=0.05)
    combo_df = pd.concat([boring_sample, interesting_df], ignore_index=True)

    # some of the motifs have trouble getting reattached and need to be flipped
    # see selfies_sanitizer for more detailed approach for SELFIES
    fixes = {
        "CCN=[N+]=[N-]": "[N-]=[N+]=NCC",
        "CCBr": "BrCC",
        "BrCC=O": "BrCC(=O)",
        "O=CCBr": "BrCC=O",
        "CC#C": "C#CC",
        "Oc1ccc(N=Nc2ccccc2)cc1": "Oc1ccc(cc1)N=Nc2ccccc2",
        "Nc1ccc(CO)cc1": "Nc1ccc(cc1)CO",
        "CC1c2ccccc2-c2ccccc21": "C(=O)OCC1c2ccccc2-c2ccccc21",
    }

    # update the tag_to_smiles dict with our motif fixes
    patches_applied = 0
    for tag, smi in tag_to_smiles.items():
        if smi in fixes:
            tag_to_smiles[tag] = fixes[smi]
            patches_applied += 1

    print(f"smiles fixes applied: {patches_applied}")

    # get our different reward patterns to help identify perfect linkers df
    reward_patterns = {
        "heads": [
            Chem.MolFromSmiles(s) for s in adc_motifs["heads"] if Chem.MolFromSmiles(s)
        ],
        "cleavable": [
            Chem.MolFromSmiles(s)
            for s in adc_motifs["cleavable"]
            if Chem.MolFromSmiles(s)
        ],
        "tails": [
            Chem.MolFromSmiles(s) for s in adc_motifs["tails"] if Chem.MolFromSmiles(s)
        ],
    }

    # perfect linkers have all three motif categories from our mapping
    def perfect_linkers(smi):
        # same as our calculate struct reward inside env class
        if not smi:
            return False
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return False

        has_head = False
        has_cleavable = False
        has_tail = False

        for pat in reward_patterns["heads"]:
            if mol.HasSubstructMatch(pat):
                has_head = True
                break

        if has_head:
            for pat in reward_patterns["cleavable"]:
                if mol.HasSubstructMatch(pat):
                    has_cleavable = True
                    break

        if has_head and has_cleavable:
            for pat in reward_patterns["tails"]:
                if mol.HasSubstructMatch(pat):
                    has_tail = True
                    break

        # perfect linker
        return has_head and has_cleavable and has_tail

    # filter for perfect linkers and remove boring results
    perfect_mask = combo_df["tagged_smiles"].apply(perfect_linkers)
    perfect_df = combo_df[perfect_mask].copy()
    perfect_df["data_type"] = "real"  # not really but trust

    # Create Datasets
    synth_dataset = adcDataset(
        df=combo_df,
        tokenizer=tokenizer,
        stoi_dicts=stoi_dicts,
        source_type="synthetic",
    )

    easy_dataset = adcDataset(
        df=combo_df,
        tokenizer=tokenizer,
        stoi_dicts=stoi_dicts,
        source_type="real",
    )

    perfect_dataset = adcDataset(
        df=perfect_df,
        tokenizer=tokenizer,
        stoi_dicts=stoi_dicts,
        source_type="real",
    )

    # Train the MLP classifiers for conditional rewards
    print()
    train_classifiers(df=combo_df, stoi_dicts=stoi_dicts, directory=model_directory)

    # Train the generator models...
    print()
    print("Training Generative Model")
    gen_model = LSTMGenModel(
        input_dim=128,
        hidden_dim=256,
        layer_dim=5,
        vocab_size=tokenizer.vocab_size,
        padding_idx=tokenizer.pad_token,
        output_dim=tokenizer.vocab_size,
        condition_count=5,
    )

    # Phase A: Learn Grammar (Synthetic)
    print()
    print("Phase A: Learning Grammar (Synthetic Data)...")
    gen_model = train_LSTM_gen(
        model=gen_model,
        dataset=synth_dataset,
        tokenizer=tokenizer,
        learning_rate=0.0001,
        num_epochs=30,
    )

    # Phase B: Repair Syntax (Mixed Data)
    print()
    print("Phase B: Repairing Syntax (Mixed Data)...")
    mixed_dataset = torch.utils.data.ConcatDataset([synth_dataset, easy_dataset])
    gen_model = train_LSTM_gen(
        model=gen_model,
        dataset=mixed_dataset,
        tokenizer=tokenizer,
        learning_rate=0.0001,
        num_epochs=20,
    )

    # Phase C: Polish on real data (Real Data, Low LR)
    print()
    print("Phase C: Polish on real data...")
    gen_model = train_LSTM_gen(
        model=gen_model,
        dataset=easy_dataset,
        tokenizer=tokenizer,
        learning_rate=0.00005,
        num_epochs=15,
    )
    torch.save(gen_model.state_dict(), f"{model_directory}/model_gen_weights.pth")

    # Phase D: Polish on perfect linker data
    print()
    print("Phase D: Polish on perfect linker data...")
    gen_model.load_state_dict(
        torch.load(f"{model_directory}/model_gen_weights.pth", weights_only=True)
    )
    gen_model = train_LSTM_gen(
        model=gen_model,
        dataset=perfect_dataset,
        tokenizer=tokenizer,
        learning_rate=0.0001,
        num_epochs=20,
    )
    torch.save(
        gen_model.state_dict(), f"{model_directory}/model_gen_weights_perfect.pth"
    )

    # Reinforcement Learning
    print()
    print("Starting Reinforcement Learning")

    gen_model = LSTMGenModel(
        input_dim=128,
        hidden_dim=256,
        layer_dim=5,
        vocab_size=tokenizer.vocab_size,
        padding_idx=tokenizer.pad_token,
        output_dim=tokenizer.vocab_size,
        condition_count=5,
    )

    # Re-init models to ensure clean slate loading
    critic_model = CriticModel(hidden_dim=256, output_dim=1)

    # Load Generator (Final Weights)
    gen_model.load_state_dict(
        torch.load(
            f"{model_directory}/model_gen_weights_perfect.pth", weights_only=True
        )
    )

    RL_trained_model = LSTM_model_RL(
        gen_model,
        critic_model,
        tokenizer,
        stoi_dicts=stoi_dicts,
        reward_motifs=adc_motifs,
        tags_to_smiles=tag_to_smiles,
        learning_rate=0.00005,
        temperature=0.65,
        total_frames=1_000_000,
        num_envs=32,
        frame_steps=150,
        clip_epsilon=0.1,
        kl=0.05,
        bias_strength=15.0,
        classifier_directory=model_directory,
    )

    torch.save(
        RL_trained_model.state_dict(), f"{model_directory}/model_RL_gen_weights_v2.pth"
    )
