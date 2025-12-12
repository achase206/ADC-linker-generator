import pandas as pd
import numpy as np
import torch

from LSTM_classes import (
    Tokenizer,
    LSTMGenModel,
    adcDataset,
)
from LSTM_helpers import (
    conditions_tokens,
    canonicalize_motifs,
    tag_dataset_with_motifs,
    build_tag_to_smiles_map,
)

if __name__ == "__main__":

    print("#" * 30)
    print("ADC LINKER GENERATOR")
    print("#" * 30)
    print()

    """LSTM Setup"""
    print("LSTM setup starting...")

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
    # This replace complex motifs with our dummy atoms
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

    print("LSTM setup complete")
    print()
    print("Use ctrl-C to exit at any time")
    print()

    print("Which folder has your models")
    print("'models' has default pre-trained")
    while True:
        try:
            model_directory = input("Folder name: ")
            if isinstance(model_directory, str):
                print(f"Directory set: {model_directory}")
                break
        except ValueError:
            print("Invalid directory, try again")

    valid_operation = [1, 2, 3]

    print()
    print("What would you like to do? (enter a number)")
    print("1. Generate Linkers\n2. Assess Model Performance\n3. Generate UMAPs")
    while True:
        try:
            operation = input("Operation: ")
            if operation in valid_operation:
                break
        except ValueError:
            print("Invalid operation, try again")

    if operation == 1:
        valid_operation = [1, 2, 3, 4]

        print()
        print("Select your antibody:")
        print("1. Cetuximab")
        print("2. MCLL0517A")
        print("3. Patritumab")
        print("4. Lorvotuzumab")
        while True:
            try:
                ab = input("Antibody: ")
                if ab in valid_operation:
                    break
            except ValueError:
                print("Invalid antibody, try again")

        print()
        print("Select your payload:")
        print("1. Puromycin")
        print("2. PBD dimer")
        print("3. ADC-C1 payload")
        print("4. Rachelmycin")
        while True:
            try:
                pay = input("Payload: ")
                if pay in valid_operation:
                    break
            except ValueError:
                print("Invalid payload, try again")

        print()
        print("Select your cancer target:")
        print("1. Triple negative breast cancer")
        print("2. Acute myeloid leukaemia")
        print("3. Breast cancer")
        print("4. Small cell lung cancer")
        while True:
            try:
                tar = input("Target: ")
                if tar in valid_operation:
                    break
            except ValueError:
                print("Invalid payload, try again")

        print()
        print("How many linkers to generate?")
        while True:
            try:
                num_linkers = input("Num linkers: ")
                if isinstance(num_linkers, int):
                    break
            except ValueError:
                print("Invalid number, try again")

        linkers = generation_loop(
            model=gen_model,
            tokenizer=tokenizer,
            tag_to_smiles=tag_to_smiles,
            ab=ab,
            pay=pay,
            tar=tar,
            bias_strength=10.0,
            temp=0.5,
            target_count=20,
        )

        print("------ Linker Report -------")
        print()
        print(f"Antibody: {itos_dicts[0][ab]}")
        print(f"Payload: {itos_dicts[1][pay]}")
        print(f"Cancer Target: {itos_dicts[2][tar]}")
        print()
        for i, result in enumerate(linkers):
            print(f"------- Linker {i+1} ----------")
            print(f"SMILE: {result['SMILES']}")
            print(f"SA: {result['SA']:.4f}")
            print(f"TPSA: {result['TPSA']:.4f}")
            print(f"QED: {result['QED']:.4f}")
            print(f"LogP: {result['LogP']:.4f}")
            print(f"CSSP3: {result['CSP3']:.4f}")
            print("---------------------------")

    if operation == 2:
        print("Generating 1000 random linkers...")
        gen_model = LSTMGenModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            vocab_size=tokenizer.vocab_size,
            padding_idx=tokenizer.pad_token,
            output_dim=tokenizer.vocab_size,
            condition_count=5,
        )

        gen_model.load_state_dict(
            torch.load(
                f"{model_directory}/model_RL_gen_weights_v2.pth", weights_only=True
            )
        )

        real_smiles = adc_df["smiles"]

        results = performance_metrics(
            model=gen_model,
            real_smiles_series=real_smiles,
            tokenizer=tokenizer,
            tag_to_smiles=tag_to_smiles,
            bias_strength=15.0,
            temp=0.70,
            target_count=1000,
        )

        print("Performance for 1000 Generated Linkers")
        print(f"validity = {results["validity"]:.4f}")
        print(f"novelty = {results["novelty"]:.4f}")
        print(f"unique = {results["unique"]:.4f}")
        print(f"tanimoto = {results["tanimoto"]:.4f}")
        print(f"SA Avg = {results["sa_avg"]:.4f} \tstd = {results["sa_std"]:.4f}")
        print(f"QED AVG = {results["qed_avg"]:.4f} \tstd = {results["qed_std"]:.4f}")
        print(f"LogP AVG = {results["logp_avg"]:.4f} \tstd = {results["logp_std"]:.4f}")
        print(f"TPSA AVG = {results["tpsa_avg"]:.4f} \tstd = {results["tpsa_std"]:.4f}")
        print(f"CSP3 AVG = {results["csp3_avg"]:.4f} \tstd = {results["csp3_std"]:.4f}")

    if operation == 3:
        valid_operation = [1, 2, 3]
        print("Select from three UMAP visualization options:")
        print("1. Pre-RL on real linkers")
        print("2. Post-RL on real linkers")
        print("3. Post-RL on generated linkers")

        while True:
            try:
                option = input("Option: ")
                if operation in valid_operation:
                    break
            except ValueError:
                print("Invalid option, try again")

        viz_dataset = adcDataset(
            df=combo_df,  # or perfect_df
            tokenizer=tokenizer,
            stoi_dicts=stoi_dicts,
            source_type="real",
        )

        gen_model = LSTMGenModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            vocab_size=tokenizer.vocab_size,
            padding_idx=tokenizer.pad_token,
            output_dim=tokenizer.vocab_size,
            condition_count=5,
        )

        if option == 1:
            gen_model.load_state_dict(
                torch.load(
                    f"{model_directory}/model_gen_weights_perfect.pth",
                    weights_only=True,
                )
            )
            umap_training_data(
                model=gen_model,
                dataset=viz_dataset,
                tokenizer=tokenizer,
                itos_dicts=itos_dicts,
                num_samples=2000,
                color_by="antibody",
            )

        if option == 2:
            gen_model.load_state_dict(
                torch.load(
                    f"{model_directory}/model_RL_gen_weights_v2.pth", weights_only=True
                )
            )
            umap_training_data(
                model=gen_model,
                dataset=viz_dataset,
                tokenizer=tokenizer,
                itos_dicts=itos_dicts,
                num_samples=2000,
                color_by="antibody",
            )

        if option == 3:
            gen_model.load_state_dict(
                torch.load(
                    f"{model_directory}/model_RL_gen_weights_v2.pth", weights_only=True
                )
            )
            umap_generated_data(
                model=gen_model,
                tokenizer=tokenizer,
                itos_dicts=itos_dicts,
                bias_strength=15.0,
                temperature=0.7,
                color_by="antibody",  # Options: "antibody", "payload", "indication"
            )
