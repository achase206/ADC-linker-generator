import os
import torch
import numpy as np
import pandas as pd

from LSTM_classes import (
    Tokenizer,
    LSTMScoreModel,
    LSTMGenModel,
    CriticModel,
    adcDataset,
    SmilesGeneratorEnv,
)

from LSTM_helpers import (
    conditions_tokens,
    train_LSTM_scores,
    test_LSTM_scores,
    train_LSTM_gen,
    LSTM_model_RL,
    generate_smiles_conditional,
    check_sequences_conditional,
)

def train(train_target, tokenizer, easy_dataset, hard_dataset, stoi_dicts):
    """
    Train scoring models OR generative model OR RL model.
    This is a clean wrapper around your helper functions.
    """

    training_options = ["scoring", "generation", "reinforce"]
    if train_target not in training_options:
        print("Training target not recognized.")
        return

    model_directory = input("Enter model directory name to save results: ")
    os.makedirs(model_directory, exist_ok=True)


    if train_target == "scoring":

        while True:
            try:
                num_epochs = int(input("Enter epochs to train: "))
                break
            except ValueError:
                print("Number of epochs must be an integer.")

        training_scores = ["SA", "TPSA", "QED", "LogP", "CSP3"]

        for score in training_scores:
            model = train_LSTM_scores(
                dataset=easy_dataset,
                score_type=score,
                tokenizer=tokenizer,
                num_epochs=num_epochs,
            )
            test_LSTM_scores(
                dataset=easy_dataset,
                model=model,
                score_type=score,
                tokenizer=tokenizer,
            )
            torch.save(
                model.state_dict(), f"{model_directory}/{score}_scores_weights.pth"
            )

    if train_target == "generation":
        # prompt for easy epochs and hard epochs, lr
        while True:
            try:
                num_epochs_easy = int(input("Enter epochs to train on easy data: "))
                break
            except ValueError:
                print("Number of epochs must be an integer.")
        while True:
            try:
                num_epochs_hard = int(input("Enter epochs to train on hard data: "))
                break
            except ValueError:
                print("Number of epochs must be an integer.")

        gen_model = LSTMGenModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            vocab_size=tokenizer.vocab_size,
            padding_idx=tokenizer.pad_token,
            output_dim=tokenizer.vocab_size,
            condition_count=4,
        )

        print("\nTraining sequence model on EASY dataset...")
        model_gen = train_LSTM_gen(
            model=gen_model,
            dataset=easy_dataset,
            tokenizer=tokenizer,
            stoi_dicts=stoi_dicts,
            num_epochs=num_epochs_easy,
        )

        torch.save(model_gen.state_dict(), f"{model_directory}/model_gen_weights.pth")
        print("Easy training saved")

        # reload before hard pass
        gen_model.load_state_dict(
            torch.load(f"{model_directory}/model_gen_weights.pth", weights_only=True)
        )

        print("\nTraining sequence model on HARD dataset...")
        model_gen = train_LSTM_gen(
            model=gen_model,
            dataset=hard_dataset,
            tokenizer=tokenizer,
            stoi_dicts=stoi_dicts,
            num_epochs=num_epochs_hard,
        )

        torch.save(model_gen.state_dict(), f"{model_directory}/model_gen_weights.pth")
        print("Hard training saved")

    if train_target == "reinforce":
        # prompt for batches, temps, entropy bonus
        while True:
            try:
                num_batches = int(input("Enter number of batches for RL training: "))
                break
            except ValueError:
                print("Number of batches must be an integer.")

        gen_model = LSTMGenModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            vocab_size=tokenizer.vocab_size,
            padding_idx=tokenizer.pad_token,
            output_dim=tokenizer.vocab_size,
            condition_count=4,
        )
        critic_model = CriticModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            vocab_size=tokenizer.vocab_size,
            padding_idx=tokenizer.pad_token,
            output_dim=1,
            condition_count=4,
        )

        SA = LSTMScoreModel(128, 256, 5, 1, tokenizer)
        TPSA = LSTMScoreModel(128, 256, 5, 1, tokenizer)
        QED = LSTMScoreModel(128, 256, 5, 1, tokenizer)
        LogP = LSTMScoreModel(128, 256, 5, 1, tokenizer)
        CSP3 = LSTMScoreModel(128, 256, 5, 1, tokenizer)

        gen_model.load_state_dict(
            torch.load(f"{model_directory}/model_gen_weights.pth", weights_only=True)
        )
        SA.load_state_dict(torch.load(f"{model_directory}/SA_scores_weights.pth", weights_only=True))
        TPSA.load_state_dict(torch.load(f"{model_directory}/TPSA_scores_weights.pth", weights_only=True))
        QED.load_state_dict(torch.load(f"{model_directory}/QED_scores_weights.pth", weights_only=True))
        LogP.load_state_dict(torch.load(f"{model_directory}/LogP_scores_weights.pth", weights_only=True))
        CSP3.load_state_dict(torch.load(f"{model_directory}/CSP3_scores_weights.pth", weights_only=True))

        RL_trained_model = LSTM_model_RL(
            SA_model=SA,
            TPSA_model=TPSA,
            QED_model=QED,
            LogP_model=LogP,
            CSP3_model=CSP3,
            gen_model=gen_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            total_frames=num_batches,
        )

        torch.save(RL_trained_model.state_dict(), f"{model_directory}/model_RL_gen_weights.pth")
        print("RL model saved.")



def test(test_target, tokenizer, easy_dataset, hard_dataset, stoi_dicts):
    testing_options = ["scoring", "generation", "reinforce"]
    if test_target not in testing_options:
        print("Test target not recognized.")
        return

    print(f"Testing {test} model:")
    model_directory = input("Enter model directory name to load models from: ")

    gen_model = LSTMGenModel(
        128, 256, 5, tokenizer.vocab_size, tokenizer.pad_token, tokenizer.vocab_size, 4
    )
    gen_RL = LSTMGenModel(
        128, 256, 5, tokenizer.vocab_size, tokenizer.pad_token, tokenizer.vocab_size, 4
    )

    gen_model.load_state_dict(
        torch.load(f"{model_directory}/model_gen_weights.pth", weights_only=True)
    )
    gen_RL.load_state_dict(
        torch.load(f"{model_directory}/model_RL_gen_weights.pth", weights_only=True)
    )

    SA = LSTMScoreModel(128, 256, 5, 1, tokenizer)
    TPSA = LSTMScoreModel(128, 256, 5, 1, tokenizer)
    QED = LSTMScoreModel(128, 256, 5, 1, tokenizer)
    LogP = LSTMScoreModel(128, 256, 5, 1, tokenizer)
    CSP3 = LSTMScoreModel(128, 256, 5, 1, tokenizer)

    SA.load_state_dict(torch.load(f"{model_directory}/SA_scores_weights.pth", weights_only=True))
    TPSA.load_state_dict(torch.load(f"{model_directory}/TPSA_scores_weights.pth", weights_only=True))
    QED.load_state_dict(torch.load(f"{model_directory}/QED_scores_weights.pth", weights_only=True))
    LogP.load_state_dict(torch.load(f"{model_directory}/LogP_scores_weights.pth", weights_only=True))
    CSP3.load_state_dict(torch.load(f"{model_directory}/CSP3_scores_weights.pth", weights_only=True))

    if test_target == "scoring":
        training_scores = ["SA", "TPSA", "QED", "LogP", "CSP3"]
        for score in training_scores:
            model = LSTMScoreModel(128, 256, 5, 1, tokenizer)
            model.load_state_dict(
                torch.load(f"{model_directory}/{score}_scores_weights.pth", weights_only=True)
            )
            test_LSTM_scores(easy_dataset, tokenizer, model, score)
        return

    
    if test_target == "generation":

        while True:
            try:
                temperature = float(input("Enter prediction temperature (0.01 - 1.0): "))
                break
            except ValueError:
                print("Temperature must be a float.\n")


        while True:
            try:
                print("-" * 30)
                print("Antibodies (indices may depend on your data!):")
                print("Cetuximab = 0")
                print("MCLL0517A = 1")
                print("Patritumab = 2")
                print("Lorvotuzumab = 3")
                print("-" * 30)
                ab_idx = int(input("Enter antibody idx (int): "))
                break
            except ValueError:
                print("Antibody idx must be an integer.\n")

        while True:
            try:
                print("-" * 30)
                print("Payloads:")
                print("Puromycin = 0")
                print("PBD dimer = 1")
                print("ADC-C1 payload = 2")
                print("Rachelmycin = 3")
                print("-" * 30)
                pay_idx = int(input("Enter payload idx (int): "))
                break
            except ValueError:
                print("Payload idx must be an integer.\n")

        while True:
            try:
                print("-" * 30)
                print("Targets / Indications:")
                print("Triple negative breast cancer = 0")
                print("Acute myeloid leukaemia       = 1")
                print("Breast cancer                 = 2")
                print("Small cell lung cancer        = 3")
                print("-" * 30)
                tar_idx = int(input("Enter target idx (int): "))
                break
            except ValueError:
                print("Target idx must be an integer.\n")

        seq = generate_smiles_conditional(
            gen_model, tokenizer, ab_idx, pay_idx, tar_idx, temperature
        )
        print("\nGenerated SMILES:\n", tokenizer.tokens_to_smiles(seq))

        # Evaluate many sequences under those conditions
        check_sequences_conditional(
            SA, TPSA, QED, LogP, CSP3,
            gen_model, tokenizer,
            ab_idx, pay_idx, tar_idx,
            temperature=temperature,
        )

    if test_target == "reinforce":

        while True:
            try:
                temperature = float(input("Enter prediction temperature (0.01–1.0): "))
                break
            except ValueError:
                print("Temperature must be a float.")

        while True:
            try:
                print("-" * 30)
                print("Antibodies:\n"
                    "Cetuximab = 0\n"
                    "MCLL0517A = 1\n"
                    "Patritumab = 2\n"
                    "Lorvotuzumab = 3")
                print("-" * 30)
                ab_idx = int(input("Enter antibody idx (int): "))
                break
            except ValueError:
                print("Antibody idx must be an integer.")


        while True:
            try:
                print("-" * 30)
                print("Payloads:\n"
                    "Puromycin = 0\n"
                    "PBD dimer = 1\n"
                    "ADC-C1 payload = 2\n"
                    "Rachelmycin = 3")
                print("-" * 30)
                pay_idx = int(input("Enter payload idx (int): "))
                break
            except ValueError:
                print("Payload idx must be an integer.")


        while True:
            try:
                print("-" * 30)
                print("Targets:\n"
                    "Triple negative breast cancer = 0\n"
                    "Acute myeloid leukemia = 1\n"
                    "Breast cancer = 2\n"
                    "Small cell lung cancer = 3")
                print("-" * 30)
                tar_idx = int(input("Enter target idx (int): "))
                break
            except ValueError:
                print("Target idx must be an integer.")

        print("\nSequence generated AFTER RL training:")
        seq = generate_smiles_conditional(
            gen_RL, tokenizer, ab_idx, pay_idx, tar_idx, temperature
        )
        print(tokenizer.tokens_to_smiles(seq))
        print("Length:", len(seq))

        print("\nPre-RL model properties:")
        check_sequences_conditional(
            SA, TPSA, QED, LogP, CSP3,
            gen_model, tokenizer,
            ab_idx, pay_idx, tar_idx,
            temperature=temperature,
        )

        print("\nPost-RL model properties:")
        check_sequences_conditional(
            SA, TPSA, QED, LogP, CSP3,
            gen_RL, tokenizer,
            ab_idx, pay_idx, tar_idx,
            temperature=temperature,
        )


if __name__ == "__main__":

    print("#" * 40)
    print("ADC LINKER GENERATOR")
    print("#" * 40)

    """LSTM Setup"""
    print("LSTM setup starting...")

    adc_directory = "data/adc_data_filtered.pkl"
    df = pd.read_pickle(adc_directory)

    tokenizer = Tokenizer(adc_directory)
    stoi_dicts, itos_dicts, dict_len = conditions_tokens(df)

    # build datasets
    easy_dataset = adcDataset(
        directory=adc_directory,
        tokenizer=tokenizer,
        stoi_dicts=stoi_dicts,
        augment=False,
    )
    hard_dataset = adcDataset(
        directory=adc_directory,
        tokenizer=tokenizer,
        stoi_dicts=stoi_dicts,
        augment=True,
    )

    print("LSTM setup complete")
    print("\nUse Ctrl-C to exit at any time.\n")

    # Choose operation
    valid_ops = ["train", "test"]

    print("Would you like to train or test a model?")
    print("Enter 'train' or 'test'")

    while True:
        op = input("Train or test: ").strip().lower()
        if op in valid_ops:
            break
        print("Invalid operation, try again.")

    # Choose target
    valid_targets = ["scoring", "generation", "reinforce"]

    print(f"\nWhich model to {op}? (scoring, generation, reinforce)")

    while True:
        target = input("Model: ").strip().lower()
        if target in valid_targets:
            break
        print("Invalid model target, try again.")

    # Call train or test
    if op == "train":
        train(
            train_target=target,
            tokenizer=tokenizer,
            easy_dataset=easy_dataset,
            hard_dataset=hard_dataset,
            stoi_dicts=stoi_dicts,
        )

    elif op == "test":
        test(
            test_target=target,
            tokenizer=tokenizer,
            easy_dataset=easy_dataset,
            hard_dataset=hard_dataset,
            stoi_dicts=stoi_dicts,
        )
