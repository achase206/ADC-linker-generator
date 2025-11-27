import pandas as pd
import numpy as np
import torch
import os

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


def train(train, tokenizer, easy_dataset, hard_dataset):

    training_options = ["scoring", "generation", "reinforce"]
    if train not in training_options:
        print("Training target not recognized")
        return
    else:

        model_directory = input("Enter model directory name to save results: ")
        os.makedirs(model_directory, exist_ok=True)

        if train == "scoring":

            while True:
                try:
                    num_epochs = input("Enter epochs to train: ")
                    num_epochs = int(num_epochs)
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

        if train == "generation":
            # prompt for easy epochs and hard epochs, lr
            while True:
                try:
                    num_epochs_easy = input("Enter epochs to train on easy data: ")
                    num_epochs_easy = int(num_epochs_easy)
                    break
                except ValueError:
                    print("Number of epochs must be an integer.")

            while True:
                try:
                    num_epochs_hard = input("Enter epochs to train on hard data: ")
                    num_epochs_hard = int(num_epochs_hard)
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
            print()
            print("Training sequence model on easy dataset")
            model_gen = train_LSTM_gen(
                model=gen_model,
                dataset=easy_dataset,
                tokenizer=tokenizer,
                stoi_dicts=stoi_dicts,
                num_epochs=num_epochs_easy,
            )
            torch.save(
                model_gen.state_dict(), f"{model_directory}/model_gen_weights.pth"
            )
            print("Easy training saved")
            gen_model.load_state_dict(
                torch.load(
                    f"{model_directory}/model_gen_weights.pth", weights_only=True
                )
            )

            print()
            print("Training sequence model on hard dataset")
            model_gen = train_LSTM_gen(
                model=gen_model,
                dataset=hard_dataset,
                tokenizer=tokenizer,
                stoi_dicts=stoi_dicts,
                num_epochs=num_epochs_hard,
            )

            torch.save(
                model_gen.state_dict(), f"{model_directory}/model_gen_weights.pth"
            )
            print("Hard training saved")

        if train == "reinforce":
            # prompt for batches, temps, entropy bonus
            while True:
                try:
                    num_RL_batches = input("Enter number of batches for RL training: ")
                    num_RL_batches = int(num_RL_batches)
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
            SA_model = LSTMScoreModel(
                input_dim=128,
                hidden_dim=256,
                layer_dim=5,
                output_dim=1,
                tokenizer=tokenizer,
            )
            TPSA_model = LSTMScoreModel(
                input_dim=128,
                hidden_dim=256,
                layer_dim=5,
                output_dim=1,
                tokenizer=tokenizer,
            )
            QED_model = LSTMScoreModel(
                input_dim=128,
                hidden_dim=256,
                layer_dim=5,
                output_dim=1,
                tokenizer=tokenizer,
            )
            LogP_model = LSTMScoreModel(
                input_dim=128,
                hidden_dim=256,
                layer_dim=5,
                output_dim=1,
                tokenizer=tokenizer,
            )
            CSP3_model = LSTMScoreModel(
                input_dim=128,
                hidden_dim=256,
                layer_dim=5,
                output_dim=1,
                tokenizer=tokenizer,
            )
            gen_model.load_state_dict(
                torch.load(
                    f"{model_directory}/model_gen_weights.pth", weights_only=True
                )
            )
            SA_model.load_state_dict(
                torch.load(
                    f"{model_directory}/SA_scores_weights.pth", weights_only=True
                )
            )
            TPSA_model.load_state_dict(
                torch.load(
                    f"{model_directory}/TPSA_scores_weights.pth", weights_only=True
                )
            )
            QED_model.load_state_dict(
                torch.load(
                    f"{model_directory}/QED_scores_weights.pth", weights_only=True
                )
            )
            LogP_model.load_state_dict(
                torch.load(
                    f"{model_directory}/LogP_scores_weights.pth", weights_only=True
                )
            )
            CSP3_model.load_state_dict(
                torch.load(
                    f"{model_directory}/CSP3_scores_weights.pth", weights_only=True
                )
            )

            RL_trained_model = LSTM_model_RL(
                SA_model=SA_model,
                TPSA_model=TPSA_model,
                QED_model=QED_model,
                LogP_model=LogP_model,
                CSP3_model=CSP3_model,
                gen_model=gen_model,
                critic_model=critic_model,
                tokenizer=tokenizer,
                total_frames=num_RL_batches,
            )

            torch.save(
                RL_trained_model.state_dict(),
                f"{model_directory}/model_RL_gen_weights.pth",
            )


def test(test, tokenizer, easy_dataset, hard_dataset):

    testing_options = ["scoring", "generation", "reinforce"]
    if test not in testing_options:
        print("Training target not recognized")
        return
    else:
        print(f"Testing {test} model:")
        model_directory = input("Enter model directory name to load models from: ")
        os.makedirs(model_directory, exist_ok=True)

        gen_model = LSTMGenModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            vocab_size=tokenizer.vocab_size,
            padding_idx=tokenizer.pad_token,
            output_dim=tokenizer.vocab_size,
            condition_count=4,
        )

        gen_RL_model = LSTMGenModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            vocab_size=tokenizer.vocab_size,
            padding_idx=tokenizer.pad_token,
            output_dim=tokenizer.vocab_size,
            condition_count=4,
        )

        gen_RL_model.load_state_dict(
            torch.load(f"{model_directory}/model_RL_gen_weights.pth", weights_only=True)
        )

        gen_model.load_state_dict(
            torch.load(f"{model_directory}/model_gen_weights.pth", weights_only=True)
        )

        SA_model = LSTMScoreModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            output_dim=1,
            tokenizer=tokenizer,
        )
        TPSA_model = LSTMScoreModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            output_dim=1,
            tokenizer=tokenizer,
        )
        QED_model = LSTMScoreModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            output_dim=1,
            tokenizer=tokenizer,
        )
        LogP_model = LSTMScoreModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            output_dim=1,
            tokenizer=tokenizer,
        )
        CSP3_model = LSTMScoreModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            output_dim=1,
            tokenizer=tokenizer,
        )
        gen_model.load_state_dict(
            torch.load(f"{model_directory}/model_gen_weights.pth", weights_only=True)
        )
        SA_model.load_state_dict(
            torch.load(f"{model_directory}/SA_scores_weights.pth", weights_only=True)
        )
        TPSA_model.load_state_dict(
            torch.load(f"{model_directory}/TPSA_scores_weights.pth", weights_only=True)
        )
        QED_model.load_state_dict(
            torch.load(f"{model_directory}/QED_scores_weights.pth", weights_only=True)
        )
        LogP_model.load_state_dict(
            torch.load(f"{model_directory}/LogP_scores_weights.pth", weights_only=True)
        )
        CSP3_model.load_state_dict(
            torch.load(f"{model_directory}/CSP3_scores_weights.pth", weights_only=True)
        )

        if test == "scoring":

            training_scores = ["SA", "TPSA", "QED", "LogP", "CSP3"]

            for score in training_scores:
                model = LSTMScoreModel(
                    input_dim=128,
                    hidden_dim=256,
                    layer_dim=5,
                    output_dim=1,
                    tokenizer=tokenizer,
                )
                model.load_state_dict(
                    torch.load(
                        f"{model_directory}/{score}_scores_weights.pth",
                        weights_only=True,
                    )
                )
                test_LSTM_scores(
                    dataset=easy_dataset,
                    model=model,
                    score_type=score,
                    tokenizer=tokenizer,
                )

        if test == "generation":

            while True:
                try:
                    temperature = input("Enter prediction temperature (0.01 - 1.0): ")
                    temperature = float(temperature)
                    break
                except ValueError:
                    print("Temperature must be a float.")

            while True:
                try:
                    print("-" * 30)
                    print(
                        "Antibodies:\nCetuximab = 0\nMCLL0517A = 1\nPatritumab = 2\nLorvotuzumab = 3"
                    )
                    print("-" * 30)
                    ab_idx = input("Enter antibody idx (int): ")
                    ab_idx = int(ab_idx)
                    break
                except ValueError:
                    print("Anitbody idx must be an integer.")

            while True:
                try:
                    print("-" * 30)
                    print(
                        "Payloads:\nPuromycin = 0\nPBD dimer = 1\nADC-C1 payload = 2\nRachelmycin = 3"
                    )
                    print("-" * 30)
                    pay_idx = input("Enter payload idx (int): ")
                    pay_idx = int(pay_idx)
                    break
                except ValueError:
                    print("Payload idx must be an integer.")

            while True:
                try:
                    print("-" * 30)
                    print(
                        "Payloads:\nTriple negative breast cancer = 0\nAcute myeloid leukaemia = 1\nBreast cancer = 2\nSmall cell lung cancer = 3"
                    )
                    print("-" * 30)
                    tar_idx = input("Enter target idx (int): ")
                    tar_idx = int(tar_idx)
                    break
                except ValueError:
                    print("Target idx must be an integer.")

            sequence = generate_smiles_conditional(
                model=gen_model,
                tokenizer=tokenizer,
                ab_idx=ab_idx,
                pay_idx=pay_idx,
                tar_idx=tar_idx,
                temperature=temperature,
            )

            print(tokenizer.tokens_to_smiles(sequence))

            check_sequences_conditional(
                SA_model=SA_model,
                TPSA_model=TPSA_model,
                QED_model=QED_model,
                LogP_model=LogP_model,
                CSP3_model=CSP3_model,
                gen_model=gen_model,
                tokenizer=tokenizer,
                ab_idx=ab_idx,
                pay_idx=pay_idx,
                tar_idx=tar_idx,
                temperature=temperature,
            )

        if test == "reinforce":

            while True:
                try:
                    temperature = input("Enter prediction temperature (0.01 - 1.0): ")
                    temperature = float(temperature)
                    break
                except ValueError:
                    print("Temperature must be a float.")

            while True:
                try:
                    print("-" * 30)
                    print(
                        "Antibodies:\nCetuximab = 0\nMCLL0517A = 1\nPatritumab = 2\nLorvotuzumab = 3"
                    )
                    print("-" * 30)
                    ab_idx = input("Enter antibody idx (int): ")
                    ab_idx = int(ab_idx)
                    break
                except ValueError:
                    print("Anitbody idx must be an integer.")

            while True:
                try:
                    print("-" * 30)
                    print(
                        "Payloads:\nPuromycin = 0\nPBD dimer = 1\nADC-C1 payload = 2\nRachelmycin = 3"
                    )
                    print("-" * 30)
                    pay_idx = input("Enter payload idx (int): ")
                    pay_idx = int(pay_idx)
                    break
                except ValueError:
                    print("Payload idx must be an integer.")

            while True:
                try:
                    print("-" * 30)
                    print(
                        "Payloads:\nTriple negative breast cancer = 0\nAcute myeloid leukaemia = 1\nBreast cancer = 2\nSmall cell lung cancer = 3"
                    )
                    print("-" * 30)
                    tar_idx = input("Enter target idx (int): ")
                    tar_idx = int(tar_idx)
                    break
                except ValueError:
                    print("Target idx must be an integer.")

            print()
            print("Sequence generated after RL training:")
            sequence = generate_smiles_conditional(
                model=gen_RL_model,
                tokenizer=tokenizer,
                ab_idx=ab_idx,
                pay_idx=pay_idx,
                tar_idx=tar_idx,
                temperature=temperature,
            )
            print(tokenizer.tokens_to_smiles(sequence))
            print(len(sequence))

            print()
            print("Pre-RL model: ")
            check_sequences_conditional(
                gen_model=gen_model,
                SA_model=SA_model,
                TPSA_model=TPSA_model,
                QED_model=QED_model,
                LogP_model=LogP_model,
                CSP3_model=CSP3_model,
                tokenizer=tokenizer,
                ab_idx=ab_idx,
                pay_idx=pay_idx,
                tar_idx=tar_idx,
                temperature=temperature,
            )

            print()
            print("RL model:")
            check_sequences_conditional(
                gen_model=gen_RL_model,
                SA_model=SA_model,
                TPSA_model=TPSA_model,
                QED_model=QED_model,
                LogP_model=LogP_model,
                CSP3_model=CSP3_model,
                tokenizer=tokenizer,
                ab_idx=ab_idx,
                pay_idx=pay_idx,
                tar_idx=tar_idx,
                temperature=temperature,
            )


def generate(): ...


if __name__ == "__main__":

    print("#" * 30)
    print("ADC LINKER GENERATOR")
    print("#" * 30)
    print()

    """LSTM Setup"""
    print("LSTM setup starting...")

    adc_directory = "data/adc_data_filtered.pkl"
    tokenizer = Tokenizer(adc_directory)
    df = pd.read_pickle(adc_directory)
    stoi_dicts, itos_dicts, dict_lengths = conditions_tokens(df)

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
    print()
    print("Use ctrl-C to exit at any time")
    print()

    valid_operation = ["train", "test", "generate"]
    valid_targets = ["scoring", "generation", "reinforce"]

    print("Would you like to train or test a model?")
    print("enter 'train' or 'test'")
    while True:
        try:
            operation = input("Train or test: ")
            if operation in valid_operation:
                break
        except ValueError:
            print("Invalid operation, try again")

    if operation == "train":
        print("Which model to train (scoring, generation, reinforce)")
        while True:
            try:
                model_target = input("Model: ")
                if model_target in valid_targets:
                    break
            except ValueError:
                print("Invalid model target, try again")
        train(
            train=model_target,
            tokenizer=tokenizer,
            easy_dataset=easy_dataset,
            hard_dataset=hard_dataset,
        )

    if operation == "test":
        print("Which model to test (scoring, generation, reinforce)")
        while True:
            try:
                model_target = input("Model: ")
                if model_target in valid_targets:
                    break
            except ValueError:
                print("Invalid model target, try again")
        test(
            test=model_target,
            tokenizer=tokenizer,
            easy_dataset=easy_dataset,
            hard_dataset=hard_dataset,
        )
