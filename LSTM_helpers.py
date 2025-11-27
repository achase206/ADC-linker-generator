# Torch modules for scoring and generative LSTMs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as TF

# TorchRL modules for reinforcement learning
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.envs import SerialEnv, EnvBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)
from tensordict.nn import TensorDictModule, TensorDictSequential

# Other modules
import numpy as np
import pandas as pd
import random
import re
import math
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit import rdBase

from LSTM_classes import (
    Tokenizer,
    LSTMScoreModel,
    LSTMGenModel,
    CriticModel,
    adcDataset,
    SmilesGeneratorEnv,
)

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
    dict_length = 4
    for condition in conditions:
        stoi, itos = conditions_mapping(condition, dict_length)
        stoi_dicts.append(stoi)
        itos_dicts.append(itos)

    return stoi_dicts, itos_dicts, dict_length


# THIS IS BROKEN RN!!!!!!!
def kfolds_LSTM_scores(directory):

    token = Tokenizer(directory)
    # add seed for reproducible results
    seed = 42
    # python
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    # for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # make PyTorch behavior deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = adcDataset(directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # k-fold cross validation
    # 5 fold split: 80/20
    kfolds = 5
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    # store validation results
    fold_losses = []

    # for avg_percent_error
    fold_models = []
    fold_val_indices = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n -- Fold {fold+1} / {kfolds} -- ")

        # make two subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # dataloader
        train_loader = DataLoader(
            train_subset, batch_size=64, shuffle=True, collate_fn=token.collate_smiles
        )
        val_loader = DataLoader(
            val_subset, batch_size=64, shuffle=False, collate_fn=token.collate_smiles
        )

        # reset model for each fold
        model = LSTMScoreModel(
            input_dim=128,
            hidden_dim=256,
            layer_dim=5,
            output_dim=1,
            directory=directory,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        # training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for i, (smile, score) in enumerate(train_loader):
                smile, score = smile.to(device), score.to(device)

                optimizer.zero_grad()
                output = model(smile)
                loss = criterion(output.float(), score.float())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} / {num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # validation loop
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for smile, score in val_loader:
                smile, score = smile.to(device), score.to(device)
                output = model(smile)
                loss = criterion(output.float(), score.float())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        fold_losses.append(avg_val_loss)
        print(f"Fold {fold+1} Validation Loss: {avg_val_loss:.4f}")

        fold_models.append(model)
        fold_val_indices.append(val_idx)

    print("\n -- K-Fold Results --")
    print("Fold Losses:", fold_losses)
    mse = np.mean(fold_losses)
    print(f"MSE Validation Loss: {mse:.4f}")
    rmse = np.sqrt(mse)
    print(f"RMSE Validation Loss: {rmse:.4f}")


def train_LSTM_scores(dataset, tokenizer, score_type, num_epochs=100):
    """
    Training loop for LSTM scoring metrics
    Used later on as critics for RL

    directory: Description
    score_type: Must be one of the following:
                SA, TPSA, QED, LogP, CSP3
    """

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=tokenizer.collate_smiles,
    )

    model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training {score_type} score on device {device}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        for i, (smile, ab, pay, target, score) in enumerate(loader):
            print(f"Batch {i+1}/{math.ceil(len(dataset)/loader.batch_size)}", end="\r")
            smile, label = smile.to(device), score.to(device)

            optimizer.zero_grad()

            output = model(smile)  # forward
            loss = criterion(output.float(), label.float())  # calculate MSE loss
            loss.backward()  # back propagation
            optimizer.step()

        print()
        print(f"Epoch {epoch+1}/{num_epochs}, MSE Loss: {loss.item():.3f}")

    return model


def test_LSTM_scores(dataset, tokenizer, model, score_type):

    loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=True,
        collate_fn=tokenizer.collate_smiles,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Testing {score_type} score model on device {device}")

    predictions = []
    true = []

    with torch.no_grad():
        for i, (smile, ab, pay, target, score) in enumerate(loader):
            smile, score = smile.to(device), score.to(device)
            output = model(smile)
            predictions += output
            true += score

    # convert from tensor to array for predicted and true scores
    predictions = np.array([pred.cpu().numpy() for pred in predictions])
    true = np.array([true.cpu().numpy() for true in true])

    mae = mean_absolute_error(true, predictions)
    print(f"Mean Absolute Error: {mae:.4f}")


def train_LSTM_gen(model, dataset, tokenizer, stoi_dicts, num_epochs=1000):
    batch_size = 64
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tokenizer.collate_smiles,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    print(f"Training on device: {device}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token
    )  # ignore the padding for loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        for i, (smile, ab, pay, target, score) in enumerate(loader):
            print(f"Batch {i+1}/{math.ceil(len(dataset)/loader.batch_size)}", end="\r")

            smile = smile.to(device)
            ab = ab.to(device)
            pay = pay.to(device)
            tar = target.to(device)

            # create a vector containing start tokens as long as the current batch
            seq_start = torch.full(
                (smile.shape[0], 1), tokenizer.start_token, dtype=torch.long
            )
            seq_start = seq_start.to(device)

            # remove the last token from input so that we can add start token to the front
            input_seq = smile[:, :-1]

            inputs = torch.cat(
                [seq_start, input_seq], dim=1
            )  # add the start tokens to beginning of input

            optimizer.zero_grad()

            # teacher forcing approach, only using ground truth for loss
            # instead of feeding models previous output we use the actual known previous token
            output, _, _ = model(sequence=inputs, antibody=ab, payload=pay, target=tar)
            loss = criterion(
                output.view(-1, tokenizer.vocab_size), smile.view(-1)
            )  # -1 flattens our sequence
            loss.backward()  # back propagation
            optimizer.step()

        print()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.3f}")

    return model


def generate_smiles(model, directory):
    token = Tokenizer(directory)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # starting generating with start token
    current_token = torch.tensor([[token.smiles_map["START"]]]).to(device)

    # initially hidden state and cell state will be zero
    # same setup as the geeks for geeks tutorial
    hidden_state, cell_state = None, None

    sequence = []

    max_smiles_length = 50
    with torch.no_grad():
        for i in range(max_smiles_length):

            # now because we don't know that the previous state was correct
            # we rely on the models actual prediction instead of relying on ground truth
            # we now feed the previous hidden and cell state into our model
            output, hidden_state, cell_state = model(
                current_token, hidden_state, cell_state
            )

            # get last token in sequence
            # same strategy as the scoring model to get the last output
            output = output[:, -1, :]

            # output is shape (batch_size, smiles_map)
            # need probabilities of each token on the smiles map thus dim=-1
            token_prob = F.softmax(output, dim=-1)

            # exclude the start and pad tokens from the next token prediction
            token_prob[0, token.smiles_map["START"]] = 0
            token_prob[0, token.smiles_map["PAD"]] = 0

            # this randomly picks one of our tokens based on their probabilities
            # token with higher probabilities are more likely to get picked
            # if we just picked the highest prob token we would probably just get the safest output
            # like CCCCC, etc.
            next_token = torch.multinomial(token_prob, num_samples=1).item()

            # if we reach the end then break and stop generating
            if next_token == token.smiles_map["END"]:
                break

            sequence.append(next_token)  # add to our sequence to be returned

            # update current token for next iteration
            # converts token ID to tensor and loads to device
            current_token = torch.tensor([[next_token]]).to(device)

    return sequence


def LSTM_model_RL(
    SA_model,
    TPSA_model,
    QED_model,
    LogP_model,
    CSP3_model,
    gen_model,
    critic_model,
    tokenizer,
    temperature=0.7,
    total_frames=500_000,
):
    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html
    # creating a dict for our LSTM generative model that the actor can then understand for RL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SA_model.to(device)
    TPSA_model.to(device)
    QED_model.to(device)
    LogP_model.to(device)
    CSP3_model.to(device)
    gen_model.to(device)
    critic_model.to(device)

    # actor is our generative model, critic is our scoring model
    # loss module uses the proximal policy optimization (PPO) from torchRL to calculate loss
    lstm_module = TensorDictModule(
        module=gen_model,
        in_keys=["observation", "antibody", "payload", "target", "hidden", "cell"],
        out_keys=["logits", "hidden", "cell"],
    )

    # temp module controls temp scaler for generation, default of 1.0 is too high most often
    temp_module = TensorDictModule(
        lambda x: x / temperature, in_keys=["logits"], out_keys=["logits"]
    )

    policy_module = TensorDictSequential(
        lstm_module, temp_module
    )  # adds temp scaler to lstm logits

    # keys are strict here, need to use predefined key names as described by ProbActor class
    actor_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    SA_module = TensorDictModule(
        module=SA_model,
        in_keys=["observation"],
        out_keys=["SA_score"],
    )

    TPSA_module = TensorDictModule(
        module=TPSA_model,
        in_keys=["observation"],
        out_keys=["TPSA_score"],
    )

    QED_module = TensorDictModule(
        module=QED_model,
        in_keys=["observation"],
        out_keys=["QED_score"],
    )

    LogP_module = TensorDictModule(
        module=LogP_model,
        in_keys=["observation"],
        out_keys=["LogP_score"],
    )

    CSP3_module = TensorDictModule(
        module=CSP3_model,
        in_keys=["observation"],
        out_keys=["CSP3_score"],
    )

    # theoretical bounds for all scoring metrics
    SA_min, SA_max = 0.0, 10.0
    TPSA_min, TPSA_max = 0.0, 700  # not theoretical, just higher than max in dataset
    QED_min, QED_max = 0.0, 1.0
    LogP_min, LogP_max = (
        -10.0,
        15.0,
    )  # not theoretical, just lower/higher than min/max in dataset
    CSP3_min, CSP3_max = 0.0, 1.0

    def aggregate_scores(SA, TPSA, QED, LogP, CSP3):
        SA_norm = (SA - SA_min) / (SA_max - SA_min)
        TPSA_norm = (TPSA - TPSA_min) / (TPSA_max - TPSA_min)
        QED_norm = (QED - QED_min) / (QED_max - QED_min)
        LogP_norm = (LogP - LogP_min) / (LogP_max - LogP_min)
        CSP3_norm = (CSP3 - CSP3_min) / (CSP3_max - CSP3_min)

        reward_SA = torch.clamp(1 - SA_norm, 0.0, 1.0)
        reward_TPSA = torch.clamp(1 - TPSA_norm, 0.0, 1.0)
        reward_QED = torch.clamp(QED_norm, 0.0, 1.0)
        reward_LogP = torch.clamp(1 - LogP_norm, 0.0, 1.0)
        reward_CSP3 = torch.clamp(CSP3_norm, 0.0, 1.0)

        return reward_SA + reward_TPSA + reward_QED + reward_LogP + reward_CSP3

    aggregate_module = TensorDictModule(
        aggregate_scores,
        in_keys=["SA_score", "TPSA_score", "QED_score", "LogP_score", "CSP3_score"],
        out_keys=["reward"],
    )

    reward_module = TensorDictSequential(
        SA_module, TPSA_module, QED_module, LogP_module, CSP3_module, aggregate_module
    )

    critic_module = ValueOperator(
        module=critic_model,
        in_keys=["observation", "antibody", "payload", "target", "hidden", "cell"],
    )

    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.collectors.SyncDataCollector.html

    environment_maker = lambda: SmilesGeneratorEnv(
        reward_module,
        vocab_size=tokenizer.vocab_size,
        max_length=50,
        num_layers=gen_model.layer_dim,
        hidden_dim=gen_model.hidden_dim,
        tokenizer=tokenizer,
    )

    # hyper params for reinforcement learning:
    learning_rate = 0.00001
    num_envs = 16  # 16 smiles generated in parallel

    env = SerialEnv(num_envs, environment_maker)

    frame_steps = 50

    collector = SyncDataCollector(
        create_env_fn=env,
        policy=actor_module,
        frames_per_batch=num_envs * frame_steps,  # perform 20 steps before updating
        total_frames=total_frames,
        device="cuda",
        storing_device="cuda",
    )

    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.objectives.value.GAE.html
    # Generalized advantage estimation balances bis and variance when estimating the advantage
    # The advantage tells us how much better this outcome was compared to the previous average outcome

    advantage_module = GAE(gamma=0.99, lmbda=0.95, value_network=None, average_gae=True)

    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.objectives.ClipPPOLoss.html
    # clips max policy loss to prevent huge shifts in the model during learning
    loss_module = ClipPPOLoss(
        actor=actor_module.to("cuda"),
        critic=critic_module.to("cuda"),
        clip_epsilon=0.2,  # default hyperparam
        entropy_bonus=0.0001,  # encourages exploration to prevent same output
        normalize_advantage=True,
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)

    # RL training loop
    print("Start RL training...")

    for i, batch in enumerate(collector):
        with torch.no_grad():
            flat_batch = batch.reshape(-1)

            critic_module(flat_batch)
            values = flat_batch["state_value"]
            reshaped_values = values.view(num_envs, frame_steps, 1)

            batch.set("state_value", reshaped_values)

            next_batch = batch["next"].reshape(-1)
            critic_module(next_batch)

            next_values = next_batch["state_value"]
            batch["next"].set("state_value", next_values.view(num_envs, frame_steps, 1))

            # calculate the advantage, how much better was this batch than average expected
            advantage_module(batch)

            # # calculate the loss using our PPO
            # loss = loss_module(batch.reshape(-1))
            # actor_loss = loss["loss_objective"]
            # critic_loss = loss["loss_critic"]
            # total_loss = actor_loss + critic_loss

            # # back propagate
            # optimizer.zero_grad()
            # total_loss.backward()
            # optimizer.step()

        ppo_epochs = 5
        for _ in range(ppo_epochs):
            loss = loss_module(batch.reshape(-1))
            actor_loss = loss["loss_objective"]
            critic_loss = loss["loss_critic"]
            total_loss = actor_loss + critic_loss

            optimizer.zero_grad()
            total_loss.backward()
            # clip the gradients for stability
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=1.0)
            optimizer.step()

        # print reward to keep track of training progress
        # avg_reward = batch["next", "reward"].mean().item()
        rewards = batch["next", "reward"].sum(dim=1)
        avg_reward = rewards.mean().item()
        print(
            f"Batch {i} || Loss: {total_loss.item():.4f}, Avg Reward: {avg_reward:.4f}"
        )

        # check how we are doing every 5 batches
        if i % 5 == 0:
            print(f"----- Batch {i} Monitor -----")
            sequence_tokens = batch["next", "observation"][0].cpu().numpy()
            smiles = tokenizer.tokens_to_smiles(list(sequence_tokens))
            print(f"SMILES: {smiles}")

            ab_id = batch["next", "antibody"][0, 0].item()
            pay_id = batch["next", "payload"][0, 0].item()
            tar_id = batch["next", "target"][0, 0].item()
            print(f"Condition: Antibody {ab_id} | Payload {pay_id} | Target {tar_id}")
            print("-----------------------------\n")

    print("RL training complete.")

    return gen_model


def check_sequences(
    SA_model,
    TPSA_model,
    QED_model,
    LogP_model,
    CSP3_model,
    gen_model,
    directory,
    num_samples=100,
):
    valid_smiles = 0
    SA_scores = []
    TPSA_scores = []
    QED_scores = []
    LogP_scores = []
    CSP3_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SA_model.to(device)
    TPSA_model.to(device)
    QED_model.to(device)
    LogP_model.to(device)
    CSP3_model.to(device)
    gen_model.to(device)

    SA_model.eval()
    TPSA_model.eval()
    QED_model.eval()
    LogP_model.eval()
    CSP3_model.eval()

    for _ in range(num_samples):
        sequence = generate_smiles(gen_model, directory)
        smiles = token.tokens_to_smiles(sequence)

        if Chem.MolFromSmiles(smiles) is not None:
            valid_smiles += 1

            seq_tensor = torch.tensor(sequence, device=device).unsqueeze(0)

            with torch.no_grad():
                SA_score = SA_model(seq_tensor)
                TPSA_score = TPSA_model(seq_tensor)
                QED_score = QED_model(seq_tensor)
                LogP_score = LogP_model(seq_tensor)
                CSP3_score = CSP3_model(seq_tensor)

                SA_scores.append(SA_score.cpu().numpy())
                TPSA_scores.append(TPSA_score.cpu().numpy())
                QED_scores.append(QED_score.cpu().numpy())
                LogP_scores.append(LogP_score.cpu().numpy())
                CSP3_scores.append(CSP3_score.cpu().numpy())

    print("-" * 30)
    print(f"Valid smiles = {valid_smiles}%")
    print(f"Avg SA = {np.mean(SA_scores):.4f}\t\tStd Dev = {np.std(SA_scores):.4f}")
    print(f"Avg TPSA = {np.mean(TPSA_scores):.4f}\tStd Dev = {np.std(TPSA_scores):.4f}")
    print(f"Avg QED = {np.mean(QED_scores):.4f}\tStd Dev = {np.std(QED_scores):.4f}")
    print(f"Avg LogP = {np.mean(LogP_scores):.4f}\tStd Dev = {np.std(LogP_scores):.4f}")
    print(f"Avg CSP3 = {np.mean(CSP3_scores):.4f}\tStd Dev = {np.std(CSP3_scores):.4f}")
    print("-" * 30)


def generate_smiles_conditional(
    model, tokenizer, ab_idx, pay_idx, tar_idx, temperature=1.0
):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # shape is [1,1] for each seeding condition, this gets passed into the model
    ab_tensor = torch.tensor([[ab_idx]], device=device, dtype=torch.long)
    pay_tensor = torch.tensor([[pay_idx]], device=device, dtype=torch.long)
    tar_tensor = torch.tensor([[tar_idx]], device=device, dtype=torch.long)

    # starting generating with start token
    current_token = torch.tensor([[tokenizer.start_token]]).to(device)

    # initially hidden state and cell state will be zero
    # same setup as the geeks for geeks tutorial
    hidden_state, cell_state = None, None

    sequence = []

    max_smiles_length = 50
    with torch.no_grad():
        for i in range(max_smiles_length):

            # now because we don't know that the previous state was correct
            # we rely on the models actual prediction instead of relying on ground truth
            # we now feed the previous hidden and cell state into our model
            output, hidden_state, cell_state = model(
                sequence=current_token,
                antibody=ab_tensor,
                payload=pay_tensor,
                target=tar_tensor,
                hidden_state=hidden_state,
                cell_state=cell_state,
            )

            # get last token in sequence
            # same strategy as the scoring model to get the last output
            output = output[:, -1, :]

            # exclude the start and pad tokens from the next token prediction
            output[0, tokenizer.start_token] = -float(
                "inf"
            )  # need neg inf to not mess up softmax
            output[0, tokenizer.pad_token] = -float("inf")

            # apply temperature
            output = output / temperature

            # apply top-k filtering
            # k = 5  # only choose between top 5 probable tokens
            # top_k_logits, top_k_indices = torch.topk(output, k)

            # output is shape (batch_size, smiles_map)
            # need probabilities of each token on the smiles map thus dim=-1
            token_prob = F.softmax(output, dim=-1)

            # this randomly picks one of our tokens based on their probabilities
            # token with higher probabilities are more likely to get picked
            # if we just picked the highest prob token we would probably just get the safest output
            # like CCCCC, etc.
            next_token = torch.multinomial(token_prob, num_samples=1).item()

            # next_token = top_k_indices.gather(1, sample_idx)

            # next_token = next_token.item()

            # if we reach the end then break and stop generating
            if next_token == tokenizer.end_token:
                break

            sequence.append(next_token)  # add to our sequence to be returned

            # update current token for next iteration
            # converts token ID to tensor and loads to device
            current_token = torch.tensor([[next_token]]).to(device)

    return sequence


def check_sequences_conditional(
    SA_model,
    TPSA_model,
    QED_model,
    LogP_model,
    CSP3_model,
    gen_model,
    tokenizer,
    ab_idx,
    pay_idx,
    tar_idx,
    num_samples=100,
    temperature=1.0,
):
    valid_smiles = 0
    SA_scores = []
    TPSA_scores = []
    QED_scores = []
    LogP_scores = []
    CSP3_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SA_model.to(device)
    TPSA_model.to(device)
    QED_model.to(device)
    LogP_model.to(device)
    CSP3_model.to(device)
    gen_model.to(device)

    SA_model.eval()
    TPSA_model.eval()
    QED_model.eval()
    LogP_model.eval()
    CSP3_model.eval()

    gen_model.to(device)

    for _ in range(num_samples):
        sequence = generate_smiles_conditional(
            model=gen_model,
            tokenizer=tokenizer,
            ab_idx=ab_idx,
            pay_idx=pay_idx,
            tar_idx=tar_idx,
            temperature=temperature,
        )
        smiles = tokenizer.tokens_to_smiles(sequence)

        if Chem.MolFromSmiles(smiles) is not None:
            valid_smiles += 1
            seq_tensor = torch.tensor(sequence, device=device).unsqueeze(0)

            with torch.no_grad():
                SA_score = SA_model(seq_tensor)
                TPSA_score = TPSA_model(seq_tensor)
                QED_score = QED_model(seq_tensor)
                LogP_score = LogP_model(seq_tensor)
                CSP3_score = CSP3_model(seq_tensor)

                SA_scores.append(SA_score.cpu().numpy())
                TPSA_scores.append(TPSA_score.cpu().numpy())
                QED_scores.append(QED_score.cpu().numpy())
                LogP_scores.append(LogP_score.cpu().numpy())
                CSP3_scores.append(CSP3_score.cpu().numpy())

    print("-" * 30)
    print(f"Valid smiles = {valid_smiles}%")
    print(f"Avg SA = {np.mean(SA_scores):.4f}\t\tStd Dev = {np.std(SA_scores):.4f}")
    print(f"Avg TPSA = {np.mean(TPSA_scores):.4f}\tStd Dev = {np.std(TPSA_scores):.4f}")
    print(f"Avg QED = {np.mean(QED_scores):.4f}\tStd Dev = {np.std(QED_scores):.4f}")
    print(f"Avg LogP = {np.mean(LogP_scores):.4f}\tStd Dev = {np.std(LogP_scores):.4f}")
    print(f"Avg CSP3 = {np.mean(CSP3_scores):.4f}\tStd Dev = {np.std(CSP3_scores):.4f}")
    print("-" * 30)
