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

# Disable error messages when a non-valid smiles is encountered
rdBase.DisableLog("rdApp.error")


class Tokenizer:
    def __init__(self, directory):
        self.SMI_REGEX = re.compile(
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|B|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )
        df = pd.read_pickle(directory)
        self.smiles_map = self.smiles_tokenizer(df)
        self.inv_smiles_map = self.smiles_detokenizer()

        self.pad_token = self.smiles_map["PAD"]
        self.start_token = self.smiles_map["START"]
        self.end_token = self.smiles_map["END"]
        self.vocab_size = len(self.smiles_map)

    def smiles_tokenizer(self, df):
        # SMILES tokenizer reference: https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html

        smiles_tokens = df["smiles"].str.findall(self.SMI_REGEX)

        dataset_tokens = set(smiles_tokens.explode().dropna().unique())

        standard_tokens = set(
            [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "(",
                ")",
                "[",
                "]",
                "=",
                "#",
                "-",
                "+",
                "\\",
                "/",
                ".",
                ":",
                "@",
            ]
        )

        smiles_tokens = sorted(
            list(dataset_tokens.union(standard_tokens))
        )  # combine the found tokens and the standard ones

        # start token dict with padding token at zero
        smiles_map = {"PAD": 0}

        for i, token in enumerate(smiles_tokens):
            smiles_map[token] = i + 1

        smiles_map["END"] = len(smiles_map)
        smiles_map["START"] = len(smiles_map)

        return smiles_map

    def smiles_detokenizer(self):
        inv_smiles_map = {token: pattern for pattern, token in self.smiles_map.items()}
        return inv_smiles_map

    def smiles_to_tokens(self, smile_pattern_list):
        smiles_tokens = [self.smiles_map[char] for char in smile_pattern_list]
        smiles_tokens.append(self.smiles_map["END"])  # add terminator token
        return smiles_tokens

    def tokens_to_smiles(self, smile_token_list):
        smiles_patterns = []
        for token in smile_token_list:
            if hasattr(token, "item"):
                token = token.item()
            smiles_patterns.append(self.inv_smiles_map[token])
        # smiles_patterns = [self.inv_smiles_map[token] for token in smile_token_list]
        return "".join(smiles_patterns)

    def collate_smiles(self, batch):

        smiles_list, ab_list, pay_list, target_list, scores = zip(*batch)

        smiles_padded = pad_sequence(
            smiles_list, batch_first=True, padding_value=self.smiles_map["PAD"]
        )

        scores_stack = torch.stack(scores)  # stack reshapes to a size of (batch,1)
        ab_stack = torch.stack(ab_list)
        pay_stack = torch.stack(pay_list)
        target_stack = torch.stack(target_list)

        return smiles_padded, ab_stack, pay_stack, target_stack, scores_stack


class LSTMScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, tokenizer):
        super(LSTMScoreModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=input_dim,
            padding_idx=tokenizer.pad_token,
        )
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # added dropout to reduce overfitting
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None, cell=None):
        embedded_seq = self.embedding(x)

        # for RL we need the hidden and cell state
        # didn't need it for initial training on entire sequence

        if hidden is not None and cell is not None:

            # Torch RL passes in batch, num_layers, hidden_dim
            # we need to reshape if that is the case for our lstm forward call
            # passed in as (16,5,256) change to (5,16,256)
            if hidden.shape[0] != self.layer_dim:
                hidden = hidden.permute(1, 0, 2).contiguous()
                cell = cell.permute(1, 0, 2).contiguous()

            out, _ = self.lstm(embedded_seq, (hidden, cell))

        else:
            out, _ = self.lstm(embedded_seq)

        out = out[:, -1, :]  # Take last time step
        out = self.dropout(out)  # apply dropout even when layer_dim = 1
        out = self.fc(out)  # final layer

        return out


class LSTMGenModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        vocab_size,
        padding_idx,
        output_dim,
        condition_count,
        condition_dim=32,
    ):
        super(LSTMGenModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # embedding for smiles tokens
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_dim,
            padding_idx=padding_idx,
        )

        # embedding for conditions: antibody, payload and target (indication)
        self.ab_embedding = nn.Embedding(
            num_embeddings=condition_count, embedding_dim=condition_dim
        )
        self.pay_embedding = nn.Embedding(
            num_embeddings=condition_count, embedding_dim=condition_dim
        )
        self.target_embedding = nn.Embedding(
            num_embeddings=condition_count, embedding_dim=condition_dim
        )

        # adjust input vector to be size of smiles dims + conditions dims
        self.input_size = input_dim + (3 * condition_dim)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

        # added dropout to reduce overfitting
        self.dropout = nn.Dropout(0.2)

    def forward(
        self, sequence, antibody, payload, target, hidden_state=None, cell_state=None
    ):
        # on first pass initialize hidden and cell states to zeros
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(
                self.layer_dim, sequence.size(0), self.hidden_dim
            ).to(sequence.device)
            cell_state = torch.zeros(
                self.layer_dim, sequence.size(0), self.hidden_dim
            ).to(sequence.device)

            # the collector changes the order, we need to adjust for LSTM input
            if hidden_state.shape[0] != self.layer_dim:
                hidden_state = hidden_state.permute(1, 0, 2).contiguous()
                cell_state = cell_state.permute(1, 0, 2).contiguous()

        # smiles sequence embedding
        embedded_seq = self.token_embedding(sequence)

        # condition embedding
        embedded_ab = self.ab_embedding(antibody)
        embedded_pay = self.pay_embedding(payload)
        embedded_target = self.target_embedding(target)

        # get the length of the current sequence being generated
        seq_len = sequence.size(1)

        # we need to repeat the condition vectors for as many tokens in current sequence
        # each iteration needs to see the conditions that it is optimizing around
        # ie the first token needs to know antibody, payload, target as does each subsequent token
        # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html
        # -1 keeps dimensions the same, just repeats for as long as the sequence length so we have
        # enough vecs to concatenate to sequence vecs
        embedded_ab = embedded_ab.expand(-1, seq_len, -1)
        embedded_pay = embedded_pay.expand(-1, seq_len, -1)
        embedded_target = embedded_target.expand(-1, seq_len, -1)

        # lastly concatenate sequence vec and condition vecs together for lstm input
        lstm_input = torch.cat(
            [embedded_seq, embedded_ab, embedded_pay, embedded_target], dim=-1
        )

        out, (hidden_state, cell_state) = self.lstm(lstm_input)
        out = self.fc(out)  # final layer

        hidden_state = hidden_state.permute(1, 0, 2)  # swap them back for tensordict
        cell_state = cell_state.permute(1, 0, 2)

        return out, hidden_state, cell_state


class CriticModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        vocab_size,
        padding_idx,
        condition_count=4,
        condition_dim=32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=input_dim,
            padding_idx=padding_idx,
        )

        # add condition embedding just like in the gen model
        self.ab_embedding = nn.Embedding(condition_count, condition_dim)
        self.pay_embedding = nn.Embedding(condition_count, condition_dim)
        self.tar_embedding = nn.Embedding(condition_count, condition_dim)

        # input size is now for all embeddings
        self.input_size = input_dim + (3 * condition_dim)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence, antibody, payload, target, hidden=None, cell=None):

        # smiles sequence embedding
        embedded_seq = self.token_embedding(sequence)

        # condition embedding
        embedded_ab = self.ab_embedding(antibody)
        embedded_pay = self.pay_embedding(payload)
        embedded_target = self.tar_embedding(target)

        # get the length of the current sequence being generated
        seq_len = sequence.size(1)

        # we need to repeat the condition vectors for as many tokens in current sequence
        # see gen model for more detailed description here
        embedded_ab = embedded_ab.expand(-1, seq_len, -1)
        embedded_pay = embedded_pay.expand(-1, seq_len, -1)
        embedded_target = embedded_target.expand(-1, seq_len, -1)

        # lastly concatenate sequence vec and condition vecs together for lstm input
        lstm_input = torch.cat(
            [embedded_seq, embedded_ab, embedded_pay, embedded_target], dim=-1
        )

        # for RL we need the hidden and cell state
        # didn't need it for initial training on entire sequence

        if hidden is not None and cell is not None:

            # Torch RL collector passes in batch, num_layers, hidden_dim
            # we need to reshape if that is the case for our lstm forward call
            # passed in as (16,5,256) change to (5,16,256)
            if hidden.shape[0] != self.layer_dim:
                hidden = hidden.permute(1, 0, 2).contiguous()
                cell = cell.permute(1, 0, 2).contiguous()

            out, _ = self.lstm(lstm_input, (hidden, cell))

        else:
            out, _ = self.lstm(lstm_input)

        out = out[:, -1, :]  # Take last time step
        out = self.fc(out)  # final layer

        return out


class adcDataset(Dataset):

    def __init__(
        self, directory, tokenizer, stoi_dicts, score_type="SA", augment=False
    ):
        self.directory = directory
        self.tokenizer = tokenizer
        self.file = pd.read_pickle(directory)
        self.smiles = (
            self.file["smiles"]
            .str.findall(self.tokenizer.SMI_REGEX)
            .apply(self.tokenizer.smiles_to_tokens)
            .tolist()
        )
        self.augment = augment

        self.mols = []
        valid_idx = []

        # go through smiles and attempt to generate molecule object for each
        for idx, smile in enumerate(self.file["smiles"]):
            mol = Chem.MolFromSmiles(smile)
            if mol:
                self.mols.append(mol)
                valid_idx.append(idx)

        # filter file just in case some of the molecules couldn't be read
        self.file = self.file.iloc[valid_idx].reset_index(drop=True)

        self.ab_map = stoi_dicts[0]
        self.pay_map = stoi_dicts[1]
        self.target_map = stoi_dicts[2]

        if score_type == "SA":
            self.score = torch.tensor(np.array((self.file["calc_SA_score"])))

        elif score_type == "TPSA":
            self.score = torch.tensor(np.array((self.file["TPSA"])))

        elif score_type == "QED":
            self.score = torch.tensor(np.array((self.file["QED"])))

        elif score_type == "LogP":
            self.score = torch.tensor(np.array((self.file["LogP"])))

        elif score_type == "CSP3":
            self.score = torch.tensor(np.array((self.file["FractionCSP3"])))

        else:
            raise ValueError("Not a valid scoring metric")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        row = self.file.iloc[idx]
        mol = self.mols[idx]

        # randomize structure of valid smiles to help with learning
        if self.augment:
            smiles_str = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        else:
            smiles_str = Chem.MolToSmiles(mol, canonical=True)

        # smile = torch.tensor(self.smiles[idx])

        smiles_token_list = self.tokenizer.SMI_REGEX.findall(smiles_str)

        smiles_tokens = self.tokenizer.smiles_to_tokens(smiles_token_list)
        smile = torch.tensor(smiles_tokens, dtype=torch.long)

        score = torch.tensor([self.score[idx]])

        ab_id = self.ab_map[row["antibody_name"]]
        pay_id = self.pay_map[row["payload_name"]]
        target_id = self.target_map[row["indication"]]

        return (
            smile,
            torch.tensor([ab_id], dtype=torch.long),
            torch.tensor([pay_id], dtype=torch.long),
            torch.tensor([target_id], dtype=torch.long),
            score,
        )


class SmilesGeneratorEnv(EnvBase):
    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.envs.EnvBase.html
    # https://docs.pytorch.org/rl/0.8/reference/generated/torchrl.data.CompositeSpec.html
    # https://docs.pytorch.org/tutorials/advanced/pendulum.html

    # documentation describes the standard boilerplate for implementing torchRL

    # EnvBase abstract environment base class defines standard environment interface for RL
    # this includes the reset and step methods that allow RL to proceed

    def __init__(
        self,
        reward_model,
        vocab_size,
        max_length,
        num_layers,
        hidden_dim,
        tokenizer,
        device="cuda",
        seed=None,
        condition_count=4,
    ):
        super().__init__(
            device=device, batch_size=[]
        )  # batch of [] means one env for serial processing

        self.rewarder = reward_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.token = tokenizer
        self.condition_count = condition_count

        # init current step to zero, this will get reset after each reset call as well
        self.current_step = 0

        # holds the sequence that we generate for each batch
        self.generated_sequence = []

        # need to define the specs for TorchRL to work
        # essentially telling the torchRL what the output from the generative model will look like
        # this is what we already defined in our implementation of the actor in the LSTM_model_RL function
        # provides a dictionary of tensors instead of an array of tensors
        self.observation_spec = CompositeSpec(
            {
                # discrete because there are a limited number of tokens that can be predicted
                "observation": DiscreteTensorSpec(
                    n=self.vocab_size, shape=(1,), dtype=torch.long
                ),
                # both the values for hidden can cell are continuous with no known min or max (unbounded)
                "hidden": UnboundedContinuousTensorSpec(shape=(num_layers, hidden_dim)),
                "cell": UnboundedContinuousTensorSpec(shape=(num_layers, hidden_dim)),
                "antibody": DiscreteTensorSpec(
                    n=self.condition_count, shape=(1,), dtype=torch.long
                ),
                "payload": DiscreteTensorSpec(
                    n=self.condition_count, shape=(1,), dtype=torch.long
                ),
                "target": DiscreteTensorSpec(
                    n=self.condition_count, shape=(1,), dtype=torch.long
                ),
            }
        )

        if seed == None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # action is predicting a new token in smiles string
        self.action = DiscreteTensorSpec(n=vocab_size, shape=(1,), dtype=torch.long)

        # reward is the score the smiles receives, we will increase this to match number of output heads eventually
        self.reward = UnboundedContinuousTensorSpec(shape=(1,))

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        self.rng = torch.manual_seed(seed)

        return seed + 1

    def _reset(self, tensordict=None):
        # after a sequence is generated, we need to reset the sequence list and counter
        self.current_step = 0
        self.generated_sequence = []

        start_token = torch.tensor([self.token.smiles_map["START"]], device=self.device)

        # select random conditions for this generation cycle
        ab_idx = torch.randint(0, self.condition_count, (1,), device=self.device)
        pay_idx = torch.randint(0, self.condition_count, (1,), device=self.device)
        tar_idx = torch.randint(0, self.condition_count, (1,), device=self.device)

        reset_observation = TensorDict(
            {
                # reset start of sequence, zero out hidden and cell states
                "observation": start_token,
                "hidden": torch.zeros(
                    self.num_layers, self.hidden_dim, device=self.device
                ),
                "cell": torch.zeros(
                    self.num_layers, self.hidden_dim, device=self.device
                ),
                # apply our random conditions
                "antibody": ab_idx,
                "payload": pay_idx,
                "target": tar_idx,
                # resent end conditions to false
                "done": torch.tensor([False], device=self.device),
                "terminated": torch.tensor([False], device=self.device),
            }
        )

        return reset_observation

    def _step(self, tensordict):
        # get the action token from the actor dict
        action_token = tensordict["action"]

        # update the state by appending the item from action to our sequence
        # also increment the step that we are on in our RL loop
        self.generated_sequence.append(action_token.item())
        self.current_step += 1

        # determine if action resulted in our end token, if so terminate loop
        # also check if we are at our max length for a possible sequence
        is_end = action_token == self.token.end_token
        is_max = self.current_step >= self.max_length
        done = is_end | is_max

        # pass in the conditions generated at reset for each token generation in step
        ab_idx = tensordict["antibody"]
        pay_idx = tensordict["payload"]
        tar_idx = tensordict["target"]

        # if we are done generating our sequence then calculate the reward
        reward = 0.0
        if done:

            # check to see if we got a real molecule, penalize if not
            smiles_string = self.token.tokens_to_smiles(self.generated_sequence)
            mol = Chem.MolFromSmiles(smiles_string)

            if mol is None:
                # an invalid smiles string was generated
                raw_reward = -0.5

            else:
                # matches model device if on cuda
                device = self.device

                # convert the sequence to a tensor to pass into scoring model
                full_sequence = torch.tensor(
                    self.generated_sequence, device=device
                ).unsqueeze(0)

                # rewarder needs to the sequence to be in the tensordict format, assign to observation
                score_input = TensorDict(
                    {"observation": full_sequence}, batch_size=[1], device=device
                )

                with torch.no_grad():
                    self.rewarder(score_input)

                raw_reward = score_input["reward"].item()

            reward = raw_reward * 10.0

        # pass along info for next iterations, observation is just our action for the step
        # reward is 0 if not done and done and terminated are false if we haven't reached "END" or max length
        next_observation = TensorDict(
            {
                # pass the observation, reward and end flags forward
                "observation": action_token.view(1),
                "reward": torch.tensor([reward], device=self.device),
                "done": torch.tensor([done], device=self.device),
                "terminated": torch.tensor([done], device=self.device),
                # pass the conditions forward too
                "antibody": ab_idx,
                "payload": pay_idx,
                "target": tar_idx,
            }
        )

        return next_observation


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


if __name__ == "__main__":

    """Load our full ADC dataset"""
    # full_dataset_directory = "data/adc_data_complete_v2.pkl"
    # df = pd.read_pickle(full_dataset_directory)

    """Filter df by top 4 for each condition"""
    # conditions = ["antibody_name", "payload_name", "indication"]
    # df_filtered = df.copy()
    # for condition in conditions:
    #     unique_cond = list(df_filtered[condition].value_counts().head(4).index)
    #     df_filtered = df_filtered[df_filtered[condition].isin(unique_cond)]

    # df_filtered.to_pickle("data/adc_data_filtered.pkl")
    """Get list of dicts for condition tokens"""
    adc_directory = "data/adc_data_filtered.pkl"

    tokenizer = Tokenizer(adc_directory)

    df = pd.read_pickle(adc_directory)

    stoi_dicts, itos_dicts, dict_lengths = conditions_tokens(df)

    # easy_dataset = adcDataset(
    #     directory=adc_directory,
    #     tokenizer=tokenizer,
    #     stoi_dicts=stoi_dicts,
    #     augment=False,
    # )

    """Perform k-folds on sequence to one model to assess hyperparameters"""
    # kfolds_LSTM_scores(adc_directory)

    """train the scoring models"""
    # training_scores = ["SA", "TPSA", "QED", "LogP", "CSP3"]

    # for score in training_scores:
    #     model = train_LSTM_scores(
    #         dataset=easy_dataset, score_type=score, tokenizer=tokenizer, num_epochs=50
    #     )
    #     test_LSTM_scores(
    #         dataset=easy_dataset, model=model, score_type=score, tokenizer=tokenizer
    #     )
    #     torch.save(model.state_dict(), f"models/{score}_scores_weights.pth")

    """train the sequence to sequence generative model"""
    # hard_dataset = adcDataset(
    #     directory=adc_directory,
    #     tokenizer=tokenizer,
    #     stoi_dicts=stoi_dicts,
    #     augment=True,
    # )

    # easy_dataset = adcDataset(
    #     directory=adc_directory,
    #     tokenizer=tokenizer,
    #     stoi_dicts=stoi_dicts,
    #     augment=False,
    # )

    # gen_model = LSTMGenModel(
    #     input_dim=128,
    #     hidden_dim=256,
    #     layer_dim=5,
    #     vocab_size=tokenizer.vocab_size,
    #     padding_idx=tokenizer.pad_token,
    #     output_dim=tokenizer.vocab_size,
    #     condition_count=4,
    # )

    # model_gen = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=easy_dataset,
    #     tokenizer=tokenizer,
    #     stoi_dicts=stoi_dicts,
    #     num_epochs=50,
    # )
    # torch.save(model_gen.state_dict(), "models/model_gen_weights.pth")
    # gen_model.load_state_dict(
    #     torch.load("models/model_gen_weights.pth", weights_only=True)
    # )

    # model_gen = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=hard_dataset,
    #     tokenizer=tokenizer,
    #     stoi_dicts=stoi_dicts,
    #     num_epochs=100,
    # )

    # torch.save(model_gen.state_dict(), "models/model_gen_weights.pth")

    """test the sequence to sequence generative model with conditions"""

    # gen_model.load_state_dict(
    #     torch.load("models/model_gen_weights.pth", weights_only=True)
    # )

    # sequence = generate_smiles_conditional(
    #     model=gen_model,
    #     tokenizer=tokenizer,
    #     ab_idx=0,
    #     pay_idx=0,
    #     tar_idx=0,
    #     temperature=0.5,
    # )

    # print(tokenizer.tokens_to_smiles(sequence))

    # check_sequences_conditional(
    #     gen_model=gen_model,
    #     tokenizer=tokenizer,
    #     ab_idx=0,
    #     pay_idx=0,
    #     tar_idx=0,
    #     temperature=0.5,
    # )

    """Perform RL training on generative model"""

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
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )

    TPSA_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )

    QED_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )

    LogP_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )

    CSP3_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )

    gen_model.load_state_dict(
        torch.load("models/model_gen_weights.pth", weights_only=True)
    )

    SA_model.load_state_dict(
        torch.load("models/SA_scores_weights.pth", weights_only=True)
    )

    TPSA_model.load_state_dict(
        torch.load("models/TPSA_scores_weights.pth", weights_only=True)
    )

    QED_model.load_state_dict(
        torch.load("models/QED_scores_weights.pth", weights_only=True)
    )

    LogP_model.load_state_dict(
        torch.load("models/LogP_scores_weights.pth", weights_only=True)
    )

    CSP3_model.load_state_dict(
        torch.load("models/CSP3_scores_weights.pth", weights_only=True)
    )

    # RL_trained_model = LSTM_model_RL(
    #     SA_model=SA_model,
    #     TPSA_model=TPSA_model,
    #     QED_model=QED_model,
    #     LogP_model=LogP_model,
    #     CSP3_model=CSP3_model,
    #     gen_model=gen_model,
    #     critic_model=critic_model,
    #     tokenizer=tokenizer,
    #     total_frames=100_000,
    # )

    # torch.save(RL_trained_model.state_dict(), "models/model_RL_gen_weights.pth")

    """Test RL model"""
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
        torch.load("models/model_RL_gen_weights.pth", weights_only=True)
    )

    print()
    print("Sequence generated after RL training:")
    sequence = generate_smiles_conditional(
        model=gen_RL_model,
        tokenizer=tokenizer,
        ab_idx=0,
        pay_idx=0,
        tar_idx=0,
        temperature=0.5,
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
        ab_idx=0,
        pay_idx=0,
        tar_idx=0,
        temperature=0.5,
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
        ab_idx=0,
        pay_idx=0,
        tar_idx=0,
        temperature=0.5,
    )

    # features notes
    # SA - minimize
    # CSP3 - maximize
    # TPSA - minimize, above 140 is bad
    # QED - maximize
    # LogP - minimize <5
