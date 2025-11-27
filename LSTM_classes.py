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
