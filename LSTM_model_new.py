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
from rdkit.Chem import AllChem

# Disable error messages when a non-valid smiles is encountered
rdBase.DisableLog("rdApp.error")


class Tokenizer:
    # SMILES tokenizer reference: https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html
    def __init__(self, df=None, tag_map=None):

        # store the motifs
        self.motif_map = tag_map if tag_map else {}
        self.SMI_REGEX = re.compile(
            r"(\[\d+Po\]|"  # Tags
            r"\[[^\]]+]|"  # <--- THIS PART captures [C@@H], [N+], etc.
            r"Br?|Cl?|N|O|S|P|F|I|B|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )

        self.vocab_list = [
            "#",
            "(",
            ")",
            "+",
            "-",
            ".",
            "/",
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
            ":",
            "=",
            "@",
            "Br",
            "C",
            "Cl",
            "F",
            "I",
            "N",
            "O",
            "P",
            "S",
            "[",
            "[10Po]",
            "[11Po]",
            "[12Po]",
            "[13Po]",
            "[14Po]",
            "[15Po]",
            "[16Po]",
            "[17Po]",
            "[18Po]",
            "[19Po]",
            "[1Po]",
            "[20Po]",
            "[21Po]",
            "[22Po]",
            "[23Po]",
            "[24Po]",
            "[25Po]",
            "[26Po]",
            "[27Po]",
            "[28Po]",
            "[29Po]",
            "[2Po]",
            "[30Po]",
            "[31Po]",
            "[32Po]",
            "[33Po]",
            "[34Po]",
            "[35Po]",
            "[36Po]",
            "[37Po]",
            "[38Po]",
            "[39Po]",
            "[3Po]",
            "[40Po]",
            "[41Po]",
            "[42Po]",
            "[43Po]",
            "[44Po]",
            "[45Po]",
            "[46Po]",
            "[47Po]",
            "[48Po]",
            "[4Po]",
            "[5Po]",
            "[6Po]",
            "[7Po]",
            "[8Po]",
            "[9Po]",
            "[BH3-]",
            "[C@@H]",
            "[C@@]",
            "[C@H]",
            "[C@]",
            "[CH2]",
            "[CH3]",
            "[CH]",
            "[N+]",
            "[N-]",
            "[NH+]",
            "[O-]",
            "[S@@]",
            "[S@]",
            "[SH]",
            "[Se]",
            "[c-]",
            "[n+]",
            "[nH]",
            "[se]",
            "\\",
            "]",
            "c",
            "n",
            "o",
            "p",
            "s",
            "[49Po]",
            "[50Po]",
            "[51Po]",
            "[52Po]",
            "[F]",
            "[Cl]",
            "[Br]",
            "[I]",
            "[B]",
            "[NH2]",
            "[S]",
            "[OH]",
            "[O]",
            "[NH]",
            "[Si]",
            "[o]",
            "[cH]",
            "[UNK]",
        ]

        # Sort to ensure determinism
        self.vocab_list = sorted(self.vocab_list)

        # Build Map
        self.smiles_map = {"PAD": 0}
        for i, token in enumerate(self.vocab_list):
            self.smiles_map[token] = i + 1

        self.smiles_map["END"] = len(self.smiles_map)
        self.smiles_map["START"] = len(self.smiles_map)

        self.pad_token = self.smiles_map["PAD"]
        self.start_token = self.smiles_map["START"]
        self.end_token = self.smiles_map["END"]

        self.unk_token = self.smiles_map.get("[UNK]", 0)

        # Reverse Map
        self.inv_smiles_map = {v: k for k, v in self.smiles_map.items()}
        self.vocab_size = len(self.smiles_map)

    def smiles_detokenizer(self):
        inv_smiles_map = {token: pattern for pattern, token in self.smiles_map.items()}
        return inv_smiles_map

    def smiles_to_tokens(self, smile_pattern_list):
        smiles_tokens = []
        for char in smile_pattern_list:
            # 3. USE .get() TO PREVENT CRASHES
            # If 'char' isn't found, return self.unk_token instead of KeyError
            token_id = self.smiles_map.get(char, self.unk_token)
            smiles_tokens.append(token_id)

        smiles_tokens.append(self.smiles_map["END"])
        return smiles_tokens

    def tokens_to_smiles(self, smile_token_list):
        smiles_patterns = []
        for token in smile_token_list:
            if token == self.end_token:
                break
            if token in [self.start_token, self.pad_token]:
                continue
            if hasattr(token, "item"):
                token = token.item()

            token_str = self.inv_smiles_map[token]
            smiles_patterns.append(token_str)
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
        # need to cast these to ensure that indices are ints, not floats, was getting a weird error
        sequence = sequence.long()
        antibody = antibody.long()
        payload = payload.long()
        target = target.long()

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

        out[:, :, 0] = -1e9  # mask pad token

        start_idx = out.size(-1) - 1
        out[:, :, start_idx] = -1e9  # mask the start token

        hidden_state = hidden_state.permute(1, 0, 2)  # swap them back for tensordict
        cell_state = cell_state.permute(1, 0, 2)

        return out, hidden_state, cell_state


class CriticModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        # using to judge actors current memory state
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden, **kwargs):
        # we only need the hidden state, kwargs allows us to keep forward pass logic
        # the same and pass in the sequence, cell, etc. without affecting this call
        current_state_rep = hidden[:, -1, :]

        x = self.fc1(current_state_rep)
        x = self.relu(x)
        value = self.fc2(x)

        return value


class adcDataset(Dataset):

    def __init__(
        self,
        df,
        tokenizer,
        stoi_dicts,
        score_type="SA",
        augment=False,
        source_type="real",
    ):
        self.df = df[df["data_type"] == source_type]
        self.tokenizer = tokenizer
        self.file = self.df
        self.smiles = (
            self.file["tagged_smiles"]
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
        # NEW: Pass the generator so we can prime the state
        generator_model,
        device="cuda",
        seed=None,
        condition_count=5,
    ):
        super().__init__(device=device, batch_size=[])
        self.rewarder = reward_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.token = tokenizer
        self.condition_count = condition_count
        self.current_step = 0
        self.episode_count = 0
        self.generated_sequence = []

        # Save model for state priming
        self.gen_model = generator_model

        self.head_ids = set()
        self.cleavable_ids = set()
        self.tail_ids = set()
        self.head_tags = []

        if hasattr(tokenizer, "motif_map"):
            for tag, name in tokenizer.motif_map.items():
                if tag in tokenizer.smiles_map:
                    idx = tokenizer.smiles_map[tag]
                    if "HEAD" in name:
                        self.head_ids.add(idx)
                        self.head_tags.append(tag)
                    elif "CLEAVABLE" in name:
                        self.cleavable_ids.add(idx)
                    elif "TAIL" in name:
                        self.tail_ids.add(idx)

        self.observation_spec = CompositeSpec(
            {
                "observation": DiscreteTensorSpec(
                    n=self.vocab_size, shape=(1,), dtype=torch.long
                ),
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

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        self.action = DiscreteTensorSpec(n=vocab_size, shape=(1,), dtype=torch.long)
        self.reward = UnboundedContinuousTensorSpec(shape=(1,))

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        return seed + 1

    def _reset(self, tensordict=None):
        self.current_step = 0
        self.generated_sequence = []

        # 1. Randomize Conditions
        ab_idx = torch.randint(1, self.condition_count, (1,), device=self.device)
        pay_idx = torch.randint(1, self.condition_count, (1,), device=self.device)
        tar_idx = torch.randint(1, self.condition_count, (1,), device=self.device)

        # 2. Decide on Prompting
        use_prompt = torch.rand(1).item() < 0.8

        # 3. Calculate Initial Hidden State
        # We MUST run the 'START' token through the model to get the correct
        # context for the prompt. Otherwise the model thinks [1Po] is the start.

        # Feed START
        start_input = torch.tensor([[self.token.start_token]], device=self.device)

        # Ensure model is in eval mode for this quick check (no grads needed here)
        # We act as the "Environment" setting up the board
        with torch.no_grad():
            _, h_prime, c_prime = self.gen_model(
                sequence=start_input,
                antibody=ab_idx.unsqueeze(0),  # Model expects batch dim
                payload=pay_idx.unsqueeze(0),
                target=tar_idx.unsqueeze(0),
            )
            # Remove batch dim from hidden states for TensorDict
            # Model returns (Layers, Batch, Dim), we need (Layers, Dim) for spec
            h_prime = h_prime.squeeze(0)
            c_prime = c_prime.squeeze(0)

        if use_prompt and len(self.head_tags) > 0:
            prompt_tag = random.choice(self.head_tags)
            prompt_id = self.token.smiles_map[prompt_tag]

            # Start sequence with the prompt
            self.generated_sequence = [prompt_id]
            self.current_step = 1

            # The Agent sees the PROMPT, and the HIDDEN STATE derived from START
            # This perfectly mimics the training sequence: START -> [1Po] -> [Agent decides next]
            start_token = torch.tensor([prompt_id], device=self.device)
            hidden_state = h_prime
            cell_state = c_prime
        else:
            # Standard start: Agent sees START, and ZERO hidden state
            start_token = torch.tensor(
                [self.token.start_token], device=self.device, dtype=torch.long
            )
            hidden_state = torch.zeros(
                self.num_layers, self.hidden_dim, device=self.device
            )
            cell_state = torch.zeros(
                self.num_layers, self.hidden_dim, device=self.device
            )

        return TensorDict(
            {
                "observation": start_token,
                "hidden": hidden_state,
                "cell": cell_state,
                "antibody": ab_idx,
                "payload": pay_idx,
                "target": tar_idx,
                "done": torch.tensor([False], device=self.device),
                "terminated": torch.tensor([False], device=self.device),
            }
        )

    def calculate_structural_reward(self, sequence_list):
        reward = 0.0

        # Count specific occurrences
        head_count = 0
        cleavable_count = 0
        tail_count = 0

        for token in sequence_list:
            if hasattr(token, "item"):
                token = token.item()

            if token in self.head_ids:
                head_count += 1
            elif token in self.cleavable_ids:
                cleavable_count += 1
            elif token in self.tail_ids:
                tail_count += 1

        # Do you have at least one of these?
        if head_count >= 1:
            reward += 5.0
        if cleavable_count >= 1:
            reward += 5.0
        if tail_count >= 1:
            reward += 5.0

        if head_count >= 1 and cleavable_count >= 1:
            reward += 10.0

        if head_count >= 1 and cleavable_count >= 1 and tail_count >= 1:
            if head_count == 1 and cleavable_count == 1 and tail_count == 1:
                reward += 50.0  # Perfection
            else:
                reward += 25.0  # Good but has duplicates

        return reward

    def _step(self, tensordict):
        action_token = tensordict["action"]
        self.generated_sequence.append(action_token.item())
        self.current_step += 1

        repetition_penalty = 0.0
        force_terminate = False
        if len(self.generated_sequence) >= 5:
            last_five = self.generated_sequence[-5:]
            if len(set(last_five)) == 1:
                repetition_penalty = 1.0
                force_terminate = True

        token_id = action_token.item()
        is_end = token_id == self.token.end_token
        is_max = self.current_step >= self.max_length
        done = is_end | is_max | force_terminate

        ab_idx = tensordict["antibody"]
        pay_idx = tensordict["payload"]
        tar_idx = tensordict["target"]
        next_hidden = tensordict["hidden"]
        next_cell = tensordict["cell"]

        reward = -repetition_penalty

        if done:
            self.episode_count += 1
            smiles_string = self.token.tokens_to_smiles(self.generated_sequence)
            mol = Chem.MolFromSmiles(smiles_string)

            if force_terminate:
                total_reward = -1.0
            elif mol is None:
                struct_bonus = self.calculate_structural_reward(self.generated_sequence)
                # Invalid? Small penalty (-1), but keep bonus to encourage motifs
                total_reward = -1.0 + (struct_bonus * 0.1)
            else:
                # Valid? Baseline (+1) + Bonus
                total_reward = 1.0
                struct_bonus = self.calculate_structural_reward(self.generated_sequence)
                total_reward += struct_bonus * 0.5

            reward += total_reward

            if self.episode_count % 100 == 0:
                struct_debug = self.calculate_structural_reward(self.generated_sequence)
                print(f"--- Ep {self.episode_count} ---")
                print(f"SMILES: {smiles_string}")
                print(f"Reward: {reward:.2f} (Struct: {struct_debug})")
                print("-----------------------------")

        return TensorDict(
            {
                "observation": action_token.view(1),
                "reward": torch.tensor([reward], device=self.device),
                "done": torch.tensor([done], device=self.device),
                "terminated": torch.tensor([done], device=self.device),
                "antibody": ab_idx,
                "payload": pay_idx,
                "target": tar_idx,
                "hidden": next_hidden,
                "cell": next_cell,
            }
        )


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

    print(
        f"DEBUG: Initialized {score_type} Model with Vocab Size: {tokenizer.vocab_size}"
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


def train_LSTM_gen(
    model,
    dataset,
    tokenizer,
    learning_rate=0.001,
    num_epochs=1000,
    use_weighted_loss=True,
):
    batch_size = 64
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tokenizer.collate_smiles,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"DEBUG: Initialized Gen Model with Vocab Size: {tokenizer.vocab_size}")

    model.to(device)
    print(f"Training on device: {device}")

    if use_weighted_loss:
        print("calculating class weights to fix imbalance")
        class_weights = calculate_class_weights(dataset, tokenizer, device)

        criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token, weight=class_weights
        )

    else:
        criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    max_smiles_length = 100
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
    learning_rate=0.00001,
    temperature=0.7,
    total_frames=500_000,
    num_envs=64,
    frame_steps=150,
    clip_epsilon=0.025,
    kl=0.05,
):
    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html
    # creating a dict for our LSTM generative model that the actor can then understand for RL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_model.train()

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

    # critic forward pass
    # def forward(self, batch_size, sequence, antibody, payload, target, hidden=None, cell=None)

    critic_module = ValueOperator(
        module=critic_model,
        in_keys=["hidden"],
    )

    # critic_module = ValueOperator(
    #     module=critic_model,
    #     in_keys=["observation", "antibody", "payload", "target", "hidden", "cell"],
    # )

    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.collectors.SyncDataCollector.html

    environment_maker = lambda: SmilesGeneratorEnv(
        reward_module,
        vocab_size=tokenizer.vocab_size,
        max_length=100,
        num_layers=gen_model.layer_dim,
        hidden_dim=gen_model.hidden_dim,
        tokenizer=tokenizer,
        generator_model=gen_model,
    )

    # hyper params for reinforcement learning:
    # learning_rate = 0.00001

    env = SerialEnv(num_envs, environment_maker)

    # anchor model to prevent moving too far from starting model
    ref_model = LSTMGenModel(
        input_dim=128,
        hidden_dim=256,
        layer_dim=5,
        vocab_size=tokenizer.vocab_size,
        padding_idx=tokenizer.pad_token,
        output_dim=tokenizer.vocab_size,
        condition_count=5,
    )

    ref_model.load_state_dict(gen_model.state_dict())
    ref_model.to(device)
    ref_model.eval()

    for param in ref_model.parameters():
        param.requires_grad = False

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
        clip_epsilon=clip_epsilon,  # default hyperparam
        entropy_bonus=0.0,  # encourages exploration to prevent same output
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

        kl_coeff = kl  # for KL divergence, prevents too large a shift in actor

        ppo_epochs = 1
        for _ in range(ppo_epochs):
            loss = loss_module(batch.reshape(-1))
            actor_loss = loss["loss_objective"]
            critic_loss = loss["loss_critic"]

            with torch.no_grad():
                # extract inputs from this batch
                inp_obs = batch["observation"].reshape(-1, 1)
                inp_ab = batch["antibody"].reshape(-1, 1)
                inp_pay = batch["payload"].reshape(-1, 1)
                inp_tar = batch["target"].reshape(-1, 1)

                # get the hidden and cell states too
                inp_hidden = batch["hidden"].reshape(
                    -1, gen_model.layer_dim, gen_model.hidden_dim
                )
                inp_cell = batch["cell"].reshape(
                    -1, gen_model.layer_dim, gen_model.hidden_dim
                )

                # Permute for model input
                inp_hidden = inp_hidden.permute(1, 0, 2).contiguous()
                inp_cell = inp_cell.permute(1, 0, 2).contiguous()

                # get reference model logits
                ref_logits, _, _ = ref_model(
                    sequence=inp_obs,
                    antibody=inp_ab,
                    payload=inp_pay,
                    target=inp_tar,
                    hidden_state=inp_hidden,
                    cell_state=inp_cell,
                )

            # next get the actor logits to compare against
            actor_logits, _, _ = gen_model(
                sequence=inp_obs,
                antibody=inp_ab,
                payload=inp_pay,
                target=inp_tar,
                hidden_state=inp_hidden,
                cell_state=inp_cell,
            )

            # compute the KL loss
            actor_log_probs = F.log_softmax(actor_logits, dim=-1)
            ref_probs = F.softmax(ref_logits, dim=-1)

            kl_div = F.kl_div(actor_log_probs, ref_probs, reduction="batchmean")

            total_loss = actor_loss + critic_loss + (kl_coeff * kl_div)

            if i % 10 == 0:
                print(f"KL Div: {kl_div.item():.4f} | Act Loss {actor_loss.item():.4f}")

            optimizer.zero_grad()
            total_loss.backward()
            # clip the gradients for stability
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=0.5)
            optimizer.step()

        # print reward to keep track of training progress
        # avg_reward = batch["next", "reward"].mean().item()
        rewards = batch["next", "reward"].sum(dim=1)
        avg_reward = rewards.mean().item()
        print(
            f"Batch {i} || Loss: {total_loss.item():.4f}, Avg Reward: {avg_reward:.4f}"
        )

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

    max_smiles_length = 100
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
    adc_motifs,
    num_samples=100,
    temperature=1.0,
):
    valid_smiles = 0
    SA_scores = []
    TPSA_scores = []
    QED_scores = []
    LogP_scores = []
    CSP3_scores = []

    count_heads = 0
    count_tails = 0
    count_cleavable = 0
    count_perfect = 0

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

    def get_ids(motif_list):
        ids = set()
        for smi in motif_list:
            if smi in tokenizer.smiles_map:
                ids.add(tokenizer.smiles_map[smi])
        return ids

    head_ids = get_ids(adc_motifs["heads"])
    cleavable_ids = get_ids(adc_motifs["cleavable"])
    tail_ids = get_ids(adc_motifs["tails"])

    for _ in range(num_samples):
        sequence = generate_smiles_conditional(
            model=gen_model,
            tokenizer=tokenizer,
            ab_idx=ab_idx,
            pay_idx=pay_idx,
            tar_idx=tar_idx,
            temperature=temperature,
        )

        seq_set = set(sequence)

        # "not isdisjoint" means they share at least one element
        has_head = not head_ids.isdisjoint(seq_set)
        has_cleavable = not cleavable_ids.isdisjoint(seq_set)
        has_tail = not tail_ids.isdisjoint(seq_set)

        if has_head:
            count_heads += 1
        if has_cleavable:
            count_cleavable += 1
        if has_tail:
            count_tails += 1
        if has_head and has_cleavable and has_tail:
            count_perfect += 1
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

    total = num_samples if num_samples > 0 else 1

    print("-" * 30)
    print(f"Valid smiles = {valid_smiles}%")
    print(f"Contains Head:      {count_heads}/{total} ({count_heads/total*100:.1f}%)")
    print(
        f"Contains Cleavable: {count_cleavable}/{total} ({count_cleavable/total*100:.1f}%)"
    )
    print(f"Contains Tail:      {count_tails}/{total} ({count_tails/total*100:.1f}%)")
    print(
        f"PERFECT (All 3):    {count_perfect}/{total} ({count_perfect/total*100:.1f}%)"
    )
    print(f"Avg SA = {np.mean(SA_scores):.4f}\t\tStd Dev = {np.std(SA_scores):.4f}")
    print(f"Avg TPSA = {np.mean(TPSA_scores):.4f}\tStd Dev = {np.std(TPSA_scores):.4f}")
    print(f"Avg QED = {np.mean(QED_scores):.4f}\tStd Dev = {np.std(QED_scores):.4f}")
    print(f"Avg LogP = {np.mean(LogP_scores):.4f}\tStd Dev = {np.std(LogP_scores):.4f}")
    print(f"Avg CSP3 = {np.mean(CSP3_scores):.4f}\tStd Dev = {np.std(CSP3_scores):.4f}")
    print("-" * 30)


def generate_valid_linkers(
    adc_motifs,
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
    target_count=10,
    max_attempts=10000,
    temperature=1.0,
):
    valid_linkers = []
    attempts = 0
    failures = {"structure": 0, "invalid": 0}

    # Setup Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in [SA_model, TPSA_model, QED_model, LogP_model, CSP3_model, gen_model]:
        model.to(device).eval()

    # --- 1. PREPARE TAG LOOKUPS ---
    tag_to_real = {}
    counter = 1
    for smiles_list in adc_motifs.values():
        for s in smiles_list:
            tag_to_real[f"[{counter}Po]"] = s
            counter += 1

    # Get sets of IDs for checking structure
    head_ids, cleavable_ids, tail_ids = set(), set(), set()
    head_tags = []  # We need this list to pick prompts

    if hasattr(tokenizer, "motif_map"):
        for tag, name in tokenizer.motif_map.items():
            if tag in tokenizer.smiles_map:
                idx = tokenizer.smiles_map[tag]
                if "HEAD" in name:
                    head_ids.add(idx)
                    head_tags.append(tag)
                elif "CLEAVABLE" in name:
                    cleavable_ids.add(idx)
                elif "TAIL" in name:
                    tail_ids.add(idx)

    # --- 2. HELPER: CONVERT TAGS ---
    def convert_tags_to_real(tagged_smi):
        real_smi = tagged_smi
        for tag, real in tag_to_real.items():
            real_smi = real_smi.replace(tag, real)
        return real_smi

    # --- 3. HELPER: CHECK STRUCTURE ---
    def check_structure(sequence_list):
        head_c, cleav_c, tail_c = 0, 0, 0
        for token in sequence_list:
            if hasattr(token, "item"):
                token = token.item()
            if token in head_ids:
                head_c += 1
            elif token in cleavable_ids:
                cleav_c += 1
            elif token in tail_ids:
                tail_c += 1

        # We need exactly 1 of each
        if head_c == 1 and cleav_c == 1 and tail_c == 1:
            return True
        return False

    print(f"Generating linkers (Target: {target_count})...")

    # --- 4. GENERATION LOOP ---
    while (len(valid_linkers) < target_count) and (attempts < max_attempts):
        attempts += 1

        # --- CRITICAL FIX: MANUAL PROMPTING ---
        # We manually start the generation loop with a random HEAD tag
        # This mimics the RL environment's behavior.

        prompt_tag = random.choice(head_tags)
        prompt_id = tokenizer.smiles_map[prompt_tag]

        # We need to manually "prime" the model with START -> PROMPT
        # Note: We can't use generate_smiles_conditional directly because it starts with START
        # So we create a slightly modified call logic here:

        # A. Feed START to get hidden state
        start_input = torch.tensor([[tokenizer.start_token]], device=device)
        ab_t = torch.tensor([[ab_idx]], device=device)
        pay_t = torch.tensor([[pay_idx]], device=device)
        tar_t = torch.tensor([[tar_idx]], device=device)

        with torch.no_grad():
            _, hidden, cell = gen_model(start_input, ab_t, pay_t, tar_t)

        # B. Feed PROMPT (The Head) to start the real generation
        current_token = torch.tensor([[prompt_id]], device=device)
        sequence = [prompt_id]  # Start list with our prompt

        # C. Run Generation Loop
        for _ in range(100):  # max length
            with torch.no_grad():
                out, hidden, cell = gen_model(
                    current_token, ab_t, pay_t, tar_t, hidden, cell
                )

                # Sampling
                logits = out[:, -1, :]
                logits[0, tokenizer.start_token] = -1e9
                logits[0, tokenizer.pad_token] = -1e9

                # Apply Temperature
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                if next_token == tokenizer.end_token:
                    break

                sequence.append(next_token)
                current_token = torch.tensor([[next_token]], device=device)

        # --- END GENERATION ---

        # 5. VALIDATE
        if check_structure(sequence):
            tagged_smiles = tokenizer.tokens_to_smiles(sequence)

            if Chem.MolFromSmiles(tagged_smiles):
                real_smiles = convert_tags_to_real(tagged_smiles)
                valid_linkers.append(real_smiles)

                # Print success to confirm it's working
                print(f"  [FOUND!] {tagged_smiles}")
            else:
                failures["invalid"] += 1
        else:
            failures["structure"] += 1
            # Debug: Print what it generated so we know why it failed
            if attempts % 100 == 0:
                print(f"  [Fail Struct] {tokenizer.tokens_to_smiles(sequence)}")

    # Reporting...
    print("-" * 30)
    print(f"Total Attempts: {attempts}")
    print(f"Found: {len(valid_linkers)}")
    return valid_linkers


def compile_motifs(motif_dict):
    patterns = {"heads": [], "cleavable": [], "tails": []}
    for category, smiles in motif_dict.items():
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                patterns[category].append(mol)
    return patterns


# need to canonicalize motifs so that pattern matching works in tokenization
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


def verify_model_vocabulary_corrected(gen_model, tokenizer, motif_dict=None):
    print("\n--- VOCABULARY SANITY CHECK (TAGGED VERSION) ---")
    gen_model.eval()
    device = next(gen_model.parameters()).device

    # 1. Check Motif Coverage
    # We now check if the TAGS (e.g. [1Po]) are in the vocab, not the SMILES
    print(f"Vocab Size: {tokenizer.vocab_size}")

    if not hasattr(tokenizer, "motif_map") or not tokenizer.motif_map:
        print("CRITICAL ERROR: Tokenizer has no 'motif_map'. Tags were not loaded.")
        return

    found_count = 0
    total_motifs = len(tokenizer.motif_map)

    for tag, name in tokenizer.motif_map.items():
        if tag in tokenizer.smiles_map:
            found_count += 1

    print(f"Motif Coverage: {found_count}/{total_motifs} tags present in vocabulary.")

    # 2. Ask Model to Predict
    start_token = torch.tensor([[tokenizer.start_token]]).to(device)

    # Use dummy conditions (IDs 1)
    ab = torch.tensor([[1]]).to(device)
    pay = torch.tensor([[1]]).to(device)
    tar = torch.tensor([[1]]).to(device)

    with torch.no_grad():
        output, _, _ = gen_model(start_token, ab, pay, tar)
        probs = F.softmax(output[:, -1, :], dim=-1)

    top_probs, top_indices = torch.topk(probs, 10)

    print("\nModel's Top 10 predictions after START:")

    for i in range(10):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()

        # Get the token string (which might be a tag like [1Po])
        token_str = tokenizer.inv_smiles_map.get(idx, "UNKNOWN")

        # Check if this token is a known motif tag
        is_motif = token_str in tokenizer.motif_map

        # Get the readable name (<HEAD_0>) from the tokenizer's map
        label = tokenizer.motif_map[token_str] if is_motif else "Atom/Char"

        print(f"   Rank {i+1}: ID {idx} | Type: {label} | Prob: {prob:.4f}")
        if is_motif:
            print(f"      -> Tag: {token_str}")

    print("-------------------------------------------\n")


def calculate_class_weights(dataset, tokenizer, device):
    # 1. Count all tokens
    token_counts = np.zeros(tokenizer.vocab_size)

    # Iterate through the dataset to count tokens
    # dataset.smiles is a list of token lists
    all_tokens = [t for seq in dataset.smiles for t in seq]

    unique, counts = np.unique(all_tokens, return_counts=True)
    for u, c in zip(unique, counts):
        token_counts[u] = c

    # Handle tokens that might not appear (avoid divide by zero)
    # Set count to 1 for unseen tokens so they get high weight but don't crash
    token_counts[token_counts == 0] = 1.0

    # 2. Inverse Class Frequency
    # Total tokens / (n_classes * frequency)
    # This balances the impact of each class
    total_samples = sum(token_counts)
    weights = total_samples / (len(token_counts) * token_counts)

    # 3. Normalize or Clip
    # We don't want weights to be TOO extreme (e.g., 10,000x)
    # Clip weights to a reasonable range (e.g., max 10.0 or 20.0)
    weights = np.clip(weights, 0.01, 20.0)

    # 4. Convert to Tensor
    weight_tensor = torch.FloatTensor(weights).to(device)

    # Ensure PAD token has 0 weight (ignored anyway by ignore_index, but good practice)
    weight_tensor[tokenizer.pad_token] = 0.0

    return weight_tensor


def inspect_dataset_tokenization(dataset, tokenizer, num_samples=5):
    print(f"\n--- INSPECTING DATASET TOKENIZATION (N={num_samples}) ---")

    found_motifs = 0
    total_tokens_checked = 0

    # Randomly sample indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i in indices:
        # 1. Get the Raw SMILES from the dataframe
        raw_smiles = dataset.file.iloc[i]["smiles"]

        # 2. Run the Tokenizer's Regex manually
        tokens = tokenizer.SMI_REGEX.findall(raw_smiles)

        # 3. Check for Motifs
        print(f"\nSample {i}:")
        print(f"Raw: {raw_smiles[:50]}...")  # Truncated

        # Reconstruct readable tokens
        readable_tokens = []
        has_motif = False
        for t in tokens:
            if t in tokenizer.smiles_map:
                # Check if it's a motif token
                # In your code, motifs are the KEYS of motif_map (the smiles strings)
                # But we want to know if the regex matched a long string or single chars
                if len(t) > 2 and "Br" not in t and "Cl" not in t and "[" not in t:
                    # Heuristic: Long tokens are likely motifs
                    readable_tokens.append(f"[[MOTIF: {t[:10]}...]]")
                    has_motif = True
                    found_motifs += 1
                elif t in tokenizer.motif_map:  # Direct lookup if you stored it
                    readable_tokens.append(f"[[MOTIF_ID]]")
                    has_motif = True
                    found_motifs += 1
                else:
                    readable_tokens.append(t)
            else:
                readable_tokens.append(f"UNKNOWN({t})")

        print(f"Tokens: {readable_tokens}")

        total_tokens_checked += len(tokens)

    print(f"\nSUMMARY: Found {found_motifs} motifs in {num_samples} samples.")
    if found_motifs == 0:
        print("DIAGNOSIS: The Regex is NOT matching the motifs in the dataset strings.")
        print("The model is learning atoms because it is being fed atoms.")
    else:
        print(
            "DIAGNOSIS: The Regex IS working. The issue is Class Imbalance/Weighting."
        )


# crucial need to keep
def tag_dataset_with_motifs(df, motif_dict):
    print("Tagging dataset with Graph-Based Replacements...")

    # 1. Build a Lookup: Pattern Molecule -> Replacement Tag
    # We use Polonium [Po] with isotopes 1, 2, 3... as unique IDs
    replacements = []
    counter = 1

    # Keep track of what ID maps to what name for the tokenizer later
    tag_map = {}

    for category, smiles_list in motif_dict.items():
        for i, smi in enumerate(smiles_list):
            pattern = Chem.MolFromSmiles(smi)
            if pattern is None:
                continue

            # Create a dummy atom with a specific isotope
            # e.g., [1Po], [2Po], [3Po]...
            tag_smiles = f"[{counter}Po]"
            replacement = Chem.MolFromSmiles(tag_smiles)

            token_name = f"<{category.upper()}_{i}>"

            # Store tuple: (Pattern, Replacement, Name, TagString)
            replacements.append((pattern, replacement, token_name, tag_smiles))
            tag_map[tag_smiles] = token_name
            counter += 1

    # 2. Define the row processor
    def process_row(smi):
        if not smi:
            return ""
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return ""

        # Try to replace every motif we know
        # We sort by number of atoms in pattern (descending) to match largest motifs first
        # (This prevents a small fragment match inside a larger one)
        sorted_replacements = sorted(
            replacements, key=lambda x: x[0].GetNumAtoms(), reverse=True
        )

        for pat, rep, name, tag in sorted_replacements:
            if mol.HasSubstructMatch(pat):
                try:
                    # Replace ALL instances of this motif
                    mol = AllChem.ReplaceSubstructs(mol, pat, rep, replaceAll=True)[0]
                except:
                    # Sometimes replacement fails if valences are weird, skip
                    pass

        try:
            # Return the new SMILES (now containing [1Po], [2Po] etc)
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return ""

    # 3. Apply to DataFrame
    df["tagged_smiles"] = df["smiles"].apply(process_row)

    # Filter out failures
    df = df[df["tagged_smiles"] != ""]

    print("Tagging Complete.")
    return df, tag_map


def check_motif_usage_in_generation(gen_model, tokenizer, num_samples=10):
    print(f"\n--- GENERATING {num_samples} SAMPLES TO CHECK MOTIF USAGE ---")
    gen_model.eval()
    device = next(gen_model.parameters()).device

    # Dummy conditions
    ab = torch.tensor([[1]]).to(device)
    pay = torch.tensor([[1]]).to(device)
    tar = torch.tensor([[1]]).to(device)

    found_motifs = 0

    for i in range(num_samples):
        # Generate sequence
        seq = generate_smiles_conditional(
            gen_model, tokenizer, 1, 1, 1, temperature=1.0
        )

        # Check for motif tags in the output sequence
        # We look at the IDs in 'seq' and check if they are in tokenizer.motif_map
        motifs_in_seq = []
        for token_id in seq:
            token_str = tokenizer.inv_smiles_map.get(token_id, "")
            if token_str in tokenizer.motif_map:
                motifs_in_seq.append(tokenizer.motif_map[token_str])

        smiles = tokenizer.tokens_to_smiles(seq)

        if motifs_in_seq:
            found_motifs += 1
            print(f"Sample {i+1}: FOUND {len(motifs_in_seq)} MOTIFS -> {motifs_in_seq}")
            print(f"   SMILES: {smiles[:50]}...")  # Truncated
        else:
            print(f"Sample {i+1}: NO MOTIFS")

    print(
        f"\nSuccess Rate: {found_motifs}/{num_samples} generated molecules contained motifs."
    )


def test_prompted_generation(gen_model, tokenizer):
    print("\n--- PROMPTED GENERATION TEST ---")
    gen_model.eval()
    device = next(gen_model.parameters()).device

    # 1. Find a real Head Tag (e.g., [1Po])
    head_tag = None
    head_name = None
    for tag, name in tokenizer.motif_map.items():
        if "HEAD" in name:
            head_tag = tag
            head_name = name
            break

    if not head_tag:
        print("Error: No HEAD tags found in map.")
        return

    print(f"Prompting model with: {head_name} ({head_tag})")

    # 2. Prepare Inputs
    head_id = tokenizer.smiles_map[head_tag]
    # We feed [START, HEAD_TAG] as the input sequence
    current_token = torch.tensor([[tokenizer.start_token, head_id]]).to(device)

    # Dummy conditions
    ab = torch.tensor([[1]]).to(device)
    pay = torch.tensor([[1]]).to(device)
    tar = torch.tensor([[1]]).to(device)

    # 3. Generate the rest
    sequence = [tokenizer.start_token, head_id]

    # We need to handle the LSTM state carefully.
    # The simplest way with your current code is to just run the loop manually from the 2nd token.
    # But let's use a simplified loop for this test:

    hidden, cell = None, None

    # First pass: Feed START to get state
    input_1 = torch.tensor([[tokenizer.start_token]]).to(device)
    _, hidden, cell = gen_model(input_1, ab, pay, tar, hidden, cell)

    # Second pass: Feed HEAD to get next state
    input_2 = torch.tensor([[head_id]]).to(device)
    out, hidden, cell = gen_model(input_2, ab, pay, tar, hidden, cell)

    print("Generating...")
    for i in range(50):
        # Predict next
        probs = F.softmax(out[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        sequence.append(next_token)
        if next_token == tokenizer.end_token:
            break

        # Feed next
        input_next = torch.tensor([[next_token]]).to(device)
        out, hidden, cell = gen_model(input_next, ab, pay, tar, hidden, cell)

    # 4. Decode
    smiles = tokenizer.tokens_to_smiles(sequence)
    print(f"\nResult: {smiles}")

    if Chem.MolFromSmiles(smiles):
        print("SUCCESS: Valid molecule generated from prompt!")
    else:
        print("FAILURE: Invalid molecule.")


def generate_robust_static_vocab(tag_map):
    # 1. Standard Atoms (Hardcoded)
    # These are the atoms RDKit might generate
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
            "c",
            "n",
            "o",
            "s",
            "p",
            "C",
            "N",
            "O",
            "S",
            "P",
            "F",
            "I",
            "Cl",
            "Br",
            "[C@H]",
            "[C@@H]",
            "[C@]",
            "[C@@]",
            "[nH]",
            "[N+]",
            "[N-]",
            "[O-]",
            "[S@]",
            "[S@@]",
            "[n+]",
            "[c-]",
            "[CH]",
            "[CH2]",
            "[se]",
            "[Se]",
            "[BH3-]",
            "[NH+]",
            "[SH]",
        ]
    )

    # 2. Add ALL Tags from the Map
    # tag_map keys are '[1Po]', '[2Po]', etc.
    # This ensures [18Po] is included even if it's not in the dataframe!
    if tag_map:
        standard_tokens.update(tag_map.keys())

    # 3. Sort and Print
    final_vocab = sorted(list(standard_tokens))

    print("\n--- ROBUST STATIC VOCABULARY LIST ---")
    print(final_vocab)
    print(f"Total Size: {len(final_vocab)}")
    return final_vocab


if __name__ == "__main__":
    adc_motifs = {
        "cleavable": [
            "c1ccccc1",
            "SS",
            "CC(C)[C@@H]C=O",
            "CCC(=O)NNC=O",
            "CC[C@@H]C=O",
            "OC(=O)[C@@H](CCC)C=O",
            "Oc1ccc(N=Nc2ccccc2)cc1",
            "O=[N+]([O-])c1ccccc1",
            "NC(=O)[C@@H](CCCC)C=O",
            "CC(C)CC=O",
            "OC(=O)[C@@H](CCC=O)CC=O",
            "OC(=O)[C@@H](CCCC)C=O",
        ],
        "heads": [
            "O=C1C=CC(=O)N1",
            "O=C1CCC(=O)N1O",
            "O=C1CC(S(=O)(=O)O)C(=O)N1O",
            "CCBr",
            "O=CCBr",
            "O=C1N[C@@H]2[C@H](S1)NC2",
            "CCC(=O)NN",
            "O=Cc1ccccc1",
            "CCN=[N+]=[N-]",
            "CC#C",
            "C#Cc1ccccc1",
            "C1=CCCC=CC1",
            "Cc1nnc[nH]1",
            "C#CC1CC2CC1C2",
            "C#CC1CCCCC1",
            "Fc1c(F)c(F)c(F)c(F)F",
            "Fc1cc(F)c(F)c1F",
            "O=[N+]([O-])c1ccccc1",
            "O=[N+]([O-])c1cc([N+](=O)[O-])ccc1",
            "Cc1ccc(S(=O)(=O)O)cc1",
            "CS(=O)(=O)O",
        ],
        "tails": [
            "CCC(=O)O",
            "CC(=O)O",
            "CCN",
            "CN",
            "c1ccccc1",
            "CCS",
            "CC(=O)S",
            "CO",
            "CC(C)(C)C",
            "CC1c2ccccc2-c2ccccc21",
            "CCO",
            "CO",
            "CC",
            "c1ccncc1",
            "NC=O",
            "CCS(=O)(=O)O",
            "CO",
            "CC=O",
            "CC(=O)CCC=O",
        ],
    }

    adc_directory = "data/adc_data_filtered.pkl"
    synthetic_directory = "data/synthetic_data.pkl"

    adc_df = pd.read_pickle(adc_directory)
    synth_df = pd.read_pickle(synthetic_directory)
    adc_df["data_type"] = "real"
    synth_df["data_type"] = "synthetic"

    combo_df = pd.concat([adc_df, synth_df])
    combo_df["indication"] = combo_df["indication"].fillna("unknown")
    combo_df["antibody_name"] = combo_df["antibody_name"].fillna("unknown")
    combo_df["payload_name"] = combo_df["payload_name"].fillna("unknown")
    combo_df["data_type"] = combo_df["data_type"].fillna("real")
    combo_df = combo_df.fillna(0.0)

    # Clean Motifs
    adc_motifs_clean = canonicalize_motifs(adc_motifs)

    # Clean Dataframe Strings
    def strict_canonicalize(smi):
        if not smi:
            return ""
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return ""

    combo_df["smiles"] = combo_df["smiles"].apply(strict_canonicalize)
    combo_df = combo_df[combo_df["smiles"] != ""]

    # Apply Graph Tags
    combo_df, tag_map = tag_dataset_with_motifs(combo_df, adc_motifs_clean)

    # Init Tokenizer
    tokenizer = Tokenizer(combo_df, tag_map=tag_map)
    stoi_dicts, itos_dicts, dict_lengths = conditions_tokens(combo_df)

    # Create Datasets
    synth_dataset = adcDataset(
        df=combo_df,
        tokenizer=tokenizer,
        stoi_dicts=stoi_dicts,
        augment=False,
        source_type="synthetic",
    )
    easy_dataset = adcDataset(
        df=combo_df,
        tokenizer=tokenizer,
        stoi_dicts=stoi_dicts,
        augment=False,
        source_type="real",
    )

    # # --- 2. TRAIN SCORING MODELS ---
    # print("--- Training Scoring Models ---")
    # training_scores = ["SA", "TPSA", "QED", "LogP", "CSP3"]
    # for score in training_scores:
    #     model = train_LSTM_scores(
    #         dataset=easy_dataset, score_type=score, tokenizer=tokenizer, num_epochs=20
    #     )
    #     torch.save(model.state_dict(), f"models/{score}_scores_weights.pth")

    # # # --- 3. TRAIN GENERATOR (The Sandwich Method) ---
    # print("--- Training Generative Model ---")

    # gen_model = LSTMGenModel(
    #     input_dim=128,
    #     hidden_dim=256,
    #     layer_dim=5,
    #     vocab_size=tokenizer.vocab_size,
    #     padding_idx=tokenizer.pad_token,
    #     output_dim=tokenizer.vocab_size,
    #     condition_count=5,
    # )

    # # Phase A: Learn Grammar (Synthetic)
    # print("Phase A: Learning Grammar (Synthetic Data)...")
    # gen_model = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=synth_dataset,
    #     tokenizer=tokenizer,
    #     learning_rate=0.001,
    #     num_epochs=30,
    #     use_weighted_loss=False,
    # )

    # # Phase B: Learn Motifs (Real Data + Weighted Loss)
    # # This forces the model to use the rare [1Po] tags
    # print("Phase B: Forcing Motif Learning (Weighted Loss)...")
    # gen_model = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=easy_dataset,
    #     tokenizer=tokenizer,
    #     learning_rate=0.001,
    #     num_epochs=50,
    #     use_weighted_loss=True,
    # )
    # torch.save(gen_model.state_dict(), "models/model_gen_weights_weighted.pth")

    # # Phase C: Repair Syntax (Mixed Data)
    # print("Phase C: Repairing Syntax (Mixed Data)...")
    # mixed_dataset = torch.utils.data.ConcatDataset([synth_dataset, easy_dataset])
    # gen_model = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=mixed_dataset,
    #     tokenizer=tokenizer,
    #     learning_rate=0.0001,
    #     num_epochs=20,
    #     use_weighted_loss=False,
    # )

    # # Phase D: Final Polish (Real Data, Low LR)
    # print("Phase D: Final Polish (Real Data)...")
    # gen_model = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=easy_dataset,
    #     tokenizer=tokenizer,
    #     learning_rate=0.00005,
    #     num_epochs=15,
    #     use_weighted_loss=False,
    # )
    # torch.save(gen_model.state_dict(), "models/model_gen_weights_final.pth")

    # --- 4. REINFORCEMENT LEARNING ---
    print("--- Starting Reinforcement Learning ---")

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

    # Load Scoring Models
    SA_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )
    SA_model.load_state_dict(
        torch.load("models/SA_scores_weights.pth", weights_only=True)
    )

    TPSA_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )
    TPSA_model.load_state_dict(
        torch.load("models/TPSA_scores_weights.pth", weights_only=True)
    )

    QED_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )
    QED_model.load_state_dict(
        torch.load("models/QED_scores_weights.pth", weights_only=True)
    )

    LogP_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )
    LogP_model.load_state_dict(
        torch.load("models/LogP_scores_weights.pth", weights_only=True)
    )

    CSP3_model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, tokenizer=tokenizer
    )
    CSP3_model.load_state_dict(
        torch.load("models/CSP3_scores_weights.pth", weights_only=True)
    )

    # Load Generator (Final Weights)
    gen_model.load_state_dict(
        torch.load("models/model_gen_weights_final.pth", weights_only=True)
    )

    # need to tune these params
    RL_trained_model = LSTM_model_RL(
        SA_model,
        TPSA_model,
        QED_model,
        LogP_model,
        CSP3_model,
        gen_model,
        critic_model,
        tokenizer,
        learning_rate=0.00001,
        temperature=0.7,
        total_frames=500_000,
        num_envs=64,
        frame_steps=150,
        clip_epsilon=0.025,
        kl=0.05,
    )

    torch.save(RL_trained_model.state_dict(), "models/model_RL_gen_weights.pth")

    # gen_model.load_state_dict(
    #     torch.load("models/model_RL_gen_weights.pth", weights_only=True)
    # )

    # linkers = generate_valid_linkers(
    #     adc_motifs,
    #     SA_model,
    #     TPSA_model,
    #     QED_model,
    #     LogP_model,
    #     CSP3_model,
    #     gen_model,
    #     tokenizer,
    #     ab_idx=1,
    #     pay_idx=2,
    #     tar_idx=1,
    #     target_count=1,
    #     max_attempts=1000,
    #     temperature=1.0,
    # )

    # print(linkers)
