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
import pickle
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, rdFingerprintGenerator
import sascorer

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import selfies as sf

# Disable error messages when a non-valid smiles is encountered
rdBase.DisableLog("rdApp.error")


class Tokenizer:
    def __init__(self, df=None, tag_map=None):

        # store the motifs
        self.motif_map = tag_map if tag_map else {}

        self.vocab_set = set()
        self.max_len = 0

        # helper tokens
        self.pad_token_str = "[PAD]"
        self.start_token_str = "[START]"
        self.end_token_str = "[END]"
        self.unk_token_str = "[UNK]"

        self.vocab_set.update(
            [
                self.pad_token_str,
                self.start_token_str,
                self.end_token_str,
                self.unk_token_str,
            ]
        )

        if df is not None and "selfies" in df.columns:
            self.build_vocab(df["selfies"].tolist())

        self.depro_map = {
            # Fmoc (Carbamate): Includes the Nitrogen in your string
            # Replace the whole group with just "N" (Primary Amine)
            "NC(=O)OCC1c2ccccc2-c2ccccc21": "N",
            # Boc (Carbamate): Attached to an N
            # Replace with nothing (leaves the N it was attached to)
            "C(=O)OC(C)(C)C": "",
            # Cbz (Carbamate): Attached to an N
            # Replace with nothing (leaves the N it was attached to)
            "C(=O)OCc1ccccc1": "",
            # Thioacetate (Thiol protection): Includes the Sulfur in your string
            # Replace "S-Acetyl" with just "S" (Free Thiol)
            "CC(=O)S": "S",
            # Levulinyl (Alcohol/Amine protection)
            # Replace with nothing (leaves the O/N it was attached to)
            "CC(=O)CCC(=O)": "",
        }

    def build_vocab(self, selfies_list):
        for s in selfies_list:
            try:
                # split selfies by brackets into tokens
                tokens = list(sf.split_selfies(s))
                self.vocab_set.update(tokens)  # add them to our vocab set
                self.max_len = max(self.max_len, len(tokens))
            except:
                pass  # just in case there is an encoding error to selfies

        # add our linker motifs to the vocab set as well
        self.vocab_set.update(self.motif_map.keys())

        # sort our vocabulary set
        self.vocab_list = sorted(list(self.vocab_set))

        # generate smiles map and inverse map
        self.smiles_map = {token: i for i, token in enumerate(self.vocab_list)}
        self.inv_smiles_map = {i: token for token, i in self.smiles_map.items()}

        # map the special helper IDs
        self.pad_token = self.smiles_map.get(self.pad_token_str, 0)
        self.start_token = self.smiles_map.get(self.start_token_str, 0)
        self.end_token = self.smiles_map.get(self.end_token_str, 0)
        self.unk_token = self.smiles_map.get(self.unk_token_str, 0)

        # save attribute for final vocab size after generation
        self.vocab_size = len(self.vocab_list)

    def selfies_to_ids(self, selfies_string):
        try:
            tokens = list(sf.split_selfies(selfies_string))
            ids = []
            for t in tokens:
                # Get ID
                token_id = self.smiles_map.get(t, self.unk_token)

                # --- DEBUG BLOCK ---
                if not isinstance(token_id, int):
                    print(f"CRITICAL ERROR: Token '{t}' mapped to Non-Int '{token_id}'")
                    print(f"Type: {type(token_id)}")
                    # Force it to unk_token (0) to prevent crash so we can see logs
                    token_id = self.unk_token
                # -------------------

                ids.append(token_id)

            ids.append(self.end_token)
            return ids
        except Exception as e:
            print(f"Tokenization error: {e}")
            return [self.unk_token]

    def ids_to_selfies(self, token_ids):
        tokens = []
        for tid in token_ids:
            # handles case where token is tensor scalar
            if hasattr(tid, "item"):
                tid = tid.item()

            # check for end token
            if tid == self.end_token:
                break

            # check for start or pad tokens and ignore
            if tid in [self.start_token, self.pad_token]:
                continue

            # convert the token ids to selfies
            token_str = self.inv_smiles_map.get(tid, "")
            tokens.append(token_str)

        # join the selfies together and return selfies sequence
        return "".join(tokens)

    def collate_smiles(self, batch):
        """Kept the old name from using SMILES, now collates our selfies tokens"""
        smiles_list, ab_list, pay_list, target_list = zip(*batch)

        smiles_padded = pad_sequence(
            smiles_list, batch_first=True, padding_value=self.pad_token
        )

        ab_stack = torch.stack(ab_list).long()
        pay_stack = torch.stack(pay_list).long()
        target_stack = torch.stack(target_list).long()

        return smiles_padded, ab_stack, pay_stack, target_stack


class adcDataset(Dataset):

    def __init__(
        self,
        df,
        tokenizer,
        stoi_dicts,
        source_type="real",
    ):
        self.df = df[df["data_type"] == source_type].copy()
        self.tokenizer = tokenizer

        # lists for successful selfies tokenization and their indices
        self.tokenized_data = []
        valid_indices = []

        # pre-tokenize data so we don't have to do this on the fly during training
        # convert to tensor scalar to work with pytorch
        for idx, row in self.df.iterrows():
            sel = row["selfies"]
            ids = self.tokenizer.selfies_to_ids(sel)
            self.tokenized_data.append(torch.tensor(ids, dtype=torch.long))
            valid_indices.append(idx)

        self.df = self.df.loc[valid_indices].reset_index(drop=True)

        self.ab_map = stoi_dicts[0]
        self.pay_map = stoi_dicts[1]
        self.target_map = stoi_dicts[2]

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # pull from our pre-tokenized data list
        smile_tensor = self.tokenized_data[idx]

        ab_id = self.ab_map[row["antibody_name"]]
        pay_id = self.pay_map[row["payload_name"]]
        target_id = self.target_map[row["indication"]]

        return (
            smile_tensor,
            torch.tensor([ab_id], dtype=torch.long),
            torch.tensor([pay_id], dtype=torch.long),
            torch.tensor([target_id], dtype=torch.long),
        )


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

        # out[:, :, 0] = -1e9  # mask pad token

        # start_idx = out.size(-1) - 1
        # out[:, :, start_idx] = -1e9  # mask the start token

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


class ADCClassifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=5):
        super(ADCClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class LogitBiasModule(nn.Module):
    def __init__(
        self,
        cleavable_ids,
        tail_ids,
        bias_strength=5.0,
        min_steps_before_cleav=3,
        min_steps_after_cleav=3,
        verbose=False,
    ):
        super().__init__()
        # Ensure we have long tensors for indexing
        self.register_buffer(
            "cleavable_idx", torch.tensor(list(cleavable_ids), dtype=torch.long)
        )
        self.register_buffer("tail_idx", torch.tensor(list(tail_ids), dtype=torch.long))
        self.bias = bias_strength
        self.verbose = verbose
        self.min_steps_before_cleav = min_steps_before_cleav
        self.min_steps_after_cleav = min_steps_after_cleav

    def forward(self, logits, motif_status, step_count):
        # logits shape: (Batch, 1, Vocab)
        # motif_status shape: (Batch, 3) [Head, Cleav, Tail]

        # 1. Create a zero-filled bias mask of shape (Batch, Vocab)
        # We match the device and dtype of the input logits
        bias_mask = torch.zeros(
            logits.size(0), logits.size(2), device=logits.device, dtype=logits.dtype
        )

        # 2. Determine boolean conditions
        has_head = motif_status[:, 0]
        has_cleav = motif_status[:, 1]
        has_tail = motif_status[:, 2]

        # Logic: If Head exists but No Cleavable -> Boost Cleavable
        wait_for_cleav = step_count.squeeze() > self.min_steps_before_cleav
        boost_cleav = has_head & (~has_cleav) & wait_for_cleav

        # Logic: If Cleavable exists but No Tail -> Boost Tail
        min_tail_step = self.min_steps_before_cleav + self.min_steps_after_cleav
        wait_for_tail = step_count.squeeze() > min_tail_step
        boost_tail = has_cleav & (~has_tail) & wait_for_tail

        if self.verbose and boost_cleav.any():
            rows = torch.where(boost_cleav)[0]

            # 1. Get the max logit (the token the model WANTS to pick)
            # logits is (Batch, 1, Vocab), squeeze to (Batch, Vocab)
            current_logits = logits[rows, 0, :]
            max_val, max_idx = torch.max(current_logits, dim=-1)

            # 2. Get the logit of the token we represent (just picking the first cleavable_id to test)
            target_idx = self.cleavable_idx[0]
            target_val = current_logits[:, target_idx]

        # 3. Apply Bias to Mask
        # "rows" gives us the batch indices that need boosting
        if boost_cleav.any():
            rows = torch.where(boost_cleav)[0]
            # Advanced Indexing: [Rows, Column_Indices] = Bias
            # We use broadcasting to apply bias to all cleavable_ids for the selected rows
            bias_mask[rows[:, None], self.cleavable_idx] = self.bias

            if self.verbose:
                print(f"DEBUG: Boosting Cleavables for {len(rows)} sequences")

        if boost_tail.any():
            rows = torch.where(boost_tail)[0]
            bias_mask[rows[:, None], self.tail_idx] = self.bias

            if self.verbose:
                print(f"DEBUG: Boosting Tails for {len(rows)} sequences")

        # 4. Add Mask to Original Logits
        # logits is (Batch, 1, Vocab), bias_mask is (Batch, Vocab)
        # We unsqueeze mask to (Batch, 1, Vocab) to match dimensions
        return logits + bias_mask.unsqueeze(1)


class SmilesGeneratorEnv(EnvBase):
    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.envs.EnvBase.html
    # https://docs.pytorch.org/rl/0.8/reference/generated/torchrl.data.CompositeSpec.html
    # https://docs.pytorch.org/tutorials/advanced/pendulum.html

    # documentation describes the standard boilerplate for implementing torchRL

    # EnvBase abstract environment base class defines standard environment interface for RL
    # this includes the reset and step methods that allow RL to proceed

    def __init__(
        self,
        # reward_model,
        vocab_size,
        max_length,
        num_layers,
        hidden_dim,
        tokenizer,
        generator_model,
        tags_to_smiles,
        reward_motifs=None,
        device="cuda",
        seed=None,
        condition_count=5,
        canonical_tuples=None,
    ):
        super().__init__(device=device, batch_size=[])
        # self.rewarder = reward_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.token = tokenizer
        self.condition_count = condition_count
        self.current_step = 0
        self.episode_count = 0
        self.generated_sequence = []
        self.gen_model = generator_model
        self.tags_to_smiles = tags_to_smiles
        self.head_ids = set()
        self.cleavable_ids = set()
        self.tail_ids = set()
        self.head_tags = []

        self.head_ids = set()
        self.cleavable_ids = set()
        self.tail_ids = set()
        self.head_tags = []

        self.classifiers = {}
        self.label_maps = {}

        self.canonical_tuples = torch.tensor(
            canonical_tuples, device=self.device, dtype=torch.long
        )

        for name in ["antibody", "payload", "target"]:
            # Adjust path if your folder is different
            with open(f"classifiers/map_{name}.json", "r") as f:
                raw_map = json.load(f)
                self.label_maps[name] = {int(k): v for k, v in raw_map.items()}

            num_classes = len(self.label_maps[name])
            model = ADCClassifier(num_classes=num_classes).to(self.device)

            model.load_state_dict(torch.load(f"classifiers/mlp_{name}.pt"))
            model.eval()
            self.classifiers[name] = model

        self.bounds = {
            "TPSA": (0, 140),
            "LogP": (-2, 5),
            "SA": (1, 10),
            "QED": (0, 1),
            "CSP3": (0, 1),
        }

        if hasattr(tokenizer, "motif_map"):
            for tag, name in tokenizer.motif_map.items():
                if tag in tokenizer.smiles_map:
                    idx = tokenizer.smiles_map[tag]

                    # Populate the sets based on the category name
                    if "HEAD" in name:
                        self.head_ids.add(idx)
                        self.head_tags.append(tag)
                    elif "CLEAVABLE" in name:
                        self.cleavable_ids.add(idx)
                    elif "TAIL" in name:
                        self.tail_ids.add(idx)

        # Verify we actually found them
        if len(self.head_ids) == 0:
            print("WARNING: No Head IDs found in Environment! Bias will fail.")

        # compile the reward patterns from our rewards motif dict
        # this is what we will use to produce structural rewards for good linkers
        self.reward_patterns = {"heads": [], "cleavable": [], "tails": []}
        if reward_motifs:
            for category in self.reward_patterns.keys():
                for smi in reward_motifs[category]:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        self.reward_patterns[category].append(mol)

        # get all of our head tags from the adc motifs dict
        # these have our dummy atom labels
        # during prompting we will periodically start with a valid head
        self.head_tags = []
        if hasattr(tokenizer, "motif_map"):
            for tag, name in tokenizer.motif_map.items():
                if tag in tokenizer.smiles_map and "HEAD" in name:
                    self.head_tags.append(tag)

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
                # keeps track of if we have generated the necessary motifs
                "motif_status": DiscreteTensorSpec(
                    n=2, shape=(3,), dtype=torch.bool, device=self.device
                ),
                "step_count": UnboundedContinuousTensorSpec(
                    shape=(1,), dtype=torch.long, device=self.device
                ),
            }
        )

        # honestly no clue why we need this, got an error message saying it was required for torchRL
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

        # randomize the ab, pay and target conditions on reset
        idx = torch.randint(0, len(self.canonical_tuples), (1,), device=self.device)
        selected_combo = self.canonical_tuples[idx]
        ab_idx = selected_combo[:, 0]
        pay_idx = selected_combo[:, 1]
        tar_idx = selected_combo[:, 2]

        # determine whether we will prompt a head token
        use_prompt = torch.rand(1).item() < 1.0

        # Feed START token
        start_input = torch.tensor([[self.token.start_token]], device=self.device)

        with torch.no_grad():
            _, h, c = self.gen_model(
                sequence=start_input,
                antibody=ab_idx.unsqueeze(0),  # Model expects batch dim
                payload=pay_idx.unsqueeze(0),
                target=tar_idx.unsqueeze(0),
            )
            # Remove batch dim from hidden states for TensorDict
            # Model returns (Layers, Batch, Dim), we need (Layers, Dim) for spec
            h = h.squeeze(0)
            c = c.squeeze(0)

        if use_prompt and len(self.head_tags) > 0:
            prompt_tag = random.choice(self.head_tags)
            prompt_id = self.token.smiles_map[prompt_tag]

            # Start sequence with the prompt
            self.generated_sequence = [prompt_id]
            self.current_step = 1

            start_token = torch.tensor([prompt_id], device=self.device)
            hidden_state = h
            cell_state = c
        else:
            # Standard start without the prompt injection
            start_token = torch.tensor(
                [self.token.start_token], device=self.device, dtype=torch.long
            )
            hidden_state = torch.zeros(
                self.num_layers, self.hidden_dim, device=self.device
            )
            cell_state = torch.zeros(
                self.num_layers, self.hidden_dim, device=self.device
            )

        # get the motif status for logit biasing
        token_val = start_token.item()
        status = torch.tensor(
            [
                token_val in self.head_ids,
                token_val in self.cleavable_ids,
                token_val in self.tail_ids,
            ],
            device=self.device,
            dtype=torch.bool,
        )

        return TensorDict(
            {
                "observation": start_token,
                "hidden": hidden_state,
                "cell": cell_state,
                "antibody": ab_idx,
                "motif_status": status,
                "step_count": torch.tensor(
                    [self.current_step], device=self.device, dtype=torch.long
                ),
                "payload": pay_idx,
                "target": tar_idx,
                "done": torch.tensor([False], device=self.device),
                "terminated": torch.tensor([False], device=self.device),
            },
            batch_size=[],
        )

    def calculate_chemical_reward(self, mol, mol_depro):
        if mol is None:
            return 0.0

        try:
            tpsa = Descriptors.TPSA(mol_depro)
            logp = Descriptors.MolLogP(mol_depro)
            qed = QED.qed(mol_depro)
            csp3 = Descriptors.FractionCSP3(mol_depro)
            sa = sascorer.calculateScore(mol)

            if sa is None:
                sa = 10.0  # Assume worst score (hard to synthesize) if failed
            if tpsa is None:
                tpsa = 0.0
            if logp is None:
                logp = 0.0
            if qed is None:
                qed = 0.0
            if csp3 is None:
                csp3 = 0.0

        except Exception as e:
            return 0.0

        reward_tpsa = 1.0 - (abs(tpsa - 100) / 100)
        reward_tpsa = max(0.0, reward_tpsa)

        # targeting specific range
        reward_logp = 1.0 if (0 <= logp <= 5) else 0.5

        # already 0 to 1, higher is better
        reward_qed = qed
        reward_csp3 = csp3

        # convert 1-10 scale to 0-1
        reward_sa = (10 - sa) / 9.0
        reward_sa = max(0.0, min(1.0, reward_sa))

        total = (
            (reward_tpsa * 0.2)
            + (reward_logp * 0.2)
            + (reward_qed * 0.2)
            + (reward_csp3 * 0.2)
            + (reward_sa * 0.2)
        )

        # check for excessive peroxides and apply penalty
        peroxide_matches = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#8]-[#8]")))
        if peroxide_matches > 0:
            # Exponential penalty: -2.0 for 1 match, -8.0 for 2 matches
            # This ensures O-O-O (2 matches) is punished much harder than O-O
            penalty = (peroxide_matches**2) * 2.0
            total -= penalty

        return total

    def calculate_structural_reward(self, mol):
        reward = 0.0

        has_head = False
        has_cleavable = False
        has_tail = False

        for pattern in self.reward_patterns["heads"]:
            if mol.HasSubstructMatch(pattern):
                has_head = True
                break

        for pattern in self.reward_patterns["cleavable"]:
            if mol.HasSubstructMatch(pattern):
                has_cleavable = True
                break

        for pattern in self.reward_patterns["tails"]:
            if mol.HasSubstructMatch(pattern):
                has_tail = True
                break

        # Do you have at least one of these?
        if has_head:
            reward += 1.0
        if has_cleavable:
            reward += 1.0
        if has_tail:
            reward += 1.0

        # Did we get a combination?
        if has_head and has_cleavable:
            reward += 2.0
        if has_tail and has_cleavable:
            reward += 2.0

        if has_head and has_cleavable and has_tail:
            reward += 3.0

        return reward

    def calculate_conditional_reward(self, mol, ab_id, pay_id, tar_id):
        if mol is None:
            return 0.0

        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = mfgen.GetFingerprintAsNumPy(mol)

        fp_tensor = torch.tensor(
            np.array(fp), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        total_prob = 0.0
        active_models = 0

        targets = {"antibody": ab_id, "payload": pay_id, "target": tar_id}

        with torch.no_grad():
            for name, target_val in targets.items():
                if name not in self.classifiers:
                    continue

                if target_val in self.label_maps[name]:
                    network_idx = self.label_maps[name][target_val]

                    logits = self.classifiers[name](fp_tensor)

                    probs = torch.softmax(logits, dim=1)

                    total_prob += probs[0, network_idx].item()
                    active_models += 1

        return (total_prob / active_models) if active_models > 0 else 0.0

    def _step(self, tensordict):
        action_token = tensordict["action"]
        self.generated_sequence.append(action_token.item())
        self.current_step += 1

        # progressive length penalty
        step_penalty = 0.0
        if self.current_step > 30:
            step_penalty = 0.1
        if self.current_step > 50:
            step_penalty = 0.5

        # if we are getting lots of repeats terminate generation
        # apply small penalty
        repetition_penalty = 0.0
        force_terminate = False
        if len(self.generated_sequence) >= 15:
            last_five = self.generated_sequence[-10:]
            if len(set(last_five)) == 1:
                repetition_penalty = 0.25
                force_terminate = True

        token_id = action_token.item()
        is_tail = token_id in self.tail_ids
        is_end = token_id == self.token.end_token
        is_max = self.current_step >= self.max_length
        done = is_end | is_max | force_terminate | is_tail

        ab_idx = tensordict["antibody"]
        pay_idx = tensordict["payload"]
        tar_idx = tensordict["target"]
        next_hidden = tensordict["hidden"]
        next_cell = tensordict["cell"]

        prev_status = tensordict["motif_status"]
        token_val = action_token.item()

        new_head = prev_status[0] | (token_val in self.head_ids)
        new_cleav = prev_status[1] | (token_val in self.cleavable_ids)
        new_tail = prev_status[2] | (token_val in self.tail_ids)

        current_status = torch.tensor(
            [new_head, new_cleav, new_tail], device=self.device, dtype=torch.bool
        )

        # reward = -(repetition_penalty + step_penalty)
        reward = -repetition_penalty + step_penalty

        if done:
            self.episode_count += 1

            # convert the token ids to selfies string
            selfies_string = self.token.ids_to_selfies(self.generated_sequence)

            # convert selfies to smiles for property rewards
            try:
                smiles_string = sf.decoder(selfies_string)
            except:
                smiles_string = ""  # handle failed decode just in case

            smiles_string = selfie_sanitizer(smiles_string, self.tags_to_smiles)

            # generate a deprotected version of the molecule for certain scores
            depro_smiles = smiles_string
            for protected, deprotected in tokenizer.depro_map.items():
                if protected in smiles_string:
                    depro_smiles = smiles_string.replace(protected, deprotected)

            mol = Chem.MolFromSmiles(smiles_string)
            mol_depro = Chem.MolFromSmiles(depro_smiles)

            if mol is None:
                total_reward = -0.5
            else:
                # baseline reward for valid smiles is +1.0
                total_reward = 1.0
                struct_bonus = self.calculate_structural_reward(mol)
                total_reward += struct_bonus * 0.5  # reward for adc motifs

                chemical_bonus = self.calculate_chemical_reward(mol, mol_depro)
                total_reward += chemical_bonus * 0.75

                # if the final valid molecule is huge cut the score in half
                if mol.GetNumAtoms() > 60:
                    total_reward = total_reward * 0.5

                ab_id = ab_idx.item()
                pay_id = pay_idx.item()
                tar_id = tar_idx.item()

                cond_reward = self.calculate_conditional_reward(
                    mol, ab_id, pay_id, tar_id
                )
                total_reward += cond_reward * 1.5

            reward += total_reward

            if (self.episode_count % 100 == 0) and (mol != None):
                struct_reward_info = self.calculate_structural_reward(mol)
                chemical_reward_info = self.calculate_chemical_reward(mol, mol_depro)
                conditional_reward_info = self.calculate_conditional_reward(
                    mol, ab_id, pay_id, tar_id
                )
                print()
                print(f"--- Ep {self.episode_count} ---")
                print(f"SMILES: {smiles_string}")
                print(f"Depro SMILES: {depro_smiles}")
                print(
                    f"Reward: {reward:.2f} (Struct: {struct_reward_info} | Chem: {chemical_reward_info:.4f} | Cond: {conditional_reward_info:.4f})"
                )
                print("-----------------------------")
                print()

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
                "motif_status": current_status,
                "step_count": torch.tensor(
                    [self.current_step], device=self.device, dtype=torch.long
                ),
            },
            batch_size=[],
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


def train_LSTM_gen(
    model,
    dataset,
    tokenizer,
    learning_rate=0.001,
    num_epochs=1000,
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

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (smile, ab, pay, target) in enumerate(loader):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.3f}")

    return model


def LSTM_model_RL(
    gen_model,
    critic_model,
    tokenizer,
    reward_motifs,
    tags_to_smiles,
    learning_rate=0.00001,
    temperature=1.0,
    total_frames=500_000,
    num_envs=64,
    frame_steps=150,
    clip_epsilon=0.025,
    kl=0.05,
    bias_strength=5.0,
):
    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html
    # creating a dict for our LSTM generative model that the actor can then understand for RL
    # similar tensor dicts are used for passing around data for RL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_model.train()

    gen_model.to(device)
    critic_model.to(device)

    # load in cleavables and tails from tokenizer
    # this is necessary for the logit biasing
    cleav_ids_set = set()
    tail_ids_set = set()

    # generate cleavable and tail id set
    for tag, name in tokenizer.motif_map.items():
        if tag in tokenizer.smiles_map:
            idx = tokenizer.smiles_map[tag]
            # Note: Tag names are uppercase "CLEAVABLE_0", "TAILS_0"
            if "CLEAVABLE" in name:
                cleav_ids_set.add(idx)
            elif "TAIL" in name:
                tail_ids_set.add(idx)

    # actor is our generative model, critic is our scoring model
    # loss module uses the proximal policy optimization (PPO) from torchRL to calculate loss
    lstm_module = TensorDictModule(
        module=gen_model,
        in_keys=["observation", "antibody", "payload", "target", "hidden", "cell"],
        out_keys=["logits", "hidden", "cell"],
    )

    # initializing out bias module for introducing logit bias
    bias_net = LogitBiasModule(cleav_ids_set, tail_ids_set, bias_strength=bias_strength)

    bias_module = TensorDictModule(
        bias_net,
        in_keys=["logits", "motif_status", "step_count"],
        out_keys=["logits"],
    )

    # temp module controls temp scaler for generation, default of 1.0 is too high most often
    temp_module = TensorDictModule(
        lambda x: x / temperature, in_keys=["logits"], out_keys=["logits"]
    )

    policy_module = TensorDictSequential(
        lstm_module, bias_module, temp_module
    )  # adds temp scaler to lstm logits

    # keys are strict here, need to use predefined key names as described by ProbActor class
    actor_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    critic_module = ValueOperator(
        module=critic_model,
        in_keys=["hidden"],
    )

    valid_combos_str = [
        ("Cetuximab", "Puromycin", "Triple negative breast cancer"),
        ("MCLL0517A", "PBD dimer", "Acute myeloid leukaemia"),
        ("Patritumab", "ADC-C1 payload", "Breast cancer"),
        ("Lorvotuzumab", "Rachelmycin", "Small cell lung cancer"),
    ]

    canonical_list = []
    for ab, pay, tar in valid_combos_str:
        try:
            tup = (
                tokenizer.ab_stoi[ab],
                tokenizer.pay_stoi[pay],
                tokenizer.tar_stoi[tar],
            )
            canonical_list.append(tup)
        except KeyError as e:
            print(f"KeyError: {e}")

    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.collectors.SyncDataCollector.html
    # creates our environment that handles the reset and step in RL loop
    environment_maker = lambda: SmilesGeneratorEnv(
        vocab_size=tokenizer.vocab_size,
        max_length=100,
        num_layers=gen_model.layer_dim,
        hidden_dim=gen_model.hidden_dim,
        tokenizer=tokenizer,
        generator_model=gen_model,
        reward_motifs=reward_motifs,
        tags_to_smiles=tags_to_smiles,
        canonical_tuples=canonical_list,
    )

    env = SerialEnv(num_envs, environment_maker)

    # anchor model to prevent moving too far from starting model
    # this works with kl_div, it is our baseline that we can fall back on
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
    ref_model.eval()  # ensure grads don't update for reference model

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

    training_history = []

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

                ###### BUG ################
                # indentation error here need to fix, KL not being applied
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
            # prevents updating RL model if change is too great
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
            actor_log_probs = F.log_softmax(actor_logits, dim=-1)
            ref_probs = F.softmax(ref_logits, dim=-1)
            kl_div = F.kl_div(actor_log_probs, ref_probs, reduction="batchmean")
            total_loss = actor_loss + critic_loss + (kl_coeff * kl_div)

            if i % 10 == 0:
                print(f"KL Div: {kl_div.item():.4f}")

            optimizer.zero_grad()
            total_loss.backward()
            # clip the gradients for stability
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=0.5)
            optimizer.step()

        # print reward to keep track of training progress
        # avg_reward = batch["next", "reward"].mean().item()
        rewards = batch["next", "reward"].sum(dim=1)
        avg_reward = rewards.mean().item()

        # 2. Completed Episode Average (The "Final Exam Score")
        # We look for steps where done=True. These contain the final validation reward.
        done_mask = batch["next", "done"].squeeze(-1)

        if done_mask.any():
            # Extract rewards only from the steps where the episode finished
            final_rewards = batch["next", "reward"][done_mask]
            avg_final_reward = final_rewards.mean().item()
            valid_count = (final_rewards > 0).float().sum()
            valid_pct = (valid_count / len(final_rewards)) * 100

            print(
                f"Batch {i} || Avg Reward: {avg_final_reward:.4f} | Valid: {valid_pct:.1f}% | KL: {kl_div.item():.4f}"
            )

            training_history.append(
                {
                    "batch": i,
                    "avg_reward": avg_final_reward,
                    "valid_pct": valid_pct.item(),
                    "kl_div": kl_div.item(),
                    "total_loss": total_loss.item(),
                }
            )

        else:
            print(
                f"Batch {i} || Loss: {total_loss.item():.4f} | Avg Batch Reward: {avg_reward:.4f} (No episodes finished)"
            )

    print("RL training complete.")

    if training_history:
        df_log = pd.DataFrame(training_history)
        df_log.to_csv("rl_training_log.csv", index=False)
        print(f"Training log saved")

    return gen_model


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


def build_tag_to_smiles_map(motif_dict):
    tag_to_smiles = {}
    counter = 1

    for category in ["cleavable", "heads", "tails"]:
        for smi in motif_dict[category]:
            tag = f"[{counter}Po]"
            tag_to_smiles[tag] = smi
            counter += 1

    return tag_to_smiles


def setup_bias(tokenizer, bias_strength=15.0):
    head_ids = set()
    cleavable_ids = set()
    tail_ids = set()

    # get all of our motifs
    for tag, name in tokenizer.motif_map.items():
        if tag in tokenizer.smiles_map:
            idx = tokenizer.smiles_map[tag]

            if "HEAD" in name:
                head_ids.add(idx)
            elif "CLEAVABLE" in name:
                cleavable_ids.add(idx)
            elif "TAIL" in name:
                tail_ids.add(idx)

    # instantiate the bias module
    bias_module = LogitBiasModule(
        cleavable_ids=cleavable_ids,
        tail_ids=tail_ids,
        bias_strength=bias_strength,
        min_steps_before_cleav=3,
        min_steps_after_cleav=3,
        verbose=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bias_module.to(device)

    return bias_module, head_ids, cleavable_ids, tail_ids


def generate_biased_sequence(
    gen_model,
    bias_module,
    tokenizer,
    id_sets,  # (head_ids, cleav_ids, tail_ids) from setup function
    ab_idx,
    pay_idx,
    tar_idx,
    temperature=1.0,
    max_len=100,
    use_prompt=True,
):

    # this is basically the same loop for applying bias as in our RL loop
    device = next(gen_model.parameters()).device
    head_ids, cleav_ids, tail_ids = id_sets

    # initialize the state tracking for bias module
    motif_status = torch.tensor(
        [[False, False, False]], device=device, dtype=torch.bool
    )
    step_count = torch.tensor([0], device=device, dtype=torch.long)

    # set the conditional seeds for generation
    current_token = torch.tensor([[tokenizer.start_token]], device=device)
    ab_t = torch.tensor([[ab_idx]], device=device)
    pay_t = torch.tensor([[pay_idx]], device=device)
    tar_t = torch.tensor([[tar_idx]], device=device)

    hidden, cell = None, None
    sequence = []

    gen_model.eval()

    start_input = torch.tensor([[tokenizer.start_token]], device=device)

    with torch.no_grad():
        # Prime the model with START input
        _, hidden, cell = gen_model(
            start_input, ab_t, pay_t, tar_t, hidden_state=hidden, cell_state=cell
        )

        if use_prompt and len(head_ids) > 0:
            # pick random head to start
            prompt_id = random.choice(list(head_ids))

            # set head as the current token
            current_token = torch.tensor([[prompt_id]], device=device)

            # update the trackers for generation
            sequence.append(prompt_id)
            motif_status[0, 0] = True  # We now have a head
            step_count += 1
        else:
            # else if no prompt scenario
            current_token = start_input

    hidden_states_list = []
    with torch.no_grad():
        for i in range(max_len):

            # get the raw output
            output, hidden, cell = gen_model(
                current_token, ab_t, pay_t, tar_t, hidden_state=hidden, cell_state=cell
            )

            current_hidden = hidden[0, -1, :]
            hidden_states_list.append(current_hidden.cpu())

            # apply the bias strength modifier
            if bias_module is not None:
                output = bias_module(output, motif_status, step_count)

            # grab the next token
            logits = output[:, -1, :]

            # mask the pad and start tokens like usual
            logits[0, tokenizer.start_token] = -1e9
            logits[0, tokenizer.pad_token] = -1e9

            # temp and sample application
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == tokenizer.end_token:
                break

            # update the internal state of the module
            sequence.append(next_token)
            current_token = torch.tensor([[next_token]], device=device)
            step_count += 1

            # check if the next picked token is one of our desired motifs
            is_head = next_token in head_ids
            is_cleav = next_token in cleav_ids
            is_tail = next_token in tail_ids

            # update motif status tracker if we found them
            if is_head:
                motif_status[0, 0] = True
            if is_cleav:
                motif_status[0, 1] = True
            if is_tail:
                motif_status[0, 2] = True

            # stop generating once we reach a tail
            if is_tail:
                break

    # did we get all three motifs
    success = motif_status[0].all().item()
    # last_hidden = hidden[:, -1, :].cpu().numpy()

    if len(hidden_states_list) > 0:
        all_states = torch.stack(hidden_states_list)
        mean_pooled = torch.mean(all_states, dim=0).numpy()
    else:
        # default if failed sequence
        mean_pooled = np.zeros(hidden[0, -1, :])

    return sequence, success, mean_pooled


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


def generation_loop(
    model,
    tokenizer,
    tag_to_smiles,
    ab,
    pay,
    tar,
    bias_strength=15.0,
    temp=0.8,
    target_count=10,
):

    # initialize our bias module
    bias_net, h_ids, c_ids, t_ids = setup_bias(tokenizer, bias_strength=bias_strength)

    # start generating
    print("Generating Valid Linkers with Bias...")

    valid_linkers = []
    attempts = 0
    max_attempts = target_count * 1000

    while len(valid_linkers) < target_count and attempts < max_attempts:
        attempts += 1
        print(f"Attemps: {attempts}/{max_attempts}", end="\r")
        seq, success, _ = generate_biased_sequence(
            gen_model=model,
            bias_module=bias_net,
            tokenizer=tokenizer,
            id_sets=(h_ids, c_ids, t_ids),
            ab_idx=ab,
            pay_idx=pay,
            tar_idx=tar,
            temperature=temp,
        )

        if success:
            tagged_selfies = tokenizer.ids_to_selfies(seq)
            tagged_smi = sf.decoder(tagged_selfies)

            real_smi = selfie_sanitizer(tagged_smi, tag_to_smiles)

            mol = Chem.MolFromSmiles(real_smi)

            if mol is None:
                # Debugging block for invalid molecules
                print(f"\n[DEBUG] FAILED SMILES: {real_smi}")
                try:
                    mol_unsanitized = Chem.MolFromSmiles(real_smi, sanitize=False)
                    if mol_unsanitized:
                        print("  -> Structure exists but failed sanitization")
                        problems = Chem.DetectChemistryProblems(mol_unsanitized)
                        for p in problems:
                            print(f"  -> Problem: {p.GetType()}")
                    else:
                        print("  -> Grammar error")
                except:
                    pass
                continue  # Skip to next attempt
            # -----------------------------------

            if Chem.MolFromSmiles(real_smi) is not None:
                # Basic deduplication
                if real_smi not in valid_linkers:
                    print(f"Gen {len(valid_linkers)+1}:")
                    # valid_linkers.append(real_smi)

                try:
                    mol = Chem.MolFromSmiles(real_smi)
                    tpsa = Descriptors.TPSA(mol)
                    logp = Descriptors.MolLogP(mol)
                    qed = QED.qed(mol)
                    csp3 = Descriptors.FractionCSP3(mol)
                    sa = sascorer.calculateScore(mol)

                    results_entry = {
                        "SMILES": real_smi,
                        "SA": sa,
                        "TPSA": tpsa,
                        "QED": qed,
                        "LogP": logp,
                        "CSP3": csp3,
                    }

                    valid_linkers.append(results_entry)

                except Exception as e:
                    print("chemical calculation error...")
                    pass

            else:
                pass

    if attempts >= max_attempts:
        print("Warning: Hit max attempts limit.")
    return valid_linkers


def performance_metrics(
    model,
    real_smiles_series,
    tokenizer,
    tag_to_smiles,
    bias_strength=15.0,
    temp=0.8,
    target_count=10,
):

    # initialize our bias module
    bias_net, h_ids, c_ids, t_ids = setup_bias(tokenizer, bias_strength=bias_strength)

    # start generating
    print("Generating Valid Linkers with Bias...")

    unique_linkers = set()
    valid_smiles = 0
    novel_linkers = 0
    valid_linkers = 0
    sa_total = 0
    sa_list = []
    qed_total = 0
    qed_list = []
    logp_total = 0
    logp_list = []
    tpsa_total = 0
    tpsa_list = []
    csp3_total = 0
    csp3_list = []

    unique_real_smiles = set()
    for s in real_smiles_series:
        m = Chem.MolFromSmiles(s)
        if m:
            unique_real_smiles.add(Chem.MolToSmiles(m, canonical=True))

    attempts = 0

    while valid_linkers < target_count:
        attempts += 1
        ab = random.randint(1, 4)
        pay = random.randint(1, 4)
        tar = random.randint(1, 4)

        print(
            f"attempts: {attempts+1}, valid linkers: {valid_linkers}, valid smiles: {valid_smiles}",
            end="\r",
        )

        seq, success, _ = generate_biased_sequence(
            gen_model=model,  # Use your RL trained model
            bias_module=bias_net,
            tokenizer=tokenizer,
            id_sets=(h_ids, c_ids, t_ids),
            ab_idx=ab,
            pay_idx=pay,
            tar_idx=tar,  # Change conditions as needed
            temperature=temp,  # Lower temp slightly for stability
        )

        if success:
            tagged_selfies = tokenizer.ids_to_selfies(seq)

            try:
                tagged_smi = sf.decoder(tagged_selfies)
                real_smi = selfie_sanitizer(tagged_smi, tag_to_smiles)
            except:
                real_smi = ":("

            mol = Chem.MolFromSmiles(real_smi)

            if mol is not None:

                canon_smi = Chem.MolToSmiles(mol, canonical=True)
                valid_smiles += 1
                valid_linkers += 1
                unique_linkers.add(canon_smi)

                if canon_smi not in unique_real_smiles:
                    novel_linkers += 1

                try:
                    depro_smiles = real_smi
                    for protected, deprotected in tokenizer.depro_map.items():
                        if protected in real_smi:
                            depro_smiles = real_smi.replace(protected, deprotected)
                    mol = Chem.MolFromSmiles(real_smi)
                    depro_mol = Chem.MolFromSmiles(depro_smiles)

                    logp = Descriptors.MolLogP(depro_mol)
                    qed = QED.qed(depro_mol)
                    tpsa = Descriptors.TPSA(depro_mol)
                    csp3 = Descriptors.FractionCSP3(depro_mol)

                    sa = sascorer.calculateScore(mol)

                    sa_list.append(sa)
                    qed_list.append(qed)
                    logp_list.append(logp)
                    tpsa_list.append(tpsa)
                    csp3_list.append(csp3)

                    sa_total += sa
                    qed_total += qed
                    logp_total += logp
                    tpsa_total += tpsa
                    csp3_total += csp3

                except Exception as e:
                    print("chemical calculation error...")
                    pass

            else:
                # print("Failed validity check")
                pass
        else:
            tagged_selfies = tokenizer.ids_to_selfies(seq)

            try:
                tagged_smi = sf.decoder(tagged_selfies)
                real_smi = selfie_sanitizer(tagged_smi, tag_to_smiles)
            except:
                real_smi = ":("

            if Chem.MolFromSmiles(real_smi) is not None:
                # Basic deduplication
                valid_smiles += 1

    mols = [Chem.MolFromSmiles(s) for s in unique_linkers]
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [mfgen.GetFingerprint(m) for m in mols if m is not None]

    avg_tanimoto = 0.0
    if len(fps) > 1:
        sims = []
        for i in range(len(fps)):
            s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
            sims.extend(s)

        avg_tanimoto = np.mean(sims) if sims else 0.0

    results = {
        "validity": valid_smiles / attempts,
        "novelty": novel_linkers / valid_linkers,
        "unique": len(unique_linkers) / valid_linkers,
        "tanimoto": avg_tanimoto,
        "sa_avg": sa_total / valid_linkers,
        "qed_avg": qed_total / valid_linkers,
        "logp_avg": logp_total / valid_linkers,
        "tpsa_avg": tpsa_total / valid_linkers,
        "csp3_avg": csp3_total / valid_linkers,
        "sa_std": np.std(sa_list),
        "qed_std": np.std(qed_list),
        "logp_std": np.std(logp_list),
        "tpsa_std": np.std(tpsa_list),
        "csp3_std": np.std(csp3_list),
    }

    return results


def umap_training_data(
    model,
    dataset,
    tokenizer,
    itos_dicts,
    num_samples=2000,
    color_by="antibody",  # Options: "antibody", "payload", "indication"
):

    print(f"Producing UMAP for {color_by}")
    model.eval()
    device = next(model.parameters()).device

    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=tokenizer.collate_smiles
    )

    vectors = []
    labels = []

    map_idx = 0
    if color_by == "payload":
        map_idx = 1
    elif color_by == "indication":
        map_idx = 2

    current_map = itos_dicts[map_idx]

    count = 0
    with torch.no_grad():
        for i, (smile, ab, pay, tar) in enumerate(loader):
            if count >= num_samples:
                break

            smile = smile.to(device)
            ab = ab.to(device)
            pay = pay.to(device)
            tar = tar.to(device)

            emb_seq = model.token_embedding(smile)

            seq_len = smile.size(1)
            emb_ab = model.ab_embedding(ab).expand(-1, seq_len, -1)
            emb_pay = model.pay_embedding(pay).expand(-1, seq_len, -1)
            emb_tar = model.target_embedding(tar).expand(-1, seq_len, -1)

            lstm_input = torch.cat([emb_seq, emb_ab, emb_pay, emb_tar], dim=-1)

            out, (h, c) = model.lstm(lstm_input)

            last_hidden = out[:, -1, :].cpu().numpy()

            vectors.extend(last_hidden)

            if color_by == "antibody":
                ids = ab.cpu().numpy().flatten()
            elif color_by == "payload":
                ids = pay.cpu().numpy().flatten()
            else:
                ids = tar.cpu().numpy().flatten()

            for cid in ids:
                name = current_map.get(cid, "Unknown")
                labels.append(name)

            count += len(smile)
            print(f"Processed {count}/{num_samples} molecules...", end="\r")

    print("\nRunning UMAP dimensionality reduction... (This may take a moment)")

    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, metric="cosine", random_state=42)
    embedding = reducer.fit_transform(vectors)

    df = pd.DataFrame(embedding, columns=["x", "y"])
    df["Condition"] = labels[: len(df)]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="Condition",
        style="Condition",
        palette="tab10",
        s=100,
        alpha=0.8,
    )

    plt.title(f"Molecule Latent Space conditioned by {color_by.title()}", fontsize=18)
    plt.legend(fontsize=16.0)
    plt.tight_layout()
    plt.show()


def umap_generated_data(
    model,
    tokenizer,
    itos_dicts,
    stoi_dicts,
    bias_strength=15.0,
    temperature=0.7,
    color_by="antibody",  # Options: "antibody", "payload", "indication"
):

    print(f"Producing UMAP for ADC combos")

    canonical_configs = [
        {"ab": "Cetuximab", "pay": "Puromycin", "tar": "Triple negative breast cancer"},
        {"ab": "MCLL0517A", "pay": "PBD dimer", "tar": "Acute myeloid leukaemia"},
        {"ab": "Patritumab", "pay": "ADC-C1 payload", "tar": "Breast cancer"},
        {"ab": "Lorvotuzumab", "pay": "Rachelmycin", "tar": "Small cell lung cancer"},
    ]

    ab_stoi, pay_stoi, tar_stoi = stoi_dicts

    vectors = []
    labels = []

    # initialize our bias module
    bias_net, h_ids, c_ids, t_ids = setup_bias(tokenizer, bias_strength=bias_strength)

    target_count = 500  # generate 500 per condition
    max_attempts = target_count * 1000

    for config in canonical_configs:
        valid_linkers = 0
        attempts = 0

        ab = ab_stoi[config["ab"]]
        pay = pay_stoi[config["pay"]]
        tar = tar_stoi[config["tar"]]

        condition_label = config["ab"]

        while valid_linkers < target_count and attempts < max_attempts:
            attempts += 1
            print(f"Attemps: {attempts}/{max_attempts}", end="\r")

            _, success, last_hidden = generate_biased_sequence(
                gen_model=model,
                bias_module=bias_net,
                tokenizer=tokenizer,
                id_sets=(h_ids, c_ids, t_ids),
                ab_idx=ab,
                pay_idx=pay,
                tar_idx=tar,
                temperature=temperature,
            )

            if success:
                valid_linkers += 1
                vectors.append(last_hidden)
                labels.append(condition_label)

                print(
                    f"Combo {config}: {valid_linkers}/{target_count} molecules...",
                    end="\r",
                )

    print("\nRunning UMAP...")

    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, metric="cosine", random_state=42)
    embedding = reducer.fit_transform(vectors)

    df = pd.DataFrame(embedding, columns=["x", "y"])
    df["Condition"] = labels[: len(df)]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="Condition",
        style="Condition",
        palette="tab10",
        s=100,
        alpha=0.8,
    )

    plt.title(f"Molecule Latent Space conditioned by {color_by.title()}", fontsize=18)
    plt.legend(fontsize=16.0)
    plt.tight_layout()
    plt.show()


def compute_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=np.float32)

    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = mfgen.GetFingerprintAsNumPy(mol)
    # need float 32 for pytorch compatibility
    return np.array(fp, dtype=np.float32)


def train_classifiers(
    df,
    stoi_dicts,
    smiles_col="smiles",
    ab_col="antibody_name",
    pay_col="payload_name",
    tar_col="indication",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Training on device: {device}")

    # prepare valid data that has conditions
    valid_data = df[
        df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    ].copy()

    # create our data matrix, samples by fingerprint bits (N_samples, 2048)
    X_raw = np.stack(valid_data[smiles_col].apply(compute_fingerprint).values)

    ab_stoi, pay_stoi, tar_stoi = stoi_dicts

    # Convert strings to initial integer IDs
    y_ab_full = valid_data[ab_col].map(ab_stoi).dropna().astype(int).values
    y_pay_full = valid_data[pay_col].map(pay_stoi).dropna().astype(int).values
    y_tar_full = valid_data[tar_col].map(tar_stoi).dropna().astype(int).values

    tasks = [
        ("antibody", y_ab_full, valid_data[ab_col].map(ab_stoi).notna()),
        ("payload", y_pay_full, valid_data[pay_col].map(pay_stoi).notna()),
        ("target", y_tar_full, valid_data[tar_col].map(tar_stoi).notna()),
    ]

    trained_models = {}

    for name, y_target, mask in tasks:
        print(f"Training {name} Model")

        # Filter X to match rows
        X_curr = X_raw[mask]

        # 2. FILTER UNKNOWNS (Class 0)
        # We only want to train on real classes
        real_mask = y_target != 0
        X_curr = X_curr[real_mask]
        y_target = y_target[real_mask]

        if len(y_target) < 50:
            print(f"Skipping {name}: Not enough data.")
            continue

        # 3. CRITICAL: Create Label Mapping (Real ID -> Network Index)
        # PyTorch needs labels 0, 1, 2... but your IDs might be 1, 5, 8.
        unique_classes = sorted(np.unique(y_target))
        label_map = {int(real_id): idx for idx, real_id in enumerate(unique_classes)}

        # Transform y_target to 0..N-1
        y_mapped = np.array([label_map[y] for y in y_target])
        num_classes = len(unique_classes)

        print(f"Classes: {num_classes} | Samples: {len(y_target)}")
        print(f"ID Mapping: {label_map}")

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X_curr, y_mapped, test_size=0.2, random_state=42
        )

        # Convert to Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

        # 4. Initialize Model & Optimizer
        model = ADCClassifier(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 50  # Small dataset, converges fast
        batch_size = 32

        model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(X_train_t.size()[0])

            epoch_loss = 0.0
            for i in range(0, X_train_t.size()[0], batch_size):
                indices = permutation[i : i + batch_size]
                batch_x, batch_y = X_train_t[indices], y_train_t[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_test_t).float().mean().item()
            print(f"--> Accuracy: {acc:.4f}")

        # save the model
        torch.save(model.state_dict(), f"classifiers/mlp_{name}.pt")

        # save mapping so network knows which conditions correspond to what in environment class
        with open(f"classifiers/map_{name}.json", "w") as f:
            json.dump(label_map, f)

        trained_models[name] = model

    return trained_models


if __name__ == "__main__":

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

    # previously tried having smiles augmentation but it confused model
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

    # Train the random forest classifiers for conditional rewards
    # train_classifiers(df=combo_df, stoi_dicts=stoi_dicts)

    # # Train the generator models...
    # print()
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
    #     learning_rate=0.0001,
    #     num_epochs=30,
    # )

    # # Phase B: Repair Syntax (Mixed Data)
    # print("Phase B: Repairing Syntax (Mixed Data)...")
    # mixed_dataset = torch.utils.data.ConcatDataset([synth_dataset, easy_dataset])
    # gen_model = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=mixed_dataset,
    #     tokenizer=tokenizer,
    #     learning_rate=0.0001,
    #     num_epochs=20,
    # )

    # # Phase C: Polish on real data (Real Data, Low LR)
    # print("Phase C: Polish on real data...")
    # gen_model = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=easy_dataset,
    #     tokenizer=tokenizer,
    #     learning_rate=0.00005,
    #     num_epochs=15,
    # )
    # torch.save(gen_model.state_dict(), "models/model_gen_weights.pth")

    # # Phase D: Polish on perfect linker data
    # print("Phase D: Polish on perfect linker data...")
    # gen_model.load_state_dict(
    #     torch.load("models/model_gen_weights.pth", weights_only=True)
    # )
    # gen_model = train_LSTM_gen(
    #     model=gen_model,
    #     dataset=perfect_dataset,
    #     tokenizer=tokenizer,
    #     learning_rate=0.0001,
    #     num_epochs=20,
    # )
    # torch.save(gen_model.state_dict(), "models/model_gen_weights_perfect.pth")

    # # --- 3. REINFORCEMENT LEARNING ---
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

    # Load Generator (Final Weights)
    gen_model.load_state_dict(
        torch.load("models/model_gen_weights_perfect.pth", weights_only=True)
    )

    # need to tune these params
    RL_trained_model = LSTM_model_RL(
        gen_model,
        critic_model,
        tokenizer,
        reward_motifs=adc_motifs,
        tags_to_smiles=tag_to_smiles,
        learning_rate=0.00005,  # .00005
        temperature=0.65,  # 0.65
        total_frames=1_000_000,
        num_envs=32,
        frame_steps=150,
        clip_epsilon=0.1,  # .1
        kl=0.005,  # .2
        bias_strength=15.0,  # 15.0
    )

    torch.save(RL_trained_model.state_dict(), "models/model_RL_gen_weights_v2.pth")

    gen_model.load_state_dict(
        torch.load("models/model_RL_gen_weights_v2.pth", weights_only=True)
    )

    # gen_model.load_state_dict(
    #     torch.load("models/model_gen_weights_perfect.pth", weights_only=True)
    # )

    # ab = 2
    # pay = 2
    # tar = 2

    # linkers = generation_loop(
    #     model=gen_model,
    #     tokenizer=tokenizer,
    #     tag_to_smiles=tag_to_smiles,
    #     ab=ab,
    #     pay=pay,
    #     tar=tar,
    #     bias_strength=10.0,
    #     temp=0.5,
    #     target_count=20,
    # )

    # print("------ Linker Report -------")
    # print()
    # print(f"Antibody: {itos_dicts[0][ab]}")
    # print(f"Payload: {itos_dicts[1][pay]}")
    # print(f"Cancer Target: {itos_dicts[2][tar]}")
    # print()
    # for i, result in enumerate(linkers):
    #     print(f"------- Linker {i+1} ----------")
    #     print(f"SMILE: {result['SMILES']}")
    #     print(f"SA: {result['SA']:.4f}")
    #     print(f"TPSA: {result['TPSA']:.4f}")
    #     print(f"QED: {result['QED']:.4f}")
    #     print(f"LogP: {result['LogP']:.4f}")
    #     print(f"CSSP3: {result['CSP3']:.4f}")
    #     print("---------------------------")

    # gen_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # viz_dataset = adcDataset(
    #     df=combo_df,  # or perfect_df
    #     tokenizer=tokenizer,
    #     stoi_dicts=stoi_dicts,
    #     source_type="real",
    # )

    # umap_training_data(
    #     model=gen_model,
    #     dataset=viz_dataset,
    #     tokenizer=tokenizer,
    #     itos_dicts=itos_dicts,
    #     num_samples=2000,
    #     color_by="antibody",
    # )

    # gen_model.load_state_dict(
    #     torch.load("models/model_RL_gen_weights.pth", weights_only=True)
    # )

    # gen_model.load_state_dict(
    #     torch.load("models/model_gen_weights_perfect.pth", weights_only=True)
    # )

    # real_smiles = adc_df["smiles"]

    # results = performance_metrics(
    #     model=gen_model,
    #     real_smiles_series=real_smiles,
    #     tokenizer=tokenizer,
    #     tag_to_smiles=tag_to_smiles,
    #     bias_strength=15.0,
    #     temp=0.80,
    #     target_count=1000,
    # )

    # print(f"validity = {results["validity"]:.4f}")
    # print(f"novelty = {results["novelty"]:.4f}")
    # print(f"unique = {results["unique"]:.4f}")
    # print(f"tanimoto = {results["tanimoto"]:.4f}")
    # print(f"SA Avg = {results["sa_avg"]:.4f} \tstd = {results["sa_std"]:.4f}")
    # print(f"QED AVG = {results["qed_avg"]:.4f} \tstd = {results["qed_std"]:.4f}")
    # print(f"LogP AVG = {results["logp_avg"]:.4f} \tstd = {results["logp_std"]:.4f}")
    # print(f"TPSA AVG = {results["tpsa_avg"]:.4f} \tstd = {results["tpsa_std"]:.4f}")
    # print(f"CSP3 AVG = {results["csp3_avg"]:.4f} \tstd = {results["csp3_std"]:.4f}")

    # gen_model.load_state_dict(
    #     torch.load("models/model_RL_gen_weights.pth", weights_only=True)
    # )

    # umap_generated_data(
    #     model=gen_model,
    #     tokenizer=tokenizer,
    #     itos_dicts=itos_dicts,
    #     stoi_dicts=stoi_dicts,
    #     bias_strength=15.0,
    #     temperature=0.7,
    #     color_by="payload",  # Options: "antibody", "payload", "indication"
    # )
