# Torch modules for scoring and generative LSTMs
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# TorchRL modules for reinforcement learning
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)

# Other modules
import numpy as np
import pandas as pd
import random
import json
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator
import sascorer

import pandas as pd
import numpy as np
import os

import selfies as sf

# Disable error messages when a non-valid smiles is encountered
rdBase.DisableLog("rdApp.error")

os.environ["RL_WARNINGS"] = "False"

from LSTM_utils import selfie_sanitizer


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
        classifier_directory=None,
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

        self.classifier_directory = classifier_directory

        self.canonical_tuples = torch.tensor(
            canonical_tuples, device=self.device, dtype=torch.long
        )

        for name in ["antibody", "payload", "target"]:
            # Adjust path if your folder is different
            with open(f"{self.classifier_directory}/map_{name}.json", "r") as f:
                raw_map = json.load(f)
                self.label_maps[name] = {int(k): v for k, v in raw_map.items()}

            num_classes = len(self.label_maps[name])
            model = ADCClassifier(num_classes=num_classes).to(self.device)

            model.load_state_dict(
                torch.load(f"{self.classifier_directory}/mlp_{name}.pt")
            )
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
            for protected, deprotected in self.token.depro_map.items():
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
