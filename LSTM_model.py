import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as TF
import numpy as np
import pandas as pd
import re


df = pd.read_pickle("data/adc_data_complete_v2.pkl")

# SMILES tokenizer reference: https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html

SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|B|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
smiles_tokens = df["smiles"].str.findall(SMI_REGEX)

# smiles tokens is a series of lists for all patterns found per smiles string
# explode converts into a single series with all pattern matches
unique_tokens = smiles_tokens.explode().unique()
smiles_map = {"PAD": 0}
smiles_map_rest = {char: i + 1 for i, char in enumerate(unique_tokens)}
smiles_map.update(smiles_map_rest)
# smiles_map


embedding_dim = 16


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(
            num_embeddings=len(unique_tokens),
            embedding_dim=embedding_dim,
            padding_idx=smiles_map["PAD"],
        )
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded_seq = self.embedding(x)
        out, (hn, cn) = self.lstm(embedded_seq)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out


model = LSTMModel(input_dim=embedding_dim, hidden_dim=100, layer_dim=5, output_dim=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on device: {device}")


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def smiles_to_tokens(smile_pattern_list):
    return [smiles_map[char] for char in smile_pattern_list]


def collate_smiles(batch):
    smiles = [item[0] for item in batch]
    scores = [item[1] for item in batch]

    smiles_padded = pad_sequence(
        smiles, batch_first=True, padding_value=smiles_map["PAD"]
    )

    scores_stack = torch.stack(
        scores
    )  # scores is a list from batch, stack reshapes to a size of (batch,1)

    return smiles_padded, scores_stack


class adcDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.file = pd.read_pickle(directory)
        self.smiles = df["smiles"].str.findall(SMI_REGEX).apply(smiles_to_tokens)
        # self.smiles = torch.tensor(self.smiles)
        self.SA_score = torch.tensor(np.array((self.file["calc_SA_score"])))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = torch.tensor(self.smiles[idx])
        score = torch.tensor([self.SA_score[idx]])

        return smile, score


dataset = adcDataset("data/adc_data_complete_v2.pkl")
loader = DataLoader(
    dataset,
    batch_size=100,
    shuffle=True,
    num_workers=2,  # more processes for cpu to load data for gpu
    pin_memory=True,  # speeds up cpu-gpu transfer
    persistent_workers=True,  # keeps the same workers between epochs
    collate_fn=collate_smiles,
)


num_epochs = 20

for epoch in range(num_epochs):
    for i, (smile, score) in enumerate(loader):
        print(f"Batch {i+1}/25", end="\r")
        smile, label = smile.to(device), score.to(device)

        optimizer.zero_grad()

        output = model(smile)  # forward
        loss = criterion(output.float(), label.float())  # calculate cross entropy loss
        loss.backward()  # back propagation
        optimizer.step()

    print()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.3f}")
