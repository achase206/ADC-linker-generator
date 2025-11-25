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
from tensordict.nn import TensorDictModule

# Other modules
import numpy as np
import pandas as pd
import random
import re
import math
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit import rdBase

# Disable error messages when a non-valid smiles is encountered
rdBase.DisableLog("rdApp.error")


class Tokenizer:
    def __init__(self, directory):
        df = pd.read_pickle(directory)
        self.smiles_map = self.smiles_tokenizer(df)
        self.inv_smiles_map = self.smiles_detokenizer()

    def smiles_tokenizer(self, df):
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
        smiles_patterns = [self.inv_smiles_map[token] for token in smile_token_list]
        return "".join(smiles_patterns)

    def collate_smiles(self, batch):
        # this adds the padding, short smiles will get padded to length of longest smile in batch
        smiles = [item[0] for item in batch]
        scores = [item[1] for item in batch]

        smiles_padded = pad_sequence(
            smiles, batch_first=True, padding_value=self.smiles_map["PAD"]
        )

        scores_stack = torch.stack(scores)  # stack reshapes to a size of (batch,1)

        return smiles_padded, scores_stack


class LSTMScoreModel(nn.Module, Tokenizer):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, directory):
        super(LSTMScoreModel, self).__init__()
        Tokenizer.__init__(self, directory)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(
            num_embeddings=len(self.smiles_map),
            embedding_dim=input_dim,
            padding_idx=self.smiles_map["PAD"],
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


class LSTMGenModel(nn.Module, Tokenizer):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, directory):
        super(LSTMGenModel, self).__init__()
        Tokenizer.__init__(self, directory)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(
            num_embeddings=len(self.smiles_map),
            embedding_dim=input_dim,
            padding_idx=self.smiles_map["PAD"],
        )
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # added dropout to reduce overfitting
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden_state=None, cell_state=None):
        # on first pass initialize hidden and cell states to zeros
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(
                x.device
            )
            cell_state = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(
                x.device
            )
            if hidden_state.shape[0] != self.layer_dim:
                hidden_state = hidden_state.permute(1, 0, 2).contiguous()
                cell_state = cell_state.permute(1, 0, 2).contiguous()

        embedded_seq = self.embedding(x)
        out, (hidden_state, cell_state) = self.lstm(embedded_seq)
        out = self.fc(out)  # final layer

        temperature = 0.25
        out = out / temperature

        hidden_state = hidden_state.permute(1, 0, 2)  # swap them back for tensordict
        cell_state = cell_state.permute(1, 0, 2)

        return out, hidden_state, cell_state


class adcDataset(Dataset, Tokenizer):

    SMI_REGEX = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|B|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    def __init__(self, directory):
        Tokenizer.__init__(self, directory)
        self.directory = directory
        self.file = pd.read_pickle(directory)
        self.smiles = (
            self.file["smiles"]
            .str.findall(adcDataset.SMI_REGEX)
            .apply(self.smiles_to_tokens)
        )
        self.SA_score = torch.tensor(np.array((self.file["calc_SA_score"])))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = torch.tensor(self.smiles[idx])
        score = torch.tensor([self.SA_score[idx]])

        return smile, score


class SmilesGeneratorEnv(EnvBase):
    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.envs.EnvBase.html
    # https://docs.pytorch.org/rl/0.8/reference/generated/torchrl.data.CompositeSpec.html
    # https://docs.pytorch.org/tutorials/advanced/pendulum.html

    # documentation describes the standard boilerplate for implementing torchRL

    # EnvBase abstract environment base class defines standard environment interface for RL
    # this includes the reset and step methods that allow RL to proceed

    def __init__(
        self,
        score_model,
        vocab_size,
        max_length,
        num_layers,
        hidden_dim,
        token,
        device="cuda",
        seed=None,
    ):
        super().__init__(
            device=device, batch_size=[]
        )  # batch of [] means one env for serial processing
        self.observation_spec = CompositeSpec()

        self.scorer = score_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.token = token

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
                "observation": DiscreteTensorSpec(
                    n=self.vocab_size, shape=(1,), dtype=torch.long
                ),  # discrete because there are a limited number of tokens that can be predicted
                "hidden": UnboundedContinuousTensorSpec(
                    shape=(num_layers, hidden_dim)
                ),  # both the values for hidden can cell are continuous with no known min or max (unbounded)
                "cell": UnboundedContinuousTensorSpec(shape=(num_layers, hidden_dim)),
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

        reset_observation = TensorDict(
            {
                "observation": start_token,
                "hidden": torch.zeros(
                    self.num_layers, self.hidden_dim, device=self.device
                ),
                "cell": torch.zeros(
                    self.num_layers, self.hidden_dim, device=self.device
                ),
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
        is_end = action_token == token.smiles_map["END"]
        is_max = self.current_step >= self.max_length
        done = is_end | is_max

        # if we are done generating our sequence then calculate the reward
        reward = 0.0
        if done:

            # check to see if we got a real molecule, penalize if not
            smiles_string = token.tokens_to_smiles(self.generated_sequence)
            mol = Chem.MolFromSmiles(smiles_string)

            if mol is None:
                # an invalid smiles string was generated
                reward = -20.0

            else:
                # matches model device if on cuda
                device = next(self.scorer.parameters()).device

                # convert the sequence to a tensor to pass into scoring model
                full_sequence = torch.tensor(
                    self.generated_sequence, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    reward = -1 * self.scorer(full_sequence).item()

        # pass along info for next iterations, observation is just our action for the step
        # reward is 0 if not done and done and terminated are false if we haven't reached "END" or max length
        next_observation = TensorDict(
            {
                "observation": action_token.view(1),
                "reward": torch.tensor([reward], device=self.device),
                "done": torch.tensor([done], device=self.device),
                "terminated": torch.tensor([done], device=self.device),
            }
        )

        return next_observation


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


def train_LSTM_scores(directory):
    token = Tokenizer(directory)
    dataset = adcDataset(directory)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=token.collate_smiles,
    )

    model = LSTMScoreModel(
        input_dim=128, hidden_dim=256, layer_dim=5, output_dim=1, directory=directory
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 20

    for epoch in range(num_epochs):
        for i, (smile, score) in enumerate(loader):
            print(f"Batch {i+1}/{math.ceil(len(dataset)/loader.batch_size)}", end="\r")
            smile, label = smile.to(device), score.to(device)

            optimizer.zero_grad()

            output = model(smile)  # forward
            loss = criterion(output.float(), label.float())  # calculate MSE loss
            loss.backward()  # back propagation
            optimizer.step()

        print()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.3f}")

    return model


def test_LSTM_scores(directory, model):

    token = Tokenizer(directory)
    dataset = adcDataset(directory)
    loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=True,
        collate_fn=token.collate_smiles,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    SA_predictions = []
    SA_true = []

    with torch.no_grad():
        for i, (smile, score) in enumerate(loader):
            smile, score = smile.to(device), score.to(device)
            output = model(smile)
            SA_predictions += output
            SA_true += score

    # convert from tensor to array for predicted and true scores
    SA_predictions = np.array([pred.cpu().numpy() for pred in SA_predictions])
    SA_true = np.array([true.cpu().numpy() for true in SA_true])

    avg_percent_error = np.mean(
        np.abs((SA_true - SA_predictions) / (SA_true + 1e-8)) * 100
    )
    print(f"Mean Absolute Percent Error: {avg_percent_error:.2f}%")


def train_LSTM_gen(directory, num_epochs=1000):
    batch_size = 64
    token = Tokenizer(directory)
    dataset = adcDataset(directory)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=token.collate_smiles,
    )

    model = LSTMGenModel(
        input_dim=128,
        hidden_dim=256,
        layer_dim=5,
        output_dim=len(token.smiles_map),
        directory=directory,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=token.smiles_map["PAD"]
    )  # ignore the padding for loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        for i, (smile, score) in enumerate(loader):
            print(f"Batch {i+1}/{math.ceil(len(dataset)/loader.batch_size)}", end="\r")

            target = smile.to(device)

            # create a vector containing start tokens as long as the current batch
            seq_start = torch.full(
                (target.shape[0], 1), token.smiles_map["START"], dtype=torch.long
            )
            seq_start = seq_start.to(device)

            # remove the last token from input so that we can add start token to the front
            input_seq = target[:, :-1]

            inputs = torch.cat(
                [seq_start, input_seq], dim=1
            )  # add the start tokens to beginning of input

            optimizer.zero_grad()

            # teacher forcing approach, only using ground truth for loss
            # instead of feeding models previous output we use the actual known previous token
            output, _, _ = model(inputs)
            loss = criterion(
                output.view(-1, len(token.smiles_map)), target.view(-1)
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


def LSTM_model_RL(score_model, gen_model, token, total_frames=500_000):
    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.modules.tensordict_module.ProbabilisticActor.html
    # creating a dict for our LSTM generative model that the actor can then understand for RL

    # actor is our generative model, critic is our scoring model
    # loss module uses the proximal policy optimization (PPO) from torchRL to calculate loss
    lstm_module = TensorDictModule(
        module=gen_model,
        in_keys=["observation", "hidden", "cell"],
        out_keys=["logits", "hidden", "cell"],
    )

    # keys are strict here, need to use predefined key names as described by ProbActor class
    actor_module = ProbabilisticActor(
        module=lstm_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    score_module = TensorDictModule(
        module=score_model,
        in_keys=["observation", "hidden", "cell"],
        out_keys=["out"],
    )

    critic_module = ValueOperator(
        module=score_module, in_keys=["observation", "hidden", "cell"]
    )

    # https://docs.pytorch.org/rl/main/reference/generated/torchrl.collectors.SyncDataCollector.html

    environment_maker = lambda: SmilesGeneratorEnv(
        score_model,
        vocab_size=len(token.smiles_map),
        max_length=50,
        num_layers=gen_model.layer_dim,
        hidden_dim=gen_model.hidden_dim,
        token=token,
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
        entropy_bonus=0.001,  # encourages exploration to prevent same output
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

        # calculate the loss using our PPO
        loss = loss_module(batch.reshape(-1))
        actor_loss = loss["loss_objective"]
        critic_loss = loss["loss_critic"]
        total_loss = actor_loss + critic_loss

        # back propagate
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print reward to keep track of training progress
        avg_reward = batch["next", "reward"].mean().item()
        print(
            f"Batch {i} || Loss: {total_loss.item():.4f}, Avg Reward: {avg_reward:.4f}"
        )

    print("RL training complete.")

    return gen_model


def check_sequences(score_model, gen_model, directory, num_samples=100):
    valid_smiles = 0
    scores = []

    device = next(score_model.parameters()).device
    score_model.eval()

    for _ in range(num_samples):
        sequence = generate_smiles(gen_model, directory)
        smiles = token.tokens_to_smiles(sequence)

        if Chem.MolFromSmiles(smiles) is not None:
            valid_smiles += 1

            seq_tensor = torch.tensor(sequence, device=device).unsqueeze(0)

            with torch.no_grad():
                score = score_model(seq_tensor)
                scores.append(score)

    print("-" * 30)
    print(f"Valid smiles = {valid_smiles}%")
    print(f"Average SA score = {np.mean(scores):.4f}")
    print(f"Std Dev SA score = {np.std(scores):.4f}")
    print("-" * 30)


if __name__ == "__main__":
    # Load our ADC dataset
    adc_directory = "data/adc_data_complete_v2.pkl"
    df = pd.read_pickle(adc_directory)

    # # Perform k-folds on sequence to one model to assess hyperparameters
    # kfolds_LSTM_scores(adc_directory)

    # # train the model on the entire dataset and save the result
    # model_scores = train_LSTM_scores(adc_directory)
    # test_LSTM_scores(adc_directory, model_scores)
    # torch.save(model_scores.state_dict(), "models/model_scores_weights.pth")

    # # train the sequence to sequence generative model
    # model_gen = train_LSTM_gen(adc_directory, num_epochs=200)
    # torch.save(model_gen.state_dict(), "models/model_gen_weights.pth")

    # generate a new sequence from our trained generator model
    token = Tokenizer(adc_directory)
    model_gen = LSTMGenModel(
        input_dim=128,
        hidden_dim=256,
        layer_dim=5,
        output_dim=len(token.smiles_map),
        directory=adc_directory,
    )
    model_gen.load_state_dict(
        torch.load("models/model_gen_weights.pth", weights_only=True)
    )
    # sequence = generate_smiles(model_gen, adc_directory)
    # print(token.tokens_to_smiles(sequence))
    # print(len(sequence))

    model_score = LSTMScoreModel(
        input_dim=128,
        hidden_dim=256,
        layer_dim=5,
        output_dim=1,
        directory=adc_directory,
    )

    model_score.load_state_dict(
        torch.load("models/model_scores_weights.pth", weights_only=True)
    )

    # RL_trained_model = LSTM_model_RL(model_score, model_gen, token)
    # torch.save(RL_trained_model.state_dict(), "models/model_RL_gen_weights.pth")

    model_gen_RL = LSTMGenModel(
        input_dim=128,
        hidden_dim=256,
        layer_dim=5,
        output_dim=len(token.smiles_map),
        directory=adc_directory,
    )

    model_gen_RL.load_state_dict(
        torch.load("models/model_RL_gen_weights.pth", weights_only=True)
    )

    # print()
    # print("Sequence generated after RL training:")
    # sequence = generate_smiles(model_gen_RL, adc_directory)
    # print(token.tokens_to_smiles(sequence))
    # print(len(sequence))

    print("Pre-RL model: ")
    check_sequences(model_score, model_gen, directory=adc_directory)

    print()
    print("RL model:")
    check_sequences(model_score, model_gen_RL, directory=adc_directory)
