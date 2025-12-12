# Torch modules for scoring and generative LSTMs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# TorchRL modules for reinforcement learning
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.envs import SerialEnv
from tensordict.nn import TensorDictModule, TensorDictSequential

# Other modules
import numpy as np
import pandas as pd
import random
import re
import math
import json
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, rdFingerprintGenerator
import sascorer

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import os

import selfies as sf

# Disable error messages when a non-valid smiles is encountered
rdBase.DisableLog("rdApp.error")

from LSTM_classes import (
    ADCClassifier,
    LSTMGenModel,
    SmilesGeneratorEnv,
    LogitBiasModule,
)

from LSTM_utils import compute_fingerprint, selfie_sanitizer

# Disable error messages when a non-valid smiles is encountered
rdBase.DisableLog("rdApp.error")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchrl")
os.environ["RL_WARNINGS"] = "False"


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
    stoi_dicts,
    learning_rate=0.00001,
    temperature=1.0,
    total_frames=500_000,
    num_envs=64,
    frame_steps=150,
    clip_epsilon=0.025,
    kl=0.05,
    bias_strength=5.0,
    classifier_directory=None,
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
                stoi_dicts[0][ab],
                stoi_dicts[1][pay],
                stoi_dicts[2][tar],
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
        classifier_directory=classifier_directory,
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
    pool_size=100,
):

    # initialize our bias module
    bias_net, h_ids, c_ids, t_ids = setup_bias(tokenizer, bias_strength=bias_strength)

    # start generating
    print("Generating Valid Linkers with Bias...")

    valid_linkers = []
    seen_smiles = set()

    attempts = 0
    max_attempts = pool_size * 1000

    while len(valid_linkers) < pool_size and attempts < max_attempts:
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
            try:
                tagged_selfies = tokenizer.ids_to_selfies(seq)
                tagged_smi = sf.decoder(tagged_selfies)

                real_smi = selfie_sanitizer(tagged_smi, tag_to_smiles)

                mol = Chem.MolFromSmiles(real_smi)

                if Chem.MolFromSmiles(real_smi) is not None:
                    # Basic deduplication
                    if real_smi not in seen_smiles:
                        mol = Chem.MolFromSmiles(real_smi)

                        if mol is not None:

                            # filter away boring small stuff
                            if mol.GetNumHeavyAtoms() < 25:
                                continue
                            mol = Chem.MolFromSmiles(real_smi)
                            tpsa = Descriptors.TPSA(mol)
                            logp = Descriptors.MolLogP(mol)
                            qed = QED.qed(mol)
                            csp3 = Descriptors.FractionCSP3(mol)
                            sa = sascorer.calculateScore(mol)

                            norm_sa = (10 - sa) / 9.0
                            score = (qed + norm_sa) / 2.0

                            results_entry = {
                                "SMILES": real_smi,
                                "Score": score,
                                "SA": sa,
                                "TPSA": tpsa,
                                "QED": qed,
                                "LogP": logp,
                                "CSP3": csp3,
                            }

                            valid_linkers.append(results_entry)
                            seen_smiles.add(real_smi)

            except:
                print("chemical calculation error...")
                pass

            else:
                pass

    if attempts >= max_attempts:
        print("Warning: Hit max attempts limit.")

    sorted_linkers = sorted(valid_linkers, key=lambda x: x["Score"], reverse=True)

    top_linkers = sorted_linkers[:target_count]

    return top_linkers


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

    print("\nPerforming UMAP...")

    reducer = umap.UMAP(
        n_neighbors=50, min_dist=0.1, metric="euclidean", random_state=42
    )
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

    plt.title(f"Real Linkers: {color_by.title()}", fontsize=18)
    plt.legend(fontsize=16.0)
    plt.tight_layout()
    plt.show()


def umap_generated_data(
    model,
    tokenizer,
    stoi_dicts,
    bias_strength=15.0,
    temperature=0.7,
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
            print(f"Attempts: {attempts}/{max_attempts}", end="\r")

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

    print("\nRunning UMAP...")

    reducer = umap.UMAP(
        n_neighbors=50, min_dist=0.1, metric="euclidean", random_state=42
    )
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

    plt.title(f"Generated Linkers: Antibody Combos", fontsize=18)
    plt.legend(fontsize=16.0)
    plt.tight_layout()
    plt.show()


def train_classifiers(
    df,
    stoi_dicts,
    directory,
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
            print(f"Accuracy: {acc:.4f}")

        # save the model
        torch.save(model.state_dict(), f"{directory}/mlp_{name}.pt")

        # save mapping so network knows which conditions correspond to what in environment class
        with open(f"{directory}/map_{name}.json", "w") as f:
            json.dump(label_map, f)

        trained_models[name] = model

    return trained_models
