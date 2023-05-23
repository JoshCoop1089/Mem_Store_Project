"""demo: train a DND LSTM on a contextual choice task
"""
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sl_model import DNDLSTM as Agent
from sl_model.utils import (
    compute_a2c_loss,
    compute_returns,
    get_reward_from_assumed_barcode,
    vectorize_cos_sim,
    freeze_linear_params
)
from task.ContextBandits import ContextualBandit
from sklearn.cluster import KMeans

def barcodes_are_different(barcode_strings):
    percent_diff = 0.0
    x = barcode_strings
    diff = [int(x[i][0]==x[i-1][0]) for i, _ in enumerate(x)]
    percent_diff = sum(diff)/len(barcode_strings)
    print(percent_diff)
    return
def calc_contrastive_loss(input_stuff):
    return episode_loss

def run_experiment_sl(exp_settings):
    """
    exp_settings is a dict with parameters as keys:

    randomize: Boolean (for performing multiple trials to average results or not)
    epochs: int (number of times to wipe memory and rerun learning)
    kernel: string (should be either 'l2' or 'cosine')
    agent_input: string (choose between passing obs/context, or only obs into LSTM)
    mem_store: string (what is used as keys in the memory)
    num_arms: int (number of unique arms to choose from)
    barcode_size: int (dimension of barcode used to specify good arm)
    num_barcodes: int (number of unique contexts to define)
    pulls_per_episode: int (how many arm pulls are given to each unique barcode episode)
    perfect_info: Boolean (True -> arms only give reward if best arm is pulled, False -> 90%/10% chances on arms as usual)
    noise_percent: float (between 0 and 1 to make certain percent of observations useless)
    embedding_size: int (how big is the embedding model size)
    reset_barcodes_per_epoch: Boolean (create a new random set of barcodes at the start of every epoch, or keep the old one)
    reset_arms_per_epoch: Boolean (remap arms to barcodes at the start of every epoch, or keep the old one)
    lstm_learning_rate: float (learning rate for the LSTM-DND main agent optimizer)
    embedder_learning_rate: float (learning rate for the Embedder-Barcode Prediction optimizer)
    task_version: string (bandit or original QiHong task)
    """
    # Tensorboard viewing
    if exp_settings["tensorboard_logging"]:
        cur_date = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        tb = SummaryWriter(log_dir=f"runs/{exp_settings['exp_name']}_{cur_date}")

    # See Experimental parameters for GPU vs CPU choices
    if exp_settings["torch_device"] == "CPU":
        device = torch.device("cpu")
    elif exp_settings["torch_device"] == "GPU":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError("Incorrect Torch Device set")
    print(f"Device: {device}")

    if not exp_settings["randomize"]:
        seed_val = 0
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)

    # Full training and noise eval length
    n_epochs = exp_settings["epochs"] + exp_settings["noise_eval_epochs"] * len(
        exp_settings["noise_percent"]
    )

    """init task"""
    # input/output/hidden/memory dim
    num_arms = exp_settings["num_arms"]
    barcode_size = exp_settings["barcode_size"]
    num_barcodes = exp_settings["num_barcodes"]

    # Arm pulls per single barcode episode
    pulls_per_episode = exp_settings["pulls_per_episode"]

    # Arm rewards can be deterministic for debugging
    perfect_info = exp_settings["perfect_info"]

    # Cluster barcodes at the start (Only use one per experiment)
    hamming_threshold = exp_settings["hamming_threshold"]
    assert (hamming_threshold == 0) or (
        hamming_threshold > 0 and 3 * hamming_threshold < barcode_size
    )

    # How many extra bits to add per barcode, and where
    noise_train_percent = exp_settings['noise_train_percent']
    noise_train_type = exp_settings['noise_train_type']

    # Task Init
    # Example: 4 unique barcodes -> 16 total barcodes in epoch, 4 trials of each unique barcode
    episodes_per_epoch = num_barcodes**2

    task = ContextualBandit(
        pulls_per_episode,
        episodes_per_epoch,
        num_arms,
        num_barcodes,
        barcode_size,
        noise_train_percent,
        noise_train_type,
        hamming_threshold,
        device,
        perfect_info,
    )

    # LSTM Chooses which arm to pull
    dim_output_lstm = num_arms
    dict_len = pulls_per_episode * (num_barcodes**2)
    value_weight = exp_settings["value_error_coef"]
    entropy_weight = exp_settings["entropy_error_coef"]

    # Input is obs/context/reward triplet
    dim_input_lstm = num_arms + len(task.cluster_lists[0][0]) + 1
    dim_hidden_lstm = exp_settings["dim_hidden_lstm"]
    learning_rate = exp_settings["lstm_learning_rate"]

    # init agent / optimizer
    agent = Agent(
        dim_input_lstm, dim_hidden_lstm, dim_output_lstm, dict_len, exp_settings, device
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()), lr=learning_rate
    )

    # Freeze a portion of the layers to simulate keeping the R-Gates open for the first portion of training
    if exp_settings['freeze_r_gates'] > 0 and exp_settings['epochs'] > 0:
        r_gate_indicies = [x for x in range(3*dim_hidden_lstm,4*dim_hidden_lstm)]
        i2h_weight_hook, i2h_bias_hook = freeze_linear_params(agent.i2h, r_gate_indicies)
        h2h_weight_hook, h2h_bias_hook = freeze_linear_params(agent.h2h, r_gate_indicies)

    if exp_settings["load_pretrained_model"]:
        # Load trained weights
        agent.load_state_dict(torch.load(f"model/{exp_settings['exp_name']}.pt", map_location = device))
        if exp_settings["mem_store"] == "embedding":
            agent.dnd.embedder.load_state_dict(
                torch.load(f"model/{exp_settings['exp_name']}_embedder.pt", map_location = device)
            )

        # Load clusters of barcodes
        task.cluster_lists = np.load(f"model/{exp_settings['exp_name']}.npz")[
            "cluster_lists"
        ].tolist()
        print("--> Model loaded from disk <--")

    # Timing
    run_time = np.zeros(
        n_epochs,
    )

    # Results for TB or Graphing
    log_keys = []
    log_return = np.zeros(
        n_epochs,
    )
    log_memory_accuracy = np.zeros(
        n_epochs,
    )
    log_embedder_accuracy = np.zeros(
        n_epochs,
    )
    log_loss_value = np.zeros(
        n_epochs,
    )
    log_loss_policy = np.zeros(
        n_epochs,
    )
    log_loss_total = np.zeros(
        n_epochs,
    )
    log_bc_guess_accuracy = np.zeros(
        n_epochs,
    )
    log_r_gate_sum = np.zeros(
        n_epochs,
    )

    # Save keys during training at 0%, 33%, 66% and 100% of total train time
    early_keys = [int(x * exp_settings['epochs'] // 20) for x in range(1,6)]
    train_epochs = [int(x * exp_settings["epochs"] // 3) for x in range(3)]
    train_epochs.extend([exp_settings["epochs"] - 1])
    train_epochs.extend(early_keys)
    train_epochs.sort()

    # Save keys at end of different noise epochs
    noise_epochs = [
        x * exp_settings["noise_eval_epochs"] + exp_settings["epochs"] - 1
        for x in range(1, len(exp_settings["noise_percent"]) + 1)
    ]

    if exp_settings["load_pretrained_model"]:
        key_save_epochs = noise_epochs
    else:
        key_save_epochs = train_epochs + noise_epochs

    # Data stroage for K-Means Barcode guesses
    avg_inputs = np.zeros((episodes_per_epoch*pulls_per_episode, dim_input_lstm))

    print("\n", "-*-_-*- " * 3, "\n")
    # loop over epoch
    for i in range(n_epochs):
        # Set to no_grad if in noise eval portion
        with torch.no_grad() if i >= exp_settings['epochs'] else nullcontext():
            time_start = time.perf_counter()

            # get data for this epoch
            (
                observations_barcodes_rewards,
                epoch_mapping,
                barcode_strings,
                barcode_tensors,
                barcode_id,
                arm_id,
            ) = task.sample()
            agent.dnd.mapping = epoch_mapping

            # # How many times is a barcode the same between episodes
            # barcodes_are_different(barcode_strings)

            # flush hippocampus
            agent.reset_memory()
            agent.turn_on_retrieval()

            # Training with noise on?
            if exp_settings["noise_train_percent"] > 0:
                noise_barcode_flip_locs = int(
                    exp_settings["noise_train_percent"] * exp_settings['barcode_size']
                )

            # How much noise is needed in the evaluation stages?
            apply_noise = i - exp_settings["epochs"]
            if apply_noise >= 0:
                noise_idx = apply_noise // exp_settings["noise_eval_epochs"]
                noise_percent = exp_settings["noise_percent"][noise_idx]
                noise_barcode_flip_locs = int(noise_percent * exp_settings['barcode_size'])

            # loop over the training set
            for m in range(episodes_per_epoch):

                # prealloc
                agent.dnd.pred_accuracy = 0
                agent.unsure_bc_guess = torch.Tensor([0.0]).to(device)
                memory_accuracy = 0
                cumulative_reward = 0
                probs, rewards, values, entropies = [], [], [], []
                h_t, c_t = agent.get_init_states()
                if exp_settings['mem_store'] == 'embedding':
                    emb_model = agent.dnd.embedder
                    emb_model.h_lstm, emb_model.c_lstm = emb_model.emb_get_init_states()

                # Always use ground truth bc for reward eval
                real_bc = barcode_strings[m][0][0]

                # Non-noised barcode for accuracy measurements, not used in model calculations
                raw_bc = real_bc[:-int(exp_settings["noise_train_percent"] * exp_settings['barcode_size'])]

                # Training Signal for Embedder
                cross_ent_loss_tensor = barcode_id[m]

                # Clearing the per trial hidden state buffer
                agent.flush_trial_buffer()

                # Noisy Barcodes are constant across an episode if needed
                if apply_noise >= 0 or exp_settings["noise_train_percent"] > 0:
                    apply_noise_again = True
                    obs = observations_barcodes_rewards[m][0]
                    action = obs[0:num_arms].view(1, -1)
                    original_bc = obs[num_arms:-1].view(1, -1)
                    reward = obs[-1].view(1, -1)

                    # No noise, eval phase of model
                    if not exp_settings["noise_type"]:
                        apply_noise_again = False
                        noisy_bc = original_bc.detach().clone()

                    # Noise applied to BC at beginning of episode
                    while apply_noise_again:
                        apply_noise_again = False

                        # What indicies need to be randomized?
                        idx = random.sample(
                            range(
                                len(task.cluster_lists[0][0])), noise_barcode_flip_locs
                        )

                        # Coin Flip to decide whether to flip the values at the indicies
                        if not exp_settings["perfect_noise"]:
                            mask = torch.tensor(
                                [random.randint(0, 1) for _ in idx], device=device
                            )

                        # Always flip the value at the indicies
                        else:
                            mask = torch.tensor([1 for _ in idx], device=device)

                        noisy_bc = original_bc.detach().clone()

                        # Applying the mask to the barcode at the idx
                        if exp_settings["noise_type"] == "random":
                            for idx1, mask1 in zip(idx, mask):
                                noisy_bc[0][idx1] = float(
                                    torch.ne(mask1, noisy_bc[0][idx1])
                                )

                        # Applying a continuous block starting on the left end of bc
                        elif exp_settings["noise_type"] == "left_mask":
                            for idx1, mask1 in enumerate(mask):
                                noisy_bc[0][idx1] = float(
                                    torch.ne(mask1, noisy_bc[0][idx1])
                                )

                        # Applying a continuous block covering the center of bc
                        elif exp_settings["noise_type"] == "center_mask":
                            # Find center
                            center = len(task.cluster_lists[0][0]) // 2

                            # Find edges of window
                            start = center - noise_barcode_flip_locs // 2
                            end = center + noise_barcode_flip_locs // 2

                            idx = np.arange(start, end)
                            for idx1, mask1 in zip(idx, mask):
                                noisy_bc[0][idx1] = float(
                                    torch.ne(mask1, noisy_bc[0][idx1])
                                )

                        # Applying an continuous block starting on the right end of bc
                        elif exp_settings["noise_type"] == "right_mask":
                            for idx1, mask1 in enumerate(mask):
                                loc = len(task.cluster_lists[0][0]) - 1 - idx1
                                noisy_bc[0][loc] = float(torch.ne(mask1, noisy_bc[0][loc]))

                        # Even distribution of noise across bc
                        elif exp_settings["noise_type"] == "checkerboard":
                            if noise_percent != 0:
                                idx = np.arange(
                                    0, len(task.cluster_lists[0][0]), int(1 / noise_percent)
                                )
                                for idx1, mask1 in zip(idx, mask):
                                    noisy_bc[0][idx1] = float(
                                        torch.ne(mask1, noisy_bc[0][idx1])
                                    )

                        # Cosine similarity doesn't like all 0's for matching in memory
                        if torch.sum(noisy_bc) == 0:
                            apply_noise_again = True

                    # Remake the input
                    noisy_init_input = torch.cat(
                        (action, noisy_bc, reward.view(1, 1)), dim=1
                    )

                # loop over time, for one training example
                for t in range(pulls_per_episode):

                    # only save memory at the last time point
                    agent.turn_off_encoding()
                    if t == pulls_per_episode - 1 and m < episodes_per_epoch:
                        agent.turn_on_encoding()

                    # First input when not noisy comes from task.sample
                    if t == 0:
                        if (
                            i < exp_settings["epochs"]
                            and exp_settings["noise_train_percent"] == 0
                        ):
                            input_to_lstm = observations_barcodes_rewards[m]
                        else:
                            input_to_lstm = noisy_init_input

                    # Using the output action and reward of the last step of the LSTM as the next input
                    else:  # t != 0:
                        input_to_lstm = last_action_output

                    # What is being stored for Ritter?
                    mem_key = (
                        barcode_tensors[m]
                        if (
                            i < exp_settings["epochs"]
                            and exp_settings["noise_train_percent"] == 0
                        )
                        else noisy_bc
                    )

                    output_t, cache = agent(
                        input_to_lstm,          #Tensor made of arm choice, noised BC, and reward
                        raw_bc,                 #String version of context BC, no noise
                        real_bc,                #String version of context BC, with noise
                        mem_key,                #Tensor version of context BC, with noise
                        cross_ent_loss_tensor,  #BC Class ID for embedder loss
                        h_t,                    #Hidden state of LSTM
                        c_t,                    #Cell State of LSTM
                    )

                    a_t, assumed_barcode_string, prob_a_t, v_t, entropy, h_t, c_t = output_t
                    f_t, i_t, o_t, r_gate, m_t, sim_score = cache

                    # compute immediate reward for actor network
                    r_t = get_reward_from_assumed_barcode(
                        a_t, real_bc, epoch_mapping, device, perfect_info
                    )

                    # Does the predicted context match the actual context?
                    memory_accuracy += int(raw_bc == assumed_barcode_string)

                    probs.append(prob_a_t)
                    rewards.append(r_t)
                    values.append(v_t)
                    entropies.append(entropy)
                    cumulative_reward += r_t

                    # Store inputs to id barcodes with k-means
                    if i == 0 and exp_settings['emb_loss'] == 'kmeans':
                        avg_inputs[m*pulls_per_episode + t] += input_to_lstm.view(-1).cpu().numpy()

                    # Inputs to LSTM come from predicted actions and rewards of last time step
                    one_hot_action = torch.zeros(
                        (1, num_arms), dtype=torch.float32, device=device
                    )
                    one_hot_action[0][a_t] = 1.0
                    next_bc = barcode_tensors[m]

                    # Add noise to the barcode at the right moments in experiment
                    if (
                        # Noise during training phase if noise being used in training
                        exp_settings["noise_train_percent"] and i < exp_settings["epochs"]
                        ) or (
                        # Noise during eval phase
                        i >= exp_settings["epochs"]
                    ):
                        next_bc = noisy_bc

                    # Create next input to feed back into LSTM
                    last_action_output = torch.cat(
                        (one_hot_action, next_bc, r_t.view(1, 1)), dim=1
                    )

                    log_r_gate_sum[i] += torch.div(torch.sum(r_gate), exp_settings['dim_hidden_lstm']*episodes_per_epoch*pulls_per_episode)

                    # Look at R-Gate values in the final training epoch
                    if (
                        exp_settings["tensorboard_logging"]
                        and i == exp_settings["epochs"] - 1
                    ):
                        tb.add_histogram(
                            "R-Gate Sigmoided Train Final Epoch",
                            r_gate,
                            t + m * pulls_per_episode,
                        )

                # LSTM/A2C Loss for Episode
                returns = compute_returns(rewards, device, gamma=0.0)
                loss_policy, loss_value, entropies_tensor = compute_a2c_loss(
                    probs, values, returns, entropies
                )
                loss = (
                    loss_policy
                    + value_weight * loss_value
                    - entropy_weight * entropies_tensor
                )

                # Only perform model updates during train phase
                if apply_noise < 0:
                    if exp_settings["mem_store"] == "embedding":
                        # Embedder Loss for Episode
                        a_dnd = agent.dnd
                        
                        # Need to do a math sweep over this code implementation.  Also prob should reduce loss by a constant factor
                        # if exp_settings['emb_loss'] == 'contrastive':
                        """
                        embA are from current episode
                        create all pairs from embA, label as similar
                        if not first episode:
                            retrieve embeddings from last episode (embB)
                            create all pairs from embA and embB, label as different
                        """
                        embA = [x[0].view(-1) for x in a_dnd.trial_buffer]
                        embA_stack = torch.stack(embA)
                        x = vectorize_cos_sim(embA_stack, embA_stack, device, same = True)

                        # Avoid doublecounting positive pairs
                        x_dist = (torch.ones_like(x, device = device) - x)/2

                        pos_output = torch.sum(x_dist)
                        neg_output = torch.tensor(0, device = device)
                        if m > 0:
                            negs = vectorize_cos_sim(
                                embA_stack, embB_stack, device, same = False)
                            
                            # Far apart negatives should not contribute to loss bc cos will be closer to 0
                            neg_dist = torch.ones_like(negs, device = device)-negs  

                            # print(neg_dist)
                            neg_output = torch.sum(neg_dist)
                        embB_stack = embA_stack.detach().clone()
                        episode_loss = torch.div((pos_output+neg_output), 2).detach().clone().requires_grad_(True)
                        a_dnd.contrastive_loss[i] += torch.div(episode_loss, episodes_per_epoch)

                        if exp_settings['emb_loss'] == 'kmeans' or exp_settings['emb_loss'] == 'groundtruth':
                            loss_vals = [x[2] for x in a_dnd.trial_buffer]
                            episode_loss = torch.stack(loss_vals).sum()
                            
                        a_dnd.embedder_loss[i] += torch.div(episode_loss, episodes_per_epoch)

                        # Unfreeze Embedder
                        for name, param in a_dnd.embedder.named_parameters():
                            param.requires_grad = True

                        # Freeze LSTM/A2C
                        layers = [agent.i2h, agent.h2h, agent.a2c]
                        for layer in layers:
                            for name, param in layer.named_parameters():
                                param.requires_grad = False

                        # Embedder Backprop
                        a_dnd.embed_optimizer.zero_grad()
                        episode_loss.backward(retain_graph=True)
                        a_dnd.embed_optimizer.step()
                        a_dnd.embed_optimizer.zero_grad()

                        # Freeze Embedder until next memory retrieval
                        for name, param in a_dnd.embedder.named_parameters():
                            param.requires_grad = False

                        # Unfreeze LSTM/A2C
                        for layer in layers:
                            for name, param in layer.named_parameters():
                                param.requires_grad = True

                    # LSTM and A2C Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Reset the hook on the R-Gates after having kept them frozen until a specified epoch
                    if i == exp_settings['freeze_r_gates'] and i > 0 and m == 0 and exp_settings['epochs'] > 0:
                        i2h_weight_hook.remove()
                        i2h_bias_hook.remove()
                        h2h_weight_hook.remove()
                        h2h_bias_hook.remove()
                        
                # Updating avg return per episode
                log_return[i] += torch.div(
                    cumulative_reward, (episodes_per_epoch * pulls_per_episode)
                )


                # Updating avg accuracy per episode
                log_bc_guess_accuracy[i] += torch.div(agent.unsure_bc_guess, (episodes_per_epoch*pulls_per_episode))
                log_embedder_accuracy[i] += torch.div(agent.dnd.pred_accuracy, (episodes_per_epoch*pulls_per_episode))
                log_memory_accuracy[i] += torch.div(memory_accuracy, (episodes_per_epoch*pulls_per_episode))

                # Loss Logging
                log_loss_value[i] += torch.div(loss_value, episodes_per_epoch)
                log_loss_policy[i] += torch.div(loss_policy, episodes_per_epoch)
                log_loss_total[i] += torch.div(loss, episodes_per_epoch)

            if i == 0 and exp_settings['emb_loss'] == 'kmeans':
                km = KMeans(n_clusters = num_barcodes, init = 'random', n_init= 10, max_iter = 300)
                y_km = km.fit_predict(avg_inputs)
                agent.dnd.barcode_guesses = torch.as_tensor(km.cluster_centers_, device = device)

            # Tensorboard Stuff
            if exp_settings["tensorboard_logging"]:
                tb.add_scalar("LSTM Returns", log_return[i], i)
                tb.add_scalar("Mem Retrieval Accuracy", log_memory_accuracy[i], i)
                tb.add_scalar("Emb Retrieval Accuracy", log_embedder_accuracy[i], i)
                tb.add_scalar("Emb Loss", agent.dnd.embedder_loss[i], i)
                if i < exp_settings["epochs"]:
                    tb.add_histogram("R-Gate Sigmoided Train Epochs", r_gate, i)
                else:
                    tb.add_histogram("R-Gate Sigmoided Noise Epochs", r_gate, i)

            run_time[i] = time.perf_counter() - time_start

            # Print reports every 10% of the total number of epochs
            if i % (int(n_epochs / 10)) == 0 or i == n_epochs - 1:
                if exp_settings['mem_store'] == 'embedding':
                    print(
                        "Epoch %3d | avg_return = %.2f | loss: LSTM = %.2f, Embedder = %.2f | time = %.2f"
                        % (
                            i,
                            log_return[i],
                            log_loss_total[i],
                            agent.dnd.embedder_loss[i],
                            run_time[i],
                        )
                    )
                else:
                    print(
                        "Epoch %3d | avg_return = %.2f | loss: val = %.2f, pol = %.2f, tot = %.2f | time = %.2f"
                        % (
                            i,
                            log_return[i],
                            log_loss_value[i],
                            log_loss_policy[i],
                            log_loss_total[i],
                            run_time[i],
                        )
                    )
                # Accuracy over the last 10 epochs
                if i > 11:
                    avg_acc = log_memory_accuracy[i - 9 : i + 1].mean()
                    avg_emb_acc = log_embedder_accuracy[i - 9 : i + 1].mean()
                    avg_bc_acc = log_bc_guess_accuracy[i - 9 : i + 1].mean()
                else:
                    avg_acc = log_memory_accuracy[: i + 1].mean()
                    avg_emb_acc = log_embedder_accuracy[: i + 1].mean()
                    if i == 0:
                        avg_bc_acc = 0
                    else:
                        avg_bc_acc = log_bc_guess_accuracy[1: i + 1].mean()
                    
                print("  Mem Acc:", round(avg_acc, 4), end=" | ")
                if exp_settings['mem_store'] == 'embedding':
                    print("Model Acc:", round(avg_emb_acc, 4), end=" | ")
                    if exp_settings['emb_loss'] == 'kmeans':
                        print("BC Acc:", round(avg_bc_acc, 4), end=" | ")
                print(f"Time Elapsed: {round(sum(run_time), 1)} secs")

            # Store the keys from the end of specific training epochs
            if i in key_save_epochs:
                keys, prediction_mapping = agent.get_all_mems_embedder()
                log_keys.append(keys)

    # Save model for eval on different noise types
    if exp_settings['save_model']:
        torch.save(agent.state_dict(), f"model/{exp_settings['exp_name']}.pt")
        if exp_settings["mem_store"] == "embedding":
            torch.save(agent.dnd.embedder.state_dict(), f"model/{exp_settings['exp_name']}_embedder.pt")
        
        # Save clusters of barcodes
        np.savez(f"model/{exp_settings['exp_name']}.npz", cluster_lists = task.cluster_lists)

    # Final Results printed to Console
    start = exp_settings["epochs"]
    eval_len = exp_settings["noise_eval_epochs"]
    print()
    print("- - - " * 3)
    print(f"BC Size: {exp_settings['barcode_size']}\t| Noise Added: {int(exp_settings['barcode_size']*exp_settings['noise_train_percent'])}")
    for idx, percent in enumerate(exp_settings["noise_percent"]):
        avg_returns = np.mean(log_return[start : start + eval_len])
        avg_mem_accuracy = np.mean(log_memory_accuracy[start : start + eval_len])
        acc = f"Accuracy: {round(avg_mem_accuracy,3)}"
        if exp_settings['mem_store'] == 'embedding':
            emb_model_acc = np.mean(log_embedder_accuracy[start : start + eval_len])
            acc += f" \t| Model Acc: {round(emb_model_acc,3)}"
        print(
            f"Noise Bits: {int(percent*exp_settings['barcode_size'])}\t| Returns: {round(avg_returns,3):0.3} \t| {acc}"
        )
        start += eval_len

    print("- - - " * 3)
    print("Total Time:\t", round(sum(run_time), 1), "secs")
    if exp_settings["epochs"] != 0:
        train_time = round(np.mean(run_time[: exp_settings["epochs"]]), 2)
    else:
        train_time = 0
    print("Avg Train Time:\t", train_time, "secs")
    print(
        "Avg Eval Time:\t",
        round(np.mean(run_time[exp_settings["epochs"] :]), 2),
        "secs",
    )
    print("- - - " * 3)

    logs_for_graphs = log_return, log_memory_accuracy, log_r_gate_sum, log_embedder_accuracy
    loss_logs = log_loss_value, log_loss_policy, log_loss_total, agent.dnd.embedder_loss, agent.dnd.contrastive_loss
    key_data = log_keys, epoch_mapping

    if exp_settings["tensorboard_logging"]:
        tb.flush()
        tb.close()

    return logs_for_graphs, loss_logs, key_data