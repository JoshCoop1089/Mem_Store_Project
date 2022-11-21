"""demo: train a DND LSTM on a contextual choice task
"""
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sl_model import DNDLSTM as Agent
from sl_model.utils import (
    compute_a2c_loss,
    compute_returns,
    get_reward_from_assumed_barcode,
)
from task.ContextBandits import ContextualBandit


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
    sim_threshold = exp_settings["sim_threshold"]
    noise_train_percent = exp_settings['noise_train_percent']
    hamming_threshold = exp_settings["hamming_threshold"]
    assert (hamming_threshold == 0) or (
        hamming_threshold > 0 and 3 * hamming_threshold < barcode_size
    )

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

    if exp_settings["load_pretrained_model"]:
        # Load trained weights
        agent.load_state_dict(torch.load(f"model/{exp_settings['exp_name']}.pt"))
        if exp_settings["mem_store"] == "embedding":
            agent.dnd.embedder.load_state_dict(
                torch.load(f"model/{exp_settings['exp_name']}_embedder.pt")
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
    epoch_sim_log = np.zeros(
        episodes_per_epoch * pulls_per_episode,
    )

    # Save keys during training at 0%, 33%, 66% and 100% of total train time
    train_epochs = [int(x * exp_settings["epochs"] // 3) for x in range(3)]
    train_epochs.extend([exp_settings["epochs"] - 1])

    # Save keys at end of different noise epochs
    noise_epochs = [
        x * exp_settings["noise_eval_epochs"] + exp_settings["epochs"] - 1
        for x in range(1, len(exp_settings["noise_percent"]) + 1)
    ]

    if exp_settings["load_pretrained_model"]:
        key_save_epochs = noise_epochs
    else:
        key_save_epochs = train_epochs + noise_epochs

    print("\n", "-*-_-*- " * 3, "\n")
    # loop over epoch
    for i in range(n_epochs):

        # Anneal Entropy from 1 down to 0 over course of training
        if exp_settings["entropy_error_coef"] == 1 and exp_settings['epochs'] != 0:
            # entropy_weight = (i - exp_settings["epochs"]) / exp_settings["epochs"]
            entropy_weight = (exp_settings["epochs"]-i) / exp_settings["epochs"]

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

        # flush hippocampus
        agent.reset_memory()
        agent.turn_on_retrieval()

        # Training with noise on?
        if exp_settings["noise_train_percent"] > 0:
            noise_barcode_flip_locs = int(
                exp_settings["noise_train_percent"] * len(task.cluster_lists[0][0])
            )

        # How much noise is needed in the evaluation stages?
        apply_noise = i - exp_settings["epochs"]
        if apply_noise >= 0:
            noise_idx = apply_noise // exp_settings["noise_eval_epochs"]
            noise_percent = exp_settings["noise_percent"][noise_idx]
            noise_barcode_flip_locs = int(noise_percent * len(task.cluster_lists[0][0]))

        # loop over the training set
        for m in range(episodes_per_epoch):

            # prealloc
            agent.dnd.pred_accuracy = 0
            embedder_accuracy = 0
            cumulative_reward = 0
            probs, rewards, values, entropies = [], [], [], []
            h_t, c_t = agent.get_init_states()

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
                    input_to_lstm,
                    barcode_strings[m][0][0],
                    mem_key,
                    barcode_id[m],
                    h_t,
                    c_t,
                )
                a_t, assumed_barcode_string, prob_a_t, v_t, entropy, h_t, c_t = output_t
                f_t, i_t, o_t, r_gate, m_t, sim_score = cache
                # epoch_sim_log[t+m*pulls_per_episode] += sim_score/n_epochs

                # Always use ground truth bc for reward eval
                real_bc = barcode_strings[m][0][0]

                # compute immediate reward for actor network
                r_t = get_reward_from_assumed_barcode(
                    a_t, real_bc, epoch_mapping, device, perfect_info
                )

                # Does the predicted context match the actual context?
                embedder_accuracy += int(real_bc == assumed_barcode_string)

                probs.append(prob_a_t)
                rewards.append(r_t)
                values.append(v_t)
                entropies.append(entropy)
                cumulative_reward += r_t

                # Inputs to LSTM come from predicted actions and rewards of last time step
                one_hot_action = torch.zeros(
                    (1, num_arms), dtype=torch.float32, device=device
                )
                one_hot_action[0][a_t] = 1.0
                next_bc = barcode_tensors[m]

                # Add noise to the barcode at the right moments in experiment
                if (
                    exp_settings["noise_train_percent"] and i < exp_settings["epochs"]
                ) or (i >= exp_settings["epochs"]):
                    next_bc = noisy_bc

                # Create next input to feed back into LSTM
                last_action_output = torch.cat(
                    (one_hot_action, next_bc, r_t.view(1, 1)), dim=1
                )

                # Look at R-Gate values in the final training epoch
                if (
                    exp_settings["tensorboard_logging"]
                    and i == exp_settings["epochs"] - 1
                ):
                    tb.add_histogram(
                        "R-Gate Weights Train Final Epoch",
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
                    loss_vals = [x[2] for x in a_dnd.trial_buffer]
                    episode_loss = torch.stack(loss_vals).sum()
                    a_dnd.embedder_loss[i] += episode_loss / episodes_per_epoch

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

            # Updating avg return per episode
            log_return[i] += torch.div(
                cumulative_reward, (episodes_per_epoch * pulls_per_episode)
            )

            # Updating avg accuracy per episode
            # log_embedder_accuracy[i] += torch.div(agent.dnd.pred_accuracy, (episodes_per_epoch*pulls_per_episode))
            log_embedder_accuracy[i] += torch.div(
                embedder_accuracy, (episodes_per_epoch * pulls_per_episode)
            )

            # Loss Logging
            log_loss_value[i] += torch.div(loss_value, episodes_per_epoch)
            log_loss_policy[i] += torch.div(loss_policy, episodes_per_epoch)
            log_loss_total[i] += torch.div(loss, episodes_per_epoch)

        # Tensorboard Stuff
        if exp_settings["tensorboard_logging"]:
            tb.add_scalar("LSTM Returns", log_return[i], i)
            tb.add_scalar("Mem Retrieval Accuracy", log_embedder_accuracy[i], i)
            if i < exp_settings["epochs"]:
                tb.add_histogram("R-Gate Weights Train Epochs", r_gate, i)
            else:
                tb.add_histogram("R-Gate Weights Noise Epochs", r_gate, i)

        run_time[i] = time.perf_counter() - time_start

        # Print reports every 10% of the total number of epochs
        if i % (int(n_epochs / 10)) == 0 or i == n_epochs - 1:
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
            if i > 10:
                avg_acc = log_embedder_accuracy[i - 9 : i + 1].mean()
            else:
                avg_acc = log_embedder_accuracy[: i + 1].mean()
            print("  Embedder Accuracy:", round(avg_acc, 4), end=" | ")
            print(
                "Ritter Baseline:",
                round(1 - 1 / exp_settings["num_barcodes"], 4),
                end=" | ",
            )
            print(f"Time Elapsed: {round(sum(run_time), 1)} secs")

        # Store the keys from the end of the training epochs
        if i in key_save_epochs:
            keys, prediction_mapping = agent.get_all_mems_embedder()
            log_keys.append(keys)

    # Save model for eval on different noise types
    if exp_settings['save_model']:
        torch.save(agent.state_dict(), f"model/{exp_settings['exp_name']}.pt")
        if exp_settings["mem_store"] == "embedding":
            torch.save(agent.dnd.embedder.state_dict(), f"model/{exp_settings['exp_name']}_embedder.pt")
        
        # Load clusters of barcodes
        np.savez(f"model/{exp_settings['exp_name']}.npz", cluster_lists = task.cluster_lists)

    # Final Results
    start = exp_settings["epochs"]
    eval_len = exp_settings["noise_eval_epochs"]
    print()
    print("- - - " * 3)
    for idx, percent in enumerate(exp_settings["noise_percent"]):
        no_noise_eval = np.mean(log_return[start : start + eval_len])
        no_noise_accuracy = np.mean(log_embedder_accuracy[start : start + eval_len])
        print(
            f"Noise: {int(100*percent)}%\t| Bits: {int(percent*len(task.cluster_lists[0][0]))}\t| Returns: {round(no_noise_eval,3):0.3} \t| Accuracy: {round(no_noise_accuracy,3)}"
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

    logs_for_graphs = log_return, log_embedder_accuracy, epoch_sim_log
    loss_logs = log_loss_value, log_loss_policy, log_loss_total, agent.dnd.embedder_loss
    key_data = log_keys, epoch_mapping

    if exp_settings["tensorboard_logging"]:
        tb.flush()
        tb.close()

    return logs_for_graphs, loss_logs, key_data


def run_experiment(exp_base, exp_difficulty):

    exp_settings = {}

    ### Hyperparams in BayesOpt ###
    exp_settings["dim_hidden_a2c"] = 0
    exp_settings["dim_hidden_lstm"] = 0
    exp_settings["entropy_error_coef"] = 0
    exp_settings["lstm_learning_rate"] = 0
    exp_settings["value_error_coef"] = 0
    exp_settings["embedding_size"] = 0
    exp_settings["embedder_learning_rate"] = 0
    ### End Hyperparams in BayesOpt ###

    ### Experimental Parameters ###
    exp_settings["randomize"] = True
    exp_settings["perfect_info"] = False  # Make arms 100%/0% reward instead of 90%/10%
    exp_settings["torch_device"] = "CPU"  # 'CPU' or 'GPU'
    exp_settings["load_pretrained_model"] = False

    # Task Info
    exp_settings["kernel"] = "cosine"  # Cosine, l2
    exp_settings["mem_store"] = "context"  # Context, embedding, hidden, L2RL

    # Noise Parameters
    # Always flip bit for noise instead of coin flip chance
    exp_settings["perfect_noise"] = False
    exp_settings["noise_type"] = "right_mask"
    """
    apply_noise_types = [
    False,
    "random",
    "left_mask",
    "center_mask",
    "right_mask",
    "checkerboard",]
    """

    # Task Size and Length
    exp_settings["num_arms"] = 0
    exp_settings["barcode_size"] = 0
    exp_settings["num_barcodes"] = 0
    exp_settings["pulls_per_episode"] = 0
    exp_settings["epochs"] = 0

    # Task Complexity
    # What noise percent to apply during eval phase
    exp_settings["noise_percent"] = []
    # How long to spend on a single noise percent eval
    exp_settings["noise_eval_epochs"] = 0
    # What noise percent to apply during training, if any
    exp_settings["noise_train_percent"] = 0
    # Cosine similarity threshold for single clustering
    exp_settings["sim_threshold"] = 0
    # Hamming distance for multi clustering
    exp_settings["hamming_threshold"] = 0

    # Data Logging
    exp_settings["tensorboard_logging"] = False
    ### End of Experimental Parameters ###

    # Forced Hyperparams (found after multiple passes through Bayesian Optimization)
    exp_settings["torch_device"] = "CPU"
    # exp_settings["dim_hidden_a2c"] = int(2**8.644)  # 400
    # exp_settings["dim_hidden_lstm"] = int(2**8.655)  # 403
    # exp_settings["embedder_learning_rate"] = 10**-3.0399  # 9.1e-4
    # exp_settings["embedding_size"] = int(2**8.629)  # 395
    # exp_settings["entropy_error_coef"] = 0.0391
    # exp_settings["lstm_learning_rate"] = 10**-3.332  # 4.66e-4
    # exp_settings["value_error_coef"] = 0.62

    # Task Size specific hyperparams
    # Bayes Opt to mazimize L2RL then Bayes on embedder params

    # 4a8b24s, with noisy bc right mask on init
    exp_settings['dim_hidden_a2c'] = int(2**7.772)
    exp_settings['dim_hidden_lstm'] = int(2**7.772)
    exp_settings['lstm_learning_rate'] = 10**-3.089
    exp_settings['embedding_size'] = int(2**6.236)
    exp_settings['embedder_learning_rate'] = 10**-3.2246
    exp_settings['value_error_coef'] = 0.5986
    exp_settings["entropy_error_coef"] = 0.0368

    # # 6a12b24s, with noisy bc right mask on init
    # exp_settings['dim_hidden_a2c'] = int(2**6.597)
    # exp_settings['dim_hidden_lstm'] = int(2**6.597)
    # exp_settings['lstm_learning_rate'] = 10**-3.8705
    # exp_settings['embedding_size'] = int(2**5.7278)
    # exp_settings['embedder_learning_rate'] = 10**-4.6275
    # exp_settings['value_error_coef'] = 0.4878
    # exp_settings["entropy_error_coef"] = 0.0134

    # # 8a16b24s, with noisy bc right mask on init
    # exp_settings['dim_hidden_a2c'] = int(2**8.9765)
    # exp_settings['dim_hidden_lstm'] = int(2**8.9765)
    # exp_settings['lstm_learning_rate'] = 10**-3.3663
    # exp_settings['value_error_coef'] = 0.0335
    # exp_settings["entropy_error_coef"] = 0.1626
    # exp_settings['embedding_size'] = int(2**4.4617)
    # exp_settings['embedder_learning_rate'] = 10**-4.7065

    # Experimental Variables
    (
        mem_store_types,
        exp_settings["epochs"],
        exp_settings["noise_eval_epochs"],
        exp_settings["noise_train_percent"],
        exp_settings['noise_type'],
        num_repeats,
        file_loc,
    ) = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["pulls_per_episode"],
        exp_settings["sim_threshold"],
        exp_settings["noise_percent"],
    ) = exp_difficulty

    exp_length = exp_settings["epochs"] + exp_settings["noise_eval_epochs"] * len(
        exp_settings["noise_percent"]
    )

    # Safety Assertions
    assert exp_length >= 10, "Total number of epochs must be greater than 10"
    assert (
        exp_settings["pulls_per_episode"] >= 2
    ), "Pulls per episode must be greater than 2"
    assert (
        exp_settings["barcode_size"] > 3 * exp_settings["hamming_threshold"]
    ), "Barcodes must be greater than 3*Hamming"
    assert (
        exp_settings["num_barcodes"] <= 32
    ), "Too many distinct barcodes to display with current selection of labels in T-SNE"
    assert (exp_settings['num_barcodes']%exp_settings['num_arms'] == 0), "Number of barcodes must be a multiple of number of arms"

    ### Beginning of Experimental Runs ###
    epoch_info = np.array(
        [
            exp_settings["epochs"],
            exp_settings["noise_eval_epochs"],
            exp_settings["noise_percent"],
            num_repeats,
            exp_settings['noise_type']
        ],
        dtype=object,
    )

    # Load a pretrained model if there are no training epochs
    exp_settings['load_pretrained_model'] = (exp_settings['epochs'] == 0)

    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    for idx_mem, mem_store in enumerate(mem_store_types):
        tot_rets = np.zeros(exp_length)
        tot_acc = np.zeros(exp_length)
        exp_settings["mem_store"] = mem_store
        exp_name = exp_size + exp_other + f"_{exp_settings['mem_store']}"
        exp_settings["exp_name"] = exp_name

        for i in range(num_repeats):
            print(
                f"\nNew Run --> Iteration: {i} | Exp: {exp_name} | Noise: {exp_settings['noise_type']}")

            # Save tensorboard returns, accuracy, and r-gates for last run of long tests
            exp_settings["tensorboard_logging"] = (
                i == num_repeats - 1 and exp_settings["epochs"] >= 200
            )

            # Save model for future noise evals of different types
            exp_settings['save_model'] = (i == 0 and not exp_settings['load_pretrained_model'])

            logs_for_graphs, loss_logs, key_data = run_experiment_sl(exp_settings)
            log_return, log_embedder_accuracy, epoch_sim_logs = logs_for_graphs
            log_loss_value, log_loss_policy, log_loss_total, embedder_loss = loss_logs
            log_keys, epoch_mapping = key_data

            # Find total averages over all repeats
            tot_rets += log_return / num_repeats
            tot_acc += log_embedder_accuracy / num_repeats

        if exp_length >= 200:
            if exp_settings["epochs"] < 25:
                exp_name += f"_{exp_settings['noise_type']}_noise_eval"

            # Keys will be tensors, and will save keys from only the last run of a repeated run to capture training data
            torch.save(log_keys, "..\\Mem_Store_Project\\data\\" + exp_name + ".pt")    #win
            # torch.save(log_keys, "..//Mem_Store_Project//data//" + exp_name + ".pt")  #ilab

            # Logs will be numpy arrays of returns, accuracy, BC->Arm maps (for the last run of a repetition), and epoch_info
            np.savez(
                "..\\Mem_Store_Project\\data\\" + exp_name,     #win
                # "..//Mem_Store_Project//data//" + exp_name,   #ilab
                tot_rets=tot_rets,
                tot_acc=tot_acc,
                epoch_mapping=epoch_mapping,
                epoch_info=epoch_info,
            )
    ### End of Experiment Data
