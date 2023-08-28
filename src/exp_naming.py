from contextual_choice_sl import run_experiment_sl
from hyperparams import get_hyperparameters
import numpy as np
import torch

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
    # Make arms 100%/0% reward instead of 90%/10%
    exp_settings["perfect_info"] = False
    exp_settings["torch_device"] = "CPU"  # 'CPU' or 'GPU'
    exp_settings["load_pretrained_model"] = False

    # Task Info
    exp_settings["kernel"] = "cosine"
    exp_settings["mem_store"] = "context"  # Context, embedding, L2RL

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
    exp_settings['noise_train_type'] = "right_mask"
    """
    Noise Train Types: right_mask, left_mask, none
    """

    # Hamming distance for multi clustering
    exp_settings["hamming_threshold"] = 0

    # Mem Modes: LSTM, one_layer, two_layer
    exp_settings['mem_mode'] = 'LSTM'

    # Loss Types for Embedder
    # exp_settings['emb_loss'] = 'contrastive'
    exp_settings['emb_loss'] = 'kmeans'
    # exp_settings['emb_loss'] = 'groundtruth'

    # Evaluate Emb Model without Mem
    exp_settings['emb_with_mem'] = True

    # Data Logging
    exp_settings["tensorboard_logging"] = False
    ### End of Experimental Parameters ###

    # Forced Hyperparams (found after multiple passes through Bayesian Optimization)
    # exp_settings["torch_device"] = "CPU"
    exp_settings["torch_device"] = "GPU"
    exp_settings["dim_hidden_a2c"] = int(2**8.644)  # 400
    exp_settings["dim_hidden_lstm"] = int(2**8.655)  # 403
    exp_settings["embedder_learning_rate"] = 10**-3.0399  # 9.1e-4
    exp_settings["embedding_size"] = int(2**8.629)  # 395
    exp_settings["entropy_error_coef"] = 0.0391
    exp_settings["lstm_learning_rate"] = 10**-3.332  # 4.66e-4
    exp_settings["value_error_coef"] = 0.62
    exp_settings["dropout_coef"] = 0.25
    
    # Contrastive Loss Tweaks
    exp_settings['contrastive_chunk_size'] = 10
    exp_settings['neg_pairs_only'] = False

    # Experimental Variables
    (
        mem_store_types,
        exp_settings["epochs"],
        exp_settings["noise_eval_epochs"],
        exp_settings["noise_train_percent"],
        exp_settings['noise_train_type'],
        exp_settings['noise_type'],
        num_repeats,
    ) = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["pulls_per_episode"],
        exp_settings["noise_percent"],
        exp_settings['emb_loss'],
        exp_settings['emb_with_mem'],
        exp_settings['switch_to_contrastive']
    ) = exp_difficulty

    # Task Size specific hyperparams
    exp_settings = get_hyperparameters(exp_settings)
    exp_settings['contrastive_size'] = 2**7

    if exp_settings['switch_to_contrastive']:
        exp_settings['embedder_learning_rate'] = 0.0005

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
    assert (exp_settings['num_barcodes'] % exp_settings['num_arms']
            == 0), "Number of barcodes must be a multiple of number of arms"

    ### Beginning of Experimental Runs ###

    # Info to be saves along with raw data outputs
    epoch_info = np.array(
        [
            exp_settings["epochs"],
            exp_settings["noise_eval_epochs"],
            exp_settings["noise_percent"],
            num_repeats,
            exp_settings['noise_type'],
            exp_settings['noise_train_percent'],
            exp_settings['mem_mode'],
            exp_settings['emb_loss'],
            exp_settings['emb_with_mem'],
        ],
        dtype=object,
    )

    # Load a pretrained model if there are no training epochs
    exp_settings['load_pretrained_model'] = (exp_settings['epochs'] == 0)

    # Naming convention automation
    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"

    # Iterate over every distinct type of memory, with repetition for reduced randomness
    for idx_mem, mem_store in enumerate(mem_store_types):
        tot_rets = np.zeros(exp_length)
        tot_acc = np.zeros(exp_length)
        tot_emb_acc = np.zeros(exp_length)
        tot_emb_loss = np.zeros(exp_length)
        tot_cont_loss = np.zeros(exp_length)
        tot_cont_pos_loss = np.zeros(exp_length)
        tot_cont_neg_loss = np.zeros(exp_length)

        exp_settings["mem_store"] = mem_store
        exp_name = exp_size + exp_other + f"_{exp_settings['mem_store']}"
        if exp_settings['mem_store'] == 'embedding':
            exp_name += f"_{exp_settings['mem_mode']}_{exp_settings['emb_loss']}"

        exp_settings["exp_name"] = exp_name

        # Print out current hyperparams to console
        print("\nNext Run Commencing with the following params:")
        print(
            f"Arms: {exp_settings['num_arms']} | Barcodes: {exp_settings['num_barcodes']} | Size: {exp_settings['barcode_size']} | Mem: {exp_settings['mem_store']}"
        )
        print(
            f"A2C_Size: {exp_settings['dim_hidden_a2c']} | LSTM_Size: {exp_settings['dim_hidden_lstm']} | LSTM_LR: {round(exp_settings['lstm_learning_rate'], 5)}"
        )
        print(
            f"Val_CF: {round(exp_settings['value_error_coef'], 5)} | Ent_CF: {round(exp_settings['entropy_error_coef'], 5)}"
        )
        if exp_settings["mem_store"] == "embedding":
            print(
                f"Emb_LR: {round(exp_settings['embedder_learning_rate'], 5)} | Emb_Size: {exp_settings['embedding_size']}"
            )
            print(
                f"Memory Mode: {exp_settings['mem_mode']} | Emb Loss: {exp_settings['emb_loss']}")
            print(f"Contrastive Loss Neg Pair Only: {exp_settings['neg_pairs_only']} | Con Loss Episodes used: {exp_settings['contrastive_chunk_size']}")


        for i in range(num_repeats):
            print(
                f"\nNew Run --> Iteration: {i} | Exp: {exp_name} | Noise: {exp_settings['noise_type']}")

            # Save tensorboard returns, accuracy, and r-gates for last run of long tests
            exp_settings["tensorboard_logging"] = (
                i == num_repeats - 1 and exp_length >= 200
            )

            # Save model for future noise evals of different types
            exp_settings['save_model'] = (
                i == 0 and not exp_settings['load_pretrained_model'])

            logs_for_graphs, loss_logs, key_data = run_experiment_sl(
                exp_settings)
            log_return, log_memory_accuracy, log_embedder_accuracy = logs_for_graphs
            log_loss_value, log_loss_policy, log_loss_total, embedder_loss, contrastive_loss = loss_logs
            log_keys, epoch_mapping = key_data

            # Find total averages over all repeats
            tot_rets += log_return / num_repeats
            tot_acc += log_memory_accuracy / num_repeats
            tot_emb_acc += log_embedder_accuracy / num_repeats
            tot_emb_loss += embedder_loss / num_repeats
            tot_cont_loss += contrastive_loss[0] / num_repeats
            tot_cont_pos_loss += contrastive_loss[1] / num_repeats
            tot_cont_neg_loss += contrastive_loss[2] / num_repeats

        if exp_length >= 200:
            if exp_settings["epochs"] < 25:
                exp_name += f"_{exp_settings['noise_type']}_noise_eval"
                if not exp_settings['emb_with_mem']:
                    exp_name += "_no_mem"

            # Keys will be tensors, and will save keys from only the last run of a repeated run to capture training data
            torch.save(log_keys, "..\\Mem_Store_Project\\data\\" + exp_name + ".pt")  # win
            # torch.save(log_keys, "..//Mem_Store_Project//data//" + exp_name + ".pt")  #ilab

            # Logs will be numpy arrays of returns, accuracy, BC->Arm maps (for the last run of a repetition), and epoch_info
            np.savez(
                "..\\Mem_Store_Project\\data\\" + exp_name,  # win
                # "..//Mem_Store_Project//data//" + exp_name,   #ilab
                tot_rets=tot_rets,
                tot_acc=tot_acc,
                epoch_mapping=epoch_mapping,
                epoch_info=epoch_info,
                tot_emb_acc=tot_emb_acc,
                tot_emb_loss=tot_emb_loss,
                tot_cont_loss=tot_cont_loss,
                tot_cont_pos_loss=tot_cont_pos_loss,
                tot_cont_neg_loss=tot_cont_neg_loss,
            )
    ### End of Experiment Data
