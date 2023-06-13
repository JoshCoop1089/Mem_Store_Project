import numpy as np
import sys
from contextual_choice_sl import run_experiment_sl

# [dim_hidden_lstm, lstm_lr, val_cf, ent_cf]
test_vals = [
[8.6357,    -3.501,     0.7177, 0.0004],
[8.11,      -3,         0.4495, 0.0],
[9,         -3,         0.7441, 0.0865],
[6.597,     -3.8705,    0.4878, 0.0134],
[7.479,     -3.6955,    1,      0],
[6.9958,    -3.0788,    0.9407, 0.006],
[7.0673,    -3.4393,    1.0,    0.0],
[7.418,     -3.4857,    0.5408, 0.0],
[6.42929,   -3.32629,   0.2146, 0.04],
[7.117,     -3.0818,    0.8046, 0.0446],
[8.4056,    -3.017,     0.9885, 0.049],
[7.2753,    -3.5947,    0.907,  0.04356],
[6.0873,    -3.4539,    1.0,    0.09921]]

# [raw_emb_size, raw_emb_lr, loss_switch]
test_vals = [(4+0.5*x, -5+0.2*x, 0+0.05*x) for x in range(11)]
# test_vals = [(4+0.5*x, -3-0.2*x, 0.5-0.05*x) for x in range(11)]

def run_exp(input_val):
    exp_settings = {}

    ### Experimental Parameters ###
    exp_settings["randomize"] = False
    exp_settings["perfect_info"] = False
    exp_settings['perfect_noise'] = False
    exp_settings["torch_device"] = "GPU"
    exp_settings["load_pretrained_model"] = False
    exp_settings["save_model"] = False

    # Task Info
    exp_settings["kernel"] = "cosine"
    # exp_settings["mem_store"] = "L2RL"
    exp_settings["mem_store"] = "embedding"

    exp_settings['mem_mode'] = "LSTM"

    # exp_settings['emb_loss'] = 'kmeans'
    exp_settings['emb_loss'] = 'contrastive'
    exp_settings['switch_to_contrastive'] = True

    # Task Complexity
    exp_settings["num_arms"] = 10
    exp_settings["num_barcodes"] = 10
    exp_settings["barcode_size"] = 20
    exp_settings["epochs"] = 1000
    exp_settings["noise_eval_epochs"] = 50
    exp_settings['dropout_coef'] = 0
    exp_settings['emb_with_mem'] = True

    # Noise Complexity
    exp_settings["hamming_threshold"] = 1
    exp_settings["pulls_per_episode"] = 10
    exp_settings["noise_percent"] = [0.2]
    exp_settings["noise_type"] = "random"
    exp_settings["noise_train_percent"] = 0.2
    exp_settings['noise_train_type'] = 'right_mask'
    exp_settings['perfect_noise'] = False

    # Data Logging
    exp_settings["tensorboard_logging"] = False

    # Naming convention automation
    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name = exp_size + exp_other + f"_{exp_settings['mem_store']}"
    if exp_settings['mem_store'] == 'embedding':
        exp_name += f"_{exp_settings['mem_mode']}_{exp_settings['emb_loss']}"
    exp_settings["exp_name"] = exp_name
    ### End of Experimental Parameters ###

    # # HyperParam Searches for BayesOpt #
    # raw_lstm, raw_lr, raw_val_cf, raw_ent_cf = input_val
    # exp_settings['dim_hidden_a2c'] = int(2**raw_lstm)
    # exp_settings['dim_hidden_lstm'] = int(2**raw_lstm)
    # exp_settings['lstm_learning_rate'] = 10**raw_lr
    # exp_settings['value_error_coef'] = raw_val_cf
    # exp_settings["entropy_error_coef"] = raw_ent_cf
    # exp_settings['embedder_learning_rate'] = 10**-3
    # exp_settings['embedding_size'] = int(2**7)

    # # 5a5b10s
    # exp_settings['dim_hidden_a2c'] = int(2**8.7326)
    # exp_settings['dim_hidden_lstm'] = int(2**8.7326)
    # exp_settings['lstm_learning_rate'] = 10**-3.2345
    # exp_settings['value_error_coef'] = 0.9024
    # exp_settings["entropy_error_coef"] = 0.03658

    # # # 10a10b10s
    # exp_settings['dim_hidden_a2c'] = int(2**8.6357)
    # exp_settings['dim_hidden_lstm'] = int(2**8.6357)
    # exp_settings['lstm_learning_rate'] = 10**-3.501
    # exp_settings['value_error_coef'] = 0.7177
    # exp_settings["entropy_error_coef"] = 0.0004

    # 10a10b20s
    exp_settings['dim_hidden_a2c'] = int(2**8.11)
    exp_settings['dim_hidden_lstm'] = int(2**8.11)
    exp_settings['lstm_learning_rate'] = 10**-3.0788
    exp_settings['value_error_coef'] = 0.4495
    exp_settings["entropy_error_coef"] = 0.00
    exp_settings['embedding_size'] = int(2**7.27899)

    raw_emb, raw_emb_lr, loss_switch = input_val
    exp_settings['embedder_learning_rate'] = 10**raw_emb_lr
    # exp_settings['embedding_size'] = int(2**raw_emb)
    # exp_settings['dropout_coef'] = loss_switch

    # Print out current hyperparams to console
    print("\nNext Run Commencing with the following params:")
    print(
        f"Arms: {exp_settings['num_arms']} | Barcodes: {exp_settings['num_barcodes']} | Size: {exp_settings['barcode_size']} | Mem: {exp_settings['mem_store']}"
    )
    print(
        f"A2C_Size: {exp_settings['dim_hidden_a2c']} | LSTM_Size: {exp_settings['dim_hidden_lstm']} | LSTM_LR: {round(exp_settings['lstm_learning_rate'], 5)}"
    )
    if exp_settings["mem_store"] == "embedding":
        print(
            f"Emb_LR: {round(exp_settings['embedder_learning_rate'], 5)} | Emb_Size: {exp_settings['embedding_size']}"
        )
        print(
            f"Memory Mode: {exp_settings['mem_mode']} | Emb Loss: {exp_settings['emb_loss']}")
    print(
        f"Val_CF: {round(exp_settings['value_error_coef'], 5)} | Ent_CF: {round(exp_settings['entropy_error_coef'], 5)}"
    )

    # Current function being used as maximization target is just avg of total epoch returns
    logs_for_graphs, loss_logs, key_data = run_experiment_sl(exp_settings)
    log_return, log_memory_accuracy, log_embedder_accuracy = logs_for_graphs
    log_loss_value, log_loss_policy, log_loss_total, embedder_loss, contrastive_loss = loss_logs
    log_keys, epoch_mapping = key_data

    # Focusing only on noiseless eval to maximize training
    no_noise_eval = np.mean(
        log_return[
            exp_settings["epochs"] : exp_settings["epochs"]
            + exp_settings["noise_eval_epochs"]
        ]
    )
    no_noise_accuracy = np.mean(
        log_memory_accuracy[
            exp_settings["epochs"] : exp_settings["epochs"]
            + exp_settings["noise_eval_epochs"]
        ]
    )

    # out_text = f"('target': {no_noise_eval}, 'params': ('dim_hidden_lstm': {raw_lstm}, 'entropy_error_coef': {raw_ent_cf}, 'lstm_learning_rate': {raw_lr}, 'value_error_coef': {raw_val_cf}))" 
    # out_text = f"('target': {no_noise_eval*no_noise_accuracy}, 'params': ('embedding_learning_rate': {raw_emb_lr}, 'embedding_size': {raw_emb}, 'dropout_coef': {loss_switch}))" 
    out_text = f"('target': {no_noise_eval*no_noise_accuracy}, 'params': ('embedding_learning_rate': {raw_emb_lr}))" 
    out_text_changed = out_text.replace("(","{").replace(")", "}").replace("'", "\"")
    print(out_text_changed)

# run_exp(test_vals[int(sys.argv[1])])
run_exp(test_vals[0])