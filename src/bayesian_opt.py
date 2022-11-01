import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs

from contextual_choice_sl import run_experiment_sl


def avg_returns(
    dim_hidden_lstm=0,
    lstm_learning_rate=0,
    dim_hidden_a2c=0,
    value_error_coef=0,
    entropy_error_coef=0,
    embedding_size=0,
    embedding_learning_rate=0,
):
    exp_settings = {}

    ### Experimental Parameters ###
    exp_settings["randomize"] = False
    exp_settings["perfect_info"] = False
    exp_settings["torch_device"] = "CPU"

    # Task Info
    exp_settings["kernel"] = "cosine"
    exp_settings["mem_store"] = "L2RL"

    # Task Complexity
    exp_settings["num_arms"] = 2
    exp_settings["num_barcodes"] = 4
    exp_settings["barcode_size"] = 24
    exp_settings["epochs"] = 300
    exp_settings["hamming_threshold"] = 1
    exp_settings["noise_eval_epochs"] = 50
    # exp_settings['embedder_training'] = 'arms'
    exp_settings["embedder_training"] = "barcodes"

    # Not really manipulated during bayes
    exp_settings["pulls_per_episode"] = 10
    exp_settings["noise_percent"] = [0]
    exp_settings["sim_threshold"] = 0
    exp_settings["noise_train_percent"] = 0

    # Data Logging
    exp_settings["tensorboard_logging"] = False
    ### End of Experimental Parameters ###

    # HyperParam Searches for BayesOpt #
    # Using ints in bayes-opt for better performance
    exp_settings["dim_hidden_a2c"] = int(
        2**dim_hidden_lstm
    )  # *** Forcing A2C and LSTM dimensions to be the same ***
    exp_settings["dim_hidden_lstm"] = int(2**dim_hidden_lstm)
    exp_settings["lstm_learning_rate"] = 10**lstm_learning_rate
    exp_settings["value_error_coef"] = value_error_coef
    # exp_settings['entropy_error_coef'] = entropy_error_coef
    # exp_settings['embedder_learning_rate'] = 10**embedding_learning_rate
    # exp_settings['embedding_size'] = int(2**embedding_size)
    # End HyperParam Searches for BayesOpt#

    # Static vals for L2RL testing
    # exp_settings['dim_hidden_a2c'] = int(2**6.5423)
    # exp_settings['dim_hidden_lstm'] = int(2**6.5423)
    # exp_settings['lstm_learning_rate'] = 10**-3
    # exp_settings['value_error_coef'] = 0.3887

    # exp_settings['embedder_learning_rate'] = 1e-5
    # exp_settings['embedding_size'] = 128
    exp_settings[
        "entropy_error_coef"
    ] = 1  # Annealing Entropy values from 1 to 0 linearly over training

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
        f"Val_CF: {round(exp_settings['value_error_coef'], 5)} | Ent_CF: {round(exp_settings['entropy_error_coef'], 5)}"
    )

    # Current function being used as maximization target is just avg of total epoch returns
    logs_for_graphs, loss_logs, key_data = run_experiment_sl(exp_settings)
    log_return, log_embedder_accuracy, epoch_sim_logs = logs_for_graphs
    log_loss_value, log_loss_policy, log_loss_total, embedder_loss = loss_logs
    log_keys, epoch_mapping = key_data

    # Focusing only on noiseless eval to maximize training
    no_noise_eval = np.mean(
        log_return[
            exp_settings["epochs"] : exp_settings["epochs"]
            + exp_settings["noise_eval_epochs"]
        ]
    )
    no_noise_accuracy = np.mean(
        log_embedder_accuracy[
            exp_settings["epochs"] : exp_settings["epochs"]
            + exp_settings["noise_eval_epochs"]
        ]
    )

    if exp_settings["mem_store"] == "embedding":
        bayes_target = round(no_noise_eval * no_noise_accuracy, 5)
    else:
        bayes_target = round(no_noise_eval, 5)

    print(
        f"Bayes: {round(bayes_target,3)} | Returns: {round(no_noise_eval,3)} | Accuracy: {round(no_noise_accuracy,3)}"
    )
    return bayes_target


# Bounded region of parameter space
pbounds = {
    # 'dim_hidden_a2c': (4, 8),               #transformed into 2**x in function
    "dim_hidden_lstm": (4, 8),  # transformed into 2**x in function
    # 'embedding_learning_rate': (-5, -3),    #transformed into 10**x in function
    # 'embedding_size': (4,9),                #transformed into 2**x in function
    # 'entropy_error_coef': (0, 0.5),
    "lstm_learning_rate": (-5, -3),  # transformed into 10**x in function
    "value_error_coef": (0, 0.75),
}

optimizer = BayesianOptimization(
    f=avg_returns,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)
log_name = "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_2a4n24s1h_300_epochs_entropy_annealed_loss_summed.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_2a4n24s1h_300_epochs.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_2a4n24s1h_600_epochs_arm.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs_L2RL.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs_emb.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_6a12n24s1h_750_epochs.json"

# 4a8b24s l2rl
# {"target": 0.77119, "params": {"dim_hidden_lstm": 6.5422972373862756, "lstm_learning_rate": -3.0,
# "value_error_coef": 0.3887431851179297}


# Suspend/Resume Function for longer iterations
try:
    load_logs(optimizer, logs=[log_name])
except:
    pass
logger = JSONLogger(path=log_name, reset=False)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

optimizer.maximize(
    init_points=5,
    n_iter=40,
)

print(" *-* " * 5)
print(optimizer.max)
