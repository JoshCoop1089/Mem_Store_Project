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
    dropout_coef=0,
    loss_switch=0,
    embedding_learning_rate_cont=0
):
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
    # exp_settings['mem_mode'] = "one_layer"

    # exp_settings['emb_loss'] = 'contrastive'
    exp_settings['emb_loss'] = 'kmeans'

    exp_settings['emb_with_mem'] = True

    # Task Complexity
    exp_settings["num_arms"] = 5
    exp_settings["num_barcodes"] = 10
    exp_settings["barcode_size"] = 20
    exp_settings["epochs"] = 600
    exp_settings["noise_eval_epochs"] = 30
    exp_settings['dropout_coef'] = 0

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
    ### End of Experimental Parameters ###

    # HyperParam Searches for BayesOpt #
    # Using ints in bayes-opt for better performance
    # *** Forcing A2C and LSTM dimensions to be the same ***
    # exp_settings["dim_hidden_a2c"] = int(2**dim_hidden_lstm)
    # exp_settings["dim_hidden_lstm"] = int(2**dim_hidden_lstm)
    # exp_settings["lstm_learning_rate"] = 10**lstm_learning_rate
    # exp_settings["value_error_coef"] = value_error_coef
    # exp_settings['entropy_error_coef'] = entropy_error_coef
    exp_settings['embedder_learning_rate'] = 10**embedding_learning_rate
    exp_settings['embedder_learning_rate_second_phase'] = 10**embedding_learning_rate_cont
    exp_settings['embedding_size'] = int(2**embedding_size)
    # exp_settings['dropout_coef'] = dropout_coef
    # End HyperParam Searches for BayesOpt#

    # Static vals for L2RL testing
    # 5a5b10s 800 epoch noise init 20 percen right mask
    exp_settings['dim_hidden_a2c'] = int(2**8.7326)
    exp_settings['dim_hidden_lstm'] = int(2**8.7326)
    exp_settings['lstm_learning_rate'] = 10**-3.2345
    exp_settings['value_error_coef'] = .90238
    exp_settings["entropy_error_coef"] = 0.03658

    # # 5a10b20s 1000 epoch noise init 20 percent right mask shuffle 0.7111 target #3
    # exp_settings['dim_hidden_a2c'] = int(2**7.117)
    # exp_settings['dim_hidden_lstm'] = int(2**7.117)
    # exp_settings['lstm_learning_rate'] = 10**-3.0818
    # exp_settings['value_error_coef'] = .8046
    # exp_settings["entropy_error_coef"] = 0.0446

    # # 10a20b40s 3000 epoch noise init 25 percent right mask
    # exp_settings['dim_hidden_a2c'] = int(2**5.514)
    # exp_settings['dim_hidden_lstm'] = int(2**5.514)
    # exp_settings['lstm_learning_rate'] = 10**-3.4478
    # exp_settings['value_error_coef'] = 0.75
    # exp_settings["entropy_error_coef"] = 0

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
            f"Emb_LR: {round(exp_settings['embedder_learning_rate'], 5)} | Emb_Size: {exp_settings['embedding_size']} | Contrastive LR: {round(exp_settings['embedder_learning_rate_second_phase'], 5)}"
        )
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
    # 'dim_hidden_a2c': (4, 8),                 #transformed into 2**x in function
    'embedding_learning_rate': (-5, -3),        #transformed into 10**x in function
    'embedding_learning_rate_cont': (-5, -3),   #transformed into 10**x in function
    'embedding_size': (4,9),                    #transformed into 2**x in function
    'loss_switch': (0.2,0.5)                    #Percent of training to switch to contrastive learning
    # 'dropout_coef': (0,0.5),
    # "dim_hidden_lstm": (4, 9),  # transformed into 2**x in function
    # 'entropy_error_coef': (0, 0.2),
    # "lstm_learning_rate": (-5, -3),  # transformed into 10**x in function
    # "value_error_coef": (0, 1),
}

optimizer = BayesianOptimization(
    f=avg_returns,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

# log_name = './logs_5a10b20s1h_1000_epochs_noisy_init_20_right.json'
# log_name = './logs_5a10b20s1h_600_epochs_noisy_init_10_right.json'
# log_name = './logs_5a10b20s1h_800_epochs_embedder_lstm_noisy_init_10_right.json'
log_name = './logs_5a5b10s1h_800_epochs_embedder_noisy_init_20_right_two_phase.json'

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
