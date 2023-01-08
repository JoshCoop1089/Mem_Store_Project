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
    dropout_coef=0
):
    exp_settings = {}

    ### Experimental Parameters ###
    exp_settings["randomize"] = False
    exp_settings["perfect_info"] = False
    exp_settings['perfect_noise'] = False
    exp_settings["torch_device"] = "CPU"
    exp_settings["load_pretrained_model"] = False
    exp_settings["save_model"] = False

    # Task Info
    exp_settings["kernel"] = "cosine"
    # exp_settings["mem_store"] = "L2RL"
    exp_settings["mem_store"] = "embedding"

    # Task Complexity
    exp_settings["num_arms"] = 5
    exp_settings["num_barcodes"] = 10
    exp_settings["barcode_size"] = 20
    exp_settings["epochs"] = 500
    exp_settings["noise_eval_epochs"] = 20
    exp_settings["emb_mem_limits"] = (1,9)
    exp_settings['dropout_coef'] = 0

    # Noise Complexity
    exp_settings["hamming_threshold"] = 1
    exp_settings["pulls_per_episode"] = 10
    exp_settings["noise_percent"] = [4/20,6/20,8/20]
    exp_settings["sim_threshold"] = 0
    exp_settings["noise_type"] = "right_mask"
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
    exp_settings['embedding_size'] = int(2**embedding_size)
    exp_settings['dropout_coef'] = dropout_coef
    # End HyperParam Searches for BayesOpt#

    # Static vals for L2RL testing
    # 4a8b24s 500 epoch no noise init
    # exp_settings['dim_hidden_a2c'] = int(2**6.711)
    # exp_settings['dim_hidden_lstm'] = int(2**6.711)
    # exp_settings['lstm_learning_rate'] = 10**-3
    # exp_settings['value_error_coef'] = 0.75

    # # 4a8b24s 500 epoch noise init 25 percent right mask shuffle (0.76256 returns)
    # exp_settings['dim_hidden_a2c'] = int(2**8.6357)
    # exp_settings['dim_hidden_lstm'] = int(2**8.6357)
    # exp_settings['lstm_learning_rate'] = 10**-3.501
    # exp_settings['value_error_coef'] = 0.7177
    # exp_settings['entropy_error_coef'] = 0.0004

    # # 4a8b24s 1500 epoch noise init 50 percent right mask shuffle
    # exp_settings['dim_hidden_a2c'] = 77
    # exp_settings['dim_hidden_lstm'] = 77
    # exp_settings['lstm_learning_rate'] = 0.00026
    # exp_settings['value_error_coef'] = 0.73822

    # # 4a8b40s 1000 epoch noise init 50 percent right mask shuffle
    # exp_settings['dim_hidden_a2c'] = int(2**9)
    # exp_settings['dim_hidden_lstm'] = int(2**9)
    # exp_settings['lstm_learning_rate'] = 10**-3
    # exp_settings['value_error_coef'] = 0.7441
    # exp_settings["entropy_error_coef"] = 0.0865

    # # 5a10b20s 1000 epoch noise init 20 percent right mask shuffle 0.72054 target #1
    # exp_settings['dim_hidden_a2c'] = int(2**5.396)
    # exp_settings['dim_hidden_lstm'] = int(2**5.396)
    # exp_settings['lstm_learning_rate'] = 10**-3.603
    # exp_settings['value_error_coef'] = 1
    # exp_settings["entropy_error_coef"] = 0

    # # 5a10b20s 1000 epoch noise init 20 percent right mask shuffle 0.7152 target #2
    # exp_settings['dim_hidden_a2c'] = int(2**6.6638)
    # exp_settings['dim_hidden_lstm'] = int(2**6.6638)
    # exp_settings['lstm_learning_rate'] = 10**-3.0185
    # exp_settings['value_error_coef'] = .5535
    # exp_settings["entropy_error_coef"] = 0.008

    # 5a10b20s 1000 epoch noise init 20 percent right mask shuffle 0.7111 target #3
    exp_settings['dim_hidden_a2c'] = int(2**7.117)
    exp_settings['dim_hidden_lstm'] = int(2**7.117)
    exp_settings['lstm_learning_rate'] = 10**-3.0818
    exp_settings['value_error_coef'] = .8046
    exp_settings["entropy_error_coef"] = 0.0446

    # # 6a12b24s 1000 epoch noise init 25 percent right mask
    # exp_settings['dim_hidden_a2c'] = int(2**6.597)
    # exp_settings['dim_hidden_lstm'] = int(2**6.597)
    # exp_settings['lstm_learning_rate'] = 10**-3.8705
    # exp_settings['value_error_coef'] = 0.4878
    # exp_settings["entropy_error_coef"] = 0.0134

    # # 6a12b40s 1500 epoch noise init 25 percent right mask
    # exp_settings['dim_hidden_a2c'] = int(2**6.0271)
    # exp_settings['dim_hidden_lstm'] = int(2**6.0271)
    # exp_settings['lstm_learning_rate'] = 10**-3.8511
    # exp_settings['value_error_coef'] = 0.8928
    # exp_settings["entropy_error_coef"] = 0.10297

    # # 6a12b40s 1500 epoch noise init 50 percent right mask
    # exp_settings['dim_hidden_a2c'] = int(2**7.479)
    # exp_settings['dim_hidden_lstm'] = int(2**7.479)
    # exp_settings['lstm_learning_rate'] = 10**-3.6955
    # exp_settings['value_error_coef'] = 1
    # exp_settings["entropy_error_coef"] = 0

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
            f"Emb_LR: {round(exp_settings['embedder_learning_rate'], 5)} | Emb_Size: {exp_settings['embedding_size']} | Dropout: {round(exp_settings['dropout_coef'], 3)}"
        )
    print(
        f"Val_CF: {round(exp_settings['value_error_coef'], 5)} | Ent_CF: {round(exp_settings['entropy_error_coef'], 5)}"
    )

    # Current function being used as maximization target is just avg of total epoch returns
    logs_for_graphs, loss_logs, key_data = run_experiment_sl(exp_settings)
    log_return, log_memory_accuracy, epoch_sim_logs, log_embedder_accuracy = logs_for_graphs
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
        
    start = exp_settings["epochs"] 
    end = start + exp_settings["noise_eval_epochs"]
    bayes_target = 0
    if exp_settings["mem_store"] == "embedding":
        for idx in range(len(exp_settings['noise_percent'])):
            # Weights: 1,1,2 for the rest
            weight = 1 if idx < 2 else 2
            noise_eval = np.mean(log_return[start:end])
            noise_model_acc = np.mean(log_embedder_accuracy[start:end])
            noise_mem_acc = np.mean(log_memory_accuracy[start:end])
            start = end
            end += exp_settings["noise_eval_epochs"]
            bayes_target += weight*round(noise_eval * noise_model_acc * noise_mem_acc, 5)
    else:
        bayes_target = round(no_noise_eval, 5)

    print(
        f"Bayes: {round(bayes_target,3)} | Returns: {round(no_noise_eval,3)} | Accuracy: {round(no_noise_accuracy,3)}"
    )
    return bayes_target


# Bounded region of parameter space
pbounds = {
    # 'dim_hidden_a2c': (4, 8),               #transformed into 2**x in function
    'embedding_learning_rate': (-5, -3),    #transformed into 10**x in function
    'embedding_size': (4,9),                #transformed into 2**x in function
    'dropout_coef': (0,0.5),
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
log_name = './logs_5a10b20s1h_1000_epochs_noisy_init_40_right.json'
# log_name = './logs_5a10b20s1h_1000_epochs_embedder_noisy_init_20_right.json'
# log_name = './logs_5a10b20s1h_1000_epochs_embedder_noisy_init_20_right_2.json'
# log_name = './logs_5a10b20s1h_1000_epochs_embedder_noisy_init_20_right_3.json'
# log_name = './logs_5a10b20s1h_1000_epochs_embedder_noisy_init_20_right_mem_recall_trunc_loss.json'
# log_name = './logs_5a10b20s1h_1000_epochs_embedder_noisy_init_20_right_double_layer_emb.json'
# log_name = './logs_5a10b20s1h_1000_epochs_embedder_noisy_init_20_right_double_layer_emb 19mem variable dropout.json'
# log_name = './logs_10a20n40s1h_2000_epochs_noisy_init_25_right.json'
# log_name = './logs_10a20n40s1h_3000_epochs_embedder_noisy_init_25_right.json'
# log_name = './logs_6a12b24s1h_1250_epochs_embedder_noisy_init_25_right.json'
# log_name = './logs_6a12b40s1h_2000_epochs_noisy_init_25_right.json'
# log_name = './logs_6a12b40s1h_1500_epochs_embedder_noisy_init_25_right.json'
# log_name = './logs_6a12b40s1h_1500_epochs_noisy_init_50_right.json'
# log_name = './logs_6a12b40s1h_1500_epochs_embedder_noisy_init_50_right.json'
# log_name = './logs_4a8b40s1h_750_epochs_noisy_init_25_right.json'
# log_name = './logs_4a8b40s1h_750_epochs_embedder_noisy_init_25_right.json'
# log_name = './logs_4a8b40s1h_1000_epochs_noisy_init_50_right.json'
# log_name = './logs_4a8b40s1h_1000_epochs_embedder_noisy_init_50_right.json'
# log_name = './logs_4a8b24s1h_500_epochs_noisy_init_25_right.json'
# log_name = './logs_4a8b24s1h_500_epochs_embedder_noisy_init_25_right.json'
# log_name = "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs_l2rl_noisy_bc_init.json"
# log_name = "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_1500_epochs_embedder_noisy_bc_50_init.json"
# log_name = "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_1500_epochs_embedder1_noisy_bc_50_init.json"
# log_name = "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs_embedder_entropy_annealed_loss_summed.json"
# log_name = "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_2a4n24s1h_300_epochs_entropy_annealed_loss_summed.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_2a4n24s1h_300_epochs.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_2a4n24s1h_600_epochs_arm.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs_L2RL.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs_emb.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs.json"
# log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_6a12n24s1h_750_epochs.json"
log_name = "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_5a10b20s1h_500_epochs_noisy_init_20_right_multi_noise_bayes.json"

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
    n_iter=50,
)

print(" *-* " * 5)
print(optimizer.max)
