import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs

from contextual_choice_sl import run_experiment_sl


def avg_returns(dim_hidden_lstm = 0, lstm_learning_rate = 0, dim_hidden_a2c = 0, 
                value_error_coef = 0, entropy_error_coef = 0,
                embedding_size = 0, embedding_learning_rate = 0):
    exp_settings = {}

    ### Experimental Parameters ###
    exp_settings['randomize'] = False
    exp_settings['perfect_info'] = False
    exp_settings['torch_device'] = 'CPU' 
    exp_settings['exp_name'] = 'bayes_opt'
    exp_settings['save_model'] = True           

    # Task Info
    exp_settings['kernel'] = 'cosine'               
    exp_settings['mem_store'] = 'embedding'

    # Task Complexity
    exp_settings['num_arms'] = 4
    exp_settings['num_barcodes'] = 8
    exp_settings['barcode_size'] = 24
    exp_settings['pulls_per_episode'] = 10
    exp_settings['epochs'] = 500
    exp_settings['hamming_threshold'] = 1

    # Noise eval settings not used
    exp_settings['noise_eval_epochs'] = 100
    exp_settings['noise_percent'] = [0]
    exp_settings['sim_threshold'] = 0
    exp_settings['noise_train_percent'] = 0

    # Data Logging
    exp_settings['tensorboard_logging'] = False
    ### End of Experimental Parameters ###

    # HyperParam Searches for BayesOpt #
    # Using ints in bayes-opt for better performance
    exp_settings['dim_hidden_lstm'] = int(2**dim_hidden_lstm)
    exp_settings['value_error_coef'] = value_error_coef
    exp_settings['entropy_error_coef'] = entropy_error_coef
    exp_settings['lstm_learning_rate'] = 10**lstm_learning_rate
    exp_settings['dim_hidden_a2c'] = int(2**dim_hidden_a2c)
    exp_settings['embedder_learning_rate'] = 10**embedding_learning_rate
    exp_settings['embedding_size'] = int(2**embedding_size)
    #End HyperParam Searches for BayesOpt#

    # Print out current hyperparams to console
    print("\nNext Run Commencing with the following params:")
    print(f"A2C_Size: {exp_settings['dim_hidden_a2c']} | LSTM_Size: {exp_settings['dim_hidden_lstm']} | LSTM_LR: {round(exp_settings['lstm_learning_rate'], 5)}")
    print(f"Emb_LR: {round(exp_settings['embedder_learning_rate'], 5)} | Emb_Size: {exp_settings['embedding_size']}")
    print(f"Val_CF: {round(exp_settings['value_error_coef'], 5)} | Ent_CF: {round(exp_settings['entropy_error_coef'], 5)}")
    
    # Current function being used as maximization target is just avg of total epoch returns
    logs_for_graphs, loss_logs, key_data = run_experiment_sl(exp_settings)
    log_return, log_embedder_accuracy, epoch_sim_logs = logs_for_graphs
    log_loss_value, log_loss_policy, log_loss_total, embedder_loss = loss_logs
    log_keys, epoch_mapping = key_data

    # Focusing only on noiseless eval to maximize training
    no_noise_eval = np.mean(log_return[exp_settings['epochs']:exp_settings['epochs']+exp_settings['noise_eval_epochs']])
    return no_noise_eval
    
# Bounded region of parameter space
pbounds = { 
            'dim_hidden_a2c': (5, 9),               #transformed into 2**x in function
            'dim_hidden_lstm': (5, 9),              #transformed into 2**x in function
            'embedding_learning_rate': (-5, -2),    #transformed into 10**x in function
            'embedding_size': (5,9),                #transformed into 2**x in function
            'lstm_learning_rate': (-5, -2),         #transformed into 10**x in function
            'entropy_error_coef': (0, 0.5),
            'value_error_coef': (0, 0.75),
            }

optimizer = BayesianOptimization(
    f=avg_returns,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)
log_name =  "C:\\Users\\joshc\\Google Drive\\CS Research\\Mem_Store_Project\\logs_4a8n24s1h_500_epochs.json"

# Suspend/Resume Function for longer iterations
load_logs(optimizer, logs=[log_name])
logger = JSONLogger(path=log_name, reset = False)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

optimizer.maximize(
    init_points=1,
    n_iter=1,
)

print(" *-* "*5)    
print(optimizer.max)