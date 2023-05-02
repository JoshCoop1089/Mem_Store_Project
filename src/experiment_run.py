import sys

from contextual_choice_sl import run_experiment

"""
Notes:
Barcode size needs to be at least 3 times as large as hamming_clustering
training_epochs cannot be less than 10
pulls_per_episode has to be more than 2

noise_epochs will be for one level of noise, the program will run through 4 levels of noise automatically
i'm not sure what will happen if num_barcodes isn't an integer multiple of num_arms

"""

###### Change These!!! ###########
# Experiment Type and Length
#context, embedding, L2RL

# exp_types = ['context', 'embedding']
exp_types = ['context', 'embedding', 'L2RL']
# exp_types = ['context']
# exp_types = ['embedding']

try:
    exp_type = [exp_types[int(sys.argv[1])]]
except:
    exp_type = exp_types

training_epochs = 800
noise_epochs = 20
noise_train_percent = 0.2
noise_train_type = 'right_mask'

mem_store_key = 'hidden'
# mem_store_key = 'context'
# mem_store_key = 'full'

emb_loss = 'groundtruth'
# emb_loss = 'kmeans'
# emb_loss = 'contrastive'

emb_with_mem = True

# Experiment Difficulty
hamming_clustering = 1      #Create evenly distributed clusters based on arms/barcodes
sim_threshold = 0           #Create one cluster regardless of arms/barcodes

num_arms = 5
num_barcodes = 10
barcode_size = 20

# num_arms = 8
# num_barcodes = 16
# barcode_size = 40

pulls_per_episode = 10

# try: 
#     mem_limits = (int(sys.argv[1]), pulls_per_episode - int(sys.argv[1]))
# except:
mem_limits = (0,pulls_per_episode)


noise_percent = [4/20]
# noise_percent = [8/40]

# noise_percent = [0/20]
# noise_percent = [8/20]
# noise_percent = [6/24]
# noise_percent = [12/24]
# noise_percent = [10/40]
# noise_percent = [20/40]
noise_types = [
    # False,
    # "right_mask",
    "random",
    # "left_mask",
    # "center_mask",
    # "checkerboard",
    ]

# Randomized seed changes to average for returns graph
num_repeats = 1

exp_diff_general = [hamming_clustering, num_arms, num_barcodes, barcode_size, pulls_per_episode]
exp_diff_specifics = [sim_threshold, noise_percent, mem_limits, mem_store_key, emb_loss, emb_with_mem]

# Modify this to fit your machines save paths
figure_save_location = "..\\Mem_Store_Project\\figs\\"
###### NO MORE CHANGES!!!!!!!! ##########

# Train Model
for noise_type in noise_types:
    exp_base = exp_type, training_epochs, noise_epochs, noise_train_percent, noise_train_type, noise_type, num_repeats
    exp_difficulty = exp_diff_general + exp_diff_specifics
    run_experiment(exp_base, exp_difficulty)
    
# Eval Model on different noise types
training_epochs = 0
noise_epochs = 40
num_repeats = 10

noise_percent = [4/20, 6/20, 8/20, 10/20, 12/20, 14/20]
# noise_percent = [8/40, 12/40, 16/40, 20/40, 24/40, 28/40]

# noise_percent = [0/20, 2/20, 4/20, 6/20, 8/20, 10/20]
# noise_percent = [8/20, 10/20, 12/20, 14/20, 16/20, 18/20]
# noise_percent = [6/24, 8/24, 10/24, 12/24, 14/24, 16/24]
# noise_percent = [10/40, 15/40, 20/40, 25/40, 30/40, 35/40]
# noise_percent = [20/40, 25/40, 30/40, 35/40, 40/40, 45/40]

noise_types = [
    # False,
    # "right_mask",
    "random_no_mem",
    "random",
    # "left_mask",
    # "center_mask",
    # "checkerboard",
    ]

for noise_type in noise_types:
    if 'embedding' in exp_types and "no_mem" in noise_type:
        noise_type = noise_type[:-7]
        emb_with_mem = False
        exp_diff_specifics = [sim_threshold, noise_percent,
                              mem_limits, mem_store_key, emb_loss, emb_with_mem]

    exp_base = exp_type, training_epochs, noise_epochs, noise_train_percent, noise_train_type, noise_type, num_repeats
    exp_difficulty = exp_diff_general + exp_diff_specifics
    run_experiment(exp_base, exp_difficulty)
