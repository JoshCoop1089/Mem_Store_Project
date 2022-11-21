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
#context, embedding, hidden, L2RL
#
# exp_types = ['context', 'embedding','hidden', 'L2RL']
# exp_types = ['context', 'hidden', 'L2RL']
# exp_types = ['context', 'embedding']
# exp_types = ['context']
exp_types = ['embedding']
# exp_types = ['embedding', 'hidden', 'L2RL']
# exp_types = ['hidden']
try:
    exp_type = [exp_types[int(sys.argv[1])]]
except:
    exp_type = exp_types
training_epochs = 500
noise_epochs = 50
noise_train_percent = 0.25

# Experiment Difficulty
hamming_clustering = 1     #Create evenly distributed clusters based on arms/barcodes
sim_threshold = 0           #Create one cluster regardless of arms/barcodes
num_arms = 4
num_barcodes = 8
barcode_size = 24
pulls_per_episode = 10
noise_percent = [0, 0.25, 0.5, 0.75]
noise_types = [
    # False,
    # "random",
    # "left_mask",
    # "center_mask",
    "right_mask",
    # "checkerboard",
    ]

# noise_percent = [0]

# Randomized seed changes to average for returns graph
num_repeats = 1

# Modify this to fit your machines save paths
figure_save_location = "..\\Mem_Store_Project\\figs\\"
###### NO MORE CHANGES!!!!!!!! ##########

for noise_type in noise_types:
    exp_base = exp_type, training_epochs, noise_epochs, noise_train_percent, noise_type, num_repeats, figure_save_location
    exp_difficulty = hamming_clustering, num_arms, num_barcodes, barcode_size, pulls_per_episode, sim_threshold, noise_percent
    run_experiment(exp_base, exp_difficulty)
