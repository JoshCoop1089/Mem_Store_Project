from contextual_choice_sl import run_experiment
import sys

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
# exp_types = ['context','embedding', 'hidden', 'L2RL']
exp_types = ['embedding']
try:
    exp_type = [exp_types[int(sys.argv[1])]]
except:
    exp_type = exp_types
training_epochs = 400
noise_epochs = 100
noise_train_percent = 0

# Experiment Difficulty
hamming_clustering = 1     #Create evenly distributed clusters based on arms/barcodes
sim_threshold = 0           #Create one cluster regardless of arms/barcodes
num_arms = 4
num_barcodes = 8
barcode_size = 24
pulls_per_episode = 10
# noise_percent = [0.125, 0.25, 0.5, 0.75, 0.875]
noise_percent = [0.25]

# Randomized seed changes to average for returns graph
num_repeats = 1

# Modify this to fit your machines save paths
figure_save_location = "..\\Mem_Store_Project\\figs\\"
###### NO MORE CHANGES!!!!!!!! ##########

exp_base = exp_type, training_epochs, noise_epochs, noise_train_percent, num_repeats, figure_save_location
exp_difficulty = hamming_clustering, num_arms, num_barcodes, barcode_size, pulls_per_episode, sim_threshold, noise_percent
run_experiment(exp_base, exp_difficulty)


"""
Tests to run:

2a4b, 2a8b, 4a8b, 4a12b, 6a12b
size 16, 24, 32

Barcode Number
Arm Number
Barcode Size
Baseline vs 1 Hamming Cluster

Train with noise?
"""