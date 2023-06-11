import sys
from exp_naming import run_experiment

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

training_epochs = 1000
noise_epochs = 40

noise_train_percent = 0.2
noise_train_type = 'right_mask'

# emb_loss = 'groundtruth'
emb_loss = 'kmeans'
# emb_loss = 'contrastive'

emb_with_mem = True
switch_to_contrastive = False

# Experiment Difficulty
hamming_clustering = 1      #Create evenly distributed clusters based on arms/barcodes

# num_arms = 5
# num_barcodes = 5
# barcode_size = 10
# noise_percent = [4/20]

num_arms = 10
num_barcodes = 10
barcode_size = 20
noise_percent = [8/40]

pulls_per_episode = 10

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
exp_diff_specifics = [noise_percent, emb_loss, emb_with_mem, switch_to_contrastive]

# Modify this to fit your machines save paths
figure_save_location = "..\\Mem_Store_Project\\figs\\"
###### NO MORE CHANGES!!!!!!!! ##########

# Train Model
for noise_type in noise_types:
    exp_base = exp_type, training_epochs, noise_epochs, noise_train_percent, noise_train_type, noise_type, num_repeats
    exp_difficulty = exp_diff_general + exp_diff_specifics
    run_experiment(exp_base, exp_difficulty)

# After training K-Means, load in weights and train Contrastive
switch_to_contrastive = True
emb_loss = 'contrastive'
exp_diff_specifics = [noise_percent, emb_loss, emb_with_mem, switch_to_contrastive]
training_epochs = 1000
noise_epochs = 40

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

noise_types = [
    # False,
    # "right_mask",
    "random",
    "random_no_mem",
    # "left_mask",
    # "center_mask",
    # "checkerboard",
    ]

for noise_type in noise_types:
    if 'embedding' in exp_types and "no_mem" in noise_type:
        noise_type = noise_type[:-7]
        emb_with_mem = False
        
    exp_diff_specifics = [noise_percent, emb_loss,
                          emb_with_mem, switch_to_contrastive]
    exp_base = exp_type, training_epochs, noise_epochs, noise_train_percent, noise_train_type, noise_type, num_repeats
    exp_difficulty = exp_diff_general + exp_diff_specifics
    run_experiment(exp_base, exp_difficulty)
