# Win64bit Optimizations for TSNE
from sklearn.manifold import TSNE
from sklearnex import patch_sklearn
patch_sklearn()
from statsmodels.nonparametric.smoothers_lowess import lowess
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### Graphing Helper Functions ###
# Theoretical Min/Max Return Performance
def expected_return(num_arms, perfect_info):
    if not perfect_info:
        perfect = 0.9
        random = 0.9*(1/num_arms) + 0.1*(num_arms-1)/num_arms
    else:
        perfect = 1
        random = 1/num_arms
    return perfect, random

# Adapted from https://learnopencv.com/t-sne-for-feature-visualization/
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def plot_tsne_distribution(keys, labels, mapping, fig, axes, idx_mem):
    features = np.array([y.cpu().numpy() for y in keys])
    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # Seperate by barcode
    classes = {k: [] for k in mapping.keys()}
    for idx, c_id in enumerate(labels):
        classes[c_id].append(idx)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    marker_list = ['x', '1', 'o', 'D', '*', 'p', 'X', 'h', '8',
                   '2', 'v', '.', '^', '3', '<', 'd', '>', '4', '+', 's']

    # Map each barcode as a seperate layer on the same scatterplot
    for m_id, (c_id, indices) in enumerate(classes.items()):
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # Identify the arm of the barcode
        arm = mapping[c_id]

        # Graph arms by color and barcodes by marker
        axes[idx_mem].scatter(current_tx, current_ty,
                              c=colors[arm], marker=marker_list[m_id])

    return fig, axes

def graph_with_lowess_smoothing (exp_base, exp_difficulty, graph_type, use_lowess = True):

    # Experimental Variables
    exp_settings = {}
    mem_store_types, file_loc, num_repeats = exp_base
    exp_settings['hamming_threshold'], exp_settings['num_arms'], exp_settings['num_barcodes'], exp_settings[
        'barcode_size'], exp_settings['sim_threshold'], exp_settings['noise_train_percent'] = exp_difficulty

    f, axes = plt.subplots(1, 1, figsize=(8, 6))
    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name = exp_size+exp_other

    # LOWESS Smoothed Graphs
    frac = 0.05
    marker_list = ['dashdot','solid', (0, (3, 1, 1)), 'dashed']
    for idx_mem, mem_store in enumerate(mem_store_types):
        exp_name1 = "..\\Mem_Store_Project\\data\\" + exp_name + f"_{mem_store}.npz"
        
        # Returns
        if graph_type == 'Returns':
            data = np.load(exp_name1)['tot_rets']
        
        # Accuracy
        elif graph_type == 'Accuracy':
            data = np.load(exp_name1)['tot_acc']
        
        in_array = np.arange(len(data))
        lowess_data = lowess(data, in_array,
                                frac=frac, return_sorted=False)
        if not use_lowess:
            axes.plot(data, linestyle = marker_list[idx_mem], label=f"Mem: {mem_store.capitalize()}")
        else:
            axes.plot(lowess_data, linestyle = marker_list[idx_mem], label=f"Mem: {mem_store.capitalize()}")

    exp_len = np.load(exp_name1, allow_pickle=True)['epoch_info']
    exp_settings['epochs'] = exp_len[0]
    exp_settings['noise_eval_epochs'] = exp_len[1]
    exp_settings['noise_percent'] = exp_len[2]

    # Graph Labeling and Misc Stuff
    if exp_settings['hamming_threshold']:
        cluster_info = f"Clusters: {int(exp_settings['num_barcodes']/exp_settings['num_arms'])}\nIntraCluster Dist: {exp_settings['hamming_threshold']} | InterCluster Dist: {exp_settings['barcode_size']-2*exp_settings['hamming_threshold']}"
    else:
        cluster_info = f"Similarity: {exp_settings['sim_threshold']}"

    graph_title = f""" --- {graph_type} averaged over {num_repeats} runs ---
    Arms: {exp_settings['num_arms']} | Unique Barcodes: {exp_settings['num_barcodes']} | Barcode Dim: {exp_settings['barcode_size']}
    LOWESS: {min(frac, use_lowess)} | {cluster_info}
    """
    # Noise Trained: {int(exp_settings['barcode_size']*exp_settings['noise_train_percent'])} Bits | 

    # Noise Partitions
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for idx, noise_percent in enumerate(exp_settings['noise_percent']):
        axes.axvline(x=exp_settings['epochs'] + idx*exp_settings['noise_eval_epochs'], color=colors[idx], linestyle='dashed',
                     label=f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy")

    # Random Mem Choice Info
    if graph_type == 'Accuracy':
        # axes.set_title('Barcode Prediction Accuracy from Memory Retrievals')
        axes.axhline(y=1/exp_settings['num_barcodes'], color='b',
                    linestyle='dashed', label='Random Barcode')
        axes.axhline(y=1/exp_settings['num_arms'], color='g',
                    linestyle='dashed', label='Random Arm') 

    sns.despine()
    axes.set_xlabel('Epoch')
    axes.set_ylabel(f'{graph_type}')
    axes.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
                mode="expand", borderaxespad=0, ncol=3)
    f.tight_layout()
    f.subplots_adjust(top=0.8)
    f.suptitle(graph_title)

    plt.show()

    # Graph Saving
    if len(data) >= 200:
        exp_title = file_loc + exp_name + f"{num_repeats}r_{graph_type}"+ ".png"
        f.savefig(exp_title)


def graph_keys_single_run (exp_base, exp_difficulty):

        # Experimental Variables
    exp_settings = {}
    mem_store_types, file_loc, num_repeats = exp_base
    exp_settings['hamming_threshold'], exp_settings['num_arms'], exp_settings['num_barcodes'], exp_settings[
        'barcode_size'], exp_settings['sim_threshold'], exp_settings['noise_train_percent'] = exp_difficulty

    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name1 = exp_size+exp_other

    exp_name = "..\\Mem_Store_Project\\data\\" + exp_name1 + "_" + mem_store_types

    # There will be many key chunks stored in torch.load(key_file)
    # Initial, 33%, 66%, 100% Training View
    train = [0, 33, 66, 100]
    f, axes = plt.subplots(1, 4, figsize=(20, 6))
    all_keys = torch.load(exp_name+".pt")
    epoch_mapping = np.load(exp_name+".npz", allow_pickle=True)['epoch_mapping'].reshape((1,1))
    epoch_mapping = epoch_mapping[0][0]
    for idx_mem, memory in enumerate(all_keys[0:4]):
        # T-SNE to visualize keys in memory
        embeddings = [x[0] for x in memory]
        labels = [x[1] for x in memory]

        # Artifically boost datapoint count to make tsne nicer
        while len(embeddings) < 100:
            embeddings.extend(embeddings)
            labels.extend(labels)

        f, axes = plot_tsne_distribution(
            embeddings, labels, epoch_mapping, f, axes, idx_mem)
        axes[idx_mem].xaxis.set_visible(False)
        axes[idx_mem].yaxis.set_visible(False)
        axes[idx_mem].set_title(f"{train[idx_mem]}%")

    # Keys for the end of every noise eval epoch (probably 5)
    f1, axes1 = plt.subplots(1, 5, figsize=(25, 6))
    exp_noise = np.load(exp_name+".npz", allow_pickle=True)['epoch_info'][2]

    for idx_mem, memory in enumerate(all_keys[4:]):
        # T-SNE to visualize keys in memory
        embeddings = [x[0] for x in memory]
        labels = [x[1] for x in memory]

        # Artifically boost datapoint count to make tsne nicer
        while len(embeddings) < 100:
            embeddings.extend(embeddings)
            labels.extend(labels)

        f1, axes1 = plot_tsne_distribution(
            embeddings, labels, epoch_mapping, f1, axes1, idx_mem)
        axes1[idx_mem].xaxis.set_visible(False)
        axes1[idx_mem].yaxis.set_visible(False)
        axes1[idx_mem].set_title(f"{int(exp_settings['barcode_size']*exp_noise[idx_mem])} Bits Noisy")

    f.suptitle(exp_name1 + mem_store_types)
    f1.suptitle(exp_name1 + mem_store_types)

    f.tight_layout()
    f1.tight_layout()
    plt.show()

def graph_keys_multiple_memory_types(exp_base, exp_difficulty):

    # Experimental Variables
    exp_settings = {}
    mem_store_types, file_loc, num_repeats = exp_base
    exp_settings['hamming_threshold'], exp_settings['num_arms'], exp_settings['num_barcodes'], exp_settings[
        'barcode_size'], exp_settings['sim_threshold'], exp_settings['noise_train_percent'] = exp_difficulty

    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name = exp_size+exp_other

    # Prevent graph subscripting bug if running test on only one mem_store type
    num_tsne = len(mem_store_types) if len(mem_store_types) > 2 else 2
    f, axes = plt.subplots(1, num_tsne, figsize=(5*num_tsne, 6))

    for idx_mem, mem_store in enumerate(mem_store_types):

        exp_name1 = "..\\Mem_Store_Project\\data\\" + \
            exp_name + f"_{mem_store}"
        all_keys = torch.load(exp_name1+".pt")
        epoch_mapping = np.load(exp_name1+".npz", allow_pickle=True)['epoch_mapping'].reshape((1,1))
        epoch_mapping = epoch_mapping[0][0]
        
        # Get keys from end of training
        keys = all_keys[3]

        # T-SNE to visualize keys in memory
        embeddings = [x[0] for x in keys]
        labels = [x[1] for x in keys]

        # Artifically boost datapoint count to make tsne nicer
        while len(embeddings) < 100:
            embeddings.extend(embeddings)
            labels.extend(labels)

        f, axes = plot_tsne_distribution(
            embeddings, labels, epoch_mapping, f, axes, idx_mem)
        axes[idx_mem].xaxis.set_visible(False)
        axes[idx_mem].yaxis.set_visible(False)
        if mem_store != 'L2RL':
            axes[idx_mem].set_title(mem_store.capitalize())
        else:
            axes[idx_mem].set_title('Hidden (L2RL)')

    f.tight_layout()
    plt.show()


def make_graphs(exp_base, exp_difficulty):


    # Experimental Variables
    exp_settings = {}
    mem_store_types, exp_settings['epochs'], exp_settings['noise_eval_epochs'], exp_settings[
        'noise_train_percent'], num_repeats, file_loc = exp_base
    exp_settings['hamming_threshold'], exp_settings['num_arms'], exp_settings['num_barcodes'], exp_settings[
        'barcode_size'], exp_settings['pulls_per_episode'], exp_settings['sim_threshold'] = exp_difficulty

    exp_length = exp_settings['epochs'] + \
        exp_settings['noise_eval_epochs']*len(exp_settings['noise_percent'])

    # Get data for graphs

    f, axes = plt.subplots(1, 1, figsize=(8, 6))
    f1, axs = plt.subplots(1, 1, figsize=(8, 6))
    # f2, axs2 = plt.subplots(1, 1, figsize=(8, 6))

    # Prevent graph subscripting bug if running test on only one mem_store type
    num_tsne = len(mem_store_types) if len(mem_store_types) > 2 else 2
    f3, axes3 = plt.subplots(1, num_tsne, figsize=(5*num_tsne, 6))

    for mem_store in mem_store_types:

        # LOWESS Smoothed Graphs
        frac = 0.05
        in_array = np.arange(exp_length)
        lowess_rets = lowess(tot_rets, in_array,
                             frac=frac, return_sorted=False)
        axes.plot(lowess_rets, label=f"Mem: {mem_store.capitalize()}")

        if mem_store != 'L2RL':
            lowess_acc = lowess(tot_acc, in_array,
                                frac=frac, return_sorted=False)
            axs.plot(lowess_acc, label=f"Mem: {mem_store.capitalize()}")

            # axs2.scatter(range(len(epoch_sim_logs)), epoch_sim_logs,
            #                label=f"Mem: {mem_store.capitalize()}")

        # # Rolling Window Smoothed Graphs
        # # Returns
        # smoothed_rewards = pd.Series.rolling(pd.Series(tot_rets), 5).mean()
        # smoothed_rewards = [elem for elem in smoothed_rewards]
        # axes.scatter(in_array, tot_rets, label=f"Mem: {mem_store.capitalize()}")
        # axes.plot(smoothed_rewards, label=f"Mem: {mem_store}")

        # # Embedder/Mem Accuracy
        # smoothed_accuracy = pd.Series.rolling(pd.Series(log_embedder_accuracy), 5).mean()
        # smoothed_accuracy = [elem for elem in smoothed_accuracy]
        # axs.scatter(in_array, smoothed_accuracy, label=f"Mem: {mem_store.capitalize()}")
        # axs.plot(smoothed_accuracy, label=f"Mem: {mem_store}")

    # T-SNE to visualize keys in memory
    embeddings = [x[0] for x in keys]
    labels = [x[1] for x in keys]

    # Artifically boost datapoint count to make tsne nicer
    while len(embeddings) < 100:
        embeddings.extend(embeddings)
        labels.extend(labels)

    f3, axes3 = plot_tsne_distribution(
        embeddings, labels, epoch_mapping, f3, axes3, idx_mem)
    axes3[idx_mem].xaxis.set_visible(False)
    axes3[idx_mem].yaxis.set_visible(False)
    if mem_store != 'L2RL':
        axes3[idx_mem].set_title(mem_store.capitalize())
    else:
        axes3[idx_mem].set_title('Hidden (L2RL)')

    # Theoretical Max/Min Returns
    # perfect_ret, random_ret = expected_return(exp_settings['num_arms'], exp_settings['perfect_info'])
    # axes.axhline(y=random_ret, color='b', linestyle='dashed', label = 'Random Pulls')
    # axes.axhline(y=perfect_ret, color='k', linestyle='dashed', label = 'Theoretical Max')

    # Graph Labeling and Misc Stuff
    if exp_settings['hamming_threshold']:
        cluster_info = f"IntraCluster Dist: {exp_settings['hamming_threshold']} | InterCluster Dist: {exp_settings['barcode_size']-2*exp_settings['hamming_threshold']}"
    else:
        cluster_info = f"Similarity: {exp_settings['sim_threshold']}"

    graph_title = f""" --- Returns averaged over {num_repeats} runs ---
    Arms: {exp_settings['num_arms']} | Unique Barcodes: {exp_settings['num_barcodes']} | Barcode Dim: {exp_settings['barcode_size']}
    LOWESS: {frac} | Noise Trained: {int(exp_settings['barcode_size']*exp_settings['noise_train_percent'])} bits | Clusters: {int(exp_settings['num_barcodes']/exp_settings['num_arms'])}
    {cluster_info}
    """

    # Noise Partitions
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for idx, noise_percent in enumerate(exp_settings['noise_percent']):
        axes.axvline(x=exp_settings['epochs'] + idx*exp_settings['noise_eval_epochs'], color=colors[idx], linestyle='dashed',
                     label=f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy")
        axs.axvline(x=exp_settings['epochs'] + idx*exp_settings['noise_eval_epochs'], color=colors[idx], linestyle='dashed',
                    label=f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy")

    sns.despine()

    # Returns
    axes.set_ylabel('Returns')
    axes.set_xlabel('Epoch')
    axes.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
                mode="expand", borderaxespad=0, ncol=3)
    f.tight_layout()
    f.subplots_adjust(top=0.8)
    f.suptitle(graph_title)

    # Accuracy
    axs.set_ylabel('Accuracy')
    axs.set_xlabel('Epoch')
    axs.set_title('Barcode Prediction Accuracy from Memory Retrievals')
    axs.axhline(y=1/exp_settings['num_barcodes'], color='b',
                linestyle='dashed', label='Random Choice')
    axs.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
               mode="expand", borderaxespad=0, ncol=2)
    f1.tight_layout()

    # # Memory Similarity
    # axs2.set_xlabel('Pull Number')
    # axs2.set_ylabel('Avg Similarity of Best Memory')
    # axs2.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
    #         mode="expand", borderaxespad=0, ncol=2)
    # f2.tight_layout()

    # T-SNE
    f3.tight_layout()
    f3.subplots_adjust(top=0.8)
    f3.suptitle(
        "t-SNE on keys in memory from last training epoch\nIcon indicates real barcode, color is best arm choice")

    # Graph Saving
    file_loc = file_loc
    exp_id = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s {exp_settings['hamming_threshold']} hamming {exp_settings['noise_train_percent']} noise_trained {num_repeats} run(s) "
    plot_type = ['returns', 'accuracy', 'tsne']
    if exp_length >= 200:
        for fig_num, figa in enumerate([f, f1, f3]):
            filename = file_loc + exp_id + plot_type[fig_num] + ".png"
            figa.savefig(filename)

    plt.show()

if __name__ == '__main__':
    exp_types = ['context', 'embedding', 'hidden', 'L2RL']

    # Experiment Difficulty
    num_arms = 4
    num_barcodes = 8
    barcode_size = 24
    noise_train_percent = 0
    hamming_clustering = 1     #Create evenly distributed clusters based on arms/barcodes
    sim_threshold = 0           #Create one cluster regardless of arms/barcodes

    # Randomized seed changes to average for returns graph
    num_repeats = 5

    # Modify this to fit your machines save paths
    figure_save_location = "..\\Mem_Store_Project\\figs\\"
    ###### NO MORE CHANGES!!!!!!!! ##########

    exp_base = exp_types, figure_save_location, num_repeats
    exp_difficulty = hamming_clustering, num_arms, num_barcodes, barcode_size, sim_threshold, noise_train_percent
    graph_with_lowess_smoothing(exp_base, exp_difficulty, 'Returns')
    # graph_with_lowess_smoothing(exp_base, exp_difficulty, 'Accuracy', use_lowess=False)
    # graph_with_lowess_smoothing(exp_base, exp_difficulty, 'Returns', use_lowess=False)
    # graph_with_lowess_smoothing(exp_base, exp_difficulty, 'Accuracy')
    # graph_keys_multiple_memory_types(exp_base, exp_difficulty)
    # for mem_type in exp_types:
    #     exp_base = mem_type, figure_save_location, num_repeats
    #     graph_keys_single_run(exp_base, exp_difficulty)