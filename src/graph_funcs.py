import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# Win64bit Optimizations for TSNE
from sklearn.manifold import TSNE
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.decomposition import PCA

from statsmodels.nonparametric.smoothers_lowess import lowess

# ignore all future warnings from sklearn tsne
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

### Graphing Helper Functions ###
# Theoretical Min/Max Return Performance
def expected_return(num_arms, perfect_info):
    if not perfect_info:
        perfect = 0.9
        random = 0.9 * (1 / num_arms) + 0.1 * (num_arms - 1) / num_arms
    else:
        perfect = 1
        random = 1 / num_arms
    return perfect, random

# Adapted from https://learnopencv.com/t-sne-for-feature-visualization/
def scale_to_01_range(x):
    value_range = np.max(x) - np.min(x)
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def hamming_distance(barcode1, barcode2):
    return sum(c1 != c2 for c1, c2 in zip(barcode1, barcode2))

def pca_n_component_finder(data_std):
    pca = PCA(n_components=None)
    pca.fit(data_std)

    exp_var = pca.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)

    plt.bar(range(1, data_std.shape[1]+1), exp_var, align='center',
            label='Individual explained variance')

    plt.step(range(1, data_std.shape[1]+1), cum_exp_var, where='mid',
            label='Cumulative explained variance', color='red')

    plt.ylabel('Explained variance percentage')
    plt.xlabel('Principal component index')
    plt.xticks(ticks=[x for x in range(data_std.shape[1]+1)])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_tsne_distribution(
    keys, labels, mapping, fig, axes, id_mem_x, id_mem_y, color_by="cluster"
):
    features = np.array([y.cpu().numpy() for y in keys])

    if len(features[1]) > 30:
        # pca_n_component_finder(features)
        features = PCA(n_components=30).fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=50).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    cluster_id = {k: 0 for k in mapping.keys()}
    for loc_id, c_id in enumerate(mapping.keys()):
        cluster_id[c_id] = loc_id//(len(mapping.keys())//2)

    # Seperate by barcode
    classes = {k: [] for k in mapping.keys()}
    for idx, c_id in enumerate(labels):
        classes[c_id].append(idx)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors.extend(["b", "g", "r"])

    marker_list = [
        "x",
        "1",
        "o",
        "D",
        "*",
        "p",
        "X",
        "h",
        "8",
        "2",
        "v",
        ".",
        "^",
        "3",
        "<",
        "d",
        ">",
        "4",
        "+",
        "s",
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]

    # Map each barcode as a seperate layer on the same scatterplot
    for m_id, (c_id, indices) in enumerate(classes.items()):
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        if color_by == "arms":
            # Identify the arm of the barcode
            color_id = mapping[c_id]

        elif color_by == "cluster":
            # Identify the cluster group for each barcode
            color_id = cluster_id[c_id]

        # Graph arms by color and barcodes by marker
        axes[id_mem_x][id_mem_y].scatter(
            current_tx, current_ty, c=colors[color_id], marker=marker_list[m_id]
        )

    return fig, axes


def graph_with_lowess_smoothing(exp_base, exp_difficulty, graph_type, use_lowess=True, f = None, axes = None):

    # Experimental Variables
    exp_settings = {}
    mem_store_types, noise_type, file_loc, noise_eval = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["noise_train_percent"],
    ) = exp_difficulty

    if f == None:
        f, axes = plt.subplots(1, 1, figsize=(8, 7))

    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name = exp_size + exp_other
    

    # LOWESS Smoothed Graphs
    frac = 0.05
    marker_list = ["dashdot", "solid", (0, (3, 1, 1)), (0,(2,1,2)), (0,(1,2,1))]
    for idx_mem, mem_store in enumerate(mem_store_types):
        if graph_type in ["Embedder Loss", 'Contrastive Loss', 'Contrastive Pos Loss', 'Contrastive Neg Loss'] and 'embedding' not in mem_store:
            continue
        
        # exp_name1 = "..\\Mem_Store_Project\\data\\10_Run_C_T_Right_Mask\\" + exp_name + f"_{mem_store}"
        exp_name1 = "..\\Mem_Store_Project\\data\\" + exp_name + f"_{mem_store}"
        # if mem_store == 'embedding':
        #     exp_name1 += "_LSTM_full"
        #     # exp_name1 += "_one_layer"
        #     # exp_name1 += "_two_layer"

        if noise_eval:
            exp_name1 += f"_{noise_type}_noise_eval"
            # try:
            #     exp_name1 += "_no_mem"
            # except Exception as e:
            #     pass
        exp_name1 += ".npz"
        exp_len = np.load(exp_name1, allow_pickle=True)["epoch_info"]
        exp_settings["epochs"] = exp_len[0]
        exp_settings["noise_eval_epochs"] = exp_len[1]
        exp_settings["noise_percent"] = exp_len[2]
        try:
            num_repeats = exp_len[3]
        except:
            num_repeats = 1
        try:
            noise_type = exp_len[4]
        except:
            noise_type = "random"

        # Returns
        if graph_type == "Returns":
            axes.set_ylim([0.1, 0.9])
            data = np.load(exp_name1)["tot_rets"]
            # if exp_settings['epochs']:
            #     data = data[exp_settings['epochs']:]

        # Accuracy
        elif graph_type == "Accuracy":
            axes.set_ylim([0, 1])
            data = np.load(exp_name1)["tot_acc"]
            # if exp_settings['epochs']:
            #     data = data[exp_settings['epochs']:]

        # Embedder Loss
        elif graph_type == "Embedder Loss":
            data = np.load(exp_name1)["tot_emb_loss"][:exp_settings['epochs']]

        # Contrastive Loss
        elif graph_type == "Contrastive Loss" and 'embedding' in mem_store:
            data = np.load(exp_name1)["tot_cont_loss"][:exp_settings['epochs']]
        # Contrastive Pos Loss
        elif graph_type == "Contrastive Pos Loss" and 'embedding' in mem_store:
            data = np.load(exp_name1)["tot_cont_pos_loss"][:exp_settings['epochs']]
        # Contrastive Neg Loss
        elif graph_type == "Contrastive Neg Loss" and 'embedding' in mem_store:
            data = np.load(exp_name1)["tot_cont_neg_loss"][:exp_settings['epochs']]

        in_array = np.arange(len(data))
        lowess_data = lowess(data, in_array, frac=frac, return_sorted=False)
        if not use_lowess:
            axes.plot(
                data,
                linestyle=marker_list[idx_mem],
                label=f"Mem: {mem_store.capitalize()}",
            )
        else:
            axes.plot(
                lowess_data,
                linestyle=marker_list[idx_mem],
                label=f"Mem: {mem_store.capitalize()}",
            )
            
        try:
            if mem_store == 'embedding' and graph_type == 'Accuracy':
                data = np.load(exp_name1)["tot_emb_acc"]
                in_array = np.arange(len(data))
                lowess_data = lowess(data, in_array, frac=frac, return_sorted=False)
                if not use_lowess:
                    axes.plot(
                        data,
                        linestyle=marker_list[idx_mem],
                        label="Mem: Emb Model",
                    )
                else:
                    axes.plot(
                        lowess_data,
                        linestyle=marker_list[idx_mem],
                        label="Mem: Emb Model",
                    )
        except:
            pass

    # Graph Labeling and Misc Stuff
    cluster_info = (
            f"Clusters: {int(exp_settings['num_barcodes']/exp_settings['num_arms'])}\nIntraCluster Dist: {exp_settings['hamming_threshold']}"
            + f" | InterCluster Dist: {exp_settings['barcode_size']-max(2*exp_settings['hamming_threshold'], exp_settings['num_arms']-1)}"
        )

    graph_title = f""" --- {graph_type} averaged over {num_repeats} runs ---
    Arms: {exp_settings['num_arms']} | Unique Barcodes: {exp_settings['num_barcodes']} | Barcode Dim: {exp_settings['barcode_size']}
    LOWESS: {min(frac, use_lowess)} | {cluster_info}
    Noise Applied: {noise_type} | Noise Trained: {int(exp_settings["noise_train_percent"]*exp_settings['barcode_size'])} bits
    """

    # # Random Mem Choice Info
    # if graph_type == 'Accuracy':
    #     # axes.set_title('Barcode Prediction Accuracy from Memory Retrievals')
    #     axes.axhline(y=1/exp_settings['num_barcodes'], color='b',
    #                 linestyle='dashed', label='Random Barcode')
    #     axes.axhline(y=1/exp_settings['num_arms'], color='g',
    #                 linestyle='dashed', label='Random Arm')

    # Noise Partitions
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for idx, noise_percent in enumerate(exp_settings["noise_percent"]):
        axes.axvline(
            x=exp_settings["epochs"] + idx * exp_settings["noise_eval_epochs"],
            color=colors[idx],
            linestyle="dashed",
            label=f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy" if exp_settings['epochs'] != 0 else None,
        )
    sns.despine()
    axes.set_xlabel("Epoch")
    axes.set_ylabel(f"{graph_type}")
    axes.legend(
        bbox_to_anchor=(0, -0.2, 1, 0),
        loc="upper left",
        mode="expand",
        borderaxespad=0,
        ncol=2,
    )

    # Noise Eval X-Labels
    if exp_settings['epochs'] == 0:
        x_locs = [int((x+0.5)*exp_settings['noise_eval_epochs']) for x,_ in enumerate(exp_settings['noise_percent'])]
        noise_labels = [f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy" for noise_percent in exp_settings['noise_percent']]
        axes.set_xticks(x_locs)
        axes.set_xticklabels(noise_labels)
        axes.set_xlabel(f"Ran {exp_settings['noise_eval_epochs']} epochs per Noise Level")

    f.tight_layout()
    f.subplots_adjust(top=0.8)
    f.suptitle(graph_title)

    plt.show()

    # Graph Saving
    if len(data) >= 200:
        cur_date = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        exp_title = file_loc + cur_date + "_" + exp_name + \
            f"_{noise_type}_{num_repeats}r_{graph_type}" + "_no_mem.png"
        f.savefig(exp_title)


def graph_keys_single_run(exp_base, exp_difficulty, color_by):

    # Experimental Variables
    exp_settings = {}
    mem_store_types, noise_type, file_loc, noise_eval = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["noise_train_percent"],
    ) = exp_difficulty

    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name1 = exp_size + exp_other

    # exp_name = "..\\Mem_Store_Project\\data\\" + exp_name1 + "_" + mem_store_types
    exp_name = "..\\Mem_Store_Project\\data\\10_Run_C_T_Right_Mask\\" + exp_name1 + f"_{mem_store_types}"

    # if mem_store_types == 'embedding':
    #     exp_name1 += "_LSTM_full"
    #     # exp_name1 += "_one_layer"
    #     # exp_name1 += "_two_layer"
    if noise_eval:
        exp_name += f"_{noise_type}_noise_eval"
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    all_keys = torch.load(exp_name + ".pt", map_location = device)

    if mem_store_types != "L2RL":
        title = mem_store_types.capitalize()
    else:
        title = "Hidden (L2RL)"
    data = np.load(exp_name+".npz",allow_pickle=True)
    epoch_mapping = data[
        "epoch_mapping"
    ].reshape((1, 1))
    epoch_mapping = epoch_mapping[0][0]
    train_epochs = data["epoch_info"][0]

    # There will be many key chunks stored in torch.load(key_file)
    if train_epochs != 0:
        # Initial, 33%, 66%, 100% Training View
        train = [0, 5, 10, 15, 20, 25, 33, 66, 100]

        f, axes = plt.subplots(3, 3, figsize=(12, 10))
        for idx_mem, memory in enumerate(all_keys[0:len(train)]):

            # Subplot Locations for 9 key splits
            id_mem_x = idx_mem//3
            id_mem_y = idx_mem%3

            # T-SNE to visualize keys in memory
            embeddings = [x[0] for x in memory if x != []]
            try:
                labels = [x[4] for x in memory if x != []]
            except Exception as e:
                # print(e)
                labels = [x[2] for x in memory]


            # Artifically boost datapoint count to make tsne nicer
            while len(embeddings) < 100:
                embeddings.extend(embeddings)
                labels.extend(labels)

            f, axes = plot_tsne_distribution(
                embeddings, labels, epoch_mapping, f, axes, id_mem_x, id_mem_y, color_by
            )
            axes[id_mem_x][id_mem_y].xaxis.set_visible(False)
            axes[id_mem_x][id_mem_y].yaxis.set_visible(False)
            axes[id_mem_x][id_mem_y].set_title(f"Epoch: {int(train_epochs * train[idx_mem]/100)}")

        num_noise_evals = max(len(all_keys[len(train):]), 2)
        key_start = len(train)

        f.suptitle(exp_name1 + f"_{title}: Colored by {color_by}")
        f.tight_layout()
    else:
        num_noise_evals = len(all_keys)
        key_start = 0

    # Keys for the end of every noise eval epoch
    f1, axes1 = plt.subplots(2, num_noise_evals, figsize=(5 * num_noise_evals, 12))
    exp_noise = data["epoch_info"][2]

    for idx_mem, memory in enumerate(all_keys[key_start:]):
        # T-SNE to visualize keys in memory
        embeddings = [x[0] for x in memory if x != []]
        try:
            labels = [x[4] for x in memory if x != []]
        except Exception as e:
            # print(e)
            labels = [x[2] for x in memory]
        # Artifically boost datapoint count to make tsne nicer
        while len(embeddings) < 100:
            embeddings.extend(embeddings)
            labels.extend(labels)

        f1, axes1 = plot_tsne_distribution(
            embeddings, labels, epoch_mapping, f1, axes1, 0, idx_mem, color_by
        )
        axes1[0][idx_mem].xaxis.set_visible(False)
        axes1[0][idx_mem].yaxis.set_visible(False)
        axes1[0][idx_mem].set_title(
            f"{int(exp_settings['barcode_size']*exp_noise[idx_mem])} Bits Noisy"
            # f"{int(exp_settings['barcode_size']*exp_settings['noise_train_percent']*exp_noise[idx_mem])} Bits Noisy"
        )

    f1.suptitle(exp_name1 + f"_{title}: Colored by {color_by}")
    f1.tight_layout()
    plt.show()

    # Graph Saving
    cur_date = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    exp_len = np.load(exp_name + ".npz", allow_pickle=True)["epoch_info"]
    if exp_len[0] >= 200:
        exp_title = (
            file_loc + cur_date + "_" + exp_name1 + f"_{mem_store_types}_train_tsne_{color_by}" + ".png"
        )
        f.savefig(exp_title)
    if exp_len[1] >= 50:
        exp_title1 = (
            file_loc + cur_date + "_" + exp_name1 +
            f"_{mem_store_types}_noise_tsne_{color_by}" + ".png"
        )
        f1.savefig(exp_title1)


def graph_keys_multiple_memory_types(exp_base, exp_difficulty, color_by):

    # Experimental Variables
    exp_settings = {}
    mem_store_types, noise_type, file_loc, noise_eval = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["noise_train_percent"],
    ) = exp_difficulty

    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name = exp_size + exp_other

    # Prevent graph subscripting bug if running test on only one mem_store type
    num_tsne = len(mem_store_types) if len(mem_store_types) > 2 else 2
    f, axes = plt.subplots(1, num_tsne, figsize=(5 * num_tsne, 6))

    for idx_mem, mem_store in enumerate(mem_store_types):

        exp_name1 = "..\\Mem_Store_Project\\data\\" + exp_name + f"_{mem_store}"
        all_keys = torch.load(exp_name1 + ".pt")
        epoch_mapping = np.load(exp_name1 + ".npz", allow_pickle=True)[
            "epoch_mapping"
        ].reshape((1, 1))
        epoch_mapping = epoch_mapping[0][0]
        exp_len = np.load(exp_name1 + ".npz", allow_pickle=True)["epoch_info"]

        # Get keys from end of training
        keys = all_keys[3]

        # T-SNE to visualize keys in memory
        embeddings = [x[0] for x in keys]
        labels = [x[4] for x in keys]
        # embeddings = [
        #     e[x] for x in range(len(e)) if x % 5 != 0 and x % 6 != 0 and x % 7 != 0
        # ]
        # labels = [
        #     l[x] for x in range(len(l)) if x % 5 != 0 and x % 6 != 0 and x % 7 != 0
        # ]

        # Artifically boost datapoint count to make tsne nicer
        while len(embeddings) < 100:
            embeddings.extend(embeddings)
            labels.extend(labels)

        f, axes = plot_tsne_distribution(
            embeddings, labels, epoch_mapping, f, axes, idx_mem, color_by
        )
        axes[idx_mem].xaxis.set_visible(False)
        axes[idx_mem].yaxis.set_visible(False)
        if mem_store != "L2RL":
            title = mem_store.capitalize()
            if mem_store == "embedding":
                try:
                    title += f"_{exp_len[4]}"
                except:
                    pass
        else:
            title = "Hidden (L2RL)"
        axes[idx_mem].set_title(title)

    f.subplots_adjust(top=0.8)
    f.suptitle(
        exp_name
        + f"\nKey Distribution for Memory at End of Training: Colored by {color_by}"
    )
    f.tight_layout()
    plt.show()
    exp_len = np.load(exp_name1 + ".npz", allow_pickle=True)["epoch_info"]
    if exp_len[0] >= 200:
        exp_title = file_loc + exp_name + f"_multi_tsne_{color_by}" + ".png"
        f.savefig(exp_title)


def graph_2x2(exp_base, exp_difficulty, graph_type, multi_graph, use_lowess=True):
    f, axes = plt.subplots(2, 4, figsize=(10,20))
    # noise_amounts = [0.0, 0.5, 1.0, 2.0]
    noise_amounts = [0.2, 0.4, 0.6, 0.8]
    noise_amounts = [0.25, 0.4, 0.6, 0.8]
    # noise_amounts = [0.4, 0.5, 0.6, 0.8]
    # Experimental Variables
    exp_settings = {}
    mem_store_types, noise_type, file_loc, noise_eval = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["noise_train_percent"],
    ) = exp_difficulty

    for i in range(2):
        for j in range(4):
            idx = j
            exp_settings['noise_train_percent'] = noise_amounts[idx]
            exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
            exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
            exp_name = exp_size + exp_other

            # LOWESS Smoothed Graphs
            frac = 0.1
            marker_list = ["dashdot", "solid", (0, (3, 1, 1)), (0,(2,1,2))]
            for idx_mem, mem_store in enumerate(mem_store_types):

                exp_name1 = f"..\\Mem_Store_Project\\data\\{noise_type}_data\\" + exp_name + f"_{mem_store}"
                exp_name1 += ".npz"
                if mem_store == 'L2RL_base':
                    exp_name1 = "..\\Mem_Store_Project\\data\\random_data\\" + exp_size + \
                        f"{exp_settings['hamming_threshold']}h50n" + f"_{mem_store}" + ".npz"
                exp_len = np.load(exp_name1, allow_pickle=True)["epoch_info"]
                exp_settings["epochs"] = exp_len[0]
                exp_settings["noise_eval_epochs"] = exp_len[1]

                # Returns
                if i == 0: # graph_type == "Returns":
                    axes[i][j].set_ylim([0.1, 0.9])
                    data = np.load(exp_name1)["tot_rets"]
                    if exp_settings['epochs']:
                        data = data[:exp_settings['epochs']]
                    axes[i][j].set_ylabel(f"{graph_type} per Pull")

                if i == 1: # graph_type == "Accuracy"
                    axes[i][j].set_ylim([0, 1])
                    data = np.load(exp_name1)["tot_acc"]
                    if exp_settings['epochs']:
                        data = data[:exp_settings['epochs']]
                    axes[i][j].set_ylabel(f"Memory Accuracy")

                in_array = np.arange(len(data))
                lowess_data = lowess(data, in_array, frac=frac, return_sorted=False)
                if mem_store == 'context': mem_store = "Ritter (Context)"
                elif mem_store == 'embedding_LSTM_kmeans': mem_store = "Ours (Embeddings)"
                elif mem_store == 'L2RL_base': mem_store = "L2RL"
                elif mem_store == 'L2RL_context': mem_store = "L2RL + Context"

                plot_data = data if not use_lowess else lowess_data
                if multi_graph == 'name':
                    axes[i][j].plot(
                        plot_data,
                        linestyle=marker_list[idx_mem],
                        label=f"{mem_store}",
                    )
                elif multi_graph == 'noise':
                    row = int(idx_mem > 1)
                    col = idx_mem%2
                    axes[row][col].plot(
                        plot_data,
                        linestyle=marker_list[idx_mem],
                        label=f"{exp_settings['noise_train_percent']*100} % Noise Added ({exp_settings['noise_train_percent']*exp_settings['barcode_size']} extra bits)",
                    )

                sns.despine()
            axes[i][j].set_xlabel("Epoch")

            if multi_graph == 'name':
                axes[i][j].set_title(f"{exp_settings['noise_train_percent']*100}% Noise Added ({exp_settings['noise_train_percent']*exp_settings['barcode_size']} extra bits)")
            elif multi_graph == 'noise':
                for idx, mem_store_1 in enumerate(mem_store_types):
                    row = int(idx > 1)
                    col = idx%2
                    if mem_store_1 == 'context': mem_store_1 = "Ritter (Context)"
                    elif mem_store_1 == 'embedding_LSTM_kmeans': mem_store_1 = "Ours (Embeddings)"
                    elif mem_store_1 == 'L2RL_base': mem_store_1 = "L2RL"
                    elif mem_store_1 == 'L2RL_context': mem_store_1 = "L2RL + Context"
                    axes[row][col].set_title(f"{mem_store_1}")

    handles, labels = axes[1][1].get_legend_handles_labels()
    f.legend(handles, labels, loc="lower center", ncol=2)
    # axes.legend(
    #     bbox_to_anchor=(0, -0.2, 1, 0),
    #     loc="upper left",
    #     mode="expand",
    #     borderaxespad=0,
    #     ncol=3,
    # )
    # f.subplots_adjust(top = 1.1)
    f.suptitle(f"10 arms, 10 barcodes, 20 barcode length, {noise_type} noise applied")
    # f.tight_layout()
    plt.show()

    # Graph Saving
    if len(data) >= 200:
        cur_date = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        exp_title = file_loc + cur_date + "_10a10b20s_Multi-Noise_All_mems.png"
        f.savefig(exp_title)



if __name__ == "__main__":
    # exp_types = ['context']
    # exp_types = ['context', 'embedding']
    # exp_types = ['context', 'L2RL']

    # Experiment Difficulty
    # stats = [2,4,8, 0.2]
    # stats = [8, 16, 40, 0.2]
    # stats = [5, 5, 10, 0.2]
    # stats = [10,10,10, 0.0]
    # stats = [10,10,10, 0.5]

    # stats = [5,5,20, 0.2]
    # stats = [5,10,20, 0.2]
    # stats = [10,10,20, 0.2, 1]
    # stats = [10,10,20, 0.0, 1]
    stats = [10,10,20, 0.5, 1]
    # stats = [10,10,20, 1.0, 1]
    # stats = [10,10,20, 1.0, 3]
    # stats = [10,10,20, 2.0, 1]
    # stats = [10,10,20, 3.0, 1]
    # stats = [10,10,20, 4.0, 1]
    # stats = [10,10,20, 5.0, 1]

    # stats = [5,10,10, 0.2]
    # stats = [5,10,10, 0.4]
    # stats = [5,10,20, 0]
    # stats = [5,10,20, 0.1]
    stats = [5,10,20, 0.2]
    # stats = [5,10,20, 0.4]
    # stats = [5,10,40, 0.2]
    # stats = [5,10,40, 0.4]


    # noise_eval = True
    noise_eval = False
    # exp_types = ['embedding']
    # exp_types = ['context', 'embedding_LSTM_groundtruth', 'embedding_LSTM_contrastive', 'L2RL']
    # exp_types = ['embedding_LSTM_kmeans', 'embedding_LSTM_contrastive']
    # exp_types = ['context', 'embedding_LSTM_kmeans', 'L2RL']
    # exp_types = ['context', 'L2RL_base', 'L2RL_context']
    exp_types = ['context', 'embedding_LSTM_kmeans', 'L2RL_base', 'L2RL_context']
    # exp_types = ['context', 'embedding_LSTM_contrastive', 'L2RL_base', 'L2RL_context']
    # exp_types = ['context', 'embedding_LSTM_kmeans', 'embedding_LSTM_contrastive', 'L2RL_base', 'L2RL_context']
    # exp_types = ['context', 'embedding_one_layer_kmeans', 'L2RL_base', 'L2RL_context']
    # exp_types = ['context', 'embedding_LSTM_groundtruth', 'L2RL']
    # exp_types = ['context', 'embedding_LSTM_contrastive', 'L2RL']
    # exp_types = ['embedding_LSTM_contrastive']
    # exp_types = ['embedding_LSTM_kmeans']
    # exp_types = ['embedding_LSTM_groundtruth']
    # exp_types = ['context', 'L2RL']

    num_arms = stats[0]
    num_barcodes = stats[1]
    barcode_size = stats[2]
    noise_train_percent = stats[3]
    hamming_clustering = stats[4]  # Create evenly distributed clusters based on arms/barcodes
    noise_train_type = "right_mask"
    # noise_train_type = "left_mask"
    # noise_train_type = "none"
    noise_types = [
    # False,
    # "random",
    # "left_mask",
    # "center_mask",
    "right_mask",
    # "checkerboard",
    ]

    # Modify this to fit your machines save paths
    figure_save_location = "..\\Mem_Store_Project\\figs\\"
    ###### NO MORE CHANGES!!!!!!!! ##########

    for noise_type in noise_types:
        exp_base = exp_types, noise_type, figure_save_location, noise_eval
        exp_difficulty = (
            hamming_clustering,
            num_arms,
            num_barcodes,
            barcode_size,
            noise_train_percent,
        )

        # graph_2x2(exp_base, exp_difficulty, "Returns", multi_graph='name')
        # graph_2x2(exp_base, exp_difficulty, "Returns", multi_graph='noise')
        graph_with_lowess_smoothing(exp_base, exp_difficulty, "Returns")
        graph_with_lowess_smoothing(exp_base, exp_difficulty, "Accuracy")
        # # # graph_with_lowess_smoothing(exp_base, exp_difficulty, "Returns", use_lowess=False)
        # # # graph_with_lowess_smoothing(exp_base, exp_difficulty, "Accuracy", use_lowess=False)
        # # graph_with_lowess_smoothing(exp_base, exp_difficulty, "Embedder Loss")
        # graph_with_lowess_smoothing(exp_base, exp_difficulty, "Contrastive Loss")
        # graph_with_lowess_smoothing(exp_base, exp_difficulty, "Contrastive Pos Loss")
        # graph_with_lowess_smoothing(exp_base, exp_difficulty, "Contrastive Neg Loss")

        # # graph_with_lowess_smoothing(exp_base, exp_difficulty, 'Returns', use_lowess=False)
        # # graph_with_lowess_smoothing(exp_base, exp_difficulty, 'Accuracy', use_lowess=False)

        # # # graph_keys_multiple_memory_types(exp_base, exp_difficulty, color_by = 'arms')
        # for mem_type in exp_types:
        #     exp_base = mem_type, noise_type, figure_save_location, noise_eval
        #     graph_keys_single_run(exp_base, exp_difficulty, color_by = 'arms')

        # exp_base = exp_types, noise_type, figure_save_location
        # graph_keys_multiple_memory_types(exp_base, exp_difficulty, color_by = 'cluster')
        # for mem_type in exp_types:
        #     exp_base = mem_type, noise_type, figure_save_location, noise_eval
        #     graph_keys_single_run(exp_base, exp_difficulty, color_by = 'cluster')
