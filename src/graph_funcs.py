import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Win64bit Optimizations for TSNE
from sklearn.manifold import TSNE
from sklearnex import patch_sklearn

patch_sklearn()
from statsmodels.nonparametric.smoothers_lowess import lowess


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


def plot_tsne_distribution(
    keys, labels, mapping, fig, axes, idx_mem, color_by="cluster"
):
    features = np.array([y.cpu().numpy() for y in keys])
    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    checker = list(mapping.keys())[0]
    cluster_id = {k: 0 for k in mapping.keys()}
    for c_id in mapping.keys():
        if hamming_distance(c_id, checker) > 3:
            cluster_id[c_id] = 1

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
        axes[idx_mem].scatter(
            current_tx, current_ty, c=colors[color_id], marker=marker_list[m_id]
        )

    return fig, axes


def graph_with_lowess_smoothing(exp_base, exp_difficulty, graph_type, use_lowess=True):

    # Experimental Variables
    exp_settings = {}
    mem_store_types, noise_type, file_loc = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["sim_threshold"],
        exp_settings["noise_train_percent"],
    ) = exp_difficulty

    f, axes = plt.subplots(1, 1, figsize=(8, 6))
    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name = exp_size + exp_other


    # LOWESS Smoothed Graphs
    frac = 0.05
    marker_list = ["dashdot", "solid", (0, (3, 1, 1)), "dashed"]
    for idx_mem, mem_store in enumerate(mem_store_types):
        exp_name1 = "..\\Mem_Store_Project\\data\\" + exp_name + f"_{mem_store}"
        exp_name1 += f"_{noise_type}_noise_eval"
        exp_name1 += ".npz"

        # Returns
        if graph_type == "Returns":
            axes.set_ylim([0.1, 0.9])
            data = np.load(exp_name1)["tot_rets"]

        # Accuracy
        elif graph_type == "Accuracy":
            axes.set_ylim([0, 1])
            data = np.load(exp_name1)["tot_acc"]

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

    # Graph Labeling and Misc Stuff
    if exp_settings["hamming_threshold"]:
        cluster_info = (
            f"Clusters: {int(exp_settings['num_barcodes']/exp_settings['num_arms'])}\nIntraCluster Dist: {exp_settings['hamming_threshold']}"
            + f" | InterCluster Dist: {exp_settings['barcode_size']-max(2*exp_settings['hamming_threshold'], exp_settings['num_arms']-1)}"
        )
    else:
        cluster_info = f"Similarity: {exp_settings['sim_threshold']}"

    graph_title = f""" --- {graph_type} averaged over {num_repeats} runs ---
    Arms: {exp_settings['num_arms']} | Unique Barcodes: {exp_settings['num_barcodes']} | Barcode Dim: {exp_settings['barcode_size']}
    LOWESS: {min(frac, use_lowess)} | {cluster_info}
    Noise Applied: {noise_type}
    """

    # Noise Partitions
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for idx, noise_percent in enumerate(exp_settings["noise_percent"]):
        axes.axvline(
            x=exp_settings["epochs"] + idx * exp_settings["noise_eval_epochs"],
            color=colors[idx],
            linestyle="dashed",
            label=f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy",
        )

    # # Random Mem Choice Info
    # if graph_type == 'Accuracy':
    #     # axes.set_title('Barcode Prediction Accuracy from Memory Retrievals')
    #     axes.axhline(y=1/exp_settings['num_barcodes'], color='b',
    #                 linestyle='dashed', label='Random Barcode')
    #     axes.axhline(y=1/exp_settings['num_arms'], color='g',
    #                 linestyle='dashed', label='Random Arm')

    sns.despine()
    axes.set_xlabel("Epoch")
    axes.set_ylabel(f"{graph_type}")
    axes.legend(
        bbox_to_anchor=(0, -0.2, 1, 0),
        loc="upper left",
        mode="expand",
        borderaxespad=0,
        ncol=3,
    )
    f.tight_layout()
    f.subplots_adjust(top=0.8)
    f.suptitle(graph_title)

    plt.show()

    # Graph Saving
    if len(data) >= 200:
        exp_title = file_loc + exp_name + f"{noise_type}_{num_repeats}r_{graph_type}" + ".png"
        f.savefig(exp_title)


def graph_multi_barcode_sizes(exp_base, exp_difficulty, graph_type, use_lowess=True):
    # Experimental Variables
    exp_settings = {}
    mem_store_types, noise_type, file_loc = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["sim_threshold"],
        exp_settings["noise_train_percent"],
    ) = exp_difficulty
    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"

    # LOWESS Smoothed Graphs
    frac = 0.05
    marker_list = ["dashdot", "solid", (0, (3, 1, 1)), "dashed"]

    for mem_store in mem_store_types:
        f, axes = plt.subplots(1, 1, figsize=(8, 6))

        for idx_bc, bc_size in enumerate(exp_settings["barcode_size"]):

            exp_name = exp_size + f"{bc_size}s" + exp_other
            exp_name1 = "..\\Mem_Store_Project\\data\\" + exp_name + f"_{mem_store}.npz"
            try:
                # Returns
                if graph_type == "Returns":
                    data = np.load(exp_name1)["tot_rets"]

                # Accuracy
                elif graph_type == "Accuracy":
                    data = np.load(exp_name1)["tot_acc"]
            except:
                continue

            in_array = np.arange(len(data))
            lowess_data = lowess(data, in_array, frac=frac, return_sorted=False)

            if not use_lowess:
                axes.plot(
                    data, linestyle=marker_list[idx_bc], label=f"BC_Size: {bc_size}"
                )
            else:
                axes.plot(
                    lowess_data,
                    linestyle=marker_list[idx_bc],
                    label=f"BC_Size: {bc_size}",
                )

        exp_len = np.load(exp_name1, allow_pickle=True)["epoch_info"]
        exp_settings["epochs"] = exp_len[0]
        exp_settings["noise_eval_epochs"] = exp_len[1]
        exp_settings["noise_percent"] = exp_len[2]
        try:
            num_repeats = exp_len[3]
        except:
            num_repeats = 1

        # Graph Labeling and Misc Stuff
        if exp_settings["hamming_threshold"]:
            cluster_info = (
                f"Clusters: {int(exp_settings['num_barcodes']/exp_settings['num_arms'])}\nIntraCluster Dist: {exp_settings['hamming_threshold']}"
                + f" | InterCluster Dist: {bc_size-max(2*exp_settings['hamming_threshold'], exp_settings['num_arms']-1)}"
            )
        else:
            cluster_info = f"Similarity: {exp_settings['sim_threshold']}"

        graph_title = f""" --- {graph_type} averaged over {num_repeats} runs ---
        Arms: {exp_settings['num_arms']} | Unique Barcodes: {exp_settings['num_barcodes']} | Memory Type: {mem_store}
        LOWESS: {min(frac, use_lowess)} | {cluster_info}
        """

        # Noise Partitions
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        for idx, noise_percent in enumerate(exp_settings["noise_percent"]):
            axes.axvline(
                x=exp_settings["epochs"] + idx * exp_settings["noise_eval_epochs"],
                color=colors[idx],
                linestyle="dashed",
                label=f"{int(bc_size*noise_percent)} Bits Noisy",
            )

        sns.despine()
        axes.set_xlabel("Epoch")
        axes.set_ylabel(f"{graph_type}")
        axes.legend(
            bbox_to_anchor=(0, -0.2, 1, 0),
            loc="upper left",
            mode="expand",
            borderaxespad=0,
            ncol=3,
        )
        f.tight_layout()
        f.subplots_adjust(top=0.8)
        f.suptitle(graph_title)

        plt.show()

        # Graph Saving
        if len(data) >= 200:
            exp_title = (
                file_loc
                + exp_size
                + "Xs"
                + exp_other
                + f"{num_repeats}r_{mem_store}_{graph_type}"
                + ".png"
            )
            f.savefig(exp_title)
    return


def graph_keys_single_run(exp_base, exp_difficulty, color_by):

    # Experimental Variables
    exp_settings = {}
    mem_store_types, noise_type, file_loc = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["sim_threshold"],
        exp_settings["noise_train_percent"],
    ) = exp_difficulty

    exp_size = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['barcode_size']}s"
    exp_other = f"{exp_settings['hamming_threshold']}h{int(100*exp_settings['noise_train_percent'])}n"
    exp_name1 = exp_size + exp_other

    exp_name = "..\\Mem_Store_Project\\data\\" + exp_name1 + "_" + mem_store_types

    # There will be many key chunks stored in torch.load(key_file)
    # Initial, 33%, 66%, 100% Training View
    train = [0, 33, 66, 100]
    f, axes = plt.subplots(1, 4, figsize=(20, 6))
    all_keys = torch.load(exp_name + ".pt")
    epoch_mapping = np.load(exp_name + ".npz", allow_pickle=True)[
        "epoch_mapping"
    ].reshape((1, 1))
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
            embeddings, labels, epoch_mapping, f, axes, idx_mem, color_by
        )
        axes[idx_mem].xaxis.set_visible(False)
        axes[idx_mem].yaxis.set_visible(False)
        axes[idx_mem].set_title(f"{train[idx_mem]}%")

    # Keys for the end of every noise eval epoch
    num_noise_evals = max(len(all_keys[4:]), 2)
    f1, axes1 = plt.subplots(1, num_noise_evals, figsize=(5 * num_noise_evals, 6))
    exp_noise = np.load(exp_name + ".npz", allow_pickle=True)["epoch_info"][2]
    try:
        emb_trained = np.load(exp_name + ".npz", allow_pickle=True)["epoch_info"][4]
    except:
        emb_trained = "barcodes"

    for idx_mem, memory in enumerate(all_keys[4:]):
        # T-SNE to visualize keys in memory
        embeddings = [x[0] for x in memory]
        labels = [x[1] for x in memory]

        # Artifically boost datapoint count to make tsne nicer
        while len(embeddings) < 100:
            embeddings.extend(embeddings)
            labels.extend(labels)

        f1, axes1 = plot_tsne_distribution(
            embeddings, labels, epoch_mapping, f1, axes1, idx_mem, color_by
        )
        axes1[idx_mem].xaxis.set_visible(False)
        axes1[idx_mem].yaxis.set_visible(False)
        axes1[idx_mem].set_title(
            f"{int(exp_settings['barcode_size']*exp_noise[idx_mem])} Bits Noisy"
        )

    if mem_store_types != "L2RL":
        title = mem_store_types.capitalize()
        if mem_store_types == "embedding":
            title += f"_{emb_trained}"
    else:
        title = "Hidden (L2RL)"
    f.suptitle(exp_name1 + f"_{title}: Colored by {color_by}")
    f1.suptitle(exp_name1 + f"_{title}: Colored by {color_by}")

    f.tight_layout()
    f1.tight_layout()
    plt.show()

    # Graph Saving
    exp_len = np.load(exp_name + ".npz", allow_pickle=True)["epoch_info"]
    if exp_len[0] >= 200:
        exp_title = (
            file_loc + exp_name1 + f"_{mem_store_types}_train_tsne_{color_by}" + ".png"
        )
        exp_title1 = (
            file_loc + exp_name1 + f"_{mem_store_types}_noise_tsne_{color_by}" + ".png"
        )
        f.savefig(exp_title)
        f1.savefig(exp_title1)


def graph_keys_multiple_memory_types(exp_base, exp_difficulty, color_by):

    # Experimental Variables
    exp_settings = {}
    mem_store_types, noise_type, file_loc = exp_base
    (
        exp_settings["hamming_threshold"],
        exp_settings["num_arms"],
        exp_settings["num_barcodes"],
        exp_settings["barcode_size"],
        exp_settings["sim_threshold"],
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
        e = [x[0] for x in keys]
        l = [x[1] for x in keys]
        embeddings = [
            e[x] for x in range(len(e)) if x % 5 != 0 and x % 6 != 0 and x % 7 != 0
        ]
        labels = [
            l[x] for x in range(len(l)) if x % 5 != 0 and x % 6 != 0 and x % 7 != 0
        ]

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


if __name__ == "__main__":
    # exp_types = ['embedding']
    # exp_types = ['context', 'embedding']
    # exp_types = ['context', 'hidden', 'L2RL']
    # exp_types = ['embedding', 'hidden', 'L2RL']
    exp_types = ["context", "embedding", "hidden", "L2RL"]

    # Experiment Difficulty
    num_arms = 4
    num_barcodes = 8
    barcode_size = 24
    noise_train_percent = 0
    hamming_clustering = 1  # Create evenly distributed clusters based on arms/barcodes
    sim_threshold = 0  # Create one cluster regardless of arms/barcodes

    noise_types = [
    # False,
    "random",
    "left_mask",
    "center_mask",
    "right_mask",
    # "checkerboard",
    ]

    # Modify this to fit your machines save paths
    figure_save_location = "..\\Mem_Store_Project\\figs\\"
    ###### NO MORE CHANGES!!!!!!!! ##########

    for noise_type in noise_types:
        exp_base = exp_types, noise_type, figure_save_location
        exp_difficulty = (
            hamming_clustering,
            num_arms,
            num_barcodes,
            barcode_size,
            sim_threshold,
            noise_train_percent,
        )
        graph_with_lowess_smoothing(exp_base, exp_difficulty, "Returns")
        graph_with_lowess_smoothing(exp_base, exp_difficulty, "Accuracy")
        # graph_with_lowess_smoothing(exp_base, exp_difficulty, 'Returns', use_lowess=False)
        # graph_with_lowess_smoothing(exp_base, exp_difficulty, 'Accuracy', use_lowess=False)
        # graph_keys_multiple_memory_types(exp_base, exp_difficulty, color_by = 'arms')
        # for mem_type in exp_types:
        #     exp_base = mem_type, figure_save_location
        #     graph_keys_single_run(exp_base, exp_difficulty, color_by = 'arms')

        # exp_base = exp_types, figure_save_location
        # graph_keys_multiple_memory_types(exp_base, exp_difficulty, color_by = 'cluster')
        # for mem_type in exp_types:
        #     exp_base = mem_type, figure_save_location
        #     graph_keys_single_run(exp_base, exp_difficulty, color_by = 'cluster')

        # barcode_size = [16,24]
        # exp_difficulty = hamming_clustering, num_arms, num_barcodes, barcode_size, sim_threshold, noise_train_percent
        # graph_multi_barcode_sizes(exp_base, exp_difficulty, 'Returns', use_lowess=False)
        # graph_multi_barcode_sizes(exp_base, exp_difficulty, 'Accuracy')
