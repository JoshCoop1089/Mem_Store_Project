from src.task.ContextBandits import ContextualBandit
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random


class Memory:
    def __init__(self, num_barcodes, barcode_size, device) -> None:
        self.keys = []
        self.num_barcodes = num_barcodes
        self.barcode_size = barcode_size
        self.device = device

    def save_memory_non_embedder(self, memory_key, barcode_string):
        # Append new memories at head of list to allow sim search to find these first
        self.keys = [[(torch.squeeze(memory_key.detach())), barcode_string]] + self.keys
        return

    def get_memory_non_embedder(self, query_key):
        # if no memory, return the zero vector
        n_memories = len(self.keys)
        if n_memories == 0:
            return 0 * self.barcode_size, torch.tensor(0, device=self.device)
        else:
            # compute similarity(query, memory_i ), for all i
            key_list = [self.keys[x][0] for x in range(len(self.keys))]
            similarities = compute_similarities(query_key, key_list)
            # get the best-match memory
            best_memory_id, best_sim_score = self._get_memory(similarities)
            # get the barcode for that memory
            barcode = self.keys[best_memory_id][1]
        return barcode, best_sim_score

    def _get_memory(self, similarities):
        best_sim_score = torch.max(similarities)
        best_memory_id = torch.argmax(similarities)
        return best_memory_id, best_sim_score


def compute_similarities(query_key, key_list):
    q = query_key.data.view(1, -1)
    M = torch.stack(key_list)
    similarities = F.cosine_similarity(q, M)
    return similarities


def run_experiment(
    num_arms, num_barcodes, barcode_size, mem_store, apply_noise, noise_percent
):
    # Experiment Starts here
    pulls_per_episode = 10
    episodes_per_epoch = num_barcodes**2
    sim_threshold = 0
    hamming_threshold = 1
    device = torch.device("cpu")
    perfect_info = False
    perfect_noise = False

    task = ContextualBandit(
        pulls_per_episode,
        episodes_per_epoch,
        num_arms,
        num_barcodes,
        barcode_size,
        sim_threshold,
        hamming_threshold,
        device,
        perfect_info,
    )

    memory = Memory(num_barcodes, barcode_size, device)
    (
        observations_barcodes_rewards,
        epoch_mapping,
        barcode_strings,
        barcode_tensors,
        barcode_id,
        arm_id,
    ) = task.sample()

    tot_accuracy = 0
    log_sim = []
    log_acc = []
    noise_barcode_flip_locs = int(noise_percent * memory.barcode_size)
    for m in range(episodes_per_epoch):
        accuracy = 0
        real_bc_tensor = barcode_tensors[m]
        if apply_noise:
            apply_noise_again = True
            while apply_noise_again:
                apply_noise_again = False

                # What indicies need to be randomized?
                idx = random.sample(range(memory.barcode_size), noise_barcode_flip_locs)

                # Coin Flip to decide whether to flip the values at the indicies
                if not perfect_noise:
                    mask = torch.tensor(
                        [random.randint(0, 1) for _ in idx], device=device
                    )
                else:
                    mask = torch.tensor([1 for _ in idx], device=device)

                noisy_bc = real_bc_tensor.detach().clone()

                # Applying the mask to the barcode at the idx
                if apply_noise == "random":
                    for idx1, mask1 in zip(idx, mask):
                        noisy_bc[0][idx1] = float(torch.ne(mask1, noisy_bc[0][idx1]))

                # Applying an continuous block starting on the left end of bc
                elif apply_noise == "left_mask":
                    for idx1, mask1 in enumerate(mask):
                        noisy_bc[0][idx1] = float(torch.ne(mask1, noisy_bc[0][idx1]))

                elif apply_noise == "center_mask":
                    # Find center
                    center = memory.barcode_size // 2

                    # Find edges of window
                    start = center - noise_barcode_flip_locs // 2
                    end = center + noise_barcode_flip_locs // 2

                    idx = np.arange(start, end)
                    for idx1, mask1 in zip(idx, mask):
                        noisy_bc[0][idx1] = float(torch.ne(mask1, noisy_bc[0][idx1]))

                # Applying an continuous block starting on the right end of bc
                elif apply_noise == "right_mask":
                    for idx1, mask1 in enumerate(mask):
                        noisy_bc[0][memory.barcode_size - 1 - idx1] = float(
                            torch.ne(mask1, noisy_bc[0][idx1])
                        )

                # Even distribution of noise
                elif apply_noise == "checkerboard":
                    idx = np.arange(0, memory.barcode_size, int(1 / noise_percent))
                    for idx1, mask1 in zip(idx, mask):
                        noisy_bc[0][idx1] = float(torch.ne(mask1, noisy_bc[0][idx1]))

                # Cosine similarity doesn't like all 0's for matching in memory
                if torch.sum(noisy_bc) == 0:
                    apply_noise_again = True

        else:
            noisy_bc = real_bc_tensor
        real_bc_string = barcode_strings[m]

        for t in range(pulls_per_episode):
            bc_retrieved, sim_score = memory.get_memory_non_embedder(noisy_bc)
            if bc_retrieved == real_bc_string:
                accuracy += 100 / pulls_per_episode
            log_sim.append(sim_score)
        log_acc.append(accuracy)
        tot_accuracy += accuracy
        memory.save_memory_non_embedder(noisy_bc, real_bc_string)

    # Graph all the things
    f, axes = plt.subplots(2, 1, figsize=(16, 6))
    acc_x = np.arange(0, len(log_acc))
    axes[0].scatter(acc_x, log_acc)
    axes[0].set_xlabel("Episode Number")
    axes[0].set_ylabel("Percent Correct on Retrieval")
    sim_x = np.arange(0, len(log_sim))
    axes[1].plot(log_sim)
    axes[1].set_xlabel("Pull Number")
    axes[1].set_ylabel("Best Memory Similarity ")
    f.suptitle(
        f"Noise Type: {apply_noise} | Noise Percent: {noise_percent} | Bits Flipped: {noise_barcode_flip_locs}\n"
        + f"Mem Type: {mem_store} | Overall Accuracy: {round(tot_accuracy/episodes_per_epoch, 3)}%"
    )
    f.tight_layout()
    plt.show()


# Experimental Params
num_arms = 4
num_barcodes = 8
barcode_size = 40
mem_store = "context"

apply_noise_types = [
    False,
    "random",
    "left_mask",
    "center_mask",
    "right_mask",
    "checkerboard",
]
# apply_noise_types = [False, 'random']
noise_percent_list = [0.25, 0.5]

for apply_noise in apply_noise_types:
    if not apply_noise:
        run_experiment(num_arms, num_barcodes, barcode_size, mem_store, apply_noise, 0)
    else:
        for noise_percent in noise_percent_list:
            run_experiment(
                num_arms,
                num_barcodes,
                barcode_size,
                mem_store,
                apply_noise,
                noise_percent,
            )
