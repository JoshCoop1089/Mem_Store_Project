"""
Contextual Bandit Task

One task -> one barcode

Assume three arms to pull
If barcode is 001:
    arm3 gives 90% chance of reward
    arm1 and arm2 give 10% chance of reward

Generate barcode mapping to arm pull probability
dict where key is barcode and value is what arm has 90% chance of reward

From Ritter Paper:
    Each episode consisted of a series of pulls, throughout which the agent should efficiently
    find the best arm (explore) and pull it as many times as possible (exploit).
    During each episode, a context was presented which identified the reward probabilities.

# One episode is a sequence of pulls using only one barcode, in this case 10 pulls, rewards calculated via barcode mapping
# Create 10 episodes per unique context
# Thus, one epoch is a sequence of 100 episodes, allowing for 10 repeats of 10 pulls in one context

We sampled the sequence of tasks for each epoch as follows:
we first sampled a set of unique contexts, and paired
each element of that set randomly with one of the possible
rewarding arm positions b, ensuring that each rewarding
arm position was paired with at least one context. We then
created a bag S in which each (c; b) pair was duplicated
10 times. Finally, we sampled the task sequence for the
epoch by repeatedly sampling uniformly without replacement
tasks tn = (cn; bn)  unif(S). There were 100
episodes per epoch and 10 unique contexts per epoch. Thus,
each context was presented 10 times per epoch. There were
10 arms, and episodes were 10 trials long.

LSTM Input Format:
Single Trial of 10 pulls for one barcode
Trial is a sequence of 10 one hot encoded pulls indicating the pulled arm

[100, barcode2, r:0] would be one input to the LSTM
"""
import random

import numpy as np
import torch
from numpy.linalg import norm


class ContextualBandit:
    """
    Create a Contextual Bandit Task with either deterministic arm rewards or a 90%/10% reward chance
    """

    def __init__(
        self,
        pulls_per_episode,
        episodes_per_epoch,
        num_arms,
        num_barcodes,
        barcode_size,
        noise_threshold,
        noise_type,
        hamming_threshold,
        device,
        perfect_info=False,
    ):

        # Task Specific
        self.device = device
        self.episodes_per_epoch = episodes_per_epoch

        # Looping the LSTM inputs, so only need one pull per episode to start it off
        self.pulls_per_episode = 1

        # Arm Specific
        self.num_arms = num_arms
        self.num_barcodes = num_barcodes
        self.barcode_size = barcode_size
        self.perfect_info = perfect_info

        # Arm Clustering (Forcing barcodes to be close to each other, measured by hamming distance)
        # Noise Assertions
        assert 0 <= noise_threshold < 1, "Noise should be a decimal value from [0,1)"
        assert noise_type in ['right_mask', 'left_mask', 'none'], "Noise location must be right_mask, left_mask, or none"

        self.noise_threshold = noise_threshold
        self.noise_type = noise_type
        self.hamming_threshold = hamming_threshold

        # This is assuming clustering of 1 around seed barcodes, probably should change this in the future
        self.cluster_sep = self.barcode_size - (self.num_arms - 1)
        self.cluster_lists = []

        # Barcode to Arm Mapping generated on init, can be regenerated by sample if needed
        if self.hamming_threshold:
            self.epoch_mapping = self.generate_barcode_clusters()
        else:
            self.epoch_mapping = self.generate_barcode_mapping()

    def sample(self, to_torch=True):
        """
        Get a single epochs worth of observations and rewards for input to LSTM

        Args (Defined at program runtime):
            self.reset_barcode_mapping (Boolean): Whether you recreate the set of randomized barcodes
            self.reset_arms_per_epoch (Boolean): Whether you reset the distinct mapping of barcode to good arm, while not changing the original barcode values

        Returns:
            obs_barcodes_rewards (Tensor): All pulls/barcodes/rewards for the epoch
            self.epoch_mapping (Dict (String -> Int)): What arm is best for a barcode
            barcode_strings (Numpy Array): Replication of barcodes as string for easier referencing later in program
            barcode_tensors (Tensor): Replication of barcode tiled to match length of epoch
        """

        # Reassign arms randomly within current barcode scheme
        if self.hamming_threshold:
            self.epoch_mapping = self.map_arms_to_barcode_clusters(self.cluster_lists)
        else:
            self.epoch_mapping = self.map_arms_to_barcodes(self.epoch_mapping)

        (
            observation_p1,
            reward_p1,
            barcode_p1,
            barcode_strings,
            barcode_id,
            arm_id,
        ) = self.generate_trials_info(self.epoch_mapping)
        obs_barcodes_rewards = np.dstack([observation_p1, barcode_p1, reward_p1])

        # to pytorch form
        if to_torch:
            obs_barcodes_rewards = to_pth(obs_barcodes_rewards).to(self.device)
            barcode_tensors = to_pth(barcode_p1).to(self.device)
            barcode_ids = to_pth(barcode_id, pth_dtype=torch.long).to(self.device)
            arm_ids = to_pth(arm_id, pth_dtype=torch.long).to(self.device)
        return (
            obs_barcodes_rewards,
            self.epoch_mapping,
            barcode_strings,
            barcode_tensors,
            barcode_ids,
            arm_ids,
        )

    def hamming_distance(self, barcode1, barcode2):
        return sum(c1 != c2 for c1, c2 in zip(barcode1, barcode2))

    def generate_seed_barcodes(self):
        barcode_bag = set()
        new_seed = False
        seed_count = 0

        # Generate seperated seed barcodes
        while len(barcode_bag) < self.num_barcodes / self.num_arms:
            if seed_count > 100:
                barcode_bag = set()
                seed_count = 0

            if len(barcode_bag) == 0:
                barcode = np.random.randint(0, 2, (self.barcode_size))
                if np.sum(barcode) == 0:
                    continue
                # print(barcode)

                # barcode -> string starts out at '[1 1 0]', thus the reductions on the end
                # Also, somehow for barcode sizes like 48, a random \n showed up?!?! why
                barcode_string = (
                    np.array2string(barcode)[1:-1].replace(" ", "").replace("\n", "")
                )
                barcode_bag.add(barcode_string)
            else:
                for seed_bc in barcode_bag:
                    # Noise can be used to simulate hamming distance
                    noise_idx = sorted(
                        np.random.choice(
                            range(self.barcode_size), self.cluster_sep, replace=False
                        )
                    )
                    seed = np.asarray(list(seed_bc), dtype=int)
                    for idx in noise_idx:
                        seed[idx] = seed[idx] == 0
                    barcode_string = (
                        np.array2string(seed)[1:-1].replace(" ", "").replace("\n", "")
                    )

                    for seed_bc_1 in barcode_bag:
                        h_d = self.hamming_distance(seed_bc_1, barcode_string)

                        # If the cluster seeds are too similar, throw it out
                        if h_d < self.cluster_sep:
                            new_seed = True
                            break

                # Inf Loop Prevention on higher number of clusters
                seed_count += 1

                # Current Seed is too close to other seeds
                if not new_seed:
                    barcode_bag.add(barcode_string)
                new_seed = False

        return barcode_bag

    def generate_barcode_clusters(self):
        new_seed_needed = True
        while new_seed_needed:
            new_seed_needed = False
            mapping = {}
            self.cluster_lists = []
            barcode_bag = self.generate_seed_barcodes()

            # Barcode_bag now holds distinct cluster start points, create close clusters around those points
            bc_clusters = list(barcode_bag)
            for cluster in bc_clusters:
                seed_counter = 0
                mini_cluster_bag = set()
                mini_cluster_bag.add(cluster)
                other_clusters = [
                    np.asarray(list(x), dtype=int) for x in bc_clusters if x != cluster
                ]
                cluster = np.asarray(list(cluster), dtype=int)
                while len(mini_cluster_bag) < self.num_arms:
                    if seed_counter > 10000:
                        new_seed_needed = True
                        break

                    barcode = np.copy(cluster)

                    # Noise can be used to simulate hamming distance
                    noise_idx = np.random.choice(
                        range(self.barcode_size), self.hamming_threshold, replace=False
                    )

                    # There's probably a nicer numpy way to do this but i cannot figure it out
                    # Flipping the value at the noise location
                    for idx in noise_idx:
                        barcode[idx] = barcode[idx] == 0

                    # Avoid cosine similarity bug with barcode of all 0's
                    if np.sum(barcode) == 0:
                        continue

                    other_cluster_centers = [
                        self.hamming_distance(barcode, x) for x in other_clusters
                    ]

                    # Infinite loop prevention
                    seed_counter += 1

                    # BC is too close to other cluster centers
                    dist_min = min(other_cluster_centers)
                    if dist_min < self.cluster_sep:
                        continue
                    else:
                        # barcode -> string starts out at '[1 1 0]', thus the reductions on the end
                        barcode_string = (
                            np.array2string(barcode)[1:-1]
                            .replace(" ", "")
                            .replace("\n", "")
                        )
                        mini_cluster_bag.add(barcode_string)

                if not new_seed_needed:
                    mapping = self.add_noise_to_barcodes(mini_cluster_bag, mapping)

        return mapping

    def map_arms_to_barcode_clusters(self, cluster_list):
        mapping = {}
        for cluster in cluster_list:
            mapping = mapping | self.map_arms_to_barcodes(
                mapping=None, barcode_list=cluster
            )
        return mapping

    def add_noise_to_barcodes(self, mini_cluster_bag, mapping):
        # Need to store individual cluster for arm reshuffling at every epoch
        unnoised_cluster_list = list(mini_cluster_bag)
        # print(unnoised_cluster_list)

        # Append noise onto end of barcode to test model understanding of important features
        cluster_list = [""]*len(unnoised_cluster_list)
        assert 0 <= self.noise_threshold < 1
        noise_added = int(self.barcode_size * self.noise_threshold)
        for idx, barcode in enumerate(unnoised_cluster_list):
            np_noise = np.random.randint(0, 2, noise_added)
            noise = np.array2string(np_noise)[1:-1].replace(" ", "").replace("\n", "")
            if self.noise_type == 'right_mask':
                cluster_list[idx] = barcode + noise
            elif self.noise_type == 'left_mask':
                cluster_list[idx] = noise + barcode
            elif self.noise_type == 'none':
                cluster_list[idx] = barcode
        # print(cluster_list)
        self.cluster_lists.append(cluster_list)
        cluster_mapping = self.map_arms_to_barcodes(
            barcode_list=cluster_list
        )
        mapping = mapping | cluster_mapping

        return mapping

    def generate_barcode_mapping(self):
        barcode_bag = set()
        mapping = {}
        seed_reset = 0

        # Create a set of unique binary barcodes
        # Array2String allows barcodes to be hashable in set to get uniqueness guarantees
        while len(barcode_bag) < self.num_barcodes:

            # On high similarity tests, it sometimes got stuck due to a poor initial choice
            if seed_reset > 10000:
                barcode_bag = set()
                print("stuck on old seed, reseting initial location")
                seed_reset = 0

            barcode = np.random.randint(0, 2, (self.barcode_size))
            seed_reset += 1

            # Avoid cosine similarity bug with barcode of all 0's
            if np.sum(barcode) == 0:
                continue

            if len(barcode_bag) == 0:
                seed_bc = barcode

            # if self.sim_threshold:
            #     similarity = np.dot(seed_bc, barcode) / (norm(seed_bc) * norm(barcode))
            #     if similarity < self.sim_threshold:
            #         continue

            # barcode -> string starts out at '[1 1 0]', thus the reductions on the end
            barcode_string = np.array2string(barcode)[1:-1].replace(" ", "")
            barcode_bag.add(barcode_string)

        mapping = self.add_noise_to_barcodes(barcode_bag, mapping)

        # barcode_bag_list = list(barcode_bag)
        # mapping = self.map_arms_to_barcodes(mapping=None, barcode_list=barcode_bag_list)
        return mapping

    def map_arms_to_barcodes(self, mapping=None, barcode_list=None):
        if mapping:
            barcode_list = list(mapping.keys())
        else:  # barcode_list != None is required
            mapping = {}

        # Generate mapping of barcode to good arm
        for barcode in barcode_list:
            arm = random.randint(0, self.num_arms - 1)
            mapping[barcode] = arm

        # At least one barcode for every arm gets guaranteed
        unique_guarantees = random.sample(barcode_list, self.num_arms)
        for arm, guarantee in enumerate(unique_guarantees):
            mapping[guarantee] = arm

        return mapping

    def generate_trials_info(self, mapping):

        """
        LSTM Input Format:
        Trial is a sequence of X one hot encoded pulls indicating the pulled arm
        [1001010] -> human reads this as [100, barcode2 as 101, 0] would be one pull in one trial for barcode2
        this would be a pull on arm0, and based on the mapping of barcode2, returns a reward of 0

        one episode is a sequence of 10 trials drawn for a single barcode instance from barcode bag
        one epoch is the full contents of barcode bag
        """
        # For loss calculations in the embedding model
        self.sorted_bcs = sorted(list(mapping.keys()))

        # Create the trial sample bag with num_barcode instances of each unique barcode
        # 4 unique barcodes -> 16 total barcodes in bag, 4 copies of each unique barcode
        trial_barcode_bag = []
        for barcode in mapping:
            for _ in range(self.num_barcodes):
                trial_barcode_bag.append(barcode)
        random.shuffle(trial_barcode_bag)

        observations = np.zeros(
            (self.num_barcodes**2, self.pulls_per_episode, self.num_arms)
        )
        rewards = np.zeros((self.num_barcodes**2, self.pulls_per_episode, 1))
        barcodes = np.zeros(
            (self.num_barcodes**2, self.pulls_per_episode, len(self.cluster_lists[0][0]))
        )
        barcodes_strings = np.zeros(
            (self.num_barcodes**2, self.pulls_per_episode, 1), dtype=object
        )
        barcodes_id = np.zeros((self.num_barcodes**2, 1))
        arms_id = np.zeros((self.num_barcodes**2, 1))

        for episode_num, barcode in enumerate(trial_barcode_bag):
            lstm_inputs, pre_comp_tensors = self.generate_one_episode(barcode, mapping)
            (
                observations[episode_num],
                rewards[episode_num],
                barcodes[episode_num],
            ) = lstm_inputs
            (
                barcodes_strings[episode_num],
                barcodes_id[episode_num],
                arms_id[episode_num],
            ) = pre_comp_tensors

        return observations, rewards, barcodes, barcodes_strings, barcodes_id, arms_id

    def generate_one_episode(self, barcode, mapping):
        """
        Create a single series of pulls, with rewards specified under the input barcode
        Args:
            barcode (String): Context Label for Arm ID
            mapping (Dict(String -> Int)): What arm is best for a barcode

        Returns:
            trial_pulls (Numpy Array): Contains all distinct arm pulls in order
            trial_rewards (Numpy Array): All rewards for arm pulls under input barcode
            bar_ar (Numpy Array): Input barcode tiled to match length of trial pulls
            bar_strings (Numpy Array): Input barcode as string for easier referencing later in program
            bar_id (Numpy Array): Sorted ID's for Embedder loss calculations
        """
        # Generate arm pulling sequence for single episode
        # Creates an Arms X Pulls matrix, using np.eye to onehotencode arm pulls
        trial_pulls = np.eye(self.num_arms)[
            np.random.choice(self.num_arms, self.pulls_per_episode)
        ]

        # Get reward for trial pulls
        # First pull barcode from mapping to ID good arm
        best_arm = mapping[barcode]
        trial_rewards = np.zeros((self.pulls_per_episode, 1), dtype=np.float32)
        pull_num, arm_chosen = np.where(trial_pulls == 1)

        # Good arm has 90% chance of reward, all others have 10% chance
        for pull, arm in zip(pull_num, arm_chosen):
            if self.perfect_info == False:
                if arm == best_arm:
                    reward = int(np.random.random() < 0.9)
                else:
                    reward = int(np.random.random() < 0.1)

            # Deterministic Arm Rewards (for debugging purposes)
            else:  # self.perfect_info == True
                reward = int(arm == best_arm)

            trial_rewards[pull] = float(reward)

        # Tile the barcode for all pulls in the episode
        bar_strings = np.zeros((self.pulls_per_episode, 1), dtype=object)
        bar_ar = np.zeros((self.pulls_per_episode, len(self.cluster_lists[0][0])))
        for num in range(self.pulls_per_episode):
            bar_strings[num] = barcode
            for id, val in enumerate(barcode):
                bar_ar[num][id] = int(val)

        bar_id = self.sorted_bcs.index(barcode)

        lstm_inputs = trial_pulls, trial_rewards, bar_ar
        pre_comp_tensors = bar_strings, bar_id, best_arm

        return lstm_inputs, pre_comp_tensors


def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.as_tensor(np_array).type(pth_dtype)
