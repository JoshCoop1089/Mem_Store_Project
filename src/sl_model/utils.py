import torch
import numpy as np
from torch.nn.functional import mse_loss

"""helpers"""
eps = np.finfo(np.float32).eps.item()


def compute_returns(rewards, device, gamma=0, normalize=False):
    """compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 1d array
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    1d torch.tensor
        the sequence of cumulative return

    """
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, device=device)
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def get_reward(a_t, a_t_targ):
    """define the reward function at time t

    Parameters
    ----------
    a_t : int
        action
    a_t_targ : int
        target action

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    if a_t == a_t_targ:
        r_t = 1
    else:
        r_t = 0
    return torch.tensor(r_t).type(torch.FloatTensor).data


def get_reward_from_assumed_barcode(
    a_t, assumed_barcode, mapping, device, perfect_info=False
):
    """
    Once the A2C Policy predicts an action, determine the reward for that action under a certain barcode

    Args:
        a_t (Tensor): Arm chosen by A2C policy
        assumed_barcode (String): Predicted context taken from memory of LSTM
        mapping (Dict (String->Int)): What arm is best for every barcode
        device (torch.device): CPU or GPU location for tensors
        perfect_info (bool, optional): Whether the arms are deterministic (Only right arm would give reward, no chance otherwise). Defaults to False.

    Returns:
        Tensor: Reward calculated for arm pull under assumed barcode
    """
    try:
        best_arm = torch.tensor(mapping[assumed_barcode], device=device)
        if perfect_info == False:
            if torch.equal(a_t, best_arm):
                reward = float(np.random.random() < 0.9)
            else:
                reward = float(np.random.random() < 0.1)

        # Deterministic Arm Rewards (for debugging purposes)
        else:  # perfect_info == True
            reward = float(torch.equal(a_t, best_arm))

    # Empty barcode returns for the first episode of an epoch because there is nothing in memory
    except Exception as e:
        reward = 0.0

    return torch.tensor(reward, device=device)


def compute_a2c_loss(probs, values, returns, entropy):
    """compute the objective node for policy/value networks

    Parameters
    ----------
    probs : list
        action prob at time t
    values : list
        state value at time t
    returns : list
        return at time t

    Returns
    -------
    torch.tensor, torch.tensor
        Description of returned object.

    """
    policy_grads, value_losses = [], []
    for prob_t, v_t, R_t in zip(probs, values, returns):
        A_t = R_t - v_t.item()
        policy_grads.append(-prob_t * A_t)
        value_losses.append(mse_loss(torch.squeeze(v_t), torch.squeeze(R_t)))
    loss_policy = torch.stack(policy_grads).sum()
    loss_value = torch.stack(value_losses).sum()
    entropies = torch.stack(entropy).sum()
    return loss_policy, loss_value, entropies

def vectorize_cos_sim(input1, input2, device, same=False):
    """Take two batches of vectors and find the dot product between normalized versions of each pair of vectors

    Args:
        input1 (torch.tensor): First group of vectors (1xMxN)
        input2 (torch.tensor): Second group of vectors(1xMxN)
        device (torch.device): where to store the tensors
        same (bool, optional): Are you passing in the same group of vectors. Defaults to False.

    Returns:
        _type_: _description_
    """

    norm1 = torch.linalg.norm(input1, dim=1, ord=2).to(
        device).reshape(input1.shape[0], 1)
    norm2 = norm1.clone()
    input1_normed = input1/norm1
    dot = input1_normed@input1_normed.t()
    if not same:
        norm2 = torch.linalg.norm(input2, dim=1, ord=2).to(
            device).reshape(input2.shape[0], 1)
        input2_normed = input2/norm2
        dot = input1_normed@input2_normed.t()

    s = torch.ones_like(dot, device=device)
    for i in range(input1.size()[0]):
        for j in range(input1.size()[0]):
            s[i][j] = norm1[i].item()*norm2[j].item()
    x = torch.div(dot, s)
    return dot

"""https://github.com/galidor/PyTorchPartialLayerFreezing/blob/main/partial_freezing.py

For freezing a portion of a single linear layer during updates, and not having to split the model"""
def freeze_linear_params(layer, weight_indices, bias_indices=None, weight_hook_handle=None, bias_hook_handle=None):
    if weight_hook_handle is not None:
        weight_hook_handle.remove()
    if bias_hook_handle is not None:
        bias_hook_handle.remove()

    if (weight_indices == [] or weight_indices is None) and (bias_indices == [] or bias_indices is None):
        return

    if bias_indices is None:
        bias_indices = weight_indices

    if not isinstance(layer, torch.nn.Linear):
        raise ValueError("layer must be a valid Linear layer")

    if max(weight_indices) >= layer.weight.shape[0]:
        raise IndexError(
            "weight_indices must be less than the number output channels")

    if layer.bias is not None:
        if max(bias_indices) >= layer.bias.shape[0]:
            raise IndexError(
                "bias_indices must be less than the number output channels")

    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier

    def freezing_hook_bias_full(grad, bias_multiplier):
        return grad * bias_multiplier

    weight_multiplier = torch.ones(
        layer.weight.shape[0]).to(layer.weight.device)
    weight_multiplier[weight_indices] = 0
    weight_multiplier = weight_multiplier.view(-1, 1)

    def freezing_hook_weight(grad): return freezing_hook_weight_full(
        grad, weight_multiplier)
    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)

    if layer.bias is not None:
        bias_multiplier = torch.ones(
            layer.weight.shape[0]).to(layer.bias.device)
        bias_multiplier[bias_indices] = 0

        def freezing_hook_bias(grad): return freezing_hook_bias_full(
            grad, bias_multiplier)
        bias_hook_handle = layer.bias.register_hook(freezing_hook_bias)
    else:
        bias_hook_handle = None

    return weight_hook_handle, bias_hook_handle