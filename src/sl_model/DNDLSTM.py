"""
A DND-based LSTM based on ...
Ritter, et al. (2018).
Been There, Done That: Meta-Learning with Episodic Recall.
Proceedings of the International Conference on Machine Learning (ICML).
"""
import torch
import torch.nn as nn
from sl_model.DND import DND
from sl_model.A2C import A2C, A2C_linear

# constants
# N_GATES = 4


class DNDLSTM(nn.Module):
    def __init__(
        self,
        dim_input_lstm,
        dim_hidden_lstm,
        dim_output_lstm,
        dict_len,
        exp_settings,
        device,
        bias=True,
    ):
        super(DNDLSTM, self).__init__()
        self.input_dim = dim_input_lstm
        self.dim_hidden_lstm = dim_hidden_lstm
        self.dim_hidden_a2c = exp_settings["dim_hidden_a2c"]
        self.bias = bias
        self.device = device
        self.exp_settings = exp_settings
        self.unsure_bc_guess = torch.Tensor([0.0]).to(self.device)
        self.barcode_id = torch.Tensor([-1]).to(self.device)
        self.N_GATES = 4

        # input-hidden weights
        self.i2h = nn.Linear(
            dim_input_lstm,
            (self.N_GATES + 1) * dim_hidden_lstm,
            bias=bias,
            device=self.device,
        )

        # hidden-hidden weights
        self.h2h = nn.Linear(
            dim_hidden_lstm,
            (self.N_GATES + 1) * dim_hidden_lstm,
            bias=bias,
            device=self.device,
        )
        
        # dnd
        self.dnd = DND(dict_len, dim_hidden_lstm, exp_settings, self.device)
        # policy
        self.a2c = A2C(
            dim_hidden_lstm, self.dim_hidden_a2c, dim_output_lstm, device=self.device
        )
        # self.a2c = A2C_linear(dim_hidden_lstm, dim_output_lstm, device = self.device)

        # For some reason, if this is activated, the Embedder never learns, even though the embedder layers arent touched by this
        # init
        # self.reset_parameter()

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            # print(name)
            if "weight" in name:
                torch.nn.init.orthogonal_(wts)
            elif "bias" in name:
                torch.nn.init.constant_(wts, 0)

    def forward(self, obs_bar_reward, barcode_string, barcode_string_noised, barcode_tensor, barcode_id, h, c):

        # Into LSTM
        x_t = obs_bar_reward

        # Used for memory search/storage (non embedder versions)
        if self.exp_settings["mem_store"] != "embedding":
            if self.exp_settings["mem_store"] == "context":
                q_t = barcode_tensor
            elif self.exp_settings["mem_store"] == "hidden":
                q_t = h

            # Store hidden states in memory for t-SNE later, but not used in L2RL calculations
            elif self.exp_settings["mem_store"] == "L2RL_context":
                q_t = h
            elif self.exp_settings["mem_store"] == "L2RL_base":
                q_t = h

            else:
                raise ValueError("Incorrect mem_store type")

        # transform the input info
        Wx = self.i2h(x_t)
        Wh = self.h2h(h)
        preact = Wx + Wh

        # get all gate values
        gates = preact[:, : self.N_GATES * self.dim_hidden_lstm].sigmoid()

        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, : self.dim_hidden_lstm]
        i_t = gates[:, self.dim_hidden_lstm : 2 * self.dim_hidden_lstm]
        o_t = gates[:, 2 * self.dim_hidden_lstm : 3 * self.dim_hidden_lstm]
        r_t = gates[:, -self.dim_hidden_lstm :]

        # stuff to be written to cell state
        c_t_new = preact[:, self.N_GATES * self.dim_hidden_lstm :].tanh()

        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_t_new)

        if "L2RL" in self.exp_settings["mem_store"]:
            m_t = torch.zeros_like(h, device=self.device)
            predicted_barcode = "0" * self.exp_settings["barcode_size"]
            best_mem_id = 0

        else:
            if self.exp_settings["mem_store"] == "embedding":
                # Freeze all LSTM Layers before getting memory
                layers = [self.i2h, self.h2h, self.a2c]

                # Freeze K-Means Embedder for Contrastive eval
                if self.exp_settings['emb_loss'] == 'contrastive' and self.exp_settings['switch_to_contrastive']:
                    layers.extend([self.dnd.embedder.LSTM, self.dnd.embedder.l2i])

                for layer in layers:
                    for name, param in layer.named_parameters():
                        param.requires_grad = False

                # Use K-Means Clustering to ID pseudo labels for barcodes based on inputs to LSTM
                if self.exp_settings['emb_loss'] == 'kmeans' or self.exp_settings['emb_loss'] == 'contrastive':
                    if len(self.dnd.barcode_guesses) > 0:
                        barcode_sims = torch.nn.functional.cosine_similarity(obs_bar_reward, self.dnd.barcode_guesses)
                        self.barcode_id = torch.argmax(barcode_sims).view(1)
                        self.unsure_bc_guess += max(barcode_sims)

                mem, predicted_barcode, best_mem_id = self.dnd.get_memory(
                    c_t, barcode_string, barcode_id, barcode_string_noised
                )
                m_t = mem.tanh()

                # Unfreeze LSTM
                for layer in layers:
                    for name, param in layer.named_parameters():
                        param.requires_grad = True

            else:  # mem_store == context or hidden
                mem, predicted_barcode, best_mem_id = self.dnd.get_memory_non_embedder(
                    q_t
                )
                m_t = mem.tanh()

            # gate the memory; in general, can be any transformation of it
            if self.exp_settings['emb_with_mem'] or self.exp_settings['epochs'] > 0:
                c_t = c_t + torch.mul(r_t, m_t)

        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, c_t.tanh())

        # Store the most updated hidden state in memory for future use/t-sne
        if (
            self.exp_settings["mem_store"] == "hidden"
            or self.exp_settings["mem_store"] == "L2RL_context"
            or self.exp_settings["mem_store"] == "L2RL_base"
        ):
            q_t = h_t

        # Saving memory happens once at the end of every episode
        if not self.dnd.encoding_off:
            if self.exp_settings["mem_store"] == "embedding":

                # Saving Memory (hidden state passed into embedder, embedding is key and c_t is val)
                # This doesn't save the hidden state, just used as a reminder
                self.dnd.save_memory(h_t, c_t)

            else:
                self.dnd.save_memory_non_embedder(q_t, barcode_string, barcode_string_noised, c_t)

        # policy
        pi_a_t, v_t, entropy = self.a2c.forward(c_t)
        # pick an action
        a_t, prob_a_t = self.pick_action(pi_a_t)

        # fetch activity
        output = (a_t, predicted_barcode, prob_a_t, v_t, entropy, h_t, c_t)
        cache = (f_t, i_t, o_t, r_t, m_t, self.barcode_id, best_mem_id)

        return output, cache

    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.

        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)

        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)
        """
        m = torch.distributions.Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t

    def get_init_states(self, scale=0.1):
        h_0 = torch.randn(1, self.dim_hidden_lstm, device=self.device) * scale
        c_0 = torch.randn(1, self.dim_hidden_lstm, device=self.device) * scale
        return h_0, c_0

    def flush_trial_buffer(self):
        self.dnd.trial_buffer = [()]

    def turn_off_encoding(self):
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()

    def get_all_mems_embedder(self):
        mem_keys = self.dnd.keys
        predicted_mapping_to_keys = self.dnd.key_context_map
        return mem_keys, predicted_mapping_to_keys
